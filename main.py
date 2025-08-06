import sys
import os
import time
import threading
import logging
import pytz
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import re
import traceback
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
import joblib
from tensorflow.keras.models import load_model
import queue
from collections import defaultdict

# ========================
# CONSTANTS & CONFIG
# ========================
NY_TZ = pytz.timezone("America/New_York")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Oanda configuration
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
API_KEY = os.getenv("OANDA_API_KEY")

# Instrument and timeframe configuration
INSTRUMENT = "XAU_USD"
TIMEFRAME = "M5"  # Can be "M5" or "M15"
CANDLE_COUNT = 201  # Exactly 201 candles
REALTIME_POLL_INTERVAL = 3 if TIMEFRAME == "M5" else 5  # More frequent checks for 5m
MIN_CANDLE_AGE_FOR_SIGNAL = 0.25 if TIMEFRAME == "M5" else 0.5  # 15s for 5m, 30s for 15m

# Model and scaler paths - timeframe specific
MODELS_DIR = "ml_models"
MODEL_5M = "5mbilstm_model.keras"
SCALER_5M = "scaler5mcrt.joblib"
MODEL_15M = "15mbilstm_model.keras"
SCALER_15M = "scaler15mcrt.joblib"

# Prediction threshold (0.9140 for class 1)
PREDICTION_THRESHOLD = 0.9140

# Global variables
GLOBAL_LOCK = threading.Lock()
CRT_SIGNAL_COUNT = 0
LAST_SIGNAL_TIME = 0
SIGNALS = []
REALTIME_DATA_QUEUE = queue.Queue()
MODEL = None
SCALER = None
SIGNAL_FOUND_THIS_CANDLE = False
NEXT_CANDLE_TIME = None
SCAN_COUNT_THIS_CANDLE = 0

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Oanda API
oanda_api = API(access_token=API_KEY, environment="practice")

# Dictionary to store trades
active_trades = {}

# ========================
# UTILITY FUNCTIONS
# ========================
def parse_oanda_time(time_str):
    """Parse Oanda's timestamp with variable fractional seconds"""
    logger.debug(f"Parsing Oanda timestamp: {time_str}")
    try:
        if '.' in time_str and len(time_str.split('.')[1]) > 7:
            time_str = re.sub(r'\.(\d{6})\d+', r'.\1', time_str)
        result = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.utc).astimezone(NY_TZ)
        logger.debug(f"Parsed time: {result}")
        return result
    except Exception as e:
        logger.error(f"Error parsing time {time_str}: {str(e)}")
        return datetime.now(NY_TZ)

def send_telegram(message):
    """Send formatted message to Telegram with detailed error handling and retries"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set, skipping message")
        return False
        
    logger.info(f"Attempting to send Telegram message: {message}")
    if len(message) > 4000:
        message = message[:4000] + "... [TRUNCATED]"
    
    message = message.replace('_', '\_')  # Escape underscores for Markdown
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json={
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown'
            }, timeout=10)
            
            logger.info(f"Telegram response: {response.status_code} - {response.text}")
            
            if response.status_code == 200 and response.json().get('ok'):
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram error: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        except Exception as e:
            logger.error(f"Telegram connection failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
    logger.error(f"Failed to send Telegram message after {max_retries} attempts")
    return False

def fetch_candles(last_time=None):
    """Fetch exactly 201 candles for XAU_USD with full precision"""
    logger.info(f"Fetching {CANDLE_COUNT} candles for {INSTRUMENT} with timeframe {TIMEFRAME}")
    params = {
        "granularity": TIMEFRAME,
        "count": CANDLE_COUNT,
        "price": "M",  # Mid prices with full precision
        "alignmentTimezone": "America/New_York",  # Ensure proper time alignment
        "includeCurrent": True  # Include incomplete current candle
    }
    if last_time:
        params["from"] = last_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    sleep_time = 10
    max_attempts = 5
    
    for attempt in range(max_attempts):
        try:
            request = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
            logger.debug(f"API request: {request}")
            response = oanda_api.request(request)
            logger.debug(f"API response: {response}")
            candles = response.get('candles', [])
            
            if not candles:
                logger.warning("No candles received from Oanda")
                continue
            
            data = []
            for candle in candles:
                price_data = candle.get('mid', {})
                if not price_data:
                    logger.warning(f"Skipping candle with missing price data: {candle}")
                    continue
                
                try:
                    parsed_time = parse_oanda_time(candle['time'])
                    is_complete = candle.get('complete', False)
                    
                    data.append({
                        'time': parsed_time,
                        'open': float(price_data['o']),  # Full precision
                        'high': float(price_data['h']),  # Full precision
                        'low': float(price_data['l']),   # Full precision
                        'close': float(price_data['c']), # Full precision
                        'volume': int(candle.get('volume', 0)),
                        'complete': is_complete,
                        'is_current': not is_complete  # Mark incomplete candles
                    })
                except Exception as e:
                    logger.error(f"Error processing candle: {str(e)}")
            
            if not data:
                logger.warning("No valid candles found in response")
                continue
                
            df = pd.DataFrame(data).drop_duplicates(subset=['time'], keep='last')
            if last_time:
                df = df[df['time'] > last_time].sort_values('time')
            logger.info(f"Successfully fetched {len(df)} candles (including current)")
            return df
            
        except V20Error as e:
            if "rate" in str(e).lower():
                logger.warning(f"Rate limit hit, sleeping {sleep_time}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(sleep_time)
                sleep_time *= 2
            elif e.status == 502:
                wait_time = sleep_time * (2 ** attempt)
                logger.error(f"Oanda API error: 502 Bad Gateway (Attempt {attempt+1}/{max_attempts}), retrying in {wait_time}s")
                time.sleep(wait_time)
            else:
                logger.error(f"Oanda API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error fetching candles: {str(e)}")
            logger.error(traceback.format_exc())
    
    logger.error(f"Failed to fetch candles after {max_attempts} attempts")
    return pd.DataFrame()

# ========================
# FEATURE ENGINEER WITH VOLUME IMPUTATION
# ========================
class FeatureEngineer:
    def __init__(self):
        # Base features without minute dummies
        self.base_features = [
            'adj close', 'garman_klass_vol', 'rsi_20', 'bb_low', 'bb_mid', 'bb_high',
            'atr_z', 'macd_z', 'dollar_volume', 'ma_10', 'ma_100', 'vwap', 'vwap_std',
            'rsi', 'ma_20', 'ma_30', 'ma_40', 'ma_60', 'trend_strength_up',
            'trend_strength_down', 'sl_price', 'tp_price', 'prev_volume', 'sl_distance',
            'tp_distance', 'rrr', 'log_sl', 'prev_body_size', 'prev_wick_up',
            'prev_wick_down', 'is_bad_combo', 'price_div_vol', 'rsi_div_macd',
            'price_div_vwap', 'sl_div_atr', 'tp_div_atr', 'rrr_div_rsi',
            'day_Friday', 'day_Monday', 'day_Sunday', 'day_Thursday', 'day_Tuesday',
            'day_Wednesday', 'session_q1', 'session_q2', 'session_q3', 'session_q4',
            'rsi_zone_neutral', 'rsi_zone_overbought', 'rsi_zone_oversold',
            'rsi_zone_unknown', 'trend_direction_downtrend', 'trend_direction_sideways',
            'trend_direction_uptrend', 'crt_BUY', 'crt_SELL', 'trade_type_BUY',
            'trade_type_SELL', 'combo_flag_dead', 'combo_flag_fair', 'combo_flag_fine',
            'combo_flag2_dead', 'combo_flag2_fair', 'combo_flag2_fine'
        ]
        
        # Timeframe-specific minute closed features
        if TIMEFRAME == "M5":
            self.features = self.base_features + [
                'minutes,closed_0', 'minutes,closed_5', 'minutes,closed_10', 
                'minutes,closed_15', 'minutes,closed_20', 'minutes,closed_25', 
                'minutes,closed_30', 'minutes,closed_35', 'minutes,closed_40', 
                'minutes,closed_45', 'minutes,closed_50', 'minutes,closed_55'
            ]
        else:  # M15 timeframe
            self.features = self.base_features + [
                'minutes,closed_0', 'minutes,closed_15', 
                'minutes,closed_30', 'minutes,closed_45'
            ]
        
        # Features to shift
        self.shift_features = [
            'garman_klass_vol', 'rsi_20', 'bb_low', 'bb_mid', 'bb_high',
            'atr_z', 'macd_z', 'dollar_volume', 'ma_10', 'ma_100',
            'vwap', 'vwap_std', 'rsi', 'ma_20', 'ma_30', 'ma_40', 'ma_60',
            'trend_strength_up', 'trend_strength_down', 'prev_volume', 'body_size', 
            'wick_up', 'wick_down', 'prev_body_size', 'prev_wick_up', 'prev_wick_down', 
            'is_bad_combo', 'price_div_vol', 'rsi_div_macd', 'price_div_vwap', 
            'sl_div_atr', 'rrr_div_rsi', 'rsi_zone_neutral', 'rsi_zone_overbought', 
            'rsi_zone_oversold', 'rsi_zone_unknown', 'combo_flag_dead', 'combo_flag_fair',
            'combo_flag_fine', 'combo_flag2_dead', 'combo_flag2_fair', 'combo_flag2_fine'
        ]
        
        # Store historical volume data for imputation
        self.historical_volumes = defaultdict(list)

    def get_same_period_candles(self, df, current_time):
        """Get candles from same time period in previous days"""
        # Create time key (hour:minute)
        time_key = current_time.strftime('%H:%M')
        
        # If we have historical data, use it
        if time_key in self.historical_volumes and len(self.historical_volumes[time_key]) > 0:
            logger.debug(f"Using historical volume data for time: {time_key}")
            return self.historical_volumes[time_key]
        
        # Otherwise, build historical data from existing df
        logger.info(f"Building historical volume data for time: {time_key}")
        same_period = []
        
        # Only consider complete candles for historical data
        complete_df = df[df['complete'] == True]
        
        for idx, row in complete_df.iterrows():
            row_time = row['time']
            if row_time.strftime('%H:%M') == time_key:
                same_period.append(row['volume'])
        
        # Store for future use
        self.historical_volumes[time_key] = same_period
        return same_period

    def calculate_crt_signal(self, df):
        """Robust CRT signal calculation with validation using direct indexing"""
        logger.info("CRT signal calculation with validation")
        
        # Ensure we have at least 3 candles
        if len(df) < 3:
            logger.warning(f"Insufficient data: {len(df)} rows, need at least 3")
            return None, None
            
        # Create working copy with explicit index reset
        crt_df = df.tail(3).copy().reset_index(drop=True)
        
        # Verify we have exactly 3 candles
        if len(crt_df) < 3:
            logger.warning(f"Only {len(crt_df)} rows after slicing, need 3")
            return None, None
            
        # Verify chronological order
        if not crt_df['time'].is_monotonic_increasing:
            logger.error("Candles not in chronological order! Re-sorting...")
            crt_df = crt_df.sort_values('time').reset_index(drop=True)
        
        try:
            # CORRECTED CANDLE REFERENCES:
            c1 = crt_df.iloc[0]  # Reference candle (two candles back)
            c2 = crt_df.iloc[1]  # Breakout candle (previous candle)
            c3 = crt_df.iloc[2]  # Current candle
            
            # Extract prices
            c1_low = c1['low']
            c1_high = c1['high']
            c2_low = c2['low']
            c2_high = c2['high']
            c2_close = c2['close']
            c3_open = c3['open']
            
            # Calculate candle metrics
            c2_range = c2_high - c2_low
            c2_mid = c2_low + (0.5 * c2_range)

            # Vectorized conditions with explicit validation
            buy_condition = (
                (c2_low < c1_low) and 
                (c2_close > c1_low) and 
                (c3_open > c2_mid)
            )

            sell_condition = (
                (c2_high > c1_high) and 
                (c2_close < c1_high) and 
                (c3_open < c2_mid)
            )
            
            # Extract signal for current candle
            if buy_condition:
                # Log detailed validation
                logger.info(f"✅ BUY VALIDATION| "
                            f"C2_Low:{c2_low:.5f} < C1_Low:{c1_low:.5f}| "
                            f"C2_Close:{c2_close:.5f} > C1_Low:{c1_low:.5f}| "
                            f"C3_Open:{c3_open:.5f} > C2_Mid:{c2_mid:.5f}")
                
                signal_type = 'BUY'
                entry = c3_open
                sl = c2_low
                risk = abs(entry - sl)
                tp = entry + 4 * risk
                logger.info(f"BUY signal validated")
                
            elif sell_condition:
                # Log detailed validation
                logger.info(f"✅ SELL VALIDATION| "
                            f"C2_High:{c2_high:.5f} > C1_High:{c1_high:.5f}| "
                            f"C2_Close:{c2_close:.5f} < C1_High:{c1_high:.5f}| "
                            f"C3_Open:{c3_open:.5f} < C2_Mid:{c2_mid:.5f}")
                
                signal_type = 'SELL'
                entry = c3_open
                sl = c2_high
                risk = abs(sl - entry)
                tp = entry - 4 * risk
                logger.info(f"SELL signal validated")
                
            else:
                # Log why no signal was detected
                logger.info("❌ No signal detected:")
                if not (c2_low < c1_low):
                    logger.info(f"  - C2_Low:{c2_low:.5f} >= C1_Low:{c1_low:.5f}")
                if not (c2_close > c1_low):
                    logger.info(f"  - C2_Close:{c2_close:.5f} <= C1_Low:{c1_low:.5f}")
                if not (c3_open > c2_mid):
                    logger.info(f"  - C3_Open:{c3_open:.5f} <= C2_Mid:{c2_mid:.5f}")
                if not (c2_high > c1_high):
                    logger.info(f"  - C2_High:{c2_high:.5f} <= C1_High:{c1_high:.5f}")
                if not (c2_close < c1_high):
                    logger.info(f"  - C2_Close:{c2_close:.5f} >= C1_High:{c1_high:.5f}")
                if not (c3_open < c2_mid):
                    logger.info(f"  - C3_Open:{c3_open:.5f} >= C2_Mid:{c2_mid:.5f}")
                    
                return None, None
            
            logger.info(f"Detected signal: {signal_type} at {c3['time']}")
            return signal_type, {'entry': entry, 'sl': sl, 'tp': tp, 'time': c3['time']}
            
        except KeyError as e:
            logger.error(f"Missing price data in candle: {str(e)}")
            return None, None

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators with volume imputation"""
        logger.info("Calculating technical indicators with volume imputation")
        df = df.copy().drop_duplicates(subset=['time'], keep='last')
        
        # Apply volume imputation to incomplete candles
        if not df.empty and not df.iloc[-1]['complete']:
            current_candle = df.iloc[-1]
            current_time = current_candle['time']
            
            # Get historical volumes for same time period
            same_period_volumes = self.get_same_period_candles(df, current_time)
            
            if len(same_period_volumes) > 0:
                # Calculate average volume for this time period
                avg_volume = np.mean(same_period_volumes)
                
                # Get current volume
                current_volume = current_candle['volume']
                
                # Calculate volume ratio
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                # Cap the ratio to avoid extreme values
                volume_ratio = min(3.0, max(0.1, volume_ratio))
                
                # Estimate final volume
                estimated_volume = avg_volume * volume_ratio
                
                logger.info(f"Volume imputation: Current={current_volume}, "
                            f"Avg={avg_volume:.2f}, Ratio={volume_ratio:.2f}, "
                            f"Estimated={estimated_volume:.2f}")
                
                # Apply imputation
                df.at[df.index[-1], 'volume'] = estimated_volume
        
        # Continue with indicator calculations
        df['adj close'] = df['open']
        logger.debug("Adjusted close calculated")
        
        df['garman_klass_vol'] = (
            ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 -
            (2 * np.log(2) - 1) * ((np.log(df['adj close']) - np.log(df['open'])) ** 2)
        )
        logger.debug("Garman-Klass volatility calculated")
        
        df['rsi_20'] = ta.rsi(df['adj close'], length=20)
        df['rsi'] = ta.rsi(df['close'], length=14)
        logger.debug("RSI calculated")
        
        bb = ta.bbands(np.log1p(df['adj close']), length=20)
        df['bb_low'] = bb['BBL_20_2.0']
        df['bb_mid'] = bb['BBM_20_2.0']
        df['bb_high'] = bb['BBU_20_2.0']
        logger.debug("Bollinger Bands calculated")
        
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_z'] = (atr - atr.mean()) / atr.std()
        logger.debug("ATR z-score calculated")
        
        macd = ta.macd(df['adj close'], fast=12, slow=26, signal=9)
        df['macd_z'] = (macd['MACD_12_26_9'] - macd['MACD_12_26_9'].mean()) / macd['MACD_12_26_9'].std()
        logger.debug("MACD z-score calculated")
        
        df['dollar_volume'] = (df['adj close'] * df['volume']) / 1e6
        logger.debug("Dollar volume calculated")
        
        df['ma_10'] = df['adj close'].rolling(window=10).mean()
        df['ma_100'] = df['adj close'].rolling(window=100).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()
        df['ma_40'] = df['close'].rolling(window=40).mean()
        df['ma_60'] = df['close'].rolling(window=60).mean()
        logger.debug("Moving averages calculated")
        
        vwap_num = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
        vwap_den = df['volume'].cumsum()
        df['vwap'] = vwap_num / vwap_den
        df['vwap_std'] = df['vwap'].rolling(window=20).std()
        logger.debug("VWAP and VWAP STD calculated")
        
        return df

    def calculate_trade_features(self, df, signal_type, entry):
        logger.info(f"Calculating trade features for signal_type: {signal_type}")
        df = df.copy()
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row
        logger.debug(f"Last row: {last_row}, Prev row: {prev_row}")
        
        if signal_type == 'SELL':
            df['sl_price'] = prev_row['high']
            risk = abs(entry - df['sl_price'].iloc[-1])
            df['tp_price'] = entry - 4 * risk
        else:  # BUY
            df['sl_price'] = prev_row['low']
            risk = abs(entry - df['sl_price'].iloc[-1])
            df['tp_price'] = entry + 4 * risk
        logger.debug(f"SL: {df['sl_price'].iloc[-1]}, TP: {df['tp_price'].iloc[-1]}")
        
        df['sl_distance'] = abs(entry - df['sl_price']) * 10
        df['tp_distance'] = abs(df['tp_price'] - entry) * 10
        df['rrr'] = df['tp_distance'] / df['sl_distance'].replace(0, np.nan)
        df['log_sl'] = np.log1p(df['sl_price'])
        logger.debug(f"SL Distance: {df['sl_distance'].iloc[-1]}, TP Distance: {df['tp_distance'].iloc[-1]}, RRR: {df['rrr'].iloc[-1]}")
        
        return df

    def calculate_categorical_features(self, df):
        logger.info("Calculating categorical features")
        df = df.copy()
        
        df['day'] = df['time'].dt.day_name()
        all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']
        for day in all_days:
            df[f'day_{day}'] = 0
        today = datetime.now(NY_TZ).strftime('%A')
        df[f'day_{today}'] = 1
        logger.debug(f"Day dummies set for today: {today}")
        
        def get_session(hour):
            if 0 <= hour < 6:
                return 'q2'
            elif 6 <= hour < 12:
                return 'q3'
            elif 12 <= hour < 18:
                return 'q4'
            else:
                return 'q1'
        df['session'] = df['time'].dt.hour.apply(get_session)
        df = pd.get_dummies(df, columns=['session'], prefix='session', drop_first=False)
        logger.debug("Session dummies created")
        
        def rsi_zone(rsi):
            if pd.isna(rsi):
                return 'unknown'
            elif rsi < 30:
                return 'oversold'
            elif rsi > 70:
                return 'overbought'
            else:
                return 'neutral'
        df['rsi_zone'] = df['rsi'].apply(rsi_zone)
        df = pd.get_dummies(df, columns=['rsi_zone'], prefix='rsi_zone', drop_first=False)
        logger.debug("RSI zone dummies created")
        
        def is_bullish_stack(row):
            return int(row['ma_20'] > row['ma_30'] > row['ma_40'] > row['ma_60'])
        def is_bearish_stack(row):
            return int(row['ma_20'] < row['ma_30'] < row['ma_40'] < row['ma_60'])
        
        df['trend_strength_up'] = df.apply(is_bullish_stack, axis=1).astype(float)
        df['trend_strength_down'] = df.apply(is_bearish_stack, axis=1).astype(float)
        logger.debug("Trend strength calculated")
        
        def get_trend(row):
            if row['trend_strength_up'] > row['trend_strength_down']:
                return 'uptrend'
            elif row['trend_strength_down'] > row['trend_strength_up']:
                return 'downtrend'
            else:
                return 'sideways'
        df['trend_direction'] = df.apply(get_trend, axis=1)
        df = pd.get_dummies(df, columns=['trend_direction'], prefix='trend_direction', drop_first=False)
        logger.debug("Trend direction dummies created")
        
        return df

    def calculate_minutes_closed(self, df, minutes_closed):
        logger.info(f"Calculating minutes closed: {minutes_closed}")
        df = df.copy()
        
        # Create minute bucket columns based on timeframe
        if TIMEFRAME == "M5":
            minute_buckets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
            minute_cols = [f'minutes,closed_{bucket}' for bucket in minute_buckets]
        else:  # M15 timeframe
            minute_buckets = [0, 15, 30, 45]
            minute_cols = [f'minutes,closed_{bucket}' for bucket in minute_buckets]
            
        # Initialize all to 0
        for col in minute_cols:
            df[col] = 0
        
        # Find the appropriate bucket
        current_bucket = (minutes_closed // (5 if TIMEFRAME == "M5" else 15)) * (5 if TIMEFRAME == "M5" else 15)
        max_bucket = max(minute_buckets)
        if current_bucket > max_bucket:
            current_bucket = max_bucket
            
        # Set the corresponding bucket to 1
        bucket_col = f'minutes,closed_{current_bucket}'
        if bucket_col in df.columns:
            df[bucket_col] = 1
            logger.debug(f"Minutes closed set: {bucket_col} = 1")
        else:
            logger.error(f"Invalid minute bucket: {current_bucket}")
            
        return df

    def generate_features(self, df, signal_type, minutes_closed):
        logger.info(f"Generating features for signal_type: {signal_type}, minutes closed: {minutes_closed}")
        if len(df) < 200:
            logger.warning(f"Insufficient data: {len(df)} rows, need 200")
            return None
        
        df = df.tail(200).copy()
        df = self.calculate_technical_indicators(df)
        df = self.calculate_trade_features(df, signal_type, df.iloc[-1]['open'])
        df = self.calculate_categorical_features(df)
        df = self.calculate_minutes_closed(df, minutes_closed)
        
        df['prev_volume'] = df['volume'].shift(1)
        df['body_size'] = abs(df['close'] - df['open'])
        df['wick_up'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['wick_down'] = df[['close', 'open']].min(axis=1) - df['low']
        df['prev_body_size'] = df['body_size'].shift(1)
        df['prev_wick_up'] = df['wick_up'].shift(1)
        df['prev_wick_down'] = df['wick_down'].shift(1)
        logger.debug("Candle and volume features calculated")
        
        df['price_div_vol'] = df['adj close'] / (df['garman_klass_vol'] + 1e-6)
        df['rsi_div_macd'] = df['rsi'] / (df['macd_z'] + 1e-6)
        df['price_div_vwap'] = df['adj close'] / (df['vwap'] + 1e-6)
        df['sl_div_atr'] = df['sl_distance'] / (df['atr_z'] + 1e-6)
        df['tp_div_atr'] = df['tp_distance'] / (df['atr_z'] + 1e-6)
        df['rrr_div_rsi'] = df['rrr'] / (df['rsi'] + 1e-6)
        logger.debug("Derived metrics calculated")
        
        combo_key = f"{df['rsi'].iloc[-1]:.2f}_{df['macd_z'].iloc[-1]:.2f}_{df['atr_z'].iloc[-1]:.2f}"
        logger.debug(f"Combo key calculated: {combo_key}")
        combo_flags = {'combo_flag_dead': 0, 'combo_flag_fair': 0, 'combo_flag_fine': 0}
        combo_flags2 = {'combo_flag2_dead': 0, 'combo_flag2_fair': 0, 'combo_flag2_fine': 0}
        if df['rsi'].iloc[-1] < 30 or df['macd_z'].iloc[-1] < -1:
            combo_flags['combo_flag_dead'] = 1
            combo_flags2['combo_flag2_dead'] = 1
        elif df['rsi'].iloc[-1] > 70 or df['macd_z'].iloc[-1] > 1:
            combo_flags['combo_flag_fine'] = 1
            combo_flags2['combo_flag2_fine'] = 1
        else:
            combo_flags['combo_flag_fair'] = 1
            combo_flags2['combo_flag2_fair'] = 1
        for flag, value in combo_flags.items():
            df[flag] = value
        for flag, value in combo_flags2.items():
            df[flag] = value
        logger.debug(f"Combo flags set: {combo_flags}, {combo_flags2}")
        
        df['is_bad_combo'] = 1 if combo_flags['combo_flag_dead'] == 1 else 0
        logger.debug(f"is_bad_combo set to: {df['is_bad_combo'].iloc[-1]}")
        
        df['crt_BUY'] = int(signal_type == 'BUY')
        df['crt_SELL'] = int(signal_type == 'SELL')
        df['trade_type_BUY'] = int(signal_type == 'BUY')
        df['trade_type_SELL'] = int(signal_type == 'SELL')
        logger.debug("CRT and trade type encoding applied")
        
        features = pd.Series(index=self.features, dtype=float)
        for feat in self.features:
            if feat in df.columns:
                features[feat] = df[feat].iloc[-1]
            else:
                logger.warning(f"Feature {feat} not found, setting to 0")
                features[feat] = 0
        
        # Apply feature shifting - use previous candle's values for specific features
        if len(df) >= 2:
            prev_candle = df.iloc[-2]
            for feat in self.shift_features:
                if feat in features.index and feat in prev_candle:
                    features[feat] = prev_candle[feat]
                    logger.debug(f"Shifted feature {feat} to previous candle's value")
        
        # Handle missing values
        if features.isna().any():
            missing = features[features.isna()].index.tolist()
            logger.warning(f"Missing features: {missing}")
            for col in missing:
                features[col] = 0
                logger.warning(f"Filled missing feature {col} with default value 0")
        
        logger.info("Feature generation completed successfully")
        return features

# ========================
# REAL-TIME DETECTOR
# ========================
class RealTimeDetector:
    def __init__(self, detector):
        self.detector = detector
        self.current_candle_time = None
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        logger.info("Real-time detector started")

    def run(self):
        """Optimized scanning with sleep management"""
        global SCAN_COUNT_THIS_CANDLE, SIGNAL_FOUND_THIS_CANDLE, NEXT_CANDLE_TIME
        
        while self.running:
            try:
                current_time = datetime.now(NY_TZ)
                
                # Skip if no data
                if self.detector.data.empty:
                    time.sleep(REALTIME_POLL_INTERVAL)
                    continue
                    
                # Get latest candle safely
                with GLOBAL_LOCK:
                    if self.detector.data.empty:
                        continue
                    latest_candle = self.detector.data.iloc[-1].copy()
                
                # Reset state for new candle
                if self.current_candle_time != latest_candle['time']:
                    self.current_candle_time = latest_candle['time']
                    SCAN_COUNT_THIS_CANDLE = 0
                    SIGNAL_FOUND_THIS_CANDLE = False
                    logger.info(f"New candle detected at {self.current_candle_time}, resetting scan count")
                
                # Sleep until next candle if signal found
                if SIGNAL_FOUND_THIS_CANDLE and NEXT_CANDLE_TIME:
                    sleep_seconds = (NEXT_CANDLE_TIME - current_time).total_seconds()
                    if sleep_seconds > 0:
                        logger.info(f"Signal found, sleeping {sleep_seconds:.1f}s until next candle")
                        time.sleep(sleep_seconds)
                    continue
                
                # Sleep if no signal after 2 scans
                if SCAN_COUNT_THIS_CANDLE >= 2 and not SIGNAL_FOUND_THIS_CANDLE and NEXT_CANDLE_TIME:
                    sleep_seconds = (NEXT_CANDLE_TIME - current_time).total_seconds()
                    if sleep_seconds > 0:
                        logger.info(f"No signal after 2 scans, sleeping {sleep_seconds:.1f}s until next candle")
                        time.sleep(sleep_seconds)
                    continue
                
                # Only scan if candle is ready and we have scans remaining
                candle_age = (current_time - latest_candle['time']).total_seconds() / 60.0
                if latest_candle['is_current'] and candle_age >= MIN_CANDLE_AGE_FOR_SIGNAL and SCAN_COUNT_THIS_CANDLE < 2:
                    logger.info(f"Scanning candle (scan {SCAN_COUNT_THIS_CANDLE+1}/2)")
                    self.detector.process_signals(0, pd.DataFrame([latest_candle]))
                    SCAN_COUNT_THIS_CANDLE += 1
                
                time.sleep(REALTIME_POLL_INTERVAL)
                
            except Exception as e:
                logger.error(f"Real-time detector error: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(REALTIME_POLL_INTERVAL)

    def stop(self):
        self.running = False
        self.thread.join(timeout=5)
        logger.info("Real-time detector stopped")

# ========================
# TRADING DETECTOR
# ========================
class TradingDetector:
    def __init__(self):
        logger.info("Initializing TradingDetector")
        self.data = pd.DataFrame()
        self.feature_engineer = FeatureEngineer()
        self.candle_duration = 5 if TIMEFRAME == "M5" else 15
        self.scheduler = CandleScheduler(timeframe=self.candle_duration)
        self.last_signal_candle = None
        self.realtime_detector = None
        
        logger.info("Loading initial 201 candles")
        self.data = self.fetch_initial_candles()
        
        if self.data.empty or len(self.data) < 200:
            logger.error("Failed to load sufficient initial candles")
            raise RuntimeError("Initial candle fetch failed or insufficient data")
            
        logger.info(f"Initial data loaded with {len(self.data)} rows")
        self.scheduler.register_callback(self.process_signals)
        logger.info("Starting scheduler thread")
        self.scheduler.start()
        
        # Start real-time detector
        self.realtime_detector = RealTimeDetector(self)
        logger.info("TradingDetector initialized")

    def fetch_initial_candles(self):
        logger.info("Fetching initial 201 candles")
        for attempt in range(5):
            df = fetch_candles()
            if not df.empty and len(df) >= 200:
                logger.info(f"Successfully fetched {len(df)} initial candles")
                return df
            logger.warning(f"Attempt {attempt+1} failed, retrying in 10s")
            time.sleep(10)
        logger.error("Failed to fetch initial 201 candles after 5 attempts")
        return pd.DataFrame()

    def calculate_candle_age(self, current_time, candle_time):
        """Calculate age of the latest candle in minutes"""
        elapsed = (current_time - candle_time).total_seconds() / 60
        return min(self.candle_duration, max(0, elapsed))

    def _get_next_candle_time(self, current_time):
        """Calculate the next candle open time"""
        minute = current_time.minute
        if TIMEFRAME == "M5":
            remainder = minute % 5
            if remainder == 0:
                return current_time.replace(second=0, microsecond=0) + timedelta(minutes=5)
            next_minute = minute - remainder + 5
            if next_minute >= 60:
                return current_time.replace(hour=current_time.hour + 1, minute=0, second=0, microsecond=0)
            return current_time.replace(minute=next_minute, second=0, microsecond=0)
        else:  # M15 timeframe
            remainder = minute % 15
            if remainder == 0:
                return current_time.replace(second=0, microsecond=0) + timedelta(minutes=15)
            next_minute = minute - remainder + 15
            if next_minute >= 60:
                return current_time.replace(hour=current_time.hour + 1, minute=0, second=0, microsecond=0)
            return current_time.replace(minute=next_minute, second=0, microsecond=0)

    def update_data(self, df_new):
        """Update the data with new candles"""
        logger.info(f"Updating data with new dataframe of size {len(df_new)}")
        if df_new.empty:
            logger.warning("Received empty dataframe in update_data")
            return
        
        with GLOBAL_LOCK:
            if self.data.empty:
                self.data = df_new.dropna(subset=['time', 'open', 'high', 'low', 'close']).tail(201)
                logger.debug("Initialized data with new dataframe")
            else:
                last_existing_time = self.data['time'].max()
                new_data = df_new[df_new['time'] > last_existing_time]
                same_time_data = df_new[df_new['time'] == last_existing_time]
                
                # Update existing candle if time matches
                if not same_time_data.empty:
                    self.data = self.data[self.data['time'] < last_existing_time]
                    self.data = pd.concat([self.data, same_time_data])
                
                if not new_data.empty:
                    self.data = pd.concat([self.data, new_data])
                
                # Ensure we keep only the last 201 candles
                self.data = self.data.sort_values('time').reset_index(drop=True).tail(201)
                logger.debug(f"Combined data shape: {self.data.shape}, latest time: {self.data['time'].max()}, last 3 rows: {self.data.tail(3)}")

    def predict_single_model(self, features_df):
        """Predict using the single model for the current timeframe"""
        logger.info(f"Running prediction with {TIMEFRAME} model")
        
        # Get expected feature count for current timeframe
        expected_features = len(self.feature_engineer.features)
        
        # Validate input shape
        if features_df.shape[1] != expected_features:
            logger.error(f"Feature mismatch: Expected {expected_features} features, got {features_df.shape[1]}")
            logger.error("Feature names in generated data:")
            for i, feat in enumerate(features_df.columns):
                logger.error(f"{i+1}. {feat}")
                
            logger.error("Expected feature names:")
            for i, feat in enumerate(self.feature_engineer.features):
                logger.error(f"{i+1}. {feat}")
                
            return None, None
        
        try:
            # Convert to numpy and scale
            features_array = features_df.values
            scaled_features = SCALER.transform(features_array)
            
            # Reshape for LSTM (samples, timesteps, features)
            reshaped_features = np.expand_dims(scaled_features, axis=1)
            
            # Get prediction from the model
            prob = MODEL.predict(reshaped_features, verbose=0)[0][0]
            final_pred = 1 if prob >= PREDICTION_THRESHOLD else 0
            outcome = "Worth Taking" if final_pred == 1 else "Likely Loss"
            
            logger.info(f"Model prediction: Prob={prob:.6f}, Outcome={outcome}")
            return prob, outcome
        
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None

    def process_signals(self, minutes_closed, latest_candles):
        logger.info(f"Processing signals, minutes closed: {minutes_closed}, candles: {len(latest_candles)}")
        global SIGNAL_FOUND_THIS_CANDLE, NEXT_CANDLE_TIME, SCAN_COUNT_THIS_CANDLE
        
        if not latest_candles.empty:
            logger.info(f"Updating data with {len(latest_candles)} new candles")
            self.update_data(latest_candles)
            logger.debug(f"Updated data shape: {self.data.shape}, latest time: {self.data['time'].max()}, last 3 rows: {self.data.tail(3)}")
        else:
            logger.warning("No new candles, using existing data")

        if self.data.empty or len(self.data) < 3:
            logger.warning(f"Insufficient data: {len(self.data)} rows, need at least 3")
            return

        latest_candle_time = self.data.iloc[-1]['time']
        current_time = datetime.now(NY_TZ)
        candle_age = self.calculate_candle_age(current_time, latest_candle_time)
        logger.info(f"Candle age: {candle_age:.2f} minutes")
        
        # Calculate next candle time
        NEXT_CANDLE_TIME = self._get_next_candle_time(latest_candle_time)
        logger.info(f"Next candle time: {NEXT_CANDLE_TIME}")

        # CRT SIGNAL DETECTION (using candle 3 as current)
        signal_type, signal_data = self.feature_engineer.calculate_crt_signal(self.data)
        if signal_type and signal_data:
            current_candle = self.data.iloc[-1]
            # Check if this is a new signal based on candle time and price change
            if (self.last_signal_candle is None or 
                current_candle['time'] > self.last_signal_candle['time'] or 
                (current_candle['time'] == self.last_signal_candle['time'] and 
                 abs(current_candle['close'] - self.last_signal_candle['close']) > 0.5)):
                self.last_signal_candle = current_candle
                
                # Check if this signal matches an existing trade
                is_new_trade = True
                for trade_id, trade in list(active_trades.items()):
                    if trade['sl'] == signal_data['sl'] and trade.get('outcome') is None:
                        is_new_trade = False
                        logger.info(f"Matching SL found, skipping reprocessing for trade {trade_id}")
                        break
                
                if is_new_trade:
                    SIGNAL_FOUND_THIS_CANDLE = True
                    logger.info(f"New signal validated: {signal_type}")
                    alert_time = signal_data['time'].astimezone(NY_TZ)
                    setup_msg = (
                        f"🔔 *SETUP* {INSTRUMENT.replace('_','/')} {signal_type}\n"
                        f"Timeframe: {TIMEFRAME}\n"
                        f"Time: {alert_time.strftime('%Y-%m-%d %H:%M')} NY\n"
                        f"Entry: {signal_data['entry']:.5f}\n"
                        f"TP: {signal_data['tp']:.5f}\n"
                        f"SL: {signal_data['sl']:.5f}\n"
                        f"Candle Age: {candle_age:.2f} minutes"
                    )
                    if send_telegram(setup_msg):
                        features = self.feature_engineer.generate_features(self.data, signal_type, minutes_closed)
                        if features is not None:
                            feature_msg = f"📊 *FEATURES* {INSTRUMENT.replace('_','/')} {signal_type}\n"
                            formatted_features = []
                            for feat, val in features.items():
                                escaped_feat = feat.replace('_', '\\_')
                                formatted_features.append(f"{escaped_feat}: {val:.6f}")
                            feature_msg += "\n".join(formatted_features)
                            if not send_telegram(feature_msg):
                                logger.error("Failed to send features after retries")

                        if SCALER is not None and MODEL is not None:
                            features_df = pd.DataFrame([features], columns=self.feature_engineer.features)
                            
                            # Use single model prediction
                            prob, outcome = self.predict_single_model(features_df)
                            
                            if prob is not None:
                                pred_msg = f"🤖 *MODEL PREDICTION* {INSTRUMENT.replace('_','/')} {signal_type}\n"
                                pred_msg += f"Probability: {prob:.6f}\n"
                                pred_msg += f"Decision: {outcome}\n"
                                pred_msg += f"Model: {MODEL_5M if TIMEFRAME == 'M5' else MODEL_15M}"
                                
                                if not send_telegram(pred_msg):
                                    logger.error("Failed to send model prediction")
                                
                                # Store new trade with prediction
                                trade_id = f"{signal_type}_{current_time.timestamp()}"
                                active_trades[trade_id] = {
                                    'entry': signal_data['entry'],
                                    'sl': signal_data['sl'],
                                    'tp': signal_data['tp'],
                                    'time': current_time,
                                    'signal_time': signal_data['time'],  # Store signal candle time
                                    'prediction': prob,
                                    'outcome': None
                                }
                                logger.info(f"New trade stored: {trade_id} with prediction {prob:.6f}")
                        else:
                            logger.error("No scaler or model loaded")
            else:
                logger.debug("Signal skipped due to similar candle conditions")

        # TRADE OUTCOME CHECKING (only on completed candles)
        if len(self.data) > 0 and minutes_closed == self.candle_duration:
            latest_candle = self.data.iloc[-1]
            for trade_id, trade in list(active_trades.items()):
                if trade.get('outcome') is None:
                    entry, sl, tp = trade['entry'], trade['sl'], trade['tp']
                    logger.info(f"Checking outcome for trade {trade_id}: entry={entry}, sl={sl}, tp={tp}")
                    
                    # SELL trade: entry > sl
                    if entry > sl:
                        if latest_candle['high'] >= sl:
                            trade['outcome'] = 'Hit SL (Loss)'
                            logger.info(f"SELL trade {trade_id} outcome: Hit SL at {latest_candle['high']:.5f}")
                        elif latest_candle['low'] <= tp:
                            trade['outcome'] = 'Hit TP (Win)'
                            logger.info(f"SELL trade {trade_id} outcome: Hit TP at {latest_candle['low']:.5f}")
                    # BUY trade: entry < sl
                    else:
                        if latest_candle['low'] <= sl:
                            trade['outcome'] = 'Hit SL (Loss)'
                            logger.info(f"BUY trade {trade_id} outcome: Hit SL at {latest_candle['low']:.5f}")
                        elif latest_candle['high'] >= tp:
                            trade['outcome'] = 'Hit TP (Win)'
                            logger.info(f"BUY trade {trade_id} outcome: Hit TP at {latest_candle['high']:.5f}")
                    
                    # If outcome determined, send notification and remove trade
                    if trade.get('outcome'):
                        outcome_msg = (
                            f"📈 *Trade Outcome*\n"
                            f"Signal Time: {trade['signal_time'].strftime('%Y-%m-%d %H:%M')} NY\n"
                            f"Entry: {entry:.5f}\n"
                            f"SL: {sl:.5f}\n"
                            f"TP: {tp:.5f}\n"
                            f"Prediction: {trade['prediction']:.6f}\n"
                            f"Outcome: {trade['outcome']}\n"
                            f"Detected at: {current_time.strftime('%Y-%m-%d %H:%M')} NY"
                        )
                        if not send_telegram(outcome_msg):
                            logger.error(f"Failed to send outcome for trade {trade_id}")
                        # Remove the trade from active_trades
                        del active_trades[trade_id]

# ========================
# CANDLE SCHEDULER
# ========================
class CandleScheduler(threading.Thread):
    def __init__(self, timeframe=15):
        super().__init__(daemon=True)
        self.timeframe = timeframe
        self.callback = None
        self.active = True
        self.event = threading.Event()
    
    def register_callback(self, callback):
        self.callback = callback
        
    def calculate_next_candle(self):
        now = datetime.now(NY_TZ)
        current_minute = now.minute
        remainder = current_minute % self.timeframe
        if remainder == 0:
            return now.replace(second=0, microsecond=0) + timedelta(minutes=self.timeframe)
        next_minute = current_minute - remainder + self.timeframe
        if next_minute >= 60:
            return now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
        return now.replace(minute=next_minute, second=0, microsecond=0)
    
    def calculate_minutes_closed(self, latest_time):
        if latest_time is None:
            return 0
            
        logger.info(f"Calculating minutes closed for latest time: {latest_time}")
        now = datetime.now(NY_TZ)
        elapsed = (now - latest_time).total_seconds() / 60
        logger.debug(f"Elapsed time since last candle: {elapsed:.2f} minutes")
        max_closed = 4.9 if self.timeframe == 5 else 14.9
        return min(max_closed, max(0, elapsed))
    
    def run(self):
        logger.info("Starting CandleScheduler thread")
        while self.active:
            try:
                logger.info("Fetching latest candle data")
                df_candles = fetch_candles()
                
                if df_candles.empty:
                    logger.warning("No candles fetched, forcing callback with existing data")
                    if hasattr(self, 'data') and self.data is not None and len(self.data) >= 3:
                        latest_time = self.data['time'].max()
                        minutes_closed = self.calculate_minutes_closed(latest_time)
                        if self.callback:
                            logger.info(f"Forcing callback with minutes closed: {minutes_closed}")
                            self.callback(minutes_closed, self.data.tail(3))
                else:
                    # Filter to only complete candles
                    complete_candles = df_candles[df_candles['complete'] == True]
                    if not complete_candles.empty:
                        latest_candle = complete_candles.iloc[-1]
                        latest_time = latest_candle['time']
                        minutes_closed = self.calculate_minutes_closed(latest_time)
                        if self.callback:
                            logger.info(f"Calling callback with minutes closed: {minutes_closed}")
                            self.callback(minutes_closed, complete_candles.tail(1))
                    else:
                        logger.warning("No complete candles in fetched data")
                
                now = datetime.now(NY_TZ)
                next_run = self.calculate_next_candle()
                sleep_seconds = (next_run - now).total_seconds()
                logger.info(f"Sleeping {sleep_seconds:.1f} seconds until next candle")
                time.sleep(max(1, sleep_seconds))
                
            except Exception as e:
                logger.error(f"Scheduler error: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(60)

# ========================
# BOT INSTANCE CLASS
# ========================
class BotInstance:
    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.model = None
        self.scaler = None
        self.active_trades = {}
        self.detector = None
        self.logger = logging.getLogger(f"{__name__}.{timeframe}")
        
    def run(self):
        self.logger.info(f"Starting trading bot for {self.timeframe}")
        send_telegram(f"🚀 *Bot Started*\nInstrument: {INSTRUMENT}\nTimeframe: {self.timeframe}\nTime: {datetime.now(NY_TZ)}")
        
        # Load appropriate model and scaler
        try:
            if self.timeframe == "M5":
                model_path = os.path.join(MODELS_DIR, MODEL_5M)
                scaler_path = os.path.join(MODELS_DIR, SCALER_5M)
                expected_features = 76  # 64 base + 12 minute dummies
            else:  # M15 timeframe
                model_path = os.path.join(MODELS_DIR, MODEL_15M)
                scaler_path = os.path.join(MODELS_DIR, SCALER_15M)
                expected_features = 68  # 64 base + 4 minute dummies
                
            self.logger.info(f"Loading model: {model_path}")
            self.model = load_model(model_path)
            self.logger.info(f"Loading scaler: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            # Verify feature dimension match
            fe = FeatureEngineer()
            actual_features = len(fe.features)
            if actual_features != expected_features:
                self.logger.error(f"CRITICAL: Feature dimension mismatch! Expected {expected_features}, got {actual_features}")
                send_telegram(f"❌ *Feature Dimension Mismatch*\n"
                              f"Timeframe: {self.timeframe}\n"
                              f"Expected: {expected_features}\n"
                              f"Actual: {actual_features}\n"
                              "Please check feature engineering")
                return
            
            self.logger.info("Model and scaler loaded successfully")
            self.logger.info(f"Feature dimensions match: {actual_features} features")
            
        except Exception as e:
            self.logger.error(f"Failed to load model or scaler: {str(e)}")
            send_telegram(f"❌ *Failed to load model/scaler for {self.timeframe}*:\n{str(e)}")
            return
            
        try:
            self.detector = TradingDetector()
        except Exception as e:
            self.logger.error(f"Detector initialization failed: {str(e)}")
            send_telegram(f"❌ *Bot Failed to Start for {self.timeframe}*:\n{str(e)}")
            return
            
        self.logger.info("Bot started successfully")
        
        while True:
            try:
                self.logger.info("Running bot cycle")
                last_time = self.detector.data['time'].max() if not self.detector.data.empty else None
                df = fetch_candles(last_time)
                if not df.empty:
                    self.logger.info(f"Fetched {len(df)} new candles, updating data")
                    self.detector.update_data(df)
                else:
                    self.logger.warning("No new candles fetched in this cycle")
                time.sleep(REALTIME_POLL_INTERVAL)  # Check more frequently
            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
                self.logger.error(traceback.format_exc())
                time.sleep(REALTIME_POLL_INTERVAL)

# ========================
# MAIN FUNCTION
# ========================
def run_bot():
    # Start both timeframe bots in parallel
    bot_5m = BotInstance("M5")
    bot_15m = BotInstance("M15")
    
    # Create threads for each bot instance
    thread_5m = threading.Thread(target=bot_5m.run, daemon=True)
    thread_15m = threading.Thread(target=bot_15m.run, daemon=True)
    
    thread_5m.start()
    thread_15m.start()
    
    # Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    logger.info("Launching main application")
    run_bot()
