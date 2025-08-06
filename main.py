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
from collections import defaultdict

# ========================
# GLOBAL CONFIG
# ========================
NY_TZ = pytz.timezone("America/New_York")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
API_KEY = os.getenv("OANDA_API_KEY")
INSTRUMENT = "XAU_USD"
MODELS_DIR = "ml_models"

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

# ========================
# UTILITY FUNCTIONS
# ========================
def parse_oanda_time(time_str):
    """Parse Oanda's timestamp with variable fractional seconds"""
    try:
        if '.' in time_str and len(time_str.split('.')[1]) > 7:
            time_str = re.sub(r'\.(\d{6})\d+', r'.\1', time_str)
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.utc).astimezone(NY_TZ)
    except Exception as e:
        logger.error(f"Error parsing time {time_str}: {str(e)}")
        return datetime.now(NY_TZ)

def send_telegram(message):
    """Send formatted message to Telegram with robust error handling"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set, skipping message")
        return False
        
    # Escape MarkdownV2 special characters
    escape_chars = '_*[]()~`>#+-=|{}.!'
    for char in escape_chars:
        message = message.replace(char, '\\' + char)
    
    if len(message) > 4000:
        message = message[:4000] + "... [TRUNCATED]"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json={
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'MarkdownV2'
            }, timeout=15)
            
            if response.status_code == 200 and response.json().get('ok'):
                return True
            elif attempt < max_retries - 1:
                logger.warning(f"Telegram API error ({response.status_code}), retrying...")
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Telegram connection error: {str(e)}, retrying...")
                time.sleep(2 ** attempt)
    
    logger.error(f"Failed to send Telegram message after {max_retries} attempts")
    return False

def fetch_candles(timeframe, last_time=None, candle_count=201):
    """Fetch candles for specified timeframe with robust error handling"""
    logger.info(f"Fetching {candle_count} candles for {INSTRUMENT} with timeframe {timeframe}")
    params = {
        "granularity": timeframe,
        "count": candle_count,
        "price": "M",
        "alignmentTimezone": "America/New_York",
        "includeCurrent": True
    }
    if last_time:
        params["from"] = last_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    sleep_time = 10
    max_attempts = 5
    
    for attempt in range(max_attempts):
        try:
            request = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
            response = oanda_api.request(request)
            candles = response.get('candles', [])
            
            if not candles:
                logger.warning("No candles received from Oanda")
                time.sleep(sleep_time)
                continue
            
            data = []
            for candle in candles:
                price_data = candle.get('mid', {})
                if not price_data:
                    continue
                
                try:
                    parsed_time = parse_oanda_time(candle['time'])
                    is_complete = candle.get('complete', False)
                    
                    data.append({
                        'time': parsed_time,
                        'open': float(price_data['o']),
                        'high': float(price_data['h']),
                        'low': float(price_data['l']),
                        'close': float(price_data['c']),
                        'volume': int(candle.get('volume', 0)),
                        'complete': is_complete,
                        'is_current': not is_complete
                    })
                except Exception as e:
                    logger.error(f"Error processing candle: {str(e)}")
            
            if not data:
                logger.warning("No valid candle data processed")
                time.sleep(sleep_time)
                continue
                
            df = pd.DataFrame(data).drop_duplicates(subset=['time'], keep='last')
            if last_time:
                df = df[df['time'] > last_time].sort_values('time')
            return df
            
        except V20Error as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "too many" in error_msg:
                wait_time = sleep_time * (2 ** attempt)
                logger.warning(f"API limit, waiting {wait_time}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(wait_time)
            elif "gateway" in error_msg or "502" in error_msg:
                logger.warning("Oanda gateway error, retrying...")
                time.sleep(5)
            else:
                logger.error(f"Oanda API error: {str(e)}")
                send_telegram(f"⚠️ Oanda API Error: {str(e)}")
                time.sleep(10)
        except requests.exceptions.ConnectionError:
            logger.error("Connection error, retrying...")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Unexpected error fetching candles: {str(e)}")
            send_telegram(f"⚠️ Unexpected Candle Fetch Error: {str(e)}")
            time.sleep(10)
    
    logger.error(f"Failed to fetch candles after {max_attempts} attempts")
    send_telegram(f"❌ Failed to fetch candles after {max_attempts} attempts")
    return pd.DataFrame()

# ========================
# FEATURE ENGINEER
# ========================
class FeatureEngineer:
    def __init__(self, timeframe):
        self.timeframe = timeframe
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
        if timeframe == "M5":
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
        
        self.historical_volumes = defaultdict(list)
        self.last_signal_time = None

    def get_same_period_candles(self, df, current_time):
        """Get candles from same time period in previous days"""
        time_key = current_time.strftime('%H:%M')
        
        if time_key in self.historical_volumes and len(self.historical_volumes[time_key]) > 0:
            return self.historical_volumes[time_key]
        
        same_period = []
        complete_df = df[df['complete'] == True]
        
        for idx, row in complete_df.iterrows():
            row_time = row['time']
            if row_time.strftime('%H:%M') == time_key:
                same_period.append(row['volume'])
        
        self.historical_volumes[time_key] = same_period
        return same_period

    def calculate_crt_signal(self, df):
        """Robust CRT signal calculation with improved logic"""
        if len(df) < 3:
            return None, None
            
        try:
            # Get the last three candles: c1 is the oldest, c3 is the newest
            c1 = df.iloc[-3]
            c2 = df.iloc[-2]
            c3 = df.iloc[-1]
            
            # Skip if historical candles are incomplete
            if not c1['complete'] or not c2['complete']:
                logger.debug("Skipping incomplete historical candles")
                return None, None
                
            logger.debug(f"Checking CRT signal at {c3['time']}")
            logger.debug(f"Candle1 (t-2): time={c1['time']}, low={c1['low']}, high={c1['high']}")
            logger.debug(f"Candle2 (t-1): time={c2['time']}, low={c2['low']}, high={c2['high']}, close={c2['close']}, open={c2['open']}")
            logger.debug(f"Candle3 (current): time={c3['time']}, open={c3['open']}")

            c1_low = c1['low']
            c1_high = c1['high']
            c2_low = c2['low']
            c2_high = c2['high']
            c2_close = c2['close']
            c3_open = c3['open']
            
            c2_range = c2_high - c2_low
            c2_mid = c2_low + (0.5 * c2_range)

            # Fixed syntax error here - removed extra parentheses
            buy_condition = (
                (c2_low < c1_low) and 
                (c2_close > c1_low) and 
                (c3_open > c2_mid) and
                (c2_close > c2['open'])  # Bullish candle
            )
            
            sell_condition = (
                (c2_high > c1_high) and 
                (c2_close < c1_high) and 
                (c3_open < c2_mid) and
                (c2_close < c2['open'])  # Bearish candle
            )
            
            logger.debug(f"Buy conditions: {buy_condition}, Sell conditions: {sell_condition}")
            
            # Volume confirmation
            if buy_condition or sell_condition:
                current_time = c3['time']
                same_period_volumes = self.get_same_period_candles(df, current_time)
                
                if same_period_volumes:
                    avg_volume = np.mean(same_period_volumes)
                    if c2['volume'] < 0.8 * avg_volume:
                        logger.debug("Signal rejected: low volume")
                        return None, None
            
            if buy_condition:
                signal_type = 'BUY'
                entry = c3_open
                sl = min(c2_low, c1_low)  # Conservative SL
                risk = abs(entry - sl)
                tp = entry + 4 * risk
                logger.info(f"🔥 CRT BUY signal detected at {c3['time']}")
                return signal_type, {'entry': entry, 'sl': sl, 'tp': tp, 'time': c3['time']}
            elif sell_condition:
                signal_type = 'SELL'
                entry = c3_open
                sl = max(c2_high, c1_high)  # Conservative SL
                risk = abs(sl - entry)
                tp = entry - 4 * risk
                logger.info(f"🔥 CRT SELL signal detected at {c3['time']}")
                return signal_type, {'entry': entry, 'sl': sl, 'tp': tp, 'time': c3['time']}
            else:
                logger.debug("No CRT signal detected")
                return None, None
                
        except KeyError as e:
            logger.error(f"KeyError in CRT calculation: {str(e)}")
            logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"Unexpected error in CRT calculation: {str(e)}")
            logger.error(traceback.format_exc())
        return None, None

    def calculate_technical_indicators(self, df):
        df = df.copy().drop_duplicates(subset=['time'], keep='last')
        
        # Handle incomplete current candle
        if not df.empty and not df.iloc[-1]['complete']:
            current_candle = df.iloc[-1].copy()
            prev_candle = df.iloc[-2] if len(df) > 1 else current_candle
            
            # Estimate volume based on historical average
            current_time = current_candle['time']
            same_period_volumes = self.get_same_period_candles(df, current_time)
            
            if same_period_volumes:
                avg_volume = np.mean(same_period_volumes)
                volume_ratio = min(3.0, max(0.1, current_candle['volume'] / avg_volume))
                current_candle['volume'] = avg_volume * volume_ratio
            
            # Estimate OHLC using previous candle's close as base
            current_candle['open'] = prev_candle['close']
            current_candle['high'] = max(current_candle['high'], prev_candle['close'])
            current_candle['low'] = min(current_candle['low'], prev_candle['close'])
            current_candle['close'] = prev_candle['close']  # Temporary estimate
            
            df.iloc[-1] = current_candle
        
        df['adj close'] = df['open']
        df['garman_klass_vol'] = (
            ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 -
            (2 * np.log(2) - 1) * ((np.log(df['adj close']) - np.log(df['open'])) ** 2
        )
        
        df['rsi_20'] = ta.rsi(df['adj close'], length=20)
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        bb = ta.bbands(np.log1p(df['adj close']), length=20)
        df['bb_low'] = bb['BBL_20_2.0']
        df['bb_mid'] = bb['BBM_20_2.0']
        df['bb_high'] = bb['BBU_20_2.0']
        
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_z'] = (atr - atr.mean()) / atr.std()
        
        macd = ta.macd(df['adj close'], fast=12, slow=26, signal=9)
        df['macd_z'] = (macd['MACD_12_26_9'] - macd['MACD_12_26_9'].mean()) / macd['MACD_12_26_9'].std()
        
        df['dollar_volume'] = (df['adj close'] * df['volume']) / 1e6
        
        df['ma_10'] = df['adj close'].rolling(window=10).mean()
        df['ma_100'] = df['adj close'].rolling(window=100).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()
        df['ma_40'] = df['close'].rolling(window=40).mean()
        df['ma_60'] = df['close'].rolling(window=60).mean()
        
        vwap_num = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
        vwap_den = df['volume'].cumsum()
        df['vwap'] = vwap_num / vwap_den
        df['vwap_std'] = df['vwap'].rolling(window=20).std()
        
        return df

    def calculate_trade_features(self, df, signal_type, entry):
        df = df.copy()
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row
        
        if signal_type == 'SELL':
            df['sl_price'] = prev_row['high']
            risk = abs(entry - df['sl_price'].iloc[-1])
            df['tp_price'] = entry - 4 * risk
        else:
            df['sl_price'] = prev_row['low']
            risk = abs(entry - df['sl_price'].iloc[-1])
            df['tp_price'] = entry + 4 * risk
        
        df['sl_distance'] = abs(entry - df['sl_price']) * 10
        df['tp_distance'] = abs(df['tp_price'] - entry) * 10
        df['rrr'] = df['tp_distance'] / df['sl_distance'].replace(0, np.nan)
        df['log_sl'] = np.log1p(df['sl_price'])
        
        return df

    def calculate_categorical_features(self, df):
        df = df.copy()
        df['day'] = df['time'].dt.day_name()
        all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']
        for day in all_days:
            df[f'day_{day}'] = 0
        today = datetime.now(NY_TZ).strftime('%A')
        df[f'day_{today}'] = 1
        
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
        
        def is_bullish_stack(row):
            return int(row['ma_20'] > row['ma_30'] > row['ma_40'] > row['ma_60'])
        def is_bearish_stack(row):
            return int(row['ma_20'] < row['ma_30'] < row['ma_40'] < row['ma_60'])
        
        df['trend_strength_up'] = df.apply(is_bullish_stack, axis=1).astype(float)
        df['trend_strength_down'] = df.apply(is_bearish_stack, axis=1).astype(float)
        
        def get_trend(row):
            if row['trend_strength_up'] > row['trend_strength_down']:
                return 'uptrend'
            elif row['trend_strength_down'] > row['trend_strength_up']:
                return 'downtrend'
            else:
                return 'sideways'
        df['trend_direction'] = df.apply(get_trend, axis=1)
        df = pd.get_dummies(df, columns=['trend_direction'], prefix='trend_direction', drop_first=False)
        
        return df

    def calculate_minutes_closed(self, df, minutes_closed):
        df = df.copy()
        
        if self.timeframe == "M5":
            minute_buckets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
            minute_cols = [f'minutes,closed_{bucket}' for bucket in minute_buckets]
        else:
            minute_buckets = [0, 15, 30, 45]
            minute_cols = [f'minutes,closed_{bucket}' for bucket in minute_buckets]
            
        for col in minute_cols:
            df[col] = 0
        
        current_bucket = (minutes_closed // (5 if self.timeframe == "M5" else 15)) * (5 if self.timeframe == "M5" else 15)
        max_bucket = max(minute_buckets)
        if current_bucket > max_bucket:
            current_bucket = max_bucket
            
        bucket_col = f'minutes,closed_{current_bucket}'
        if bucket_col in df.columns:
            df[bucket_col] = 1
            
        return df

    def generate_features(self, df, signal_type, minutes_closed):
        if len(df) < 200:
            logger.warning("Not enough data for feature generation")
            return None
        
        # Create a copy and ensure we only use complete candles
        df = df[df['complete']].tail(200).copy()
        
        # Handle incomplete current candle
        if not df.empty and not df.iloc[-1]['complete']:
            current_candle = df.iloc[-1].copy()
            prev_candle = df.iloc[-2]
            
            # Estimate volume based on historical average
            current_time = current_candle['time']
            same_period_volumes = self.get_same_period_candles(df, current_time)
            
            if same_period_volumes:
                avg_volume = np.mean(same_period_volumes)
                volume_ratio = min(3.0, max(0.1, current_candle['volume'] / avg_volume))
                current_candle['volume'] = avg_volume * volume_ratio
            
            # Estimate OHLC using previous candle's close as base
            current_candle['open'] = prev_candle['close']
            current_candle['high'] = max(current_candle['high'], prev_candle['close'])
            current_candle['low'] = min(current_candle['low'], prev_candle['close'])
            current_candle['close'] = prev_candle['close']  # Temporary estimate
            
            df.iloc[-1] = current_candle

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
        
        df['price_div_vol'] = df['adj close'] / (df['garman_klass_vol'] + 1e-6)
        df['rsi_div_macd'] = df['rsi'] / (df['macd_z'] + 1e-6)
        df['price_div_vwap'] = df['adj close'] / (df['vwap'] + 1e-6)
        df['sl_div_atr'] = df['sl_distance'] / (df['atr_z'] + 1e-6)
        df['tp_div_atr'] = df['tp_distance'] / (df['atr_z'] + 1e-6)
        df['rrr_div_rsi'] = df['rrr'] / (df['rsi'] + 1e-6)
        
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
        
        df['is_bad_combo'] = 1 if combo_flags['combo_flag_dead'] == 1 else 0
        
        df['crt_BUY'] = int(signal_type == 'BUY')
        df['crt_SELL'] = int(signal_type == 'SELL')
        df['trade_type_BUY'] = int(signal_type == 'BUY')
        df['trade_type_SELL'] = int(signal_type == 'SELL')
        
        features = pd.Series(index=self.features, dtype=float)
        for feat in self.features:
            if feat in df.columns:
                features[feat] = df[feat].iloc[-1]
            else:
                features[feat] = 0
        
        if len(df) >= 2:
            prev_candle = df.iloc[-2]
            for feat in self.shift_features:
                if feat in features.index and feat in prev_candle:
                    features[feat] = prev_candle[feat]
        
        if features.isna().any():
            logger.warning("NaN values in features, replacing with 0")
            features.fillna(0, inplace=True)
        
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

    def run(self):
        while self.running:
            try:
                current_time = datetime.now(NY_TZ)
                
                with self.detector.lock:
                    if self.detector.data.empty:
                        time.sleep(self.detector.poll_interval)
                        continue
                    latest_candle = self.detector.data.iloc[-1].copy()
                
                if self.current_candle_time != latest_candle['time']:
                    self.current_candle_time = latest_candle['time']
                    self.detector.scan_count = 0
                    self.detector.signal_found = False
                
                if self.detector.signal_found and self.detector.next_candle_time:
                    sleep_seconds = (self.detector.next_candle_time - current_time).total_seconds()
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    continue
                
                if self.detector.scan_count >= 2 and not self.detector.signal_found and self.detector.next_candle_time:
                    sleep_seconds = (self.detector.next_candle_time - current_time).total_seconds()
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    continue
                
                candle_age = (current_time - latest_candle['time']).total_seconds() / 60.0
                if latest_candle['is_current'] and candle_age >= self.detector.min_candle_age and self.detector.scan_count < 2:
                    self.detector.process_signals(0, pd.DataFrame([latest_candle]))
                    self.detector.scan_count += 1
                
                time.sleep(self.detector.poll_interval)
                
            except Exception as e:
                logger.error(f"Real-time detector error: {str(e)}")
                time.sleep(self.detector.poll_interval)

    def stop(self):
        self.running = False
        self.thread.join(timeout=5)

# ========================
# TRADING DETECTOR
# ========================
class TradingDetector:
    def __init__(self, timeframe, model, scaler, prediction_threshold=0.9140):
        self.timeframe = timeframe
        self.model = model
        self.scaler = scaler
        self.prediction_threshold = prediction_threshold
        self.data = pd.DataFrame()
        self.feature_engineer = FeatureEngineer(timeframe)
        self.candle_duration = 5 if timeframe == "M5" else 15
        self.poll_interval = 3 if timeframe == "M5" else 5
        self.min_candle_age = 0.25 if timeframe == "M5" else 0.5
        self.candle_count = 201
        self.scheduler = CandleScheduler(timeframe, self.candle_duration)
        self.last_signal_candle = None
        self.realtime_detector = None
        self.lock = threading.Lock()
        self.scan_count = 0
        self.signal_found = False
        self.next_candle_time = None
        self.last_signal_time = None
        
        self.data = self.fetch_initial_candles()
        if self.data.empty or len(self.data) < 200:
            raise RuntimeError("Initial candle fetch failed")
            
        self.scheduler.register_callback(self.process_signals)
        self.scheduler.start()
        self.realtime_detector = RealTimeDetector(self)

    def fetch_initial_candles(self):
        for attempt in range(5):
            df = fetch_candles(self.timeframe, candle_count=self.candle_count)
            if not df.empty and len(df) >= 200:
                return df
            time.sleep(10)
        return pd.DataFrame()

    def calculate_candle_age(self, current_time, candle_time):
        elapsed = (current_time - candle_time).total_seconds() / 60
        return min(self.candle_duration, max(0, elapsed))

    def _get_next_candle_time(self, current_time):
        minute = current_time.minute
        if self.timeframe == "M5":
            remainder = minute % 5
            if remainder == 0:
                return current_time.replace(second=0, microsecond=0) + timedelta(minutes=5)
            next_minute = minute - remainder + 5
            if next_minute >= 60:
                return current_time.replace(hour=current_time.hour + 1, minute=0, second=0, microsecond=0)
            return current_time.replace(minute=next_minute, second=0, microsecond=0)
        else:
            remainder = minute % 15
            if remainder == 0:
                return current_time.replace(second=0, microsecond=0) + timedelta(minutes=15)
            next_minute = minute - remainder + 15
            if next_minute >= 60:
                return current_time.replace(hour=current_time.hour + 1, minute=0, second=0, microsecond=0)
            return current_time.replace(minute=next_minute, second=0, microsecond=0)

    def update_data(self, df_new):
        if df_new.empty:
            return
        
        with self.lock:
            if self.data.empty:
                self.data = df_new
                return
                
            # Get the last complete candle from existing data
            last_complete = self.data[self.data['complete']].iloc[-1] if not self.data.empty else None
            
            # Merge new data
            for _, new_row in df_new.iterrows():
                # Update existing incomplete candle
                if not new_row['complete'] and not self.data.empty:
                    current_idx = self.data.index[-1]
                    if not self.data.loc[current_idx, 'complete']:
                        self.data.loc[current_idx] = new_row
                        continue
                        
                # Add new candle if it doesn't exist
                if new_row['time'] > self.data['time'].max():
                    self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)
                    
            # Keep only the most recent 201 candles
            self.data = self.data.tail(201)

    def predict_single_model(self, features_df):
        expected_features = len(self.feature_engineer.features)
        if features_df.shape[1] != expected_features:
            logger.error(f"Feature mismatch: expected {expected_features}, got {features_df.shape[1]}")
            return None, None
        
        try:
            features_array = features_df.values
            scaled_features = self.scaler.transform(features_array)
            reshaped_features = np.expand_dims(scaled_features, axis=1)
            prob = self.model.predict(reshaped_features, verbose=0)[0][0]
            final_pred = 1 if prob >= self.prediction_threshold else 0
            outcome = "Worth Taking" if final_pred == 1 else "Likely Loss"
            return prob, outcome
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None, None

    def process_signals(self, minutes_closed, latest_candles):
        if not latest_candles.empty:
            self.update_data(latest_candles)

        if self.data.empty or len(self.data) < 3:
            logger.warning("Not enough data for signal processing")
            return

        with self.lock:
            latest_candle_time = self.data.iloc[-1]['time']
            current_time = datetime.now(NY_TZ)
            
            # Deduplication check
            if self.last_signal_time and (current_time - self.last_signal_time).total_seconds() < 60:
                logger.debug("Skipping signal processing: too recent")
                return
                
            candle_age = self.calculate_candle_age(current_time, latest_candle_time)
            self.next_candle_time = self._get_next_candle_time(latest_candle_time)

            signal_type, signal_data = self.feature_engineer.calculate_crt_signal(self.data)
            if signal_type and signal_data:
                self.last_signal_time = current_time
                current_candle = self.data.iloc[-1]
                
                is_new_trade = True
                for trade_id, trade in list(self.detector.bot.active_trades.items()):
                    if trade['sl'] == signal_data['sl'] and trade.get('outcome') is None:
                        is_new_trade = False
                        break
                
                if is_new_trade:
                    self.signal_found = True
                    alert_time = signal_data['time'].astimezone(NY_TZ)
                    setup_msg = (
                        f"🔔 *{self.timeframe} SETUP* {INSTRUMENT.replace('_','/')} {signal_type}\n"
                        f"Time: {alert_time.strftime('%Y-%m-%d %H:%M')} NY\n"
                        f"Entry: {signal_data['entry']:.5f}\n"
                        f"TP: {signal_data['tp']:.5f}\n"
                        f"SL: {signal_data['sl']:.5f}"
                    )
                    if send_telegram(setup_msg):
                        logger.info("Sent Telegram setup notification")
                    else:
                        logger.error("Failed to send Telegram setup notification")
                    
                    features = self.feature_engineer.generate_features(self.data, signal_type, minutes_closed)
                    if features is not None:
                        feature_msg = f"📊 *{self.timeframe} FEATURES* {INSTRUMENT.replace('_','/')} {signal_type}\n"
                        formatted_features = []
                        for feat, val in features.items():
                            # Skip long lists of features for Telegram
                            if 'minutes,closed' not in feat:
                                formatted_features.append(f"{feat}: {val:.6f}")
                        feature_msg += "\n".join(formatted_features[:15])  # Only show first 15 features
                        if send_telegram(feature_msg):
                            logger.info("Sent Telegram features notification")
                        else:
                            logger.error("Failed to send Telegram features notification")
                    else:
                        logger.warning("Feature generation failed")

                    if self.scaler is not None and self.model is not None:
                        features_df = pd.DataFrame([features], columns=self.feature_engineer.features)
                        prob, outcome = self.predict_single_model(features_df)
                        
                        if prob is not None:
                            pred_msg = f"🤖 *{self.timeframe} MODEL PREDICTION*\n"
                            pred_msg += f"Probability: {prob:.6f}\n"
                            pred_msg += f"Decision: {outcome}"
                            if send_telegram(pred_msg):
                                logger.info("Sent Telegram prediction notification")
                            else:
                                logger.error("Failed to send Telegram prediction notification")
                            
                            trade_id = f"{self.timeframe}_{signal_type}_{current_time.timestamp()}"
                            self.detector.bot.active_trades[trade_id] = {
                                'entry': signal_data['entry'],
                                'sl': signal_data['sl'],
                                'tp': signal_data['tp'],
                                'time': current_time,
                                'signal_time': signal_data['time'],
                                'prediction': prob,
                                'outcome': None
                            }
                        else:
                            logger.warning("Model prediction failed")
                else:
                    logger.info("Duplicate trade detected, skipping")

        # Trade outcome checking
        if len(self.data) > 0 and minutes_closed == self.candle_duration:
            with self.lock:
                latest_candle = self.data.iloc[-1]
                for trade_id, trade in list(self.detector.bot.active_trades.items()):
                    if trade.get('outcome') is None and trade_id.startswith(self.timeframe):
                        entry, sl, tp = trade['entry'], trade['sl'], trade['tp']
                        
                        if entry > sl:  # SELL trade
                            if latest_candle['high'] >= sl:
                                trade['outcome'] = 'Hit SL (Loss)'
                            elif latest_candle['low'] <= tp:
                                trade['outcome'] = 'Hit TP (Win)'
                        else:  # BUY trade
                            if latest_candle['low'] <= sl:
                                trade['outcome'] = 'Hit SL (Loss)'
                            elif latest_candle['high'] >= tp:
                                trade['outcome'] = 'Hit TP (Win)'
                        
                        if trade.get('outcome'):
                            outcome_msg = (
                                f"📈 *{self.timeframe} Trade Outcome*\n"
                                f"Signal Time: {trade['signal_time'].strftime('%Y-%m-%d %H:%M')} NY\n"
                                f"Entry: {entry:.5f}\n"
                                f"SL: {sl:.5f}\n"
                                f"TP: {tp:.5f}\n"
                                f"Prediction: {trade['prediction']:.6f}\n"
                                f"Outcome: {trade['outcome']}"
                            )
                            if send_telegram(outcome_msg):
                                logger.info("Sent Telegram outcome notification")
                            else:
                                logger.error("Failed to send Telegram outcome notification")
                            del self.detector.bot.active_trades[trade_id]

# ========================
# CANDLE SCHEDULER
# ========================
class CandleScheduler(threading.Thread):
    def __init__(self, timeframe, candle_duration):
        super().__init__(daemon=True)
        self.timeframe = timeframe
        self.candle_duration = candle_duration
        self.callback = None
        self.active = True
    
    def register_callback(self, callback):
        self.callback = callback
        
    def calculate_next_candle(self):
        now = datetime.now(NY_TZ)
        current_minute = now.minute
        remainder = current_minute % self.candle_duration
        if remainder == 0:
            return now.replace(second=0, microsecond=0) + timedelta(minutes=self.candle_duration)
        next_minute = current_minute - remainder + self.candle_duration
        if next_minute >= 60:
            return now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
        return now.replace(minute=next_minute, second=0, microsecond=0)
    
    def calculate_minutes_closed(self, latest_time):
        if latest_time is None:
            return 0
        now = datetime.now(NY_TZ)
        elapsed = (now - latest_time).total_seconds() / 60
        max_closed = 4.9 if self.candle_duration == 5 else 14.9
        return min(max_closed, max(0, elapsed))
    
    def run(self):
        while self.active:
            try:
                df_candles = fetch_candles(self.timeframe)
                
                if df_candles.empty:
                    time.sleep(60)
                    continue
                
                # Pass all candles including incomplete ones
                latest_candle = df_candles.iloc[-1]
                latest_time = latest_candle['time']
                minutes_closed = self.calculate_minutes_closed(latest_time)
                if self.callback:
                    self.callback(minutes_closed, df_candles.tail(1))
                
                now = datetime.now(NY_TZ)
                next_run = self.calculate_next_candle()
                sleep_seconds = (next_run - now).total_seconds()
                time.sleep(max(1, sleep_seconds))
                
            except Exception as e:
                logger.error(f"Candle scheduler error: {str(e)}")
                time.sleep(60)

# ========================
# TRADING BOT CLASS
# ========================
class TradingBot:
    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.model = None
        self.scaler = None
        self.detector = None
        self.active_trades = {}
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
        # Timeframe-specific configuration
        if timeframe == "M5":
            self.model_file = "5mbilstm_model.keras"
            self.scaler_file = "scaler5mcrt.joblib"
            self.candle_count = 201
            self.poll_interval = 3
            self.min_candle_age = 0.25
            self.prediction_threshold = 0.9140
        else:  # M15
            self.model_file = "15mbilstm_model.keras"
            self.scaler_file = "scaler15mcrt.joblib"
            self.candle_count = 201
            self.poll_interval = 5
            self.min_candle_age = 0.5
            self.prediction_threshold = 0.9140

    def initialize(self):
        logger.info(f"Initializing {self.timeframe} bot")
        
        # Load model and scaler
        model_path = os.path.join(MODELS_DIR, self.model_file)
        scaler_path = os.path.join(MODELS_DIR, self.scaler_file)
        
        try:
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded model and scaler for {self.timeframe}")
            
            # Initialize detector
            self.detector = TradingDetector(
                timeframe=self.timeframe,
                model=self.model,
                scaler=self.scaler,
                prediction_threshold=self.prediction_threshold
            )
            self.detector.bot = self  # Reference back to bot instance
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.timeframe} bot: {str(e)}")
            send_telegram(f"❌ *{self.timeframe} Bot Failed to Start*:\n{str(e)}")
            return False

    def run(self):
        self.running = True
        logger.info(f"Starting {self.timeframe} bot thread")
        send_telegram(f"🚀 *{self.timeframe} Bot Started*\nInstrument: {INSTRUMENT}")
        
        while self.running:
            try:
                with self.lock:
                    if self.detector.data.empty:
                        last_time = None
                    else:
                        last_time = self.detector.data['time'].max()
                
                df = fetch_candles(self.timeframe, last_time, self.candle_count)
                if not df.empty:
                    with self.lock:
                        self.detector.update_data(df)
                time.sleep(self.poll_interval)
            except V20Error as e:
                logger.error(f"Oanda API error in main loop: {str(e)}")
                time.sleep(30)
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {traceback.format_exc()}")
                send_telegram(f"⚠️ *{self.timeframe} Bot Error*:\n```\n{traceback.format_exc()}\n```")
                time.sleep(60)

    def stop(self):
        self.running = False
        if self.detector and self.detector.realtime_detector:
            self.detector.realtime_detector.stop()
        if self.thread:
            self.thread.join(timeout=5)
        send_telegram(f"🔴 *{self.timeframe} Bot Stopped*")

# ========================
# MAIN APPLICATION
# ========================
def main():
    # Enable debug logging for troubleshooting
    logger.setLevel(logging.INFO)
    
    logger.info("Launching trading bots")
    send_telegram("🚀 *Launching Trading Bots*")
    
    # Create bot instances
    m5_bot = TradingBot("M5")
    m15_bot = TradingBot("M15")
    
    # Initialize bots
    bots_initialized = True
    if not m5_bot.initialize():
        logger.error("Failed to initialize M5 bot")
        bots_initialized = False
    if not m15_bot.initialize():
        logger.error("Failed to initialize M15 bot")
        bots_initialized = False
    
    if not bots_initialized:
        send_telegram("❌ Bot initialization failed, check logs")
        return
    
    # Start bots in separate threads
    m5_thread = threading.Thread(target=m5_bot.run)
    m15_thread = threading.Thread(target=m15_bot.run)
    
    m5_thread.start()
    m15_thread.start()
    
    m5_bot.thread = m5_thread
    m15_bot.thread = m15_thread
    
    logger.info("Both bots started")
    send_telegram("✅ Both bots started successfully")
    
    # Monitor threads
    try:
        while True:
            if not m5_thread.is_alive():
                logger.warning("M5 thread died, restarting...")
                m5_bot.stop()
                m5_bot = TradingBot("M5")
                if m5_bot.initialize():
                    m5_thread = threading.Thread(target=m5_bot.run)
                    m5_thread.start()
                    m5_bot.thread = m5_thread
                    send_telegram("🔄 M5 Bot Restarted")
                else:
                    logger.error("Failed to restart M5 bot")
                    send_telegram("❌ Failed to restart M5 bot")
            
            if not m15_thread.is_alive():
                logger.warning("M15 thread died, restarting...")
                m15_bot.stop()
                m15_bot = TradingBot("M15")
                if m15_bot.initialize():
                    m15_thread = threading.Thread(target=m15_bot.run)
                    m15_thread.start()
                    m15_bot.thread = m15_thread
                    send_telegram("🔄 M15 Bot Restarted")
                else:
                    logger.error("Failed to restart M15 bot")
                    send_telegram("❌ Failed to restart M15 bot")
            
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Shutting down bots")
        m5_bot.stop()
        m15_bot.stop()
        send_telegram("🔴 All Bots Stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
