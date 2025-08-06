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

# Instrument configuration
INSTRUMENT = "XAU_USD"

# Model and scaler paths - timeframe specific
MODELS_DIR = "ml_models"
MODEL_5M = "5mbilstm_model.keras"
SCALER_5M = "scaler5mcrt.joblib"
MODEL_15M = "15mbilstm_model.keras"
SCALER_15M = "scaler15mcrt.joblib"

# Prediction threshold (0.9140 for class 1)
PREDICTION_THRESHOLD = 0.9140

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
            
            if response.status_code == 200 and response.json().get('ok'):
                return True
            else:
                logger.error(f"Telegram error: {response.status_code} - {response.text}")
                time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Telegram connection failed: {str(e)}")
            time.sleep(2 ** attempt)
    logger.error(f"Failed to send Telegram message after {max_retries} attempts")
    return False

def fetch_candles(timeframe, last_time=None):
    """Fetch exactly 201 candles for XAU_USD with full precision"""
    logger.info(f"Fetching 201 candles for {INSTRUMENT} with timeframe {timeframe}")
    params = {
        "granularity": timeframe,
        "count": 201,
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
            response = oanda_api.request(request)
            candles = response.get('candles', [])
            
            if not candles:
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
                except Exception:
                    continue
            
            if not data:
                continue
                
            df = pd.DataFrame(data).drop_duplicates(subset=['time'], keep='last')
            if last_time:
                df = df[df['time'] > last_time].sort_values('time')
            return df
            
        except V20Error as e:
            if "rate" in str(e).lower() or e.status == 502:
                wait_time = sleep_time * (2 ** attempt)
                time.sleep(wait_time)
            else:
                logger.error(f"Oanda API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error fetching candles: {str(e)}")
    
    return pd.DataFrame()

# ========================
# FEATURE ENGINEER WITH VOLUME IMPUTATION
# ========================
class FeatureEngineer:
    def __init__(self, timeframe):
        self.timeframe = timeframe
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
        time_key = current_time.strftime('%H:%M')
        
        if time_key in self.historical_volumes and self.historical_volumes[time_key]:
            return self.historical_volumes[time_key]
        
        same_period = []
        complete_df = df[df['complete'] == True]
        
        for _, row in complete_df.iterrows():
            if row['time'].strftime('%H:%M') == time_key:
                same_period.append(row['volume'])
        
        self.historical_volumes[time_key] = same_period
        return same_period

    def calculate_crt_signal(self, df):
        """Robust CRT signal calculation with validation"""
        if len(df) < 3:
            return None, None
            
        crt_df = df.tail(3).copy().reset_index(drop=True)
        
        if len(crt_df) < 3:
            return None, None
            
        if not crt_df['time'].is_monotonic_increasing:
            crt_df = crt_df.sort_values('time').reset_index(drop=True)
        
        try:
            c1 = crt_df.iloc[0]
            c2 = crt_df.iloc[1]
            c3 = crt_df.iloc[2]
            
            c1_low = c1['low']
            c1_high = c1['high']
            c2_low = c2['low']
            c2_high = c2['high']
            c2_close = c2['close']
            c3_open = c3['open']
            
            c2_range = c2_high - c2_low
            c2_mid = c2_low + (0.5 * c2_range)

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
            
            if buy_condition:
                signal_type = 'BUY'
                entry = c3_open
                sl = c2_low
                risk = abs(entry - sl)
                tp = entry + 4 * risk
            elif sell_condition:
                signal_type = 'SELL'
                entry = c3_open
                sl = c2_high
                risk = abs(sl - entry)
                tp = entry - 4 * risk
            else:
                return None, None
            
            return signal_type, {'entry': entry, 'sl': sl, 'tp': tp, 'time': c3['time']}
            
        except KeyError:
            return None, None

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators with volume imputation"""
        df = df.copy().drop_duplicates(subset=['time'], keep='last')
        
        if not df.empty and not df.iloc[-1]['complete']:
            current_candle = df.iloc[-1]
            current_time = current_candle['time']
            same_period_volumes = self.get_same_period_candles(df, current_time)
            
            if same_period_volumes:
                avg_volume = np.mean(same_period_volumes)
                current_volume = current_candle['volume']
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                volume_ratio = min(3.0, max(0.1, volume_ratio))
                estimated_volume = avg_volume * volume_ratio
                df.at[df.index[-1], 'volume'] = estimated_volume
        
        df['adj close'] = df['open']
        df['garman_klass_vol'] = (
            ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 -
            (2 * np.log(2) - 1) * ((np.log(df['adj close']) - np.log(df['open'])) ** 2)
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
        prev_row = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        
        if signal_type == 'SELL':
            df['sl_price'] = prev_row['high']
            risk = abs(entry - df['sl_price'].iloc[-1])
            df['tp_price'] = entry - 4 * risk
        else:  # BUY
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
        else:  # M15 timeframe
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
            for col in features[features.isna()].index:
                features[col] = 0
        
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
                
                if self.detector.data.empty:
                    time.sleep(self.detector.poll_interval)
                    continue
                    
                with self.detector.lock:
                    if self.detector.data.empty:
                        continue
                    latest_candle = self.detector.data.iloc[-1].copy()
                
                if self.current_candle_time != latest_candle['time']:
                    self.current_candle_time = latest_candle['time']
                    self.detector.scan_count_this_candle = 0
                    self.detector.signal_found_this_candle = False
                
                if self.detector.signal_found_this_candle and self.detector.next_candle_time:
                    sleep_seconds = (self.detector.next_candle_time - current_time).total_seconds()
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    continue
                
                if (self.detector.scan_count_this_candle >= 2 and 
                    not self.detector.signal_found_this_candle and 
                    self.detector.next_candle_time):
                    sleep_seconds = (self.detector.next_candle_time - current_time).total_seconds()
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    continue
                
                candle_age = (current_time - latest_candle['time']).total_seconds() / 60.0
                if (latest_candle['is_current'] and 
                    candle_age >= self.detector.min_candle_age and 
                    self.detector.scan_count_this_candle < 2):
                    self.detector.process_signals(0, pd.DataFrame([latest_candle]))
                    self.detector.scan_count_this_candle += 1
                
                time.sleep(self.detector.poll_interval)
                
            except Exception:
                time.sleep(self.detector.poll_interval)

    def stop(self):
        self.running = False
        self.thread.join(timeout=5)

# ========================
# TRADING DETECTOR
# ========================
class TradingDetector:
    def __init__(self, timeframe, model, scaler):
        self.timeframe = timeframe
        self.model = model
        self.scaler = scaler
        self.data = pd.DataFrame()
        self.feature_engineer = FeatureEngineer(timeframe)
        self.candle_duration = 5 if timeframe == "M5" else 15
        self.poll_interval = 3 if timeframe == "M5" else 5
        self.min_candle_age = 0.25 if timeframe == "M5" else 0.5
        self.lock = threading.Lock()
        self.scan_count_this_candle = 0
        self.signal_found_this_candle = False
        self.next_candle_time = None
        self.last_signal_candle = None
        self.active_trades = {}
        self.realtime_detector = None
        
        self.data = self.fetch_initial_candles()
        if self.data.empty or len(self.data) < 200:
            raise RuntimeError("Initial candle fetch failed")
            
        self.realtime_detector = RealTimeDetector(self)
        logger.info(f"TradingDetector initialized for {timeframe}")

    def fetch_initial_candles(self):
        for attempt in range(5):
            df = fetch_candles(self.timeframe)
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
        else:  # M15 timeframe
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
                self.data = df_new.dropna(subset=['time', 'open', 'high', 'low', 'close']).tail(201)
            else:
                last_existing_time = self.data['time'].max()
                new_data = df_new[df_new['time'] > last_existing_time]
                same_time_data = df_new[df_new['time'] == last_existing_time]
                
                if not same_time_data.empty:
                    self.data = self.data[self.data['time'] < last_existing_time]
                    self.data = pd.concat([self.data, same_time_data])
                
                if not new_data.empty:
                    self.data = pd.concat([self.data, new_data])
                
                self.data = self.data.sort_values('time').reset_index(drop=True).tail(201)

    def predict_single_model(self, features_df):
        expected_features = len(self.feature_engineer.features)
        
        if features_df.shape[1] != expected_features:
            logger.error(f"Feature mismatch: Expected {expected_features} features, got {features_df.shape[1]}")
            return None, None
        
        try:
            features_array = features_df.values
            scaled_features = self.scaler.transform(features_array)
            reshaped_features = np.expand_dims(scaled_features, axis=1)
            prob = self.model.predict(reshaped_features, verbose=0)[0][0]
            final_pred = 1 if prob >= PREDICTION_THRESHOLD else 0
            outcome = "Worth Taking" if final_pred == 1 else "Likely Loss"
            return prob, outcome
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None, None

    def process_signals(self, minutes_closed, latest_candles):
        if not latest_candles.empty:
            self.update_data(latest_candles)
            
        if self.data.empty or len(self.data) < 3:
            return

        latest_candle_time = self.data.iloc[-1]['time']
        current_time = datetime.now(NY_TZ)
        candle_age = self.calculate_candle_age(current_time, latest_candle_time)
        self.next_candle_time = self._get_next_candle_time(latest_candle_time)

        signal_type, signal_data = self.feature_engineer.calculate_crt_signal(self.data)
        if signal_type and signal_data:
            current_candle = self.data.iloc[-1]
            if (self.last_signal_candle is None or 
                current_candle['time'] > self.last_signal_candle['time'] or 
                (current_candle['time'] == self.last_signal_candle['time'] and 
                 abs(current_candle['close'] - self.last_signal_candle['close']) > 0.5)):
                self.last_signal_candle = current_candle
                
                is_new_trade = True
                for trade_id, trade in list(self.active_trades.items()):
                    if trade['sl'] == signal_data['sl'] and trade.get('outcome') is None:
                        is_new_trade = False
                        break
                
                if is_new_trade:
                    self.signal_found_this_candle = True
                    alert_time = signal_data['time'].astimezone(NY_TZ)
                    setup_msg = (
                        f"🔔 *SETUP* {INSTRUMENT.replace('_','/')} {signal_type}\n"
                        f"Timeframe: {self.timeframe}\n"
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
                            send_telegram(feature_msg)

                        # PREDICTION CALL - FIXED
                        if self.scaler is not None and self.model is not None:
                            features_df = pd.DataFrame([features], columns=self.feature_engineer.features)
                            
                            prob, outcome = self.predict_single_model(features_df)
                            
                            if prob is not None:
                                pred_msg = f"🤖 *MODEL PREDICTION* {INSTRUMENT.replace('_','/')} {signal_type}\n"
                                pred_msg += f"Probability: {prob:.6f}\n"
                                pred_msg += f"Decision: {outcome}\n"
                                pred_msg += f"Model: {MODEL_5M if self.timeframe == 'M5' else MODEL_15M}"
                                
                                if send_telegram(pred_msg):
                                    # Store new trade with prediction
                                    trade_id = f"{signal_type}_{current_time.timestamp()}"
                                    self.active_trades[trade_id] = {
                                        'entry': signal_data['entry'],
                                        'sl': signal_data['sl'],
                                        'tp': signal_data['tp'],
                                        'time': current_time,
                                        'signal_time': signal_data['time'],
                                        'prediction': prob,
                                        'outcome': None
                                    }
                                    logger.info(f"New trade stored: {trade_id} with prediction {prob:.6f}")
                        else:
                            logger.error("No scaler or model loaded")

        if len(self.data) > 0 and minutes_closed == self.candle_duration:
            latest_candle = self.data.iloc[-1]
            for trade_id, trade in list(self.active_trades.items()):
                if trade.get('outcome') is None:
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
                            f"📈 *Trade Outcome*\n"
                            f"Timeframe: {self.timeframe}\n"
                            f"Signal Time: {trade['signal_time'].strftime('%Y-%m-%d %H:%M')} NY\n"
                            f"Entry: {entry:.5f}\n"
                            f"SL: {sl:.5f}\n"
                            f"TP: {tp:.5f}\n"
                            f"Prediction: {trade['prediction']:.6f}\n"
                            f"Outcome: {trade['outcome']}\n"
                            f"Detected at: {current_time.strftime('%Y-%m-%d %H:%M')} NY"
                        )
                        if send_telegram(outcome_msg):
                            del self.active_trades[trade_id]

# ========================
# CANDLE SCHEDULER
# ========================
class CandleScheduler(threading.Thread):
    def __init__(self, granularity):
        super().__init__(daemon=True)
        self.granularity = granularity
        self.callback = None
        self.active = True
    
    def register_callback(self, callback):
        self.callback = callback
        
    def calculate_next_candle(self):
        now = datetime.now(NY_TZ)
        current_minute = now.minute
        if self.granularity == "M5":
            remainder = current_minute % 5
            if remainder == 0:
                return now.replace(second=0, microsecond=0) + timedelta(minutes=5)
            next_minute = current_minute - remainder + 5
            if next_minute >= 60:
                return now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
            return now.replace(minute=next_minute, second=0, microsecond=0)
        else:  # M15 timeframe
            remainder = current_minute % 15
            if remainder == 0:
                return now.replace(second=0, microsecond=0) + timedelta(minutes=15)
            next_minute = current_minute - remainder + 15
            if next_minute >= 60:
                return now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
            return now.replace(minute=next_minute, second=0, microsecond=0)
    
    def calculate_minutes_closed(self, latest_time):
        if latest_time is None:
            return 0
        now = datetime.now(NY_TZ)
        elapsed = (now - latest_time).total_seconds() / 60
        max_closed = 4.9 if self.granularity == "M5" else 14.9
        return min(max_closed, max(0, elapsed))
    
    def run(self):
        while self.active:
            try:
                df_candles = fetch_candles(self.granularity)
                if df_candles.empty:
                    continue
                
                complete_candles = df_candles[df_candles['complete'] == True]
                if not complete_candles.empty:
                    latest_candle = complete_candles.iloc[-1]
                    latest_time = latest_candle['time']
                    minutes_closed = self.calculate_minutes_closed(latest_time)
                    if self.callback:
                        self.callback(minutes_closed, complete_candles.tail(1))
                
                now = datetime.now(NY_TZ)
                next_run = self.calculate_next_candle()
                sleep_seconds = (next_run - now).total_seconds()
                time.sleep(max(1, sleep_seconds))
            except Exception as e:
                logger.error(f"Scheduler error: {str(e)}")
                time.sleep(60)

# ========================
# BOT INSTANCE CLASS
# ========================
class BotInstance:
    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.model = None
        self.scaler = None
        self.detector = None
        self.logger = logging.getLogger(f"{__name__}.{timeframe}")
        self.scheduler = None
        self.active_trades = {}
        
    def run(self):
        self.logger.info(f"Starting trading bot for {self.timeframe}")
        send_telegram(f"🚀 *Bot Started*\nInstrument: {INSTRUMENT}\nTimeframe: {self.timeframe}\nTime: {datetime.now(NY_TZ)}")
        
        try:
            if self.timeframe == "M5":
                model_path = os.path.join(MODELS_DIR, MODEL_5M)
                scaler_path = os.path.join(MODELS_DIR, SCALER_5M)
            else:  # M15 timeframe
                model_path = os.path.join(MODELS_DIR, MODEL_15M)
                scaler_path = os.path.join(MODELS_DIR, SCALER_15M)
                
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Create detector with timeframe-specific model and scaler
            self.detector = TradingDetector(self.timeframe, self.model, self.scaler)
            
            # Create scheduler with correct granularity string
            self.scheduler = CandleScheduler(granularity=self.timeframe)
            self.scheduler.register_callback(self.detector.process_signals)
            self.scheduler.start()
            
            self.logger.info(f"Bot started successfully for {self.timeframe}")
            
            while True:
                try:
                    last_time = self.detector.data['time'].max() if not self.detector.data.empty else None
                    df = fetch_candles(self.timeframe, last_time)
                    if not df.empty:
                        self.detector.update_data(df)
                    time.sleep(self.detector.poll_interval)
                except Exception as e:
                    self.logger.error(f"Main loop error: {e}")
                    time.sleep(10)
                    
        except Exception as e:
            self.logger.error(f"Failed to start bot: {str(e)}")
            send_telegram(f"❌ *Bot Failed to Start for {self.timeframe}*:\n{str(e)}")

# ========================
# MAIN FUNCTION
# ========================
def run_bot():
    # Start both timeframe bots in parallel
    bot_5m = BotInstance("M5")
    bot_15m = BotInstance("M15")
    
    # Create threads for each bot instance
    thread_5m = threading.Thread(target=bot_5m.run)
    thread_15m = threading.Thread(target=bot_15m.run)
    
    thread_5m.daemon = True
    thread_15m.daemon = True
    
    thread_5m.start()
    thread_15m.start()
    
    # Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    logger.info("Launching main application")
    run_bot()
