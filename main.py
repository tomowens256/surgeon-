# ====== main.py ======
# COMPATIBILITY FIXES FOR COLAB
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# NUMPY FIX
import numpy as np
try:
    np.float = float
except AttributeError:
    pass

# DISABLE UNNECESSARY LOGGING
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('ngrok').setLevel(logging.ERROR)
import sys
import time
import threading
import pytz
import pandas as pd
import pandas_ta as ta
import requests
import re
import traceback
import json
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, LayerNormalization
from collections import defaultdict
import h5py
import shutil
import tempfile

# ========================
# CONSTANTS & CONFIG - COLAB ADJUSTED
# ========================
# Use Google Drive path when in Colab
if 'COLAB_GPU' in os.environ:
    MODELS_DIR = "/content/drive/MyDrive/ml_models"
    print(f"Running in Colab, using models from: {MODELS_DIR}")
else:
    MODELS_DIR = "ml_models"

NY_TZ = pytz.timezone("America/New_York")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
API_KEY = os.getenv("OANDA_API_KEY")
INSTRUMENT = "XAU_USD"

# Model and scaler paths
MODEL_5M = "5mbilstm_model.keras"
SCALER_5M = "scaler5mcrt.joblib"
MODEL_15M = "15mbilstm_model.keras"
SCALER_15M = "scaler15mcrt.joblib"

# File size thresholds
MODEL_MIN_SIZE = 100 * 1024
SCALER_MIN_SIZE = 2 * 1024

# Prediction threshold
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

# ========================
# UTILITY FUNCTIONS
# ========================
def parse_oanda_time(time_str):
    try:
        if '.' in time_str and len(time_str.split('.')[1]) > 7:
            time_str = re.sub(r'\.(\d{6})\d+', r'.\1', time_str)
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.utc).astimezone(NY_TZ)
    except Exception as e:
        logger.error(f"Error parsing time {time_str}: {str(e)}")
        return datetime.now(NY_TZ)

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set, skipping message")
        return False
        
    logger.info(f"Attempting to send Telegram message: {message}")
    if len(message) > 4000:
        message = message[:4000] + "... [TRUNCATED]"
    
    # Escape special Markdown characters
    escape_chars = '_*[]()~`>#+-=|{}.!'
    for char in escape_chars:
        message = message.replace(char, '\\' + char)
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json={
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'MarkdownV2'
            }, timeout=10)
            
            if response.status_code == 200 and response.json().get('ok'):
                return True
            else:
                logger.error(f"Telegram error: {response.status_code} - {response.text}")
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Telegram connection failed: {str(e)}")
            time.sleep(2 ** attempt)
    logger.error(f"Failed to send Telegram message after {max_retries} attempts")
    return False

def fetch_candles(timeframe, last_time=None):
    logger.info(f"Fetching 201 candles for {INSTRUMENT} with timeframe {timeframe}")
    params = {
        "granularity": timeframe,
        "count": 201,
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
            oanda_api = API(access_token=API_KEY, environment="practice")
            request = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
            response = oanda_api.request(request)
            candles = response.get('candles', [])
            
            if not candles:
                logger.warning(f"No candles received on attempt {attempt+1}")
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
                    logger.error(f"Error parsing candle: {str(e)}")
                    continue
            
            if not data:
                logger.warning(f"Empty data after parsing on attempt {attempt+1}")
                continue
                
            df = pd.DataFrame(data).drop_duplicates(subset=['time'], keep='last')
            if last_time:
                df = df[df['time'] > last_time].sort_values('time')
            return df
            
        except V20Error as e:
            if "rate" in str(e).lower() or getattr(e, 'code', 0) in [429, 502]:
                wait_time = sleep_time * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            else:
                error_details = f"Status: {getattr(e, 'code', 'N/A')} | Message: {getattr(e, 'msg', str(e))}"
                logger.error(f"❌ Oanda API error: {error_details}")
                break
        except Exception as e:
            logger.error(f"❌ General error fetching candles: {str(e)}")
            logger.error(traceback.format_exc())
            time.sleep(sleep_time)
    
    logger.error(f"Failed to fetch candles after {max_attempts} attempts")
    return pd.DataFrame()

# ========================
# FEATURE ENGINEER WITH FIXES
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
                'minutes_closed_0', 'minutes_closed_5', 'minutes_closed_10', 
                'minutes_closed_15', 'minutes_closed_20', 'minutes_closed_25', 
                'minutes_closed_30', 'minutes_closed_35', 'minutes_closed_40', 
                'minutes_closed_45', 'minutes_closed_50', 'minutes_closed_55'
            ]
        else:  # M15 timeframe
            self.features = self.base_features + [
                'minutes_closed_0', 'minutes_closed_15', 
                'minutes_closed_30', 'minutes_closed_45'
            ]
        
        self.shift_features = [
            'garman_klass_vol', 'rsi_20', 'bb_low', 'bb_mid', 'bb_high',
            'atr_z', 'macd_z', 'dollar_volume', 'ma_10', 'ma_100',
            'vwap', 'vwap_std', 'rsi', 'ma_20', 'ma_30', 'ma_40', 'ma_60',
            'trend_strength_up', 'trend_strength_down', 'volume', 'body_size', 
            'wick_up', 'wick_down', 'prev_body_size', 'prev_wick_up', 'prev_wick_down', 
            'is_bad_combo', 'price_div_vol', 'rsi_div_macd', 'price_div_vwap', 
            'sl_div_atr', 'rrr_div_rsi', 'rsi_zone_neutral', 'rsi_zone_overbought', 
            'rsi_zone_oversold', 'rsi_zone_unknown', 'combo_flag_dead', 'combo_flag_fair',
            'combo_flag_fine', 'combo_flag2_dead', 'combo_flag2_fair', 'combo_flag2_fine'
        ]

    def calculate_crt_signal_vectorized(self, df):
        if len(df) < 3:
            return None, None
            
        df = df.copy()
        df['c1_low'] = df['low'].shift(2)
        df['c1_high'] = df['high'].shift(2)
        df['c2_low'] = df['low'].shift(1)
        df['c2_high'] = df['high'].shift(1)
        df['c2_close'] = df['close'].shift(1)
        
        df['c2_range'] = df['c2_high'] - df['c2_low']
        df['c2_mid'] = df['c2_low'] + (0.5 * df['c2_range'])
        
        buy_mask = (
            (df['c2_low'] < df['c1_low']) & 
            (df['c2_close'] > df['c1_low']) & 
            (df['open'] > df['c2_mid'])
        )
        
        sell_mask = (
            (df['c2_high'] > df['c1_high']) & 
            (df['c2_close'] < df['c1_high']) & 
            (df['open'] < df['c2_mid'])
        )
        
        last_row = df.iloc[-1]
        
        if buy_mask.iloc[-1]:
            signal_type = 'BUY'
            entry = last_row['open']
            sl = last_row['c2_low']
            risk = abs(entry - sl)
            tp = entry + 4 * risk
            return signal_type, {'entry': entry, 'sl': sl, 'tp': tp, 'time': last_row['time']}
        elif sell_mask.iloc[-1]:
            signal_type = 'SELL'
            entry = last_row['open']
            sl = last_row['c2_high']
            risk = abs(sl - entry)
            tp = entry - 4 * risk
            return signal_type, {'entry': entry, 'sl': sl, 'tp': tp, 'time': last_row['time']}
        else:
            return None, None

    def calculate_technical_indicators(self, df):
        df = df.copy().drop_duplicates(subset=['time'], keep='last')
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

    def calculate_minutes_closed(self, df):
        df = df.copy()
        
        if self.timeframe == "M5":
            minute_buckets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
            minute_cols = [f'minutes_closed_{bucket}' for bucket in minute_buckets]
        else:  # M15 timeframe
            minute_buckets = [0, 15, 30, 45]
            minute_cols = [f'minutes_closed_{bucket}' for bucket in minute_buckets]
            
        for col in minute_cols:
            df[col] = 0
        
        current_minute = df.iloc[-1]['time'].minute
        
        if self.timeframe == "M5":
            bucket = (current_minute // 5) * 5
        else:
            bucket = (current_minute // 15) * 15
            
        bucket_col = f'minutes_closed_{bucket}'
        if bucket_col in df.columns:
            df[bucket_col] = 1
            
        return df

    def generate_features(self, df, signal_type):
        if len(df) < 200:
            return None
        
        df = df.tail(200).copy()
        df = self.calculate_technical_indicators(df)
        df = self.calculate_trade_features(df, signal_type, df.iloc[-1]['open'])
        df = self.calculate_categorical_features(df)
        df = self.calculate_minutes_closed(df)
        
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

# ============= UPDATED main.py =============
# ... [previous code remains the same until BotInstance class] ...

class BotInstance:
    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.model = None
        self.scaler = None
        self.detector = None
        self.logger = logging.getLogger(f"{__name__}.{timeframe}")
        self.scheduler = None
        self.active_trades = {}
        
    def robust_model_loader(self, model_path):
        """Robust model loader with architecture reconstruction"""
        self.logger.info(f"Loading model with robust loader: {model_path}")
        
        # Determine model architecture based on timeframe
        if self.timeframe == "M5":
            input_shape = (1, 76)
            lstm_units = [512, 256]
        else:  # M15
            input_shape = (1, 68)
            lstm_units = [384, 192]
        
        try:
            # Attempt 1: Standard loading
            return load_model(model_path, compile=False)
        except Exception as e:
            self.logger.warning(f"Standard loading failed: {str(e)}")
            
            # Attempt 2: Build model from scratch
            self.logger.info("Building model from scratch...")
            model = tf.keras.Sequential()
            
            # Input layer - use Input instead of InputLayer for compatibility
            model.add(tf.keras.layers.Input(shape=input_shape, name="input_layer"))
            
            # First BiLSTM layer
            model.add(Bidirectional(
                LSTM(lstm_units[0], return_sequences=True, 
                     kernel_initializer='glorot_uniform'),
                name="bidirectional_1"
            ))
            
            # Second BiLSTM layer
            model.add(Bidirectional(
                LSTM(lstm_units[1], kernel_initializer='glorot_uniform'),
                name="bidirectional_2"
            ))
            
            # Dense layers
            model.add(Dense(128, activation='relu', name="dense_1"))
            model.add(Dense(1, activation='sigmoid', name="output"))
            
            # Load weights
            try:
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
                self.logger.warning("Loaded weights with possible mismatches")
                return model
            except Exception as e:
                self.logger.error(f"Weight loading failed: {str(e)}")
                raise RuntimeError(f"Could not load weights: {str(e)}")
        
    def run(self):
        self.logger.info(f"Starting trading bot for {self.timeframe}")
        send_telegram(f"🚀 *Bot Started*\nTimeframe: {self.timeframe}\nTime: {datetime.now(NY_TZ)}")
        
        try:
            # Configure paths
            if self.timeframe == "M5":
                model_path = os.path.join(MODELS_DIR, MODEL_5M)
                scaler_path = os.path.join(MODELS_DIR, SCALER_5M)
            else:  # M15 timeframe
                model_path = os.path.join(MODELS_DIR, MODEL_15M)
                scaler_path = os.path.join(MODELS_DIR, SCALER_15M)
                
            # Load model with robust loader
            self.model = self.robust_model_loader(model_path)
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            
            # Initialize data
            df = fetch_candles(self.timeframe)
            if df.empty:
                self.logger.error("Initial candle fetch failed")
                return
                
            # Create feature engineer
            feature_engineer = FeatureEngineer(self.timeframe)
            
            self.logger.info(f"Bot started successfully for {self.timeframe}")
            
            # Main trading loop
            while True:
                try:
                    # Check for new signals
                    signal_type, signal_data = feature_engineer.calculate_crt_signal_vectorized(df)
                    
                    if signal_type:
                        self.logger.info(f"Signal detected: {signal_type}")
                        # Generate features and predict
                        features = feature_engineer.generate_features(df, signal_type)
                        if features is not None:
                            features_df = pd.DataFrame([features], columns=feature_engineer.features)
                            prob, outcome = self.predict_single_model(features_df)
                            
                            # Send alert
                            alert_msg = (
                                f"🔔 *{signal_type} SIGNAL* {self.timeframe}\n"
                                f"Entry: {signal_data['entry']:.5f}\n"
                                f"SL: {signal_data['sl']:.5f}\n"
                                f"TP: {signal_data['tp']:.5f}\n"
                                f"Probability: {prob:.4f}\n"
                                f"Decision: {outcome}"
                            )
                            send_telegram(alert_msg)
                    
                    # Fetch new data periodically
                    if datetime.now().minute % 5 == 0:  # Refresh every 5 minutes
                        new_df = fetch_candles(self.timeframe, df['time'].max())
                        if not new_df.empty:
                            df = pd.concat([df, new_df]).tail(201)
                            
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    self.logger.error(f"Main loop error: {e}")
                    time.sleep(60)
                    
        except Exception as e:
            self.logger.error(f"Failed to start bot: {str(e)}")
            send_telegram(f"❌ *Bot Failed* {self.timeframe}:\n{str(e)}")

    # ... [rest of BotInstance remains the same] ...

    def predict_single_model(self, features_df):
        try:
            features_array = features_df.values
            scaled_features = self.scaler.transform(features_array)
            reshaped_features = np.expand_dims(scaled_features, axis=1)
            prob = self.model.predict(reshaped_features, verbose=0)[0][0]
            final_pred = 1 if prob >= PREDICTION_THRESHOLD else 0
            outcome = "Worth Taking" if final_pred == 1 else "Likely Loss"
            return prob, outcome
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return None, None

    def stop(self):
        self.active = False
        self.thread.join(timeout=5)

# ========================
# MAIN EXECUTION FUNCTION
# ========================
def run_timeframes():
    """Run both timeframes in parallel"""
    # Initialize both bots
    bot_5m = BotInstance("M5")
    bot_15m = BotInstance("M15")
    
    # Keep the main thread alive
    while True:
        time.sleep(3600)  # Sleep for 1 hour at a time

# Run the bot when executed directly
if __name__ == "__main__":
    # Force CPU usage in Colab
    tf.config.set_visible_devices([], 'GPU')
    
    # Start both timeframes
    run_timeframes()
