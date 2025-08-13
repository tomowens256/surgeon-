# ============= main.py (FULLY FIXED) =============
# COMPATIBILITY FIXES
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
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (Input, Bidirectional, LSTM, 
                                     Dense, LayerNormalization)
from collections import defaultdict
import h5py

# ========================
# CONSTANTS & CONFIG
# ========================
# Use Google Drive path when in Colab
if 'COLAB_GPU' in os.environ:
    MODELS_DIR = "/content/drive/MyDrive/ml_models"
    print(f"Running in Colab, using models from: {MODELS_DIR}")
else:
    MODELS_DIR = "ml_models"

NY_TZ = pytz.timezone("America/New_York")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7977128069:AAFMUWbOTaYj_u7WG4giGdPM0znmuUaHIqU")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1704877982")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "101-002-31411886-001")
API_KEY = os.getenv("OANDA_API_KEY", "f5c1187cda431e23a8d65fa72fe3993f-3ec29ca62a004480040d991aa91a2193")
INSTRUMENT = "XAU_USD"

# Model and scaler paths
MODEL_5M = "5mbilstm_model.keras"
SCALER_5M = "scaler5mcrt.joblib"
MODEL_15M = "15mbilstm_model.keras"
SCALER_15M = "scaler15mcrt.joblib"

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

# Verify environment variables
logger.info(f"Telegram Token: {'set' if TELEGRAM_TOKEN else 'missing'}")
logger.info(f"Telegram Chat ID: {'set' if TELEGRAM_CHAT_ID else 'missing'}")
logger.info(f"Oanda API Key: {'set' if API_KEY else 'missing'}")

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
        
    logger.info(f"Attempting to send Telegram message: {message[:100]}...")
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
# MODEL LOADING WITH ARCHITECTURE RECONSTRUCTION
# ========================
def build_model(timeframe):
    """Build model architecture from scratch based on timeframe"""
    if timeframe == "M5":
        input_shape = (x_train.shape[1], x_train.shape[2])  # (1, 76) for M5
        units = 512
    else:  # M15 timeframe
        input_shape = (x_train.shape[1], x_train.shape[2])  # (1, 68) for M15
        units = 384
    
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(units)))
    model.add(Dropout(0.15))  # From your training script
    model.add(Dense(1, activation='sigmoid'))
    return model

def load_model_with_weights(model_path, timeframe):
    """Load model by reconstructing exact training architecture"""
    # Build model architecture
    model = build_model(timeframe)
    
    # Load weights
    try:
        model.load_weights(model_path)
        return model
    except Exception as e:
        logger.error(f"Weight loading failed: {str(e)}")
        raise RuntimeError(f"Could not load weights: {str(e)}")
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
        
        # Features to shift (volume estimation replaced by shifted features)
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
        """Vectorized CRT signal calculation with precise indexing"""
        if len(df) < 3:
            return None, None
            
        # Create working copy to avoid modifying original
        df = df.copy()
        
        # Create shifted columns for previous candles
        df['c1_low'] = df['low'].shift(2)
        df['c1_high'] = df['high'].shift(2)
        df['c2_low'] = df['low'].shift(1)
        df['c2_high'] = df['high'].shift(1)
        df['c2_close'] = df['close'].shift(1)
        
        # Calculate candle metrics
        df['c2_range'] = df['c2_high'] - df['c2_low']
        df['c2_mid'] = df['c2_low'] + (0.5 * df['c2_range'])
        
        # Vectorized conditions
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
        
        # Get the last signal only
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
        """Calculate technical indicators WITHOUT volume imputation"""
        df = df.copy().drop_duplicates(subset=['time'], keep='last')
        
        # REMOVED VOLUME ESTIMATION - Using shifted features instead
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
        """Calculate minutes closed based on actual candle timestamp"""
        df = df.copy()
        
        if self.timeframe == "M5":
            minute_buckets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
            minute_cols = [f'minutes,closed_{bucket}' for bucket in minute_buckets]
        else:  # M15 timeframe
            minute_buckets = [0, 15, 30, 45]
            minute_cols = [f'minutes,closed_{bucket}' for bucket in minute_buckets]
            
        # Initialize all columns to 0
        for col in minute_cols:
            df[col] = 0
        
        # Get the current candle's minute
        current_minute = df.iloc[-1]['time'].minute
        
        # Calculate bucket based on actual minute of the hour
        if self.timeframe == "M5":
            bucket = (current_minute // 5) * 5
        else:
            bucket = (current_minute // 15) * 15
            
        bucket_col = f'minutes,closed_{bucket}'
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
        
        # Volume features now rely solely on shifted values
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
        
        # CRITICAL: Using shifted features instead of volume estimation
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
# SIMPLIFIED TRADING BOT
# ========================
class TradingBot:
    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.model = None
        self.scaler = None
        self.feature_engineer = FeatureEngineer(timeframe)
        self.data = pd.DataFrame()
        self.logger = logging.getLogger(f"Bot.{timeframe}")
        
        # Configure paths
        if timeframe == "M5":
            model_path = os.path.join(MODELS_DIR, MODEL_5M)
            scaler_path = os.path.join(MODELS_DIR, SCALER_5M)
        else:
            model_path = os.path.join(MODELS_DIR, MODEL_15M)
            scaler_path = os.path.join(MODELS_DIR, SCALER_15M)
        
        # Load model with reconstructed architecture
        try:
            self.logger.info(f"Loading model for {timeframe}")
            self.model = load_model_with_weights(model_path, timeframe)
            self.scaler = joblib.load(scaler_path)
            self.logger.info("Model and scaler loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            send_telegram(f"❌ *CRITICAL ERROR*: Failed to load model for {timeframe}\n{str(e)}")
            raise
        
        # Fetch initial data
        self.data = self.fetch_initial_candles()
        if self.data.empty:
            self.logger.error("Initial candle fetch failed")
            send_telegram(f"❌ Initial candle fetch failed for {timeframe}")
            raise RuntimeError("Candle fetch failed")
        
        send_telegram(f"🚀 *Bot Started*: {timeframe} timeframe\nTime: {datetime.now(NY_TZ)}")

    def fetch_initial_candles(self):
        self.logger.info("Fetching initial candles")
        return fetch_candles(self.timeframe)

    def run(self):
        self.logger.info(f"Starting trading loop for {self.timeframe}")
        while True:
            try:
                # Fetch new data
                last_time = self.data['time'].max() if not self.data.empty else None
                new_data = fetch_candles(self.timeframe, last_time)
                
                if not new_data.empty:
                    self.data = pd.concat([self.data, new_data]).tail(201)
                
                # Check for signals
                self.check_signals()
                
                # Sleep before next check
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Main loop error: {str(e)}")
                time.sleep(60)

    def check_signals(self):
        if len(self.data) < 3:
            return
            
        signal_type, signal_data = self.feature_engineer.calculate_crt_signal_vectorized(self.data)
        if not signal_type:
            return
            
        self.logger.info(f"Signal detected: {signal_type}")
        
        # Generate features
        features = self.feature_engineer.generate_features(self.data, signal_type)
        if features is None:
            return
            
        # Prepare for prediction
        features_df = pd.DataFrame([features])
        
        # Predict
        try:
            features_array = features_df.values
            scaled_features = self.scaler.transform(features_array)
            reshaped_features = np.expand_dims(scaled_features, axis=1)
            prob = self.model.predict(reshaped_features, verbose=0)[0][0]
            outcome = "Worth Taking" if prob >= PREDICTION_THRESHOLD else "Likely Loss"
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return
            
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

# ========================
# MAIN EXECUTION
# ========================
def run_timeframes():
    # Create and run bots in separate threads
    bot_5m = TradingBot("M5")
    bot_15m = TradingBot("M15")
    
    thread_5m = threading.Thread(target=bot_5m.run, daemon=True)
    thread_15m = threading.Thread(target=bot_15m.run, daemon=True)
    
    thread_5m.start()
    thread_15m.start()
    
    # Keep main thread alive
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    # Force CPU usage
    tf.config.set_visible_devices([], 'GPU')
    
    # Start both timeframes
    run_timeframes()
