# COMPATIBILITY FIXES
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# NUMPY FIX
import numpy as np
try:
    np.float = float  # Fix for TensorFlow 2.x
except AttributeError:
    pass

# DISABLE UNNECESSARY LOGGING
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('ngrok').setLevel(logging.ERROR)
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
import joblib
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
import tensorflow as tf
from google.colab import drive
from IPython.display import clear_output

# ========================
# SUPPRESS TENSORFLOW LOGS
# ========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info logs
tf.get_logger().setLevel('ERROR')  # Only show errors

# ========================
# CONSTANTS & CONFIG
# ========================
NY_TZ = pytz.timezone("America/New_York")
MODELS_DIR = "/content/drive/MyDrive/ml_models"  # Colab-specific path
DEBUG_MODE = True  # Enable detailed debugging

# File size thresholds
MODEL_MIN_SIZE = 100 * 1024  # 100KB for model files
SCALER_MIN_SIZE = 2 * 1024   # 2KB for scaler files

# Prediction threshold (0.9140 for class 1)
PREDICTION_THRESHOLD = 0.9140

# Initialize logging with more verbosity
log_format = '%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s'
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# ========================
# COLAB SETUP FUNCTION
# ========================
def setup_colab():
    """Set up environment for Colab without remounting"""
    logger.debug("Configuring Colab environment...")
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    logger.info("Colab environment configured")
    return logger
    
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    logger.info("Colab setup completed")
    return logger

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

def send_telegram(message, token, chat_id):
    """Send formatted message to Telegram with detailed error handling and retries"""
    logger.debug(f"Attempting to send Telegram message: {message[:50]}...")
    
    if not token or not chat_id:
        logger.error("Telegram credentials missing")
        return False
        
    if len(message) > 4000:
        message = message[:4000] + "... [TRUNCATED]"
    
    # Escape special Markdown characters
    escape_chars = '_*[]()~`>#+-=|{}.!'
    for char in escape_chars:
        message = message.replace(char, '\\' + char)
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    max_retries = 3
    
    logger.debug(f"Telegram URL: {url.split('bot')[0]}bot***")
    logger.debug(f"Chat ID: {chat_id}")
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Telegram attempt {attempt+1}/{max_retries}")
            response = requests.post(url, json={
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'MarkdownV2'
            }, timeout=10)
            
            logger.debug(f"Telegram response: {response.status_code}, {response.text[:100]}")
            
            if response.status_code == 200 and response.json().get('ok'):
                logger.info("Telegram message sent successfully")
                return True
            else:
                error_msg = f"Telegram error {response.status_code}"
                if response.text:
                    error_msg += f": {response.text[:200]}"
                logger.error(error_msg)
                time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Telegram connection failed: {str(e)}")
            time.sleep(2 ** attempt)
    logger.error(f"Failed to send Telegram message after {max_retries} attempts")
    return False

def fetch_candles(timeframe, last_time=None, api_key=None):
    """Fetch exactly 201 candles for XAU_USD with full precision and robust error handling"""
    logger.debug(f"Fetching candles for {timeframe}, last_time: {last_time}")
    
    if not api_key:
        logger.error("Oanda API key missing")
        return pd.DataFrame()
        
    try:
        api = API(access_token=api_key, environment="practice")
    except Exception as e:
        logger.error(f"Oanda API initialization failed: {str(e)}")
        return pd.DataFrame()
        
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
    
    logger.debug(f"Oanda request params: {params}")
    
    for attempt in range(max_attempts):
        try:
            logger.debug(f"Fetch attempt {attempt+1}/{max_attempts}")
            request = instruments.InstrumentsCandles(instrument="XAU_USD", params=params)
            response = api.request(request)
            candles = response.get('candles', [])
            
            logger.debug(f"Received {len(candles)} candles")
            
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
                
            logger.debug(f"Returning {len(df)} candles")
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
        logger.debug(f"Initializing FeatureEngineer for {timeframe}")
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
            'vwap', 'vwap_std', 'rsi', 'ma_20', 'ma_30', 'ma_40', 'ma_极',
            'trend_strength_up', 'trend_strength_down', 'volume', 'body_size', 
            'wick_up', 'wick_down', 'prev_body_size', 'prev_wick_up', 'prev_wick_down', 
            'is_bad_combo', 'price_div_vol', 'rsi_div_macd', 'price_div_vwap', 
            'sl_div_atr', 'rrr_div_rsi', 'rsi_zone_neutral', 'rsi_zone_overbought', 
            'rsi_zone_oversold', 'rsi_zone_unknown', 'combo_flag_dead', 'combo_flag_fair',
            'combo_flag_fine', 'combo_flag2_dead', 'combo_flag2_fair', 'combo_flag2_fine'
        ]

 
        
    def calculate_crt_signal_vectorized(self, df):
        """Vectorized CRT signal calculation matching backtesting version"""
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
        
        # Vectorized conditions - EXACTLY AS IN BACKTESTING
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
        """Vectorized CRT signal calculation with precise indexing"""
        if len(df) < 3:
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
# MODEL LOADER
# ========================
class ModelLoader:
    def __init__(self, model_path, scaler_path):
        logger.debug(f"Loading model from {model_path}")
        logger.debug(f"Loading scaler from {scaler_path}")
        try:
            logger.debug("Loading TensorFlow model...")
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.debug("Model loaded successfully")
            
            logger.debug("Loading Scikit-Learn scaler...")
            self.scaler = joblib.load(scaler_path)
            logger.debug("Scaler loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
        
    def predict(self, features):
        logger.debug("Starting prediction...")
        scaled = self.scaler.transform([features])
        reshaped = scaled.reshape(1, 1, -1)
        prediction = self.model.predict(reshaped, verbose=0)[0][0]
        logger.debug(f"Prediction complete: {prediction}")
        return prediction

# ========================
# COLAB TRADING BOT
# ========================
class ColabTradingBot:
    def __init__(self, timeframe, credentials):
        self.timeframe = timeframe
        self.credentials = credentials
        self.logger = logging.getLogger(f"{timeframe}_bot")
        self.start_time = time.time()
        self.max_duration = 11.5 * 3600
        
        logger.info(f"Initializing {timeframe} bot")
        
        # Load model
        model_path = os.path.join(MODELS_DIR, "5mbilstm_model.keras" if timeframe == "M5" else "15mbilstm_model.keras")
        scaler_path = os.path.join(MODELS_DIR, "scaler5mcrt.joblib" if timeframe == "M5" else "scaler15mcrt.joblib")
        
        logger.debug(f"Model path: {model_path}")
        logger.debug(f"Scaler path: {scaler_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            logger.error(f"Scaler file not found: {scaler_path}")
            
        try:
            self.model_loader = ModelLoader(model_path, scaler_path)
            self.feature_engineer = FeatureEngineer(timeframe)
            self.data = pd.DataFrame()
            logger.info(f"Bot initialized for {timeframe}")
        except Exception as e:
            logger.error(f"Bot initialization failed: {str(e)}")
            raise

    def calculate_next_candle_time(self):
        now = datetime.now(NY_TZ)
        minute = now.minute
        
        if self.timeframe == "M5":
            next_minute = ((minute // 5) + 1) * 5
        else:
            next_minute = ((minute // 15) + 1) * 15
            
        if next_minute >= 60:
            result = now.replace(
                hour=now.hour+1, minute=0, second=0, microsecond=0
            ) + timedelta(minutes=next_minute - 60)
        else:
            result = now.replace(minute=next_minute, second=0, microsecond=0)
            
        logger.debug(f"Next candle time: {result}")
        return result

    def run(self):
        """Main bot execution loop"""
        thread_name = threading.current_thread().name
        logger.info(f"Starting bot thread: {thread_name}")
        
        session_start = time.time()
        timeout_msg = f"⏳ {self.timeframe} bot session will expire in 30 minutes"
        start_msg = f"🚀 {self.timeframe} bot started at {datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Test credentials before starting
        logger.info("Testing credentials...")
        creds_valid = self.test_credentials()
        if not creds_valid:
            logger.error("Credentials test failed. Exiting bot.")
            return
            
        logger.info("Sending startup message...")
        telegram_sent = send_telegram(start_msg, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
        logger.info(f"Startup message {'sent' if telegram_sent else 'failed to send'}")
        
        # Initial data fetch
        logger.info("Fetching initial data...")
        self.data = fetch_candles(
            self.timeframe,
            api_key=self.credentials['oanda_api_key']
        )
        logger.info(f"Initial data fetched: {len(self.data)} records")
        
        while True:
            try:
                # Check session time remaining
                elapsed = time.time() - session_start
                logger.debug(f"Session time elapsed: {elapsed/3600:.2f} hours")
                
                if elapsed > (self.max_duration - 1800) and not hasattr(self, 'timeout_sent'):
                    logger.warning("Session nearing timeout, sending warning")
                    send_telegram(timeout_msg, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
                    self.timeout_sent = True
                    
                if elapsed > self.max_duration:
                    logger.warning("Session timeout reached, exiting")
                    end_msg = f"🔴 {self.timeframe} bot session ended after 12 hours"
                    send_telegram(end_msg, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
                    return
                
                # Calculate sleep time to next candle + 5 seconds
                now = datetime.now(NY_TZ)
                next_candle = self.calculate_next_candle_time()
                sleep_seconds = max(1, (next_candle - now).total_seconds() + 5)
                
                logger.debug(f"Sleeping for {sleep_seconds:.1f} seconds until next candle")
                time.sleep(sleep_seconds)
                
                # Fetch new data
                last_time = self.data['time'].max() if not self.data.empty else None
                logger.debug(f"Fetching new data since {last_time}")
                new_data = fetch_candles(
                    self.timeframe,
                    last_time,
                    self.credentials['oanda_api_key']
                )
                
                if not new_data.empty:
                    logger.debug(f"Received {len(new_data)} new records")
                    self.data = pd.concat([self.data, new_data]).drop_duplicates('time').sort_values('time').tail(201)
                    logger.debug(f"Total records now: {len(self.data)}")
                else:
                    logger.warning("No new data fetched")
                    continue
                
                # Detect CRT pattern
                logger.debug("Checking for CRT pattern...")
                signal_type, signal_data = self.feature_engineer.calculate_crt_signal_vectorized(self.data)
                
                if not signal_type:
                    logger.debug("No CRT pattern detected")
                    continue
                    
                logger.info(f"CRT pattern detected: {signal_type}")
                
                # Generate features
                features = self.feature_engineer.generate_features(self.data, signal_type)
                if features is None:
                    logger.warning("Feature generation failed")
                    continue
                    
                # Get prediction
                logger.debug("Getting model prediction...")
                prediction = self.model_loader.predict(features)
                logger.info(f"Prediction: {prediction:.4f}")
                
                # Send ALL signals (both high and low confidence)
                confidence_level = "HIGH" if prediction > PREDICTION_THRESHOLD else "LOW"
                emoji = "🚨" if confidence_level == "HIGH" else "⚠️"
                
                message = (
                    f"{emoji} XAU/USD Signal ({self.timeframe})\n"
                    f"Type: {signal_type}\n"
                    f"Entry: {signal_data['entry']:.5f}\n"
                    f"SL: {signal_data['sl']:.5f}\n"
                    f"TP: {signal_data['tp']:.5f}\n"
                    f"Confidence: {prediction:.4f} ({confidence_level})\n"
                    f"Time: {datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
                )
                send_telegram(message, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
                logger.info(f"Signal sent ({confidence_level} confidence)")
                    
            except Exception as e:
                error_msg = f"❌ {self.timeframe} bot error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                send_telegram(error_msg[:1000], self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
                time.sleep(60)
                
    
    def test_credentials(self):
        """Test both Telegram and Oanda credentials with detailed logging"""
        logger.info("Testing credentials...")
        
        # Test Telegram
        test_msg = f"🔧 {self.timeframe} bot credentials test"
        logger.debug(f"Sending Telegram test: {test_msg}")
        telegram_ok = send_telegram(test_msg, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
        
        # Test Oanda
        oanda_ok = False
        try:
            logger.debug("Testing Oanda API with small candle request")
            # REMOVE THE 'count' PARAMETER HERE:
            test_data = fetch_candles("M5", api_key=self.credentials['oanda_api_key'])
            oanda_ok = not test_data.empty
            logger.debug(f"Oanda test {'succeeded' if oanda_ok else 'failed'}")
        except Exception as e:
            logger.error(f"Oanda test failed: {str(e)}")
            
        if not telegram_ok:
            logger.error("Telegram credentials test failed")
            
        if not oanda_ok:
            logger.error("Oanda credentials test failed")
            
        logger.info(f"Credentials test result: {'PASS' if telegram_ok and oanda_ok else 'FAIL'}")
        return telegram_ok and oanda_ok
# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    print("===== BOT STARTING =====")
    print(f"Start time: {datetime.now(NY_TZ)}")
    
    # Force debug logging to console
    debug_handler = logging.StreamHandler()
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(debug_handler)
    
    logger.info("Starting main execution")
    
    try:
        logger = setup_colab()
    except Exception as e:
        logger.error(f"Colab setup failed: {str(e)}")
    
    # Load credentials
    credentials = {
        'telegram_token': os.getenv("TELEGRAM_BOT_TOKEN"),
        'telegram_chat_id': os.getenv("TELEGRAM_CHAT_ID"),
        'oanda_account_id': os.getenv("OANDA_ACCOUNT_ID"),
        'oanda_api_key': os.getenv("OANDA_API_KEY")
    }
    
    # Log credentials status (without values)
    logger.info("Checking credentials...")
    credentials_status = {k: "SET" if v else "MISSING" for k, v in credentials.items()}
    for k, status in credentials_status.items():
        logger.info(f"{k}: {status}")
    
    if not all(credentials.values()):
        logger.error("Missing one or more credentials in environment variables")
        # Send alert if Telegram credentials are available
        if credentials['telegram_token'] and credentials['telegram_chat_id']:
            send_telegram("❌ Bot failed to start: Missing credentials", 
                         credentials['telegram_token'], credentials['telegram_chat_id'])
        sys.exit(1)
    
    logger.info("All credentials present")
    
    try:
        # Start bots
        logger.info("Creating M5 bot")
        bot_5m = ColabTradingBot("M5", credentials)
        logger.info("Creating M15 bot")
        bot_15m = ColabTradingBot("M15", credentials)
        
        # Run in separate threads
        logger.info("Starting bot threads")
        t1 = threading.Thread(target=bot_5m.run, name="M5_Bot")
        t2 = threading.Thread(target=bot_15m.run, name="M15_Bot")
        
        t1.daemon = True
        t2.daemon = True
        
        t1.start()
        logger.info("M5 bot thread started")
        t2.start()
        logger.info("M15 bot thread started")
        
        # Keep main thread alive with status updates
        logger.info("Main thread entering monitoring loop")
        while True:
            logger.info(f"Bot status: M5 {'alive' if t1.is_alive() else 'dead'}, "
                       f"M15 {'alive' if t2.is_alive() else 'dead'}")
            time.sleep(60)
            
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}", exc_info=True)
        # Attempt to send error via Telegram if credentials are available
        if credentials.get('telegram_token') and credentials.get('telegram_chat_id'):
            send_telegram(f"❌ Bot crashed: {str(e)[:500]}", 
                         credentials['telegram_token'], credentials['telegram_chat_id'])
        sys.exit(1)
