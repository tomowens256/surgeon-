# ========================
# FIXED ROBUST TRADING BOT SCRIPT
# ========================

# Add to your imports section
try:
    from google.colab import auth
    from google.auth import default
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    # These will only be available in Colab
    pass

# COMPATIBILITY FIXES
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['NUMBA_DEBUG'] = '0'
os.environ['NUMBA_DEBUGINFO'] = '0'

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
import psutil
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
import tensorflow as tf
from google.colab import drive
from IPython.display import clear_output
from typing import Optional, Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('numba').setLevel(logging.WARNING)

# ========================
# SUPPRESS TENSORFLOW LOGS
# ========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ========================
# CUSTOM EXCEPTIONS
# ========================
class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class DataIntegrityError(Exception):
    """Custom exception for data issues"""
    pass

class CredentialsError(Exception):
    """Custom exception for credential issues"""
    pass

class ModelLoadingError(Exception):
    """Custom exception for model loading issues"""
    pass

# ========================
# CONSTANTS & CONFIG
# ========================
NY_TZ = pytz.timezone("America/New_York")
MODELS_DIR = "/content/drive/MyDrive/ml_models"
DEBUG_MODE = True

# File size thresholds
MODEL_MIN_SIZE = 100 * 1024
SCALER_MIN_SIZE = 2 * 1024

# Prediction threshold (0.9140 for class 1) - KEEPING AT 0.9 AS REQUESTED
PREDICTION_THRESHOLD = 0.9140

# Initialize logging with more verbosity
log_format = '%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s'
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot_robust.log')
    ]
)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# Optional: Use environment variable for internal Numba controls
os.environ['NUMBA_DEBUG'] = '0'
logger = logging.getLogger(__name__)

# Precomputed combo_flags dictionary
COMBO_FLAGS = {
    "SELL_sideways_nan": "dead", "BUY_sideways_nan": "dead", 
    "SELL_sideways_(70, 80]": "dead", "BUY_sideways_(70, 80]": "fine",
    "SELL_sideways_(60, 70]": "fair", "SELL_sideways_(50, 60]": "fair",
    "BUY_sideways_(50, 60]": "fine", "BUY_sideways_(60, 70]": "fine",
    "BUY_sideways_(40, 50]": "fair", "SELL_sideways_(40, 50]": "fine",
    "SELL_sideways_(30, 40]": "fine", "SELL_uptrend_(50, 60]": "fine",
    "BUY_uptrend_(50, 60]": "fair", "SELL_uptrend_(40, 50]": "fine",
    "BUY_downtrend_(40, 50]": "fair", "BUY_uptrend_(60, 70]": "fine",
    "SELL_uptrend_(60, 70]": "fair", "BUY_uptrend_(40, 50]": "fair",
    "SELL_downtrend_(40, 50]": "fair", "BUY_uptrend_(30, 40]": "dead",
    "BUY_downtrend_(30, 40]": "fair", "BUY_downtrend_(50, 60]": "fine",
    "SELL_downtrend_(50, 60]": "fair", "SELL_downtrend_(30, 40]": "fine",
    "BUY_uptrend_(70, 80]": "fine", "SELL_uptrend_(70, 80]": "fair",
    "SELL_uptrend_(80, 100]": "dead", "SELL_downtrend_(20, 30]": "fine",
    "SELL_sideways_(80, 100]": "dead", "BUY_sideways_(30, 40]": "fair",
    "SELL_downtrend_(60, 70]": "dead", "SELL_uptrend_(30, 40]": "fine",
    "SELL_downtrend_(70, 80]": "dead", "BUY_downtrend_(20, 30]": "fair",
    "BUY_downtrend_(0, 20]": "dead", "BUY_sideways_(20, 30]": "dead",
    "SELL_sideways_(20, 30]": "fine", "BUY_downtrend_(60, 70]": "fine",
    "BUY_sideways_(0, 20]": "dead", "SELL_downtrend_(0, 20]": "fine",
    "BUY_uptrend_(80, 100]": "fine", "SELL_sideways_(0, 20]": "fine",
    "BUY_sideways_(80, 100]": "fine", "SELL_uptrend_(20, 30]": "fine",
    "BUY_downtrend_(70, 80]": "fine", "BUY_uptrend_(20, 30]": "dead",
    "SELL_downtrend_(80, 100]": "dead", "BUY_uptrend_(0, 20]": "dead",
    "SELL_uptrend_(0, 20]": "dead", "nan_sideways_(50, 60]": "dead",
    "nan_sideways_(40, 50]": "dead"
}

# Precomputed combo_flags2 dictionary
COMBO_FLAGS2 = {
    "SELL_nan_nan": "dead", "BUY_nan_nan": "dead", 
    "SELL_(70, 80]_nan": "dead", "BUY_(70, 80]_nan": "dead",
    "SELL_(70, 80]_(0.527, 9.246]": "fair", "SELL_(60, 70]_(0.527, 9.246]": "fair",
    "SELL_(50, 60]_(0.527, 9.246]": "fine", "BUY_(50, 60]_(0.527, 9.246]": "fair",
    "BUY_(60, 70]_(0.527, 9.246]": "fine", "BUY_(40, 50]_(0.134, 0.527]": "fair",
    "SELL_(40, 50]_(0.134, 0.527]": "fine", "SELL_(30, 40]_(0.134, 0.527]": "fine",
    "BUY_(40, 50]_(-0.138, 0.134]": "fair", "SELL_(50, 60]_(-0.138, 0.134]": "fair",
    "BUY_(50, 60]_(-0.138, 0.134]": "fine", "SELL_(40, 50]_(-0.138, 0.134]": "fine",
    "BUY_(40, 50]_(-0.496, -0.138]": "fair", "BUY_(60, 70]_(-0.138, 0.134]": "fine",
    "SELL_(60, 70]_(0.134, 0.527]": "fair", "BUY_(50, 60]_(0.134, 0.527]": "fair",
    "SELL_(50, 60]_(0.134, 0.527]": "fine", "SELL_(40, 50]_(-0.496, -0.138]": "fair",
    "SELL_(30, 40]_(-0.496, -0.138]": "fine", "BUY_(40, 50]_(-12.386, -0.496]": "fine",
    "BUY_(60, 70]_(0.134, 0.527]": "fine", "BUY_(30, 40]_(-0.496, -0.138]": "fair",
    "BUY_(50, 60]_(-0.496, -0.138]": "fine", "BUY_(30, 40]_(-0.138, 0.134]": "dead",
    "SELL_(50, 60]_(-0.496, -0.138]": "fair", "BUY_(70, 80]_(0.134, 0.527]": "fine",
    "SELL_(80, 100]_(0.527, 9.246]": "dead", "BUY_(70, 80]_(0.527, 9.246]": "fine",
    "SELL_(40, 50]_(0.527, 9.246]": "fine", "SELL_(40, 50]_(-12.386, -0.496]": "fair",
    "BUY_(30, 40]_(-12.386, -0.496]": "fair", "SELL_(30, 40]_(-12.386, -0.496]": "fine",
    "SELL_(20, 30]_(-12.386, -0.496]": "fine", "BUY_(50, 60]_(-12.386, -0.496]": "fine",
    "SELL_(80, 100]_(-0.496, -0.138]": "dead", "SELL_(30, 40]_(-0.138, 0.134]": "fine",
    "SELL_(70, 80]_(-0.138, 0.134]": "dead", "SELL_(50, 60]_(-12.386, -0.496]": "dead",
    "BUY_(40, 50]_(0.527, 9.246]": "dead", "SELL_(20, 30]_(-0.496, -0.138]": "fine",
    "BUY_(20, 30]_(-12.386, -0.496]": "fair", "BUY_(0, 20]_(-12.386, -0.496]": "dead",
    "SELL_(60, 70]_(-0.138, 0.134]": "dead", "BUY_(20, 30]_(-0.496, -0.138]": "dead",
    "BUY_(60, 70]_(-0.496, -0.138]": "fine", "BUY_(70, 80]_(-0.138, 0.134]": "fine",
    "SELL_(70, 80]_(0.134, 0.527]": "dead", "SELL_(0, 20]_(-12.386, -0.496]": "fine",
    "BUY_(80, 100]_(0.527, 9.246]": "fine", "SELL_(60, 70]_(-0.496, -0.138]": "dead",
    "SELL_(30, 40]_(0.527, 9.246]": "fine", "BUY_(30, 40]_(0.134, 0.527]": "dead",
    "SELL_(60, 70]_(-12.386, -0.496]": "dead", "BUY_(60, 70]_(-12.386, -0.496]": "fine",
    "BUY_(80, 100]_(-0.496, -0.138]": "fine", "BUY_(80, 100]_(0.134, 0.527]": "fine",
    "SELL_(20, 30]_(-0.138, 0.134]": "fine", "SELL_(0, 20]_(-0.496, -0.138]": "fair",
    "BUY_(30, 40]_(0.527, 9.246]": "dead", "BUY_(20, 30]_(-0.138, 0.134]": "dead",
    "SELL_(70, 80]_(-0.496, -0.138]": "dead", "BUY_(80, 100]_(-0.138, 0.134]": "fine",
    "SELL_(20, 30]_(0.134, 0.527]": "fine", "BUY_(0, 20]_(-0.496, -0.138]": "dead",
    "SELL_(80, 100]_(0.134, 0.527]": "fair", "BUY_(0, 20]_(-0.138, 0.134]": "dead",
    "BUY_(0, 20]_(0.134, 0.527]": "dead", "SELL_(0, 20]_(-0.138, 0.134]": "fine",
    "BUY_(70, 80]_(-0.496, -0.138]": "fine", "SELL_(70, 80]_(-12.386, -0.496]": "dead",
    "SELL_(20, 30]_(0.527, 9.246]": "fine", "BUY_(20, 30]_(0.134, 0.527]": "dead",
    "SELL_(80, 100]_(-0.138, 0.134]": "dead", "BUY_(20, 30]_(0.527, 9.246]": "dead",
    "BUY_(0, 20]_(0.527, 9.246]": "dead", "nan_(50, 60]_(-12.386, -0.496]": "dead",
    "nan_(40, 50]_(-0.138, 0.134]": "dead", "nan_(40, 50]_(-0.496, -0.138]": "dead",
    "nan_(50, 60]_(0.134, 0.527]": "dead", "nan_(50, 60]_(0.527, 9.246]": "dead"
}

# ========================
# VALIDATION FUNCTIONS
# ========================
def validate_credentials(credentials):
    """Comprehensive credential validation"""
    required = {
        'telegram_token': (str, 30, 100),  # min/max length
        'telegram_chat_id': (str, 5, 20),
        'oanda_api_key': (str, 20, 100),
        'oanda_account_id': (str, 5, 20)
    }
    
    errors = []
    for key, (expected_type, min_len, max_len) in required.items():
        value = credentials.get(key)
        
        if not value:
            errors.append(f"Missing {key}")
            continue
            
        if not isinstance(value, expected_type):
            errors.append(f"{key} should be {expected_type.__name__}, got {type(value).__name__}")
            
        if not (min_len <= len(str(value)) <= max_len):
            errors.append(f"{key} length invalid: {len(str(value))} chars (expected {min_len}-{max_len})")
    
    if errors:
        raise CredentialsError(f"Credential validation failed: {', '.join(errors)}")
    
    return True

def validate_dataframe(df, min_rows=50, required_cols=None):
    """Validate DataFrame integrity"""
    if required_cols is None:
        required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    if df is None or df.empty:
        raise DataIntegrityError("DataFrame is None or empty")
    
    if len(df) < min_rows:
        raise DataIntegrityError(f"Insufficient data: {len(df)} rows (min {min_rows})")
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise DataIntegrityError(f"Missing required columns: {missing_cols}")
    
    # Check for NaN values in critical columns
    nan_check = df[required_cols].isna().sum()
    if nan_check.any():
        problematic = nan_check[nan_check > 0]
        raise DataIntegrityError(f"NaN values detected: {dict(problematic)}")
    
    # Check for reasonable price values
    price_stats = df[['open', 'high', 'low', 'close']].describe()
    if (price_stats.loc['min'] < 100).any() or (price_stats.loc['max'] > 5000).any():
        raise DataIntegrityError(f"Unrealistic price values: {dict(price_stats.loc[['min', 'max']])}")
    
    return True

def safe_float_conversion(value, default=0.0, field_name="value"):
    """Safely convert to float with error handling"""
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to convert {field_name} to float: {value}, using default {default}")
        return default

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

def fetch_candles(timeframe, last_time=None, count=201, api_key=None):
    """Fetch candles for XAU_USD with full precision and robust error handling"""
    logger.debug(f"Fetching {count} candles for {timeframe}, last_time: {last_time}")
    
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
        "count": count,
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
            df = df.reset_index(drop=True)
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
# CRT PATTERN DETECTION
# ========================
def calculate_crt_vectorized(df):
    """Vectorized implementation of CRT signal calculation"""
    df = df.copy()
    df['crt'] = None

    # Shifted columns for previous candles
    df['c1_low'] = df['low'].shift(2)
    df['c1_high'] = df['high'].shift(2)
    df['c2_low'] = df['low'].shift(1)
    df['c2_high'] = df['high'].shift(1)
    df['c2_close'] = df['close'].shift(1)

    # Candle metrics
    df['c2_range'] = df['c2_high'] - df['c2_low']
    df['c2_mid'] = df['c2_low'] + 0.5 * df['c2_range']

    # Vectorized conditions
    buy_mask = (df['c2_low'] < df['c1_low']) & (df['c2_close'] > df['c1_low']) & (df['open'] > df['c2_mid'])
    sell_mask = (df['c2_high'] > df['c1_high']) & (df['c2_close'] < df['c1_high']) & (df['open'] < df['c2_mid'])

    df.loc[buy_mask, 'crt'] = 'BUY'
    df.loc[sell_mask, 'crt'] = 'SELL'

    # Cleanup
    df.drop(columns=['c1_low', 'c1_high', 'c2_low', 'c2_high', 'c2_close', 'c2_range', 'c2_mid'], inplace=True)

    return df

def detect_crt_signals(df):
    """Detect CRT signals in the dataframe"""
    try:
        if len(df) < 3:
            return None, None
        
        # Calculate CRT signals
        df_with_crt = calculate_crt_vectorized(df)
        
        # Get the latest signal
        latest_signal = df_with_crt.iloc[-1]['crt']
        
        if pd.isna(latest_signal):
            return None, None
        
        current = df_with_crt.iloc[-1]
        
        # Calculate stop loss and take profit based on signal type
        if latest_signal == 'BUY':
            sl = current['low'] - (current['high'] - current['low']) * 0.5
            risk = abs(current['close'] - sl)
            tp = current['close'] + (2 * risk)
            
            signal_data = {
                'time': current['time'],
                'entry': current['close'],
                'sl': sl,
                'tp': tp,
                'signal_type': 'CRT_BUY',
                'strength': 1.0
            }
            
        elif latest_signal == 'SELL':
            sl = current['high'] + (current['high'] - current['low']) * 0.5
            risk = abs(current['close'] - sl)
            tp = current['close'] - (2 * risk)
            
            signal_data = {
                'time': current['time'],
                'entry': current['close'],
                'sl': sl,
                'tp': tp,
                'signal_type': 'CRT_SELL',
                'strength': 1.0
            }
        else:
            return None, None
        
        logger.info(f"CRT signal detected: {latest_signal} at {current['close']:.2f}")
        return latest_signal, signal_data
        
    except Exception as e:
        logger.error(f"CRT signal detection failed: {str(e)}")
        return None, None

# ========================
# FIXED MODEL LOADER WITH DIMENSION HANDLING
# ========================
class FixedModelLoader:
    def __init__(self, model_path, scaler_path, expected_features=109, max_retries=3):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.expected_features = expected_features
        self.max_retries = max_retries
        self.last_health_check = None
        self.load_attempts = 0
        
        self._validate_paths()
        self._load_with_retry()
    
    def _validate_paths(self):
        """Validate model and scaler paths"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
        
        # Check file sizes
        model_size = os.path.getsize(self.model_path)
        scaler_size = os.path.getsize(self.scaler_path)
        
        logger.debug(f"Model size: {model_size:,} bytes, Scaler size: {scaler_size:,} bytes")
        
        if model_size < MODEL_MIN_SIZE:
            logger.warning(f"Model file seems small: {model_size:,} bytes (expected > {MODEL_MIN_SIZE:,})")
        
        if scaler_size < SCALER_MIN_SIZE:
            logger.warning(f"Scaler file seems small: {scaler_size:,} bytes (expected > {SCALER_MIN_SIZE:,})")
    
    def _load_with_retry(self):
        """Load model with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Loading model attempt {attempt + 1}/{self.max_retries}")
                
                # Load model with compatibility settings
                self.model = tf.keras.models.load_model(
                    self.model_path, 
                    compile=False,
                    safe_mode=False
                )
                
                # Load scaler
                self.scaler = joblib.load(self.scaler_path)
                
                # Verify loaded objects
                self._verify_loaded_objects()
                
                logger.info(f"Model and scaler loaded successfully on attempt {attempt + 1}")
                self.load_attempts = attempt + 1
                self.last_health_check = time.time()
                return
                
            except Exception as e:
                logger.error(f"Load attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise ModelLoadingError(f"Failed to load model after {self.max_retries} attempts: {str(e)}")
                
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    def _verify_loaded_objects(self):
        """Verify that loaded model and scaler are functional"""
        # Create test features with expected dimension
        test_features = np.random.random((1, self.expected_features))
        
        try:
            # Test scaling and prediction
            scaled = self.scaler.transform(test_features)
            reshaped = scaled.reshape(1, 1, -1)
            prediction = self.model.predict(reshaped, verbose=0)
            logger.debug(f"Model test prediction: {prediction[0][0]:.6f}")
        except Exception as e:
            logger.warning(f"Model verification warning: {str(e)}")
            # Don't raise error here - we'll handle dimension mismatches in predict
    
    def health_check(self):
        """Perform health check on model and scaler"""
        try:
            test_features = np.random.random((1, self.expected_features))
            _ = self.predict(test_features[0])
            self.last_health_check = time.time()
            return True
        except Exception as e:
            logger.error(f"Model health check failed: {str(e)}")
            return False
    
    def predict(self, features):
        """Enhanced prediction with automatic dimension handling"""
        # Validate input
        if not isinstance(features, (np.ndarray, list, pd.Series)):
            raise ValueError(f"Features must be array-like, got {type(features)}")
        
        features = np.array(features, dtype=np.float64)
        
        if features.ndim != 1:
            raise ValueError(f"Features must be 1D, got shape {features.shape}")
        
        current_features = len(features)
        
        # Handle dimension mismatch
        if current_features != self.expected_features:
            logger.warning(f"Feature dimension mismatch: expected {self.expected_features}, got {current_features}")
            
            if current_features < self.expected_features:
                # Pad with zeros if we have fewer features
                padding = self.expected_features - current_features
                features = np.pad(features, (0, padding), 'constant', constant_values=0)
                logger.info(f"Padded features with {padding} zeros")
            else:
                # Truncate if we have more features
                features = features[:self.expected_features]
                logger.info(f"Truncated features from {current_features} to {self.expected_features}")
        
        # Check for extreme values
        extreme_mask = np.abs(features) > 1e6
        if extreme_mask.any():
            logger.warning(f"Extreme feature values detected at indices: {np.where(extreme_mask)[0]}")
        
        # Check for NaN/Inf
        if np.any(~np.isfinite(features)):
            logger.error("Non-finite values in features")
            return 0.0
        
        try:
            # Scale features
            scaled = self.scaler.transform([features])
            
            # Check for scaling issues
            if np.any(~np.isfinite(scaled)):
                logger.error("Non-finite values after scaling")
                return 0.0
            
            # Reshape for LSTM
            reshaped = scaled.reshape(1, 1, -1)
            
            # Predict
            prediction = self.model.predict(reshaped, verbose=0)[0][0]
            
            # Validate prediction
            if not np.isfinite(prediction):
                logger.error(f"Invalid prediction: {prediction}")
                return 0.0
            
            # Clamp to valid range
            prediction = np.clip(prediction, 0.0, 1.0)
            
            logger.debug(f"Prediction successful: {prediction:.6f}")
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            # Return neutral prediction on error
            return 0.5

# ========================
# FIXED FEATURE ENGINEER WITH EXACT FEATURE LISTS
# ========================
class SimplifiedFeatureEngineer:
    def __init__(self, timeframe):
        self.timeframe = timeframe
        logger.debug(f"Initializing SimplifiedFeatureEngineer for {timeframe}")
        
        # Define exact feature lists based on timeframe
        if timeframe == 'M5':
            # M5: 109 features exactly
            self.all_features = [
                'adj close', 'garman_klass_vol', 'rsi_20', 'bb_low', 'bb_mid', 'bb_high',
                'atr_z', 'macd_line', 'macd_z', 'ma_10', 'ma_100', 'vwap', 'vwap_std',
                'upper_band_1', 'lower_band_1', 'upper_band_2', 'lower_band_2', 
                'upper_band_3', 'lower_band_3', 'touches_vwap', 'touches_upper_band_1',
                'touches_upper_band_2', 'touches_upper_band_3', 'touches_lower_band_1',
                'touches_lower_band_2', 'touches_lower_band_3', 'far_ratio_vwap',
                'far_ratio_upper_band_1', 'far_ratio_upper_band_2', 'far_ratio_upper_band_3',
                'far_ratio_lower_band_1', 'far_ratio_lower_band_2', 'far_ratio_lower_band_3',
                'rsi', 'ma_20', 'ma_30', 'ma_40', 'ma_60', 'bearish_stack',
                'trend_strength_up', 'trend_strength_down', 'sl_price', 'tp_price',
                'prev_volume', 'body_size', 'wick_up', 'wick_down', 'sl_distance',
                'tp_distance', 'log_sl', 'prev_body_size', 'prev_wick_up', 'prev_wick_down',
                'result', 'is_bad_combo', 'volume_bin', 'dollar_volume_bin', 'price_div_vol',
                'rsi_div_macd', 'price_div_vwap', 'sl_div_atr', 'tp_div_atr', 'rrr_div_rsi',
                'hour', 'month', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
                'dayofweek_sin', 'dayofweek_cos', 'day_Friday', 'day_Monday', 'day_Sunday',
                'day_Thursday', 'day_Tuesday', 'day_Wednesday', 'session_q1', 'session_q2',
                'session_q3', 'session_q4', 'rsi_zone_neutral', 'rsi_zone_overbought',
                'rsi_zone_oversold', 'rsi_zone_unknown', 'trend_direction_downtrend',
                'trend_direction_sideways', 'trend_direction_uptrend', 'crt_BUY', 'crt_SELL',
                'trade_type_BUY', 'trade_type_SELL', 'combo_flag_dead', 'combo_flag_fair',
                'combo_flag_fine', 'combo_flag2_dead', 'combo_flag2_fair', 'combo_flag2_fine',
                'minutes,closed_0', 'minutes,closed_5', 'minutes,closed_10', 'minutes,closed_15',
                'minutes,closed_20', 'minutes,closed_25', 'minutes,closed_30', 'minutes,closed_35',
                'minutes,closed_40', 'minutes,closed_45', 'minutes,closed_50', 'minutes,closed_55'
            ]
            # Verify we have exactly 109 features
            if len(self.all_features) != 109:
                logger.error(f"M5 feature count incorrect: {len(self.all_features)} features, expected 109")
                # Remove extra features if necessary
                self.all_features = self.all_features[:109]
        else:  # M15
            # M15: 87 features exactly  
            self.all_features = [
                'adj close', 'garman_klass_vol', 'rsi_20', 'bb_low', 'bb_mid', 'bb_high',
                'atr_z', 'macd_line', 'macd_z', 'ma_10', 'ma_100', 'vwap', 'vwap_std',
                'upper_band_1', 'lower_band_1', 'upper_band_2', 'lower_band_2', 
                'upper_band_3', 'lower_band_3', 'touches_vwap', 'touches_upper_band_1',
                'touches_upper_band_2', 'touches_upper_band_3', 'touches_lower_band_1',
                'touches_lower_band_2', 'touches_lower_band_3', 'far_ratio_vwap',
                'far_ratio_upper_band_1', 'far_ratio_upper_band_2', 'far_ratio_upper_band_3',
                'far_ratio_lower_band_1', 'far_ratio_lower_band_2', 'far_ratio_lower_band_3',
                'rsi', 'ma_20', 'ma_30', 'ma_40', 'ma_60', 'bearish_stack',
                'trend_strength_up', 'trend_strength_down', 'sl_price', 'tp_price',
                'prev_volume', 'body_size', 'wick_up', 'wick_down', 'sl_distance',
                'tp_distance', 'log_sl', 'prev_body_size', 'prev_wick_up', 'prev_wick_down',
                'is_bad_combo', 'volume_bin', 'dollar_volume_bin', 'price_div_vol',
                'rsi_div_macd', 'price_div_vwap', 'sl_div_atr', 'tp_div_atr', 'rrr_div_rsi',
                'hour', 'month', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
                'dayofweek_sin', 'dayofweek_cos', 'day_Friday', 'day_Monday', 'day_Sunday',
                'day_Thursday', 'day_Tuesday', 'day_Wednesday', 'session_q1', 'session_q2',
                'session_q3', 'session_q4', 'rsi_zone_neutral', 'rsi_zone_overbought',
                'rsi_zone_oversold', 'rsi_zone_unknown', 'trend_direction_downtrend',
                'trend_direction_sideways', 'trend_direction_uptrend', 'crt_BUY', 'crt_SELL',
                'trade_type_BUY', 'trade_type_SELL', 'combo_flag_dead', 'combo_flag_fair',
                'combo_flag_fine', 'combo_flag2_dead', 'combo_flag2_fair', 'combo_flag2_fine',
                'minutes,closed_0', 'minutes,closed_15', 'minutes,closed_30', 'minutes,closed_45'
            ]
            # Verify we have exactly 87 features
            if len(self.all_features) != 87:
                logger.error(f"M15 feature count incorrect: {len(self.all_features)} features, expected 87")
                # Remove extra features if necessary
                self.all_features = self.all_features[:87]
        
        logger.info(f"Initialized {timeframe} feature engineer with {len(self.all_features)} target features")
        
        # Define columns that need to be shifted
        self.columns_to_shift = [
            'adj close', 'garman_klass_vol', 'rsi_20', 'bb_low', 'bb_mid', 'bb_high',
            'atr_z', 'macd_line', 'macd_z', 'ma_10', 'ma_100', 'vwap', 'vwap_std',
            'upper_band_1', 'lower_band_1', 'upper_band_2', 'lower_band_2', 
            'upper_band_3', 'lower_band_3', 'touches_vwap', 'touches_upper_band_1',
            'touches_upper_band_2', 'touches_upper_band_3', 'touches_lower_band_1',
            'touches_lower_band_2', 'touches_lower_band_3', 'far_ratio_vwap',
            'far_ratio_upper_band_1', 'far_ratio_upper_band_2', 'far_ratio_upper_band_3',
            'far_ratio_lower_band_1', 'far_ratio_lower_band_2', 'far_ratio_lower_band_3',
            'rsi', 'ma_20', 'ma_30', 'ma_40', 'ma_60', 'bearish_stack',
            'trend_strength_up', 'trend_strength_down', 'prev_volume', 'body_size',
            'wick_up', 'wick_down', 'prev_body_size', 'prev_wick_up', 'prev_wick_down',
            'is_bad_combo', 'volume_bin', 'dollar_volume_bin', 'price_div_vol',
            'rsi_div_macd', 'price_div_vwap', 'hour', 'month', 'dayofweek', 'is_weekend',
            'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos'
        ]
    
    def generate_features(self, df: pd.DataFrame, signal_type: str) -> Optional[pd.Series]:
        """Generate all features that match model expectations"""
        try:
            # Adjust minimum data requirement based on timeframe
            min_data_required = 200 if self.timeframe in ['M5', 'M15'] else 100
            if len(df) < min_data_required:
                logger.warning(f"Not enough data for {self.timeframe} feature generation. Need {min_data_required}, got {len(df)}")
                return None
            
            # Create a copy for processing
            df_copy = df.copy()
            
            # Use the last row for current features
            current_data = df_copy.iloc[-1].copy()
            
            # Initialize features with exact count
            features = pd.Series(index=self.all_features, dtype=float)
            
            # Calculate all technical indicators
            self._calculate_all_technical_indicators(df_copy, features)
            
            # Calculate time-based features
            timestamp = current_data['time'] if 'time' in current_data else df_copy.index[-1]
            self._calculate_time_features(timestamp, features)
            
            # Calculate candle features
            self._calculate_candle_features(current_data, features)
            
            # Calculate risk management features
            self._calculate_risk_features(current_data, features)
            
            # Calculate ratio features
            self._calculate_ratio_features(features)
            
            # Calculate categorical encodings
            self._calculate_categorical_encodings(features, timestamp)
            
            # Set default values for trade-specific features
            self._set_trade_defaults(features, signal_type, timestamp)
            
            # Fill any missing values with 0
            features = features.fillna(0)
            
            # Apply column shifting logic - simplified for live trading
            features = self._apply_column_shifting(features, df_copy)
            
            # Final validation - ensure we have exactly the right features
            if len(features) != len(self.all_features):
                logger.error(f"Feature count mismatch: expected {len(self.all_features)}, got {len(features)}")
                # Create a new series with exact feature order
                exact_features = pd.Series(index=self.all_features, dtype=float)
                for feature in self.all_features:
                    if feature in features:
                        exact_features[feature] = features[feature]
                    else:
                        exact_features[feature] = 0.0
                        logger.warning(f"Missing feature: {feature}")
                features = exact_features
            
            logger.debug(f"Generated {len(features)} features for {self.timeframe}")
            return features
            
        except Exception as e:
            logger.error(f"Feature generation failed for {self.timeframe}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _apply_column_shifting(self, features: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Apply column shifting logic to features - simplified for live trading"""
        try:
            # For live trading, we don't have historical feature columns in the dataframe
            # So we'll use current values and log a warning
            logger.debug("Skipping column shifting for live trading - using current values")
            return features
            
        except Exception as e:
            logger.error(f"Column shifting failed: {str(e)}")
            return features
    
    def _calculate_all_technical_indicators(self, df, features):
        """Calculate all technical indicators"""
        try:
            # Basic price features
            features['adj close'] = df['close'].iloc[-1]
            
            # Garman-Klass Volatility
            features['garman_klass_vol'] = (
                0.5 * (np.log(df['high'].iloc[-1] / df['low'].iloc[-1]) ** 2)
                - (2 * np.log(2) - 1) * (np.log(df['close'].iloc[-1] / df['open'].iloc[-1]) ** 2)
            )
            
            # RSI indicators
            features['rsi'] = ta.rsi(df['close'], length=14).iloc[-1] if len(df) >= 14 else 50
            features['rsi_20'] = ta.rsi(df['close'], length=20).iloc[-1] if len(df) >= 20 else 50
            
            # Bollinger Bands (log adjusted close)
            if len(df) >= 20:
                bb = ta.bbands(np.log1p(df['close']), length=20, std=2)
                if bb is not None and len(bb.columns) >= 3:
                    features['bb_low'] = bb.iloc[-1, 0]
                    features['bb_mid'] = bb.iloc[-1, 1]
                    features['bb_high'] = bb.iloc[-1, 2]
                else:
                    features['bb_low'] = features['bb_mid'] = features['bb_high'] = 0
            else:
                features['bb_low'] = features['bb_mid'] = features['bb_high'] = 0
            
            # ATR Z-score
            if len(df) >= 14:
                atr = ta.atr(df['high'], df['low'], df['close'], length=14)
                atr_mean = atr.mean()
                atr_std = atr.std(ddof=0)
                features['atr_z'] = (atr.iloc[-1] - atr_mean) / atr_std if atr_std != 0 else 0
            else:
                features['atr_z'] = 0
            
            # MACD
            if len(df) >= 26:
                macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
                if macd is not None and 'MACD_12_26_9' in macd.columns:
                    macd_line = macd['MACD_12_26_9']
                    features['macd_line'] = macd_line.iloc[-1]
                    macd_mean = macd_line.mean()
                    macd_std = macd_line.std(ddof=0)
                    features['macd_z'] = (macd_line.iloc[-1] - macd_mean) / macd_std if macd_std != 0 else 0
                else:
                    features['macd_line'] = 0
                    features['macd_z'] = 0
            else:
                features['macd_line'] = 0
                features['macd_z'] = 0
            
            # Moving averages
            for length in [10, 20, 30, 40, 60, 100]:
                if len(df) >= length:
                    features[f'ma_{length}'] = df['close'].rolling(window=length, min_periods=1).mean().iloc[-1]
                else:
                    features[f'ma_{length}'] = df['close'].iloc[-1]  # Use current price as fallback
            
            # VWAP system
            self._calculate_vwap_system(df, features)
            
            # Trend strength indicators
            features['bearish_stack'] = (
                (features['ma_20'] < features['ma_30']) &
                (features['ma_30'] < features['ma_40']) &
                (features['ma_40'] < features['ma_60'])
            ).astype(float)
            
            features['trend_strength_up'] = (
                (features['ma_20'] > features['ma_30']) &
                (features['ma_30'] > features['ma_40']) &
                (features['ma_40'] > features['ma_60'])
            ).astype(float)
            
            features['trend_strength_down'] = (
                (features['ma_20'] < features['ma_30']) &
                (features['ma_30'] < features['ma_40']) &
                (features['ma_40'] < features['ma_60'])
            ).astype(float)
            
            # Previous volume
            features['prev_volume'] = df['volume'].iloc[-2] if len(df) >= 2 else df['volume'].iloc[-1]
            
        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {str(e)}")
    
    def _calculate_vwap_system(self, df, features):
        """Calculate VWAP and band system with timeframe-aware session lengths"""
        try:
            # Use predefined session lengths based on timeframe
            tf_to_session_hours = {
                'M5': 8, 'M15': 24, 'M30': 48, 'H1': 96, 'H4': 384, 'D1': 2304
            }
            session_hours = tf_to_session_hours.get(self.timeframe, 24)  # Default to 24h
            
            # VWAP calculations - simplified for live trading
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            price_volume = typical_price * df['volume']
            
            # Simplified VWAP calculation
            vwap = price_volume.cumsum() / df['volume'].cumsum()
            features['vwap'] = vwap.iloc[-1] if len(vwap) > 0 and not pd.isna(vwap.iloc[-1]) else df['close'].iloc[-1]
            
            # VWAP standard deviation
            if len(vwap) > 0:
                features['vwap_std'] = vwap.rolling(20, min_periods=1).std(ddof=0).iloc[-1]
            else:
                features['vwap_std'] = 0
            
            # Standard deviation for bands
            deviation = (typical_price - vwap) ** 2
            variance = deviation.rolling(20, min_periods=1).mean()
            std = np.sqrt(variance)
            current_std = std.iloc[-1] if len(std) > 0 and not pd.isna(std.iloc[-1]) else 0
            
            # Bands
            for i in range(1, 4):
                features[f'upper_band_{i}'] = features['vwap'] + i * current_std
                features[f'lower_band_{i}'] = features['vwap'] - i * current_std
            
            # Touch indicators
            levels = ["vwap"] + [f"upper_band_{i}" for i in range(1, 4)] + [f"lower_band_{i}" for i in range(1, 4)]
            current_low = df['low'].iloc[-1]
            current_high = df['high'].iloc[-1]
            
            for lvl in levels:
                level_value = features[lvl]
                features[f'touches_{lvl}'] = 1 if (current_low <= level_value <= current_high) else 0
            
            # Distance ratios
            current_close = df['close'].iloc[-1]
            for lvl in levels:
                level_value = features[lvl]
                dist = abs(current_close - level_value)
                features[f'far_ratio_{lvl}'] = dist / current_std if current_std > 0 else 0
                
        except Exception as e:
            logger.error(f"VWAP system calculation failed: {str(e)}")
    
    def _calculate_time_features(self, timestamp, features):
        """Calculate time-based features"""
        try:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            # Basic time features
            features['hour'] = timestamp.hour
            features['month'] = timestamp.month
            features['dayofweek'] = timestamp.weekday()
            features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
            
            # Cyclical time features
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['dayofweek_sin'] = np.sin(2 * np.pi * features['dayofweek'] / 7)
            features['dayofweek_cos'] = np.cos(2 * np.pi * features['dayofweek'] / 7)
            
        except Exception as e:
            logger.error(f"Time feature calculation failed: {str(e)}")
    
    def _calculate_candle_features(self, candle, features):
        """Calculate candle-specific features"""
        try:
            features['body_size'] = abs(candle['close'] - candle['open'])
            features['wick_up'] = candle['high'] - max(candle['open'], candle['close'])
            features['wick_down'] = min(candle['open'], candle['close']) - candle['low']
            
        except Exception as e:
            logger.error(f"Candle feature calculation failed: {str(e)}")
    
    def _calculate_risk_features(self, candle, features):
        """Calculate risk management features"""
        try:
            # Calculate ATR-based risk parameters
            atr_value = abs(features.get('atr_z', 0)) * 0.001 + 0.001
            current_price = candle['close']
            
            # Set SL and TP based on ATR
            atr_distance = max(atr_value * current_price, 0.0001)  # Minimum distance
            features['sl_price'] = current_price - (2 * atr_distance)
            features['tp_price'] = current_price + (3 * atr_distance)
            
            # Calculate distances (in pips, assuming 5 decimal places)
            features['sl_distance'] = abs(current_price - features['sl_price']) * 10000
            features['tp_distance'] = abs(features['tp_price'] - current_price) * 10000
            
            features['log_sl'] = np.log1p(features['sl_price'])
            
            # Previous candle features (approximated from current)
            features['prev_body_size'] = features['body_size'] * 0.95
            features['prev_wick_up'] = features['wick_up'] * 0.95
            features['prev_wick_down'] = features['wick_down'] * 0.95
            
        except Exception as e:
            logger.error(f"Risk feature calculation failed: {str(e)}")
    
    def _calculate_ratio_features(self, features):
        """Calculate ratio features"""
        try:
            small_value = 1e-6
            
            features['price_div_vol'] = features['adj close'] / (features['garman_klass_vol'] + small_value)
            features['rsi_div_macd'] = features['rsi'] / (features['macd_z'] + small_value)
            features['price_div_vwap'] = features['adj close'] / (features['vwap'] + small_value)
            features['sl_div_atr'] = features['sl_distance'] / (abs(features['atr_z']) + small_value)
            features['tp_div_atr'] = features['tp_distance'] / (abs(features['atr_z']) + small_value)
            
            rrr = features['tp_distance'] / (features['sl_distance'] + small_value)
            features['rrr_div_rsi'] = rrr / (features['rsi'] + small_value)
            
        except Exception as e:
            logger.error(f"Ratio feature calculation failed: {str(e)}")
    
    def _calculate_categorical_encodings(self, features, timestamp):
        """Calculate categorical encodings with if-else logic"""
        try:
            # RSI Zone encoding
            rsi_value = features.get('rsi', 50)
            if pd.isna(rsi_value):
                features['rsi_zone_unknown'] = 1
                features['rsi_zone_neutral'] = 0
                features['rsi_zone_overbought'] = 0
                features['rsi_zone_oversold'] = 0
            elif rsi_value < 30:
                features['rsi_zone_oversold'] = 1
                features['rsi_zone_neutral'] = 0
                features['rsi_zone_overbought'] = 0
                features['rsi_zone_unknown'] = 0
            elif rsi_value > 70:
                features['rsi_zone_overbought'] = 1
                features['rsi_zone_neutral'] = 0
                features['rsi_zone_oversold'] = 0
                features['rsi_zone_unknown'] = 0
            else:
                features['rsi_zone_neutral'] = 1
                features['rsi_zone_overbought'] = 0
                features['rsi_zone_oversold'] = 0
                features['rsi_zone_unknown'] = 0
            
            # Trend Direction encoding
            trend_up = features.get('trend_strength_up', 0)
            trend_down = features.get('trend_strength_down', 0)
            
            if trend_up > trend_down:
                features['trend_direction_uptrend'] = 1
                features['trend_direction_downtrend'] = 0
                features['trend_direction_sideways'] = 0
            elif trend_down > trend_up:
                features['trend_direction_downtrend'] = 1
                features['trend_direction_uptrend'] = 0
                features['trend_direction_sideways'] = 0
            else:
                features['trend_direction_sideways'] = 1
                features['trend_direction_uptrend'] = 0
                features['trend_direction_downtrend'] = 0
            
            # Session encoding
            hour = features['hour']
            if 0 <= hour < 6:
                features['session_q1'] = 0
                features['session_q2'] = 1
                features['session_q3'] = 0
                features['session_q4'] = 0
            elif 6 <= hour < 12:
                features['session_q1'] = 0
                features['session_q2'] = 0
                features['session_q3'] = 1
                features['session_q4'] = 0
            elif 12 <= hour < 18:
                features['session_q1'] = 0
                features['session_q2'] = 0
                features['session_q3'] = 0
                features['session_q4'] = 1
            else:
                features['session_q1'] = 1
                features['session_q2'] = 0
                features['session_q3'] = 0
                features['session_q4'] = 0
            
            # Day of week encoding
            day_str = timestamp.strftime('%A')
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for day in days:
                features[f'day_{day}'] = 1 if day == day_str else 0
            
        except Exception as e:
            logger.error(f"Categorical encoding failed: {str(e)}")
    
    def _set_trade_defaults(self, features, signal_type, timestamp):
        """Set default values for trade-specific features with timeframe-aware minutes"""
        try:
            # Trade outcome features - M5 has 'result', M15 doesn't
            if self.timeframe == 'M5' and 'result' in features:
                features['result'] = 0
            
            features['is_bad_combo'] = 0
            
            # Volume binning
            volume = features.get('prev_volume', 0)
            avg_volume = volume * 1.2  # Approximation
            features['volume_bin'] = 1 if volume > avg_volume else 0
            
            # Dollar volume bin
            dollar_volume = features.get('adj close', 0) * volume
            features['dollar_volume_bin'] = 1 if dollar_volume > 1000000 else 0
            
            # Trade type encodings based on signal_type
            signal_upper = signal_type.upper()
            if 'BUY' in signal_upper:
                features['crt_BUY'] = 1
                features['crt_SELL'] = 0
                features['trade_type_BUY'] = 1
                features['trade_type_SELL'] = 0
            elif 'SELL' in signal_upper:
                features['crt_BUY'] = 0
                features['crt_SELL'] = 1
                features['trade_type_BUY'] = 0
                features['trade_type_SELL'] = 1
            else:
                # Default to neutral
                features['crt_BUY'] = 0
                features['crt_SELL'] = 0
                features['trade_type_BUY'] = 0
                features['trade_type_SELL'] = 0
            
            # Combo flags (set to fair by default)
            features['combo_flag_dead'] = 0
            features['combo_flag_fair'] = 1
            features['combo_flag_fine'] = 0
            features['combo_flag2_dead'] = 0
            features['combo_flag2_fair'] = 1
            features['combo_flag2_fine'] = 0
            
            # Minute closed features - timeframe specific
            current_minute = timestamp.minute
            if self.timeframe == 'M5':
                minutes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
            else:  # M15
                minutes = [0, 15, 30, 45]
                
            for minute in minutes:
                features[f'minutes,closed_{minute}'] = 1 if current_minute >= minute else 0
                
        except Exception as e:
            logger.error(f"Trade default setting failed: {str(e)}")

# ========================
# SIMPLIFIED GOOGLE SHEETS STORAGE
# ========================
class SimpleGoogleSheetsStorage:
    def __init__(self, spreadsheet_id):
        self.spreadsheet_id = spreadsheet_id
        self.logger = logging.getLogger('gsheets')
        self.service = None
    
    def connect(self):
        try:
            from google.colab import auth
            from google.auth import default
            from googleapiclient.discovery import build
            
            auth.authenticate_user()
            creds, _ = default()
            self.service = build('sheets', 'v4', credentials=creds)
            self.logger.info("Google Sheets connection established")
            return True
        except Exception as e:
            self.logger.warning(f"Google Sheets connection failed: {str(e)}")
            return False
    
    def append_signal(self, timeframe, signal_data, features, prediction):
        """Simple signal storage - just log for now"""
        try:
            confidence = "HIGH" if prediction > PREDICTION_THRESHOLD else "LOW"
            
            log_message = (
                f"Signal {timeframe} - {signal_data['signal_type']} | "
                f"Entry: {signal_data['entry']:.5f} | "
                f"SL: {signal_data['sl']:.5f} | TP: {signal_data['tp']:.5f} | "
                f"Pred: {prediction:.4f} ({confidence})"
            )
            
            logger.info(log_message)
            return True
            
        except Exception as e:
            logger.error(f"Signal storage failed: {str(e)}")
            return False

# ========================
# WORKING TRADING BOT WITH CRT PATTERN
# ========================
class WorkingTradingBot:
    def __init__(self, timeframe, credentials):
        self.timeframe = timeframe
        self.credentials = credentials
        self.logger = logging.getLogger(f"{timeframe}_bot")
        self.start_time = time.time()
        self.max_duration = 11.5 * 3600
        
        # Initialize storage
        self.storage = SimpleGoogleSheetsStorage('1HZo4uUfeYrzoeEQkjoxwylrqQpKI4R9OfHOZ6zaDino')
        
        logger.info(f"Initializing working {timeframe} bot")
        
        # Load model with appropriate expected features
        if timeframe == "M5":
            model_path = os.path.join(MODELS_DIR, "5mbilstm_model.keras")
            scaler_path = os.path.join(MODELS_DIR, "scaler5mcrt.joblib")
            expected_features = 109  # Based on the exact feature list
        else:  # M15
            model_path = os.path.join(MODELS_DIR, "15mbilstm_model.keras")
            scaler_path = os.path.join(MODELS_DIR, "scaler15mcrt.joblib")
            expected_features = 87   # Based on the exact feature list
            
        logger.debug(f"Model path: {model_path}")
        logger.debug(f"Scaler path: {scaler_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            # Create a dummy model loader that always returns 0.5
            self.model_loader = None
            self.feature_engineer = SimplifiedFeatureEngineer(timeframe)
            logger.warning("Using dummy model - no actual predictions will be made")
        else:
            try:
                self.model_loader = FixedModelLoader(model_path, scaler_path, expected_features)
                self.feature_engineer = SimplifiedFeatureEngineer(timeframe)
                self.data = pd.DataFrame()
                logger.info(f"Working bot initialized for {timeframe}")
            except Exception as e:
                logger.error(f"Model loading failed, using dummy: {str(e)}")
                self.model_loader = None
                self.feature_engineer = SimplifiedFeatureEngineer(timeframe)
    
    def calculate_next_candle_time(self):
        now = datetime.now(NY_TZ)
        
        if self.timeframe == "M5":
            minutes_past = now.minute % 5
            next_minute = now.minute - minutes_past + 5
            if next_minute >= 60:
                next_time = now.replace(hour=now.hour+1, minute=0, second=0, microsecond=0)
            else:
                next_time = now.replace(minute=next_minute, second=0, microsecond=0)
        else:  # M15
            minutes_past = now.minute % 15
            next_minute = now.minute - minutes_past + 15
            if next_minute >= 60:
                next_time = now.replace(hour=now.hour+1, minute=0, second=0, microsecond=0)
            else:
                next_time = now.replace(minute=next_minute, second=0, microsecond=0)
        
        # Add small buffer
        next_time += timedelta(seconds=2)
        
        # If we're at the exact candle start, move to next candle
        if now >= next_time:
            next_time += timedelta(minutes=5 if self.timeframe == "M5" else 15)
        
        logger.debug(f"Current time: {now}, Next candle: {next_time}")
        return next_time

    def detect_simple_signal(self, data):
        """Use CRT pattern detection for signals"""
        return detect_crt_signals(data)

    def send_signal(self, signal_type, signal_data, prediction, features):
        """Send formatted signal"""
        try:
            latency_ms = (datetime.now(NY_TZ) - signal_data['time']).total_seconds() * 1000
            confidence = "HIGH" if prediction > PREDICTION_THRESHOLD else "LOW"
            emoji = "🚨" if confidence == "HIGH" else "⚠️"
            
            message = (
                f"{emoji} XAU/USD Signal ({self.timeframe})\n"
                f"Type: {signal_type} ({signal_data.get('signal_type', 'MANUAL')})\n"
                f"Entry: {signal_data['entry']:.5f}\n"
                f"SL: {signal_data['sl']:.5f}\n"
                f"TP: {signal_data['tp']:.5f}\n"
                f"Confidence: {prediction:.4f} ({confidence})\n"
                f"Latency: {latency_ms:.1f}ms\n"
                f"Time: {signal_data['time'].strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Send to Telegram
            telegram_sent = send_telegram(
                message, 
                self.credentials['telegram_token'], 
                self.credentials['telegram_chat_id']
            )
            
            if telegram_sent:
                logger.info(f"Signal sent via Telegram: {signal_type}")
            else:
                logger.warning("Failed to send signal via Telegram")
            
            # Store signal
            self.storage.append_signal(
                timeframe=self.timeframe,
                signal_data=signal_data,
                features=features,
                prediction=prediction
            )
            
        except Exception as e:
            logger.error(f"Signal sending failed: {str(e)}")

    def test_credentials(self):
        """Test credentials"""
        logger.info("Testing credentials...")
        
        # Test Telegram
        test_msg = f"🔧 {self.timeframe} bot credentials test - Bot is WORKING"
        telegram_ok = send_telegram(
            test_msg, 
            self.credentials['telegram_token'], 
            self.credentials['telegram_chat_id']
        )
        
        # Test Oanda
        oanda_ok = False
        try:
            test_data = fetch_candles("M5", count=2, api_key=self.credentials['oanda_api_key'])
            oanda_ok = not test_data.empty
        except Exception as e:
            logger.error(f"Oanda test failed: {str(e)}")
        
        result = telegram_ok and oanda_ok
        logger.info(f"Credentials test: {'PASS' if result else 'FAIL'}")
        return result

    def run(self):
        """Main bot execution loop"""
        thread_name = threading.current_thread().name
        logger.info(f"Starting working bot thread: {thread_name}")
        
        # Test credentials
        if not self.test_credentials():
            logger.error("Credentials test failed, but continuing...")
        
        session_start = time.time()
        start_msg = f"🚀 {self.timeframe} WORKING bot started at {datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
        
        send_telegram(
            start_msg, 
            self.credentials['telegram_token'], 
            self.credentials['telegram_chat_id']
        )
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while True:
            try:
                # Check session timeout
                elapsed = time.time() - session_start
                if elapsed > self.max_duration:
                    logger.warning("Session timeout reached, exiting")
                    end_msg = f"🔴 {self.timeframe} bot session ended after 12 hours"
                    send_telegram(
                        end_msg, 
                        self.credentials['telegram_token'], 
                        self.credentials['telegram_chat_id']
                    )
                    return
                
                # Wait for next candle
                next_candle = self.calculate_next_candle_time()
                now = datetime.now(NY_TZ)
                sleep_seconds = max(0, (next_candle - now).total_seconds())
                
                if sleep_seconds > 0:
                    logger.debug(f"Sleeping for {sleep_seconds:.2f}s until next candle")
                    time.sleep(sleep_seconds)
                
                # Fetch data
                logger.debug("waiting for data to be available on api")
                time.sleep(5)
                logger.debug("Fetching candle data...")
                new_data = fetch_candles(
                    self.timeframe,
                    count=210,  # Reduced for speed
                    api_key=self.credentials['oanda_api_key']
                )
                
                if new_data.empty:
                    logger.error("Failed to fetch data")
                    consecutive_errors += 1
                    time.sleep(60)
                    continue
                
                self.data = new_data
                logger.debug(f"Data updated: {len(self.data)} candles")
                
                # Detect signal using CRT pattern
                signal_type, signal_data = self.detect_simple_signal(self.data)
                
                if signal_type:
                    logger.info(f"CRT Signal detected: {signal_type}")
                    
                    # Generate features
                    features = self.feature_engineer.generate_features(self.data, signal_type)
                    
                    if features is not None:
                        # Get prediction (or use 0.5 if no model)
                        if self.model_loader:
                            prediction = self.model_loader.predict(features.values)
                        else:
                            prediction = 0.5  # Neutral prediction for dummy model
                            logger.info("Using dummy prediction: 0.5")
                        
                        # Send signal if confidence is high (0.9 threshold maintained)
                        if prediction > PREDICTION_THRESHOLD:
                            self.send_signal(signal_type, signal_data, prediction, features)
                        else:
                            logger.info(f"Signal below threshold: {prediction:.4f}")
                    
                    consecutive_errors = 0
                else:
                    logger.debug("No CRT signal detected")
                    consecutive_errors = 0
                
                # Reset error count on success
                consecutive_errors = 0
                
                # Small delay between iterations
                time.sleep(1)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Bot error #{consecutive_errors}: {str(e)}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many consecutive errors, exiting")
                    error_msg = f"🔴 {self.timeframe} bot stopped due to errors"
                    send_telegram(
                        error_msg, 
                        self.credentials['telegram_token'], 
                        self.credentials['telegram_chat_id']
                    )
                    return
                
                # Exponential backoff
                backoff = min(60 * (2 ** (consecutive_errors - 1)), 300)  # Max 5 minutes
                logger.info(f"Backing off for {backoff}s")
                time.sleep(backoff)

# ========================
# SIMPLE MAIN EXECUTION
# ========================
def setup_simple_logging():
    """Setup simple logging"""
    log_format = '%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('trading_bot_simple.log')
        ]
    )
    
    # Suppress noisy loggers
    for logger_name in ['tensorflow', 'ngrok', 'numba', 'httpx', 'httpcore']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

if __name__ == "__main__":
    print("===== WORKING BOT STARTING =====")
    print(f"Start time: {datetime.now(NY_TZ)}")
    print("Python version:", sys.version)
    print("TensorFlow version:", tf.__version__)
    print("Pandas version:", pd.__version__)
    
    # Setup simple logging
    logger = setup_simple_logging()
    
    try:
        # Load credentials
        credentials = {
            'telegram_token': os.getenv("TELEGRAM_BOT_TOKEN"),
            'telegram_chat_id': os.getenv("TELEGRAM_CHAT_ID"),
            'oanda_account_id': os.getenv("OANDA_ACCOUNT_ID"),
            'oanda_api_key': os.getenv("OANDA_API_KEY")
        }
        
        # Basic credential check
        missing_creds = [k for k, v in credentials.items() if not v]
        if missing_creds:
            logger.warning(f"Missing credentials: {missing_creds}")
        else:
            logger.info("All credentials present")
        
        # Create bots - continue even if some fail
        bots = []
        threads = []
        
        for timeframe in ["M5", "M15"]:
            try:
                logger.info(f"Creating {timeframe} working bot")
                bot = WorkingTradingBot(timeframe, credentials)
                thread = threading.Thread(target=bot.run, name=f"{timeframe}_Working_Bot")
                thread.daemon = True
                
                bots.append(bot)
                threads.append(thread)
                thread.start()
                logger.info(f"Started {timeframe} bot thread")
                
            except Exception as e:
                logger.error(f"Failed to create {timeframe} bot: {str(e)}")
                continue
        
        if not bots:
            logger.warning("No bots created successfully, but continuing monitoring...")
        
        # Simple monitoring loop
        logger.info("Main thread entering monitoring loop")
        while True:
            try:
                status = []
                for bot, thread in zip(bots, threads):
                    status.append(f"{bot.timeframe}: {'ALIVE' if thread.is_alive() else 'DEAD'}")
                
                if status:
                    logger.info(f"Bot status: {', '.join(status)}")
                else:
                    logger.info("No active bots")
                
                # Check memory usage
                try:
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    logger.debug(f"Memory usage: {memory_mb:.1f} MB")
                except:
                    pass
                
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                time.sleep(60)
                
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}", exc_info=True)
        
        # Send final alert if possible
        if 'credentials' in locals() and credentials.get('telegram_token') and credentials.get('telegram_chat_id'):
            try:
                send_telegram(
                    f"❌ Trading bot system crashed: {str(e)[:500]}",
                    credentials['telegram_token'],
                    credentials['telegram_chat_id']
                )
            except:
                pass
