# ========================
# COMPLETE ROBUST TRADING BOT SCRIPT
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

# Prediction threshold (0.9140 for class 1)
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
# ROBUST MODEL LOADER WITH HEALTH CHECKS
# ========================
class RobustModelLoader:
    def __init__(self, model_path, scaler_path, max_retries=3):
        self.model_path = model_path
        self.scaler_path = scaler_path
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
                    safe_mode=False  # Disable safe mode for performance
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
        # Test model with dummy input
        if hasattr(self.scaler, 'feature_names_in_'):
            feature_count = len(self.scaler.feature_names_in_)
        else:
            # Estimate feature count from scaler parameters
            feature_count = len(self.scaler.mean_) if hasattr(self.scaler, 'mean_') else 100
        
        dummy_features = np.random.random((1, feature_count))
        dummy_scaled = self.scaler.transform(dummy_features)
        dummy_reshaped = dummy_scaled.reshape(1, 1, -1)
        
        try:
            prediction = self.model.predict(dummy_reshaped, verbose=0)
            logger.debug(f"Model test prediction: {prediction[0][0]:.6f}")
        except Exception as e:
            raise RuntimeError(f"Model verification failed: {str(e)}")
        
        # Verify scaler attributes
        required_attrs = ['mean_', 'scale_']
        for attr in required_attrs:
            if not hasattr(self.scaler, attr):
                logger.warning(f"Scaler missing attribute: {attr}")
    
    def health_check(self):
        """Perform health check on model and scaler"""
        try:
            # Test prediction with current timestamp
            if hasattr(self.scaler, 'feature_names_in_'):
                feature_count = len(self.scaler.feature_names_in_)
            else:
                feature_count = len(self.scaler.mean_) if hasattr(self.scaler, 'mean_') else 100
                
            test_features = np.random.random((1, feature_count))
            _ = self.predict(test_features[0])
            self.last_health_check = time.time()
            return True
        except Exception as e:
            logger.error(f"Model health check failed: {str(e)}")
            return False
    
    def predict(self, features):
        """Enhanced prediction with comprehensive validation"""
        # Validate input
        if not isinstance(features, (np.ndarray, list, pd.Series)):
            raise ValueError(f"Features must be array-like, got {type(features)}")
        
        features = np.array(features, dtype=np.float64)
        
        if features.ndim != 1:
            raise ValueError(f"Features must be 1D, got shape {features.shape}")
        
        # Determine expected feature count
        if hasattr(self.scaler, 'feature_names_in_'):
            expected_features = len(self.scaler.feature_names_in_)
        elif hasattr(self.scaler, 'n_features_in_'):
            expected_features = self.scaler.n_features_in_
        elif hasattr(self.scaler, 'mean_'):
            expected_features = len(self.scaler.mean_)
        else:
            expected_features = len(features)  # Assume it matches
            
        if len(features) != expected_features:
            logger.warning(f"Feature dimension mismatch: expected {expected_features}, got {len(features)}. Using available features.")
            # Truncate or pad features to match expected count
            if len(features) > expected_features:
                features = features[:expected_features]
            else:
                features = np.pad(features, (0, expected_features - len(features)), 'constant')
        
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
# ROBUST FEATURE ENGINEER
# ========================
class RobustFeatureEngineer:
    def __init__(self, timeframe):
        """
        Initialize FeatureEngineer for specific timeframe
        
        Args:
            timeframe: '15m' or '5m' timeframe
        """
        self.timeframe = timeframe
        logger.debug(f"Initializing RobustFeatureEngineer for {timeframe}")
        
        # Define features based on timeframe
        if timeframe == '15m':
            self.feature_columns = [
                'adj close', 'garman_klass_vol', 'rsi_20', 'bb_low', 'bb_mid', 'bb_high',
                'atr_z', 'macd_line', 'macd_z', 'ma_10', 'ma_100', 'vwap', 'vwap_std',
                'upper_band_1', 'lower_band_1', 'upper_band_2', 'lower_band_2', 'upper_band_3',
                'lower_band_3', 'touches_vwap', 'touches_upper_band_1', 'touches_upper_band_2',
                'touches_upper_band_3', 'touches_lower_band_1', 'touches_lower_band_2',
                'touches_lower_band_3', 'far_ratio_vwap', 'far_ratio_upper_band_1',
                'far_ratio_upper_band_2', 'far_ratio_upper_band_3', 'far_ratio_lower_band_1',
                'far_ratio_lower_band_2', 'far_ratio_lower_band_3', 'rsi', 'ma_20', 'ma_30',
                'ma_40', 'ma_60', 'bearish_stack', 'trend_strength_up', 'trend_strength_down',
                'sl_price', 'tp_price', 'prev_volume', 'body_size', 'wick_up', 'wick_down',
                'sl_distance', 'tp_distance', 'log_sl', 'prev_body_size', 'prev_wick_up',
                'prev_wick_down', 'is_bad_combo', 'volume_bin', 'dollar_volume_bin',
                'price_div_vol', 'rsi_div_macd', 'price_div_vwap', 'sl_div_atr', 'tp_div_atr',
                'rrr_div_rsi', 'hour', 'month', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
                'dayofweek_sin', 'dayofweek_cos', 'day_Friday', 'day_Monday', 'day_Sunday',
                'day_Thursday', 'day_Tuesday', 'day_Wednesday', 'session_q1', 'session_q2',
                'session_q3', 'session_q4', 'rsi_zone_neutral', 'rsi_zone_overbought',
                'rsi_zone_oversold', 'rsi_zone_unknown', 'trend_direction_downtrend',
                'trend_direction_sideways', 'trend_direction_uptrend', 'crt_BUY', 'crt_SELL',
                'trade_type_BUY', 'trade_type_SELL', 'combo_flag_dead', 'combo_flag_fair',
                'combo_flag_fine', 'combo_flag2_dead', 'combo_flag2_fair', 'combo_flag2_fine',
                'minutes,closed_0', 'minutes,closed_15', 'minutes,closed_30', 'minutes,closed_45'
            ]
        else:  # 5m
            self.feature_columns = [
                'adj close', 'garman_klass_vol', 'rsi_20', 'bb_low', 'bb_mid', 'bb_high',
                'atr_z', 'macd_line', 'macd_z', 'ma_10', 'ma_100', 'vwap', 'vwap_std',
                'upper_band_1', 'lower_band_1', 'upper_band_2', 'lower_band_2', 'upper_band_3',
                'lower_band_3', 'touches_vwap', 'touches_upper_band_1', 'touches_upper_band_2',
                'touches_upper_band_3', 'touches_lower_band_1', 'touches_lower_band_2',
                'touches_lower_band_3', 'far_ratio_vwap', 'far_ratio_upper_band_1',
                'far_ratio_upper_band_2', 'far_ratio_upper_band_3', 'far_ratio_lower_band_1',
                'far_ratio_lower_band_2', 'far_ratio_lower_band_3', 'rsi', 'ma_20', 'ma_30',
                'ma_40', 'ma_60', 'bearish_stack', 'trend_strength_up', 'trend_strength_down',
                'sl_price', 'tp_price', 'prev_volume', 'body_size', 'wick_up', 'wick_down',
                'sl_distance', 'tp_distance', 'log_sl', 'prev_body_size', 'prev_wick_up',
                'prev_wick_down', 'result', 'is_bad_combo', 'volume_bin', 'dollar_volume_bin',
                'price_div_vol', 'rsi_div_macd', 'price_div_vwap', 'sl_div_atr', 'tp_div_atr',
                'rrr_div_rsi', 'hour', 'month', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
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
        
        # Combo dictionaries
        self.combo_dict = self._initialize_combo_dictionary()
        self.combo2_dict = self._initialize_combo2_dictionary()
        
        # Bin configurations
        self.rsi_bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
        self.macd_z_bins = [-12.386, -0.496, -0.138, 0.134, 0.527, 9.246]
        
        # Timeframe configuration
        self.tf_to_session_hours = {
            '1m': 1.6, '5m': 8, '15m': 24, '30m': 48, '1h': 96, '4h': 384, '1d': 2304
        }
        
        # Validation statistics
        self.feature_validation_stats = {}

    def _initialize_combo_dictionary(self) -> Dict[str, float]:
        """Initialize combo dictionary with win rates"""
        return {
            'BUY_uptrend_(50, 60]': 0.65,
            'BUY_uptrend_(60, 70]': 0.58,
            'SELL_downtrend_(30, 40]': 0.62,
            'BUY_sideways_(40, 50]': 0.55,
            'SELL_sideways_(50, 60]': 0.53,
            'BUY_downtrend_(30, 40]': 0.45,
            'SELL_uptrend_(60, 70]': 0.48,
            'BUY_uptrend_(40, 50]': 0.59,
            'SELL_downtrend_(40, 50]': 0.56,
            'BUY_sideways_(50, 60]': 0.57,
            'SELL_sideways_(60, 70]': 0.54,
            'BUY_downtrend_(40, 50]': 0.47,
            'SELL_uptrend_(70, 80]': 0.42,
            'BUY_uptrend_(70, 80]': 0.52,
            'SELL_downtrend_(50, 60]': 0.49,
            'BUY_sideways_(60, 70]': 0.51,
            'SELL_sideways_(70, 80]': 0.46,
            'BUY_downtrend_(50, 60]': 0.43,
            'SELL_uptrend_(50, 60]': 0.44,
            'BUY_uptrend_(30, 40]': 0.41,
            'SELL_downtrend_(60, 70]': 0.39,
            'BUY_sideways_(70, 80]': 0.38,
            'SELL_sideways_(40, 50]': 0.50,
            'BUY_downtrend_(60, 70]': 0.36,
            'SELL_uptrend_(40, 50]': 0.37
        }

    def _initialize_combo2_dictionary(self) -> Dict[str, float]:
        """Initialize combo2 dictionary with win rates"""
        return {
            'BUY_(50, 60]_(-0.12, 0.126]': 0.68,
            'BUY_(60, 70]_(0.126, 0.517]': 0.61,
            'SELL_(30, 40]_(-0.489, -0.12]': 0.59,
            'BUY_(40, 50]_(-0.12, 0.126]': 0.63,
            'SELL_(50, 60]_(0.126, 0.517]': 0.55,
            'BUY_(70, 80]_(0.517, 10.271]': 0.52,
            'SELL_(40, 50]_(-0.489, -0.12]': 0.57,
            'BUY_(50, 60]_(0.126, 0.517]': 0.64,
            'SELL_(60, 70]_(-0.12, 0.126]': 0.53,
            'BUY_(30, 40]_(-0.12, 0.126]': 0.48,
            'SELL_(70, 80]_(0.517, 10.271]': 0.45,
            'BUY_(60, 70]_(-0.12, 0.126]': 0.58,
            'SELL_(50, 60]_(-0.489, -0.12]': 0.51,
            'BUY_(40, 50]_(0.126, 0.517]': 0.60,
            'SELL_(60, 70]_(0.126, 0.517]': 0.49,
            'BUY_(70, 80]_(-0.12, 0.126]': 0.47,
            'SELL_(40, 50]_(0.126, 0.517]': 0.46,
            'BUY_(30, 40]_(-0.489, -0.12]': 0.42,
            'SELL_(70, 80]_(-0.12, 0.126]': 0.44,
            'BUY_(50, 60]_(-0.489, -0.12]': 0.41,
            'SELL_(30, 40]_(0.126, 0.517]': 0.43,
            'BUY_(60, 70]_(-0.489, -0.12]': 0.39,
            'SELL_(50, 60]_(-0.12, 0.126]': 0.40,
            'BUY_(40, 50]_(-0.489, -0.12]': 0.38
        }

    def _safe_technical_calculation(self, func, **kwargs):
        """Safely calculate technical indicators with error handling"""
        try:
            result = func(**kwargs)
            if result is None or (hasattr(result, 'empty') and result.empty):
                logger.warning(f"{func.__name__} returned empty result")
                return None
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            df = df.copy()
            
            # Set adjusted close to open
            df['adj close'] = df['open']
            
            # 1. Garman-Klass Volatility
            df["garman_klass_vol"] = (
                0.5 * (np.log(df["high"] / df["low"]) ** 2) -
                (2 * np.log(2) - 1) * (np.log(df["adj close"] / df["open"]) ** 2)
            )
            
            # 2. RSI calculations
            df["rsi_20"] = ta.rsi(df["adj close"], length=20)
            df["rsi"] = ta.rsi(df["close"], length=14)
            
            # 3. Bollinger Bands (log adjusted close)
            bb = self._safe_technical_calculation(
                ta.bbands, close=np.log1p(df["adj close"]), length=20, std=2
            )
            if bb is not None:
                bb_cols = bb.columns[:3]
                bb = bb.rename(columns={bb_cols[0]: "bb_low", bb_cols[1]: "bb_mid", bb_cols[2]: "bb_high"})
                df = pd.concat([df, bb[["bb_low", "bb_mid", "bb_high"]]], axis=1)
            else:
                for col in ['bb_low', 'bb_mid', 'bb_high']:
                    df[col] = np.nan
            
            # 4. ATR (z-scored)
            atr = ta.atr(df["high"], df["low"], df["close"], length=14)
            df["atr_z"] = (atr - atr.mean()) / atr.std(ddof=0)
            
            # 5. MACD (z-scored, MACD line)
            macd = self._safe_technical_calculation(
                ta.macd, close=df["adj close"], fast=12, slow=26, signal=9
            )
            if macd is not None and "MACD_12_26_9" in macd.columns:
                df["macd_line"] = macd["MACD_12_26_9"]
                df["macd_z"] = (df["macd_line"] - df["macd_line"].mean()) / df["macd_line"].std(ddof=0)
            else:
                df["macd_line"] = np.nan
                df["macd_z"] = np.nan
            
            # 6. Moving Averages
            df["ma_10"] = df["adj close"].rolling(10, min_periods=1).mean()
            df["ma_100"] = df["adj close"].rolling(100, min_periods=1).mean()
            
            # Additional MAs on close
            for length in [20, 30, 40, 60]:
                df[f"ma_{length}"] = df["close"].rolling(window=length, min_periods=1).mean()
            
            # 7. VWAP calculations
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            vwap_num = (typical_price * df["volume"]).cumsum()
            vwap_den = df["volume"].cumsum()
            df["vwap"] = vwap_num / vwap_den
            df["vwap_std"] = df["vwap"].rolling(20, min_periods=1).std(ddof=0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_technical_indicators: {str(e)}")
            return df

    def calculate_vwap_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP bands with session-based reset"""
        try:
            # Detect timeframe
            if len(df) < 2:
                return df
                
            time_diff = (df.index[1] - df.index[0]).total_seconds() / 60
            tf_minutes = int(round(time_diff))
            
            # Get session length
            session_hours = self.tf_to_session_hours.get(self.timeframe, 24)
            freq = f"{int(session_hours * 60)}min"
            
            # VWAP calculations per session
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            price_volume = typical_price * df["volume"]
            
            # Group by session
            group = pd.Grouper(freq=freq, origin=df.index.min(), label="left")
            session_volume = df["volume"].groupby(group).cumsum()
            session_price_volume = price_volume.groupby(group).cumsum()
            
            vwap = session_price_volume / session_volume
            
            # Standard deviation within session
            deviation = (typical_price - vwap) ** 2
            dev_vol = deviation * df["volume"]
            session_dev_vol = dev_vol.groupby(group).cumsum()
            variance = session_dev_vol / session_volume
            std = np.sqrt(variance)
            
            # Create bands
            df["vwap"] = vwap
            for i in range(1, 4):
                df[f"upper_band_{i}"] = df["vwap"] + i * std
                df[f"lower_band_{i}"] = df["vwap"] - i * std
            
            # Touch indicators
            levels = ["vwap"] + [f"upper_band_{i}" for i in range(1, 4)] + [f"lower_band_{i}" for i in range(1, 4)]
            for lvl in levels:
                df[f"touches_{lvl}"] = ((df["low"] <= df[lvl]) & (df["high"] >= df[lvl])).astype(int)
            
            # Distance ratios
            for lvl in levels:
                dist = np.abs(df["close"] - df[lvl])
                df[f"far_ratio_{lvl}"] = np.where(std > 0, dist / std, 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_vwap_bands: {str(e)}")
            return df

    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend strength and direction"""
        try:
            df = df.copy()
            
            # Bearish stack
            df["bearish_stack"] = (
                (df["ma_20"] < df["ma_30"]) &
                (df["ma_30"] < df["ma_40"]) &
                (df["ma_40"] < df["ma_60"])
            ).astype(int)
            
            # Trend strength
            df['trend_strength_up'] = (
                (df['ma_20'] > df['ma_30']) &
                (df['ma_30'] > df['ma_40']) &
                (df['ma_40'] > df['ma_60'])
            ).astype(float)
            
            df['trend_strength_down'] = (
                (df['ma_20'] < df['ma_30']) &
                (df['ma_30'] < df['ma_40']) &
                (df['ma_40'] < df['ma_60'])
            ).astype(float)
            
            # Trend direction (needed for combo key)
            conditions = [
                df['trend_strength_up'] > df['trend_strength_down'],
                df['trend_strength_down'] > df['trend_strength_up']
            ]
            choices = ['uptrend', 'downtrend']
            df['trend_direction'] = np.select(conditions, choices, default='sideways')
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_trend_indicators: {str(e)}")
            return df

    def calculate_crt_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[Dict]]:
        """Calculate CRT signal with robust error handling"""
        try:
            if len(df) < 2:
                return None, None
            
            # Use the last complete candle for signal detection
            current_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            # CRT Logic
            if (prev_candle['close'] > prev_candle['open'] and  # Previous candle bullish
                current_candle['open'] > prev_candle['high'] and  # Gap up
                current_candle['close'] < current_candle['open']):  # Current candle bearish
                return 'SELL', {
                    'time': current_candle['time'],
                    'entry': current_candle['open'],
                    'sl': prev_candle['high'],
                    'tp': current_candle['open'] - 4 * (current_candle['open'] - prev_candle['high'])
                }
            
            elif (prev_candle['close'] < prev_candle['open'] and  # Previous candle bearish
                  current_candle['open'] < prev_candle['low'] and  # Gap down
                  current_candle['close'] > current_candle['open']):  # Current candle bullish
                return 'BUY', {
                    'time': current_candle['time'],
                    'entry': current_candle['open'],
                    'sl': prev_candle['low'],
                    'tp': current_candle['open'] + 4 * (prev_candle['low'] - current_candle['open'])
                }
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error in calculate_crt_signal: {str(e)}")
            return None, None

    def calculate_trade_features(self, df: pd.DataFrame, signal_type: str) -> pd.DataFrame:
        """Calculate trade-related features"""
        try:
            df = df.copy()
            
            if len(df) < 2:
                return df
            
            prev_row = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
            current_open = df.iloc[-1]['open']
            
            if signal_type == 'SELL':
                df['sl_price'] = prev_row['high']
                risk = abs(current_open - df['sl_price'].iloc[-1])
                df['tp_price'] = current_open - 4 * risk
            else:  # BUY
                df['sl_price'] = prev_row['low']
                risk = abs(current_open - df['sl_price'].iloc[-1])
                df['tp_price'] = current_open + 4 * risk
            
            # Trade management features
            df['sl_distance'] = (df['open'] - df['sl_price']).abs() * 10
            df['tp_distance'] = (df['tp_price'] - df['open']).abs() * 10
            
            # Calculate RRR
            df['rrr'] = df['tp_distance'] / df['sl_distance'].replace(0, np.nan)
            
            df['log_sl'] = np.log1p(df['sl_price'])
            
            # Set trade type
            df['trade_type'] = signal_type
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_trade_features: {str(e)}")
            return df

    def calculate_candle_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate candle body and wick metrics"""
        try:
            df = df.copy()
            
            # Current candle metrics
            df['body_size'] = (df['close'] - df['open']).abs()
            df['wick_up'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['wick_down'] = df[['close', 'open']].min(axis=1) - df['low']
            
            # Previous candle metrics
            df['prev_volume'] = df['volume'].shift(1)
            df['prev_body_size'] = df['body_size'].shift(1)
            df['prev_wick_up'] = df['wick_up'].shift(1)
            df['prev_wick_down'] = df['wick_down'].shift(1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_candle_metrics: {str(e)}")
            return df

    def calculate_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all ratio-based features"""
        try:
            df = df.copy()
            
            # Ratio features
            df['price_div_vol'] = df['adj close'] / (df['garman_klass_vol'] + 1e-6)
            df['rsi_div_macd'] = df['rsi'] / (df['macd_z'] + 1e-6)
            df['price_div_vwap'] = df['adj close'] / (df['vwap'] + 1e-6)
            df['sl_div_atr'] = df['sl_distance'] / (df['atr_z'] + 1e-6)
            df['tp_div_atr'] = df['tp_distance'] / (df['atr_z'] + 1e-6)
            df['rrr_div_rsi'] = df['rrr'] / (df['rsi'] + 1e-6)
            
            return df
        except Exception as e:
            logger.error(f"Error in calculate_ratio_features: {str(e)}")
            return df

    def calculate_advanced_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced time-based features"""
        try:
            df = df.copy()
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                    df = df.set_index("time")
                else:
                    raise ValueError("No datetime index or 'time' column found")
            
            # Basic time features
            df['hour'] = df.index.hour
            df['month'] = df.index.month
            df['dayofweek'] = df.index.dayofweek
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Cyclical time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            
            # Time difference features
            df['timestamp'] = df.index.astype('int64') // 1e9  # seconds
            df['time_diff'] = df.index.to_series().diff().dt.total_seconds().fillna(0)
            
            return df
        except Exception as e:
            logger.error(f"Error in calculate_advanced_time_features: {str(e)}")
            return df

    def calculate_session_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate session and time-based features with timeframe-specific minute encoding"""
        try:
            df = df.copy()
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                    df = df.set_index("time")
                else:
                    raise ValueError("No datetime index or 'time' column found")
            
            # Day of week
            df['day'] = df.index.day_name()
            
            # Session mapping
            def get_session(hour):
                if 0 <= hour < 6: return "q2"
                elif 6 <= hour < 12: return "q3"
                elif 12 <= hour < 18: return "q4"
                else: return "q1"
            
            df['session'] = df.index.hour.map(get_session)
            
            # Timeframe-specific minute one-hot encoding
            minutes = df.index.minute
            
            if self.timeframe == '15m':
                minute_bins = [0, 15, 30, 45, 60]
                minute_labels = ['0', '15', '30', '45']
            else:  # 5m
                minute_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
                minute_labels = ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55']
            
            df['minute_bin'] = pd.cut(minutes, bins=minute_bins, labels=minute_labels, right=False)
            
            return df
        except Exception as e:
            logger.error(f"Error in calculate_session_time_features: {str(e)}")
            return df

    def calculate_volume_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume and dollar volume bins"""
        try:
            df = df.copy()
            
            # Calculate dollar volume if not exists
            if 'dollar_volume' not in df.columns:
                df['dollar_volume'] = (df['adj close'] * df['volume']) / 1e6
            
            # Volume bins (quintiles)
            df['volume_bin'] = pd.qcut(df['volume'], q=5, duplicates='drop', labels=False)
            df['dollar_volume_bin'] = pd.qcut(df['dollar_volume'], q=5, duplicates='drop', labels=False)
            
            return df
        except Exception as e:
            logger.error(f"Error in calculate_volume_bins: {str(e)}")
            return df

    def create_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create bin features for RSI and MACD z-score"""
        try:
            df = df.copy()
            
            # RSI bins
            rsi_bin_labels = ['(0, 20]', '(20, 30]', '(30, 40]', '(40, 50]', '(50, 60]', '(60, 70]', '(70, 80]', '(80, 100]']
            df['rsi_bin'] = pd.cut(df['rsi'], bins=self.rsi_bins, labels=rsi_bin_labels, include_lowest=True)
            
            # MACD z bins
            macd_z_bin_labels = ['(-13.256, -0.489]', '(-0.489, -0.12]', '(-0.12, 0.126]', '(0.126, 0.517]', '(0.517, 10.271]']
            df['macd_z_bin'] = pd.cut(df['macd_z'], bins=self.macd_z_bins, labels=macd_z_bin_labels, include_lowest=True)
            
            return df
        except Exception as e:
            logger.error(f"Error in create_bins: {str(e)}")
            return df

    def _get_combo_key(self, row) -> str:
        """Generate combo key based on trade_type, trend_direction, and rsi_bin"""
        try:
            trade_type = row.get('trade_type', 'nan')
            trend_direction = row.get('trend_direction', 'nan')
            rsi_bin = str(row.get('rsi_bin', 'nan'))
            
            return f"{trade_type}_{trend_direction}_{rsi_bin}"
        except Exception as e:
            logger.error(f"Error generating combo key: {e}")
            return "nan_nan_nan"

    def _get_combo_key2(self, row) -> str:
        """Generate combo key2 based on trade_type, rsi_bin, and macd_z_bin"""
        try:
            trade_type = row.get('trade_type', 'nan')
            rsi_bin = str(row.get('rsi_bin', 'nan'))
            macd_z_bin = str(row.get('macd_z_bin', 'nan'))
            
            return f"{trade_type}_{rsi_bin}_{macd_z_bin}"
        except Exception as e:
            logger.error(f"Error generating combo key2: {e}")
            return "nan_nan_nan"

    def _get_combo_flag(self, win_rate: float) -> str:
        """Determine combo flag based on win rate"""
        if pd.isna(win_rate):
            return 'dead'
        elif win_rate < 0.4:
            return 'dead'
        elif win_rate < 0.55:
            return 'fair'
        else:
            return 'fine'

    def _get_combo_flag2(self, win_rate: float) -> str:
        """Determine combo flag2 based on win rate"""
        if pd.isna(win_rate):
            return 'dead'
        elif win_rate < 0.35:
            return 'dead'
        elif win_rate < 0.6:
            return 'fair'
        else:
            return 'fine'

    def calculate_combo_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate combo flags and is_bad_combo based on historical win rates"""
        try:
            df = df.copy()
            
            # Initialize combo flags
            for flag in ['dead', 'fair', 'fine']:
                df[f'combo_flag_{flag}'] = 0.0
                df[f'combo_flag2_{flag}'] = 0.0
            
            df['is_bad_combo'] = 0
            
            if len(df) == 0:
                return df
            
            # Get current row
            current_row = df.iloc[-1]
            
            # Calculate combo key and lookup win rate
            combo_key = self._get_combo_key(current_row)
            win_rate1 = self.combo_dict.get(combo_key, 0.0)
            combo_flag = self._get_combo_flag(win_rate1)
            
            # Calculate combo key2 and lookup win rate
            combo_key2 = self._get_combo_key2(current_row)
            win_rate2 = self.combo2_dict.get(combo_key2, 0.0)
            combo_flag2 = self._get_combo_flag2(win_rate2)
            
            # Set combo flags
            df.loc[df.index[-1], f'combo_flag_{combo_flag}'] = 1.0
            df.loc[df.index[-1], f'combo_flag2_{combo_flag2}'] = 1.0
            
            # Set is_bad_combo (1 if either combo is dead)
            if combo_flag == 'dead' or combo_flag2 == 'dead':
                df.loc[df.index[-1], 'is_bad_combo'] = 1
            
            # Store combo keys for debugging
            df.loc[df.index[-1], 'combo_key'] = combo_key
            df.loc[df.index[-1], 'combo_key2'] = combo_key2
            
            logger.debug(f"Combo key: {combo_key}, win rate: {win_rate1:.3f}, flag: {combo_flag}")
            logger.debug(f"Combo key2: {combo_key2}, win rate: {win_rate2:.3f}, flag: {combo_flag2}")
            logger.debug(f"Is bad combo: {df['is_bad_combo'].iloc[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_combo_flags: {str(e)}")
            return df

    def calculate_one_hot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate one-hot encoded features"""
        try:
            df = df.copy()
            
            # Day of week one-hot encoding
            days = ['Friday', 'Monday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
            for day in days:
                df[f'day_{day}'] = (df['day'] == day).astype(float)
            
            # Session one-hot encoding
            sessions = ['q1', 'q2', 'q3', 'q4']
            for session in sessions:
                df[f'session_{session}'] = (df['session'] == session).astype(float)
            
            # Minutes one-hot encoding (timeframe-specific)
            if self.timeframe == '15m':
                minute_categories = ['0', '15', '30', '45']
            else:  # 5m
                minute_categories = ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55']
            
            for minute in minute_categories:
                df[f'minutes,closed_{minute}'] = (df['minute_bin'] == minute).astype(float)
            
            # RSI zone calculation and one-hot encoding
            conditions = [
                df["rsi"].isna(),
                df["rsi"] < 30,
                df["rsi"] > 70
            ]
            choices = ["unknown", "oversold", "overbought"]
            df["rsi_zone"] = np.select(conditions, choices, default="neutral")
            
            rsi_zones = ['neutral', 'overbought', 'oversold', 'unknown']
            for zone in rsi_zones:
                df[f'rsi_zone_{zone}'] = (df['rsi_zone'] == zone).astype(float)
            
            # Trend direction one-hot encoding
            trend_directions = ['downtrend', 'sideways', 'uptrend']
            for direction in trend_directions:
                df[f'trend_direction_{direction}'] = (df['trend_direction'] == direction).astype(float)
            
            # CRT and trade type one-hot encoding
            df['crt_BUY'] = (df.get('crt', '') == 'BUY').astype(float)
            df['crt_SELL'] = (df.get('crt', '') == 'SELL').astype(float)
            df['trade_type_BUY'] = (df.get('trade_type', '') == 'BUY').astype(float)
            df['trade_type_SELL'] = (df.get('trade_type', '') == 'SELL').astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_one_hot_features: {str(e)}")
            return df

    def _safe_feature_value(self, value, feature_name):
        """Safely convert feature value with validation"""
        try:
            if pd.isna(value):
                return 0.0
            
            if isinstance(value, (bool, np.bool_)):
                return float(value)
            
            if isinstance(value, (int, np.integer, float, np.floating)):
                result = float(value)
                # Check for extreme values
                if abs(result) > 1e6:
                    logger.warning(f"Extreme value for {feature_name}: {result}")
                    result = np.clip(result, -1e6, 1e6)
                return result
            
            # Try to convert string or other types
            return float(value)
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert feature {feature_name}: {value} ({type(value)}), using 0.0")
            return 0.0

    def _validate_feature_vector(self, features):
        """Validate the final feature vector"""
        # Check for NaN values
        nan_features = features.isna()
        if nan_features.any():
            nan_names = features.index[nan_features].tolist()
            logger.warning(f"NaN values in features: {nan_names}")
            features.fillna(0.0, inplace=True)
        
        # Check for infinite values
        infinite_mask = ~np.isfinite(features)
        if infinite_mask.any():
            infinite_names = features.index[infinite_mask].tolist()
            logger.warning(f"Infinite values in features: {infinite_names}")
            features[infinite_mask] = 0.0
        
        # Update validation statistics
        for feat in features.index:
            value = features[feat]
            if feat not in self.feature_validation_stats:
                self.feature_validation_stats[feat] = {'count': 0, 'min': float('inf'), 'max': float('-inf')}
            
            stats = self.feature_validation_stats[feat]
            stats['count'] += 1
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)

    def generate_features(self, df: pd.DataFrame, signal_type: str) -> Optional[pd.Series]:
        """Robust feature generation with comprehensive validation"""
        try:
            # Input validation
            validate_dataframe(df, min_rows=100, required_cols=['open', 'high', 'low', 'close', 'volume'])
            
            if signal_type not in ['BUY', 'SELL']:
                raise ValidationError(f"Invalid signal type: {signal_type}")
            
            if len(df) < 200:
                logger.warning(f"Insufficient data: {len(df)} rows, using available data")
            
            # Use the last 200 rows or available data
            working_df = df.tail(200).copy()
            
            # Set current candle close = open for immediate processing
            if len(working_df) > 0:
                current_candle = working_df.iloc[-1].copy()
                current_candle['close'] = current_candle['open']
                working_df.iloc[-1] = current_candle
            
            # Execute feature engineering pipeline with error handling per step
            pipeline_steps = [
                ('technical_indicators', self.calculate_technical_indicators),
                ('vwap_bands', self.calculate_vwap_bands),
                ('trend_indicators', self.calculate_trend_indicators),
                ('trade_features', lambda x: self.calculate_trade_features(x, signal_type)),
                ('candle_metrics', self.calculate_candle_metrics),
                ('time_features', self.calculate_advanced_time_features),
                ('session_features', self.calculate_session_time_features),
                ('volume_bins', self.calculate_volume_bins),
                ('ratio_features', self.calculate_ratio_features),
                ('bins', self.create_bins),
                ('one_hot_features', self.calculate_one_hot_features),
                ('combo_flags', self.calculate_combo_flags)
            ]
            
            for step_name, step_func in pipeline_steps:
                try:
                    working_df = step_func(working_df)
                    if working_df is None:
                        raise DataIntegrityError(f"Pipeline step {step_name} returned None")
                    
                    # Validate step output
                    if len(working_df) == 0:
                        raise DataIntegrityError(f"Pipeline step {step_name} produced empty DataFrame")
                        
                except Exception as e:
                    logger.error(f"Pipeline step {step_name} failed: {str(e)}")
                    # Continue with available data but log error
                    continue
            
            # Create final feature vector
            features = pd.Series(index=self.feature_columns, dtype=float)
            
            for feat in self.feature_columns:
                if feat in working_df.columns:
                    value = working_df[feat].iloc[-1]
                    features[feat] = self._safe_feature_value(value, feat)
                else:
                    features[feat] = 0.0
                    logger.debug(f"Missing feature filled with 0: {feat}")
            
            # Validate feature completeness
            missing_features = set(self.feature_columns) - set(features.index)
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features, filling with zeros")
                for feat in missing_features:
                    features[feat] = 0.0
            
            # Final feature validation
            self._validate_feature_vector(features)
            
            logger.debug(f"Feature generation successful: {len(features)} features")
            return features[self.feature_columns]
            
        except Exception as e:
            logger.error(f"Feature generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return None

# ========================
# IMPROVED GOOGLE SHEETS STORAGE
# ========================
class GoogleSheetsStorage:
    def __init__(self, spreadsheet_id):
        self.spreadsheet_id = spreadsheet_id
        self.logger = logging.getLogger('gsheets')
        self.service = None
        self.headers = [
            'storage_time', 'timeframe', 'signal_time', 'signal_type', 
            'entry', 'sl', 'tp', 'prediction', 'confidence'
        ]
        
    def connect(self):
        try:
            from google.colab import auth
            from google.auth import default
            from googleapiclient.discovery import build
            
            # Authenticate with Colab
            auth.authenticate_user()
            creds, _ = default()
            
            self.service = build('sheets', 'v4', credentials=creds)
            self.logger.info("Google Sheets connection established")
            return True
        except Exception as e:
            self.logger.error(f"Google Sheets connection failed: {str(e)}")
            return False
    
    def ensure_sheet_exists(self, sheet_name):
        """Create sheet if it doesn't exist"""
        try:
            if not self.service:
                if not self.connect():
                    return False
            
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id).execute()
            
            # Check if sheet exists
            sheet_exists = any(
                sheet['properties']['title'] == sheet_name 
                for sheet in spreadsheet['sheets']
            )
            
            if not sheet_exists:
                body = {
                    'requests': [{
                        'addSheet': {
                            'properties': {'title': sheet_name}
                        }
                    }]
                }
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id, body=body).execute()
                self.logger.info(f"Created new sheet: {sheet_name}")
                
            return True
        except Exception as e:
            self.logger.error(f"Sheet creation failed: {str(e)}")
            return False
    
    def append_signal(self, timeframe, signal_data, features, prediction):
        """Store signal in both timeframe-specific sheet and combined log"""
        if not self.service and not self.connect():
            self.logger.error("Failed to connect to Google Sheets")
            return False
            
        try:
            # Prepare data row
            confidence = "HIGH" if prediction > PREDICTION_THRESHOLD else "LOW"
            base_data = [
                datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                timeframe,
                signal_data['time'].strftime('%Y-%m-%d %H:%M:%S'),
                signal_data['signal_type'],
                f"{signal_data['entry']:.5f}",
                f"{signal_data['sl']:.5f}",
                f"{signal_data['tp']:.5f}",
                f"{prediction:.4f}",
                confidence
            ]
            
            # Add all feature values
            full_row = base_data + [f"{x:.6f}" if isinstance(x, (int, float)) else str(x) for x in features.values]
            
            # Get full headers (first time only)
            full_headers = self.headers + features.index.tolist()
            
            # Store in timeframe-specific sheet
            if not self._append_to_sheet(timeframe, full_headers, full_row):
                return False
            
            # Store in combined log
            if not self._append_to_sheet("All_Signals", full_headers, full_row):
                return False
            
            self.logger.info(f"Signal stored in Google Sheets for {timeframe}")
            return True
        except Exception as e:
            self.logger.error(f"Append failed: {str(e)}")
            return False
    
    def _append_to_sheet(self, sheet_name, headers, row):
        """Internal method to write to specific sheet"""
        try:
            if not self.ensure_sheet_exists(sheet_name):
                return False
            
            # Get existing data to check headers
            range_name = f"{sheet_name}!A:ZZ"
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id, range=range_name).execute()
            values = result.get('values', [])
            
            # Write headers if first row
            if not values:
                self.service.spreadsheets().values().update(
                    spreadsheetId=self.spreadsheet_id,
                    range=f"{sheet_name}!A1",
                    valueInputOption='RAW',
                    body={'values': [headers]}
                ).execute()
                self.logger.info(f"Added headers to sheet: {sheet_name}")
            
            # Append data row
            self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=f"{sheet_name}!A:A",
                valueInputOption='USER_ENTERED',
                body={'values': [row]}
            ).execute()
            
            return True
        except Exception as e:
            self.logger.error(f"Error writing to sheet {sheet_name}: {str(e)}")
            return False

# ========================
# PREDICTION TRACKER FOR MONITORING
# ========================
class PredictionTracker:
    def __init__(self):
        self.predictions = []
        
    def add_prediction(self, prediction, signal_type, timestamp):
        self.predictions.append((timestamp, prediction, signal_type))
        # Keep only last 100 predictions
        self.predictions = self.predictions[-100:]
        
        # Log distribution every 10 predictions
        if len(self.predictions) % 10 == 0:
            preds = [p[1] for p in self.predictions]
            logger.info(f"Prediction stats - Mean: {np.mean(preds):.4f}, Std: {np.std(preds):.4f}, >0.99: {sum(p > 0.99 for p in preds)}")

# ========================
# ROBUST TRADING BOT WITH CIRCUIT BREAKERS
# ========================
class RobustColabTradingBot:
    def __init__(self, timeframe, credentials):
        self.timeframe = timeframe
        self.credentials = credentials
        self.logger = logging.getLogger(f"{timeframe}_bot")
        self.start_time = time.time()
        self.max_duration = 11.5 * 3600
        
        # Circuit breaker variables
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.circuit_breaker_tripped = False
        self.last_successful_prediction = None
        self.error_backoff_time = 60  # Start with 1 minute
        
        # Initialize Google Sheets storage
        self.storage = GoogleSheetsStorage(
            spreadsheet_id='1HZo4uUfeYrzoeEQkjoxwylrqQpKI4R9OfHOZ6zaDino'
        )
        
        # Initialize prediction tracker
        self.prediction_tracker = PredictionTracker()
        
        logger.info(f"Initializing robust {timeframe} bot")
        
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
            self.model_loader = RobustModelLoader(model_path, scaler_path)
            self.feature_engineer = RobustFeatureEngineer(timeframe)
            self.data = pd.DataFrame()
            logger.info(f"Robust bot initialized for {timeframe}")
        except Exception as e:
            logger.error(f"Bot initialization failed: {str(e)}")
            raise

    def _preflight_check(self):
        """Comprehensive preflight validation"""
        try:
            logger.info("Running preflight checks...")
            
            # Validate credentials
            validate_credentials(self.credentials)
            
            # Test APIs
            if not self.test_credentials():
                raise CredentialsError("API credential test failed")
            
            # Test model
            if not self.model_loader.health_check():
                raise RuntimeError("Model health check failed")
            
            # Test feature engineering
            test_df = pd.DataFrame({
                'open': [1800.0] * 200,
                'high': [1805.0] * 200,
                'low': [1795.0] * 200,
                'close': [1802.0] * 200,
                'volume': [1000] * 200
            })
            test_features = self.feature_engineer.generate_features(test_df, 'BUY')
            if test_features is None:
                raise RuntimeError("Feature engineering test failed")
            
            logger.info("All preflight checks passed")
            return True
            
        except Exception as e:
            logger.error(f"Preflight check failed: {str(e)}")
            self._send_alert(f"❌ {self.timeframe} bot preflight failed: {str(e)}")
            return False

    def _check_session_timeout(self, session_start):
        """Check if session should timeout"""
        elapsed = time.time() - session_start
        
        if elapsed > (self.max_duration - 1800) and not hasattr(self, 'timeout_sent'):
            logger.warning("Session nearing timeout, sending warning")
            self._send_alert(f"⏳ {self.timeframe} bot session will expire in 30 minutes")
            self.timeout_sent = True
            
        if elapsed > self.max_duration:
            logger.warning("Session timeout reached, exiting")
            self._send_alert(f"🔴 {self.timeframe} bot session ended after 12 hours")
            return True
            
        return False

    def _sleep_until_candle(self, next_candle_time):
        """Precise sleep until next candle"""
        now = datetime.now(NY_TZ)
        sleep_seconds = max(0, (next_candle_time - now).total_seconds() - 0.1)
        
        if sleep_seconds > 0:
            logger.debug(f"Sleeping for {sleep_seconds:.2f}s until next candle")
            time.sleep(sleep_seconds)
        
        # Busy-wait for precision
        while datetime.now(NY_TZ) < next_candle_time:
            time.sleep(0.001)
        
        logger.debug("Candle open detected")
        time.sleep(5)  # Wait for candle to be available

    def _fetch_and_validate_data(self):
        """Fetch and validate candle data"""
        try:
            logger.debug("Fetching candle data...")
            data = fetch_candles(
                self.timeframe,
                count=201,
                api_key=self.credentials['oanda_api_key']
            )
            
            if validate_dataframe(data, min_rows=50):
                logger.debug(f"Data validated: {len(data)} rows")
                return data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Data fetch/validation failed: {str(e)}")
            return None

    def _process_signals(self, data):
        """Process signals with enhanced error handling"""
        try:
            # Detect signal
            signal_type, signal_data = self.feature_engineer.calculate_crt_signal(data)
            if not signal_type:
                logger.debug("No signal detected")
                return True  # Not an error
            
            logger.info(f"Signal detected: {signal_type}")
            
            # Generate features
            features = self.feature_engineer.generate_features(data, signal_type)
            if features is None:
                logger.warning("Feature generation failed")
                return False
            
            # Get prediction
            prediction = self.model_loader.predict(features)
            if prediction is None:
                logger.warning("Prediction failed")
                return False
            
            # Send signal
            self.send_signal(signal_type, signal_data, prediction, features)
            self.last_successful_prediction = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Signal processing failed: {str(e)}")
            return False

    def _handle_success(self):
        """Handle successful iteration"""
        self.consecutive_errors = 0
        self.error_backoff_time = 60  # Reset backoff

    def _handle_error(self, error_msg):
        """Handle errors with circuit breaker logic"""
        logger.error(error_msg)
        self.consecutive_errors += 1
        
        # Trip circuit breaker if too many consecutive errors
        if self.consecutive_errors >= self.max_consecutive_errors:
            self.circuit_breaker_tripped = True
            alert_msg = f"🚨 {self.timeframe} bot circuit breaker tripped after {self.consecutive_errors} errors"
            self._send_alert(alert_msg)
            return
        
        # Exponential backoff
        backoff = min(self.error_backoff_time * (2 ** (self.consecutive_errors - 1)), 3600)  # Max 1 hour
        logger.info(f"Error #{self.consecutive_errors}, backing off for {backoff}s")
        time.sleep(backoff)

    def _perform_health_check(self):
        """Perform periodic health checks"""
        try:
            # Check model health every 10 minutes
            if not hasattr(self, 'last_health_check') or time.time() - self.last_health_check > 600:
                if not self.model_loader.health_check():
                    logger.error("Model health check failed")
                    self._send_alert(f"⚠️ {self.timeframe} bot model health check failed")
                self.last_health_check = time.time()
                
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")

    def _send_alert(self, message):
        """Send alert with error handling"""
        try:
            send_telegram(
                message, 
                self.credentials['telegram_token'], 
                self.credentials['telegram_chat_id']
            )
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")

    def calculate_next_candle_time(self):
        now = datetime.now(NY_TZ)
        
        if self.timeframe == "M5":
            # Calculate minutes past the hour
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
        
        # Add network latency compensation
        next_time += timedelta(seconds=0.3)
        
        # If we're at the exact candle start, move to next candle
        if now >= next_time:
            next_time += timedelta(minutes=5 if self.timeframe == "M5" else 15)
        
        logger.debug(f"Current time: {now}, Next candle: {next_time}")
        return next_time

    def send_signal(self, signal_type, signal_data, prediction, features):
        """Send formatted signal to Telegram with latency measurement"""
        latency_ms = (datetime.now(NY_TZ) - signal_data['time']).total_seconds() * 1000
        confidence = "HIGH" if prediction > PREDICTION_THRESHOLD else "LOW"
        emoji = "🚨" if confidence == "HIGH" else "⚠️"
        
        message = (
            f"{emoji} XAU/USD Signal ({self.timeframe})\n"
            f"Type: {signal_type}\n"
            f"Entry: {signal_data['entry']:.5f}\n"
            f"SL: {signal_data['sl']:.5f}\n"
            f"TP: {signal_data['tp']:.5f}\n"
            f"Confidence: {prediction:.4f} ({confidence})\n"
            f"Latency: {latency_ms:.1f}ms\n"
            f"Time: {signal_data['time'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Send to Telegram
        send_telegram(message, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
    
        # Store in Google Sheets with error handling
        sheets_success = self.storage.append_signal(
            timeframe=self.timeframe,
            signal_data=signal_data,
            features=features,
            prediction=prediction
        )
        
        if not sheets_success:
            self.logger.warning("Failed to store signal in Google Sheets")
        
        # Track prediction for monitoring
        self.prediction_tracker.add_prediction(prediction, signal_type, datetime.now(NY_TZ))
    
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
            test_data = fetch_candles("M5", count=1, api_key=self.credentials['oanda_api_key'])
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

    def run(self):
        """Enhanced main loop with circuit breakers"""
        thread_name = threading.current_thread().name
        logger.info(f"Starting robust bot thread: {thread_name}")
        
        # Validate everything before starting
        if not self._preflight_check():
            logger.error("Preflight check failed, exiting bot")
            return
        
        session_start = time.time()
        timeout_msg = f"⏳ {self.timeframe} bot session will expire in 30 minutes"
        start_msg = f"🚀 {self.timeframe} robust bot started at {datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Test credentials before starting
        logger.info("Testing credentials...")
        creds_valid = self.test_credentials()
        if not creds_valid:
            logger.error("Credentials test failed. Exiting bot.")
            return
            
        logger.info("Sending startup message...")
        telegram_sent = send_telegram(start_msg, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
        logger.info(f"Startup message {'sent' if telegram_sent else 'failed to send'}")
        
        while not self.circuit_breaker_tripped:
            try:
                # Check session timeout
                if self._check_session_timeout(session_start):
                    return
                
                # Wait for next candle
                next_candle_time = self.calculate_next_candle_time()
                self._sleep_until_candle(next_candle_time)
                
                # Fetch and validate data
                data = self._fetch_and_validate_data()
                if data is None:
                    self._handle_error("Data fetch failed")
                    continue
                
                # Process signals
                success = self._process_signals(data)
                if success:
                    self._handle_success()
                else:
                    self._handle_error("Signal processing failed")
                
                # Health check
                self._perform_health_check()
                
            except Exception as e:
                self._handle_error(f"Unexpected error in main loop: {str(e)}")

# ========================
# HEALTH MONITOR
# ========================
def monitor_bot_health(bots):
    """Monitor bot health and restart if necessary"""
    while True:
        try:
            for i, (bot, thread) in enumerate(bots):
                if not thread.is_alive():
                    logger.error(f"Bot {bot.timeframe} died, restarting...")
                    
                    # Restart bot
                    new_bot = RobustColabTradingBot(bot.timeframe, bot.credentials)
                    new_thread = threading.Thread(
                        target=new_bot.run, 
                        name=f"{bot.timeframe}_Bot_Restarted"
                    )
                    new_thread.daemon = True
                    new_thread.start()
                    
                    bots[i] = (new_bot, new_thread)
                    logger.info(f"Bot {bot.timeframe} restarted")
            
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Health monitor error: {str(e)}")
            time.sleep(60)

# ========================
# ROBUST MAIN EXECUTION
# ========================
def setup_robust_logging():
    """Setup comprehensive logging"""
    log_format = '%(asctime)s [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s'
    
    # Create handlers
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('trading_bot_robust.log')
    error_file_handler = logging.FileHandler('trading_bot_errors.log')
    
    # Set levels
    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    error_file_handler.setLevel(logging.ERROR)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    error_file_handler.setFormatter(formatter)
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_file_handler)
    
    # Suppress noisy loggers
    for logger_name in ['tensorflow', 'ngrok', 'numba', 'httpx', 'httpcore']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

if __name__ == "__main__":
    print("===== ROBUST BOT STARTING =====")
    print(f"Start time: {datetime.now(NY_TZ)}")
    print("Python version:", sys.version)
    print("TensorFlow version:", tf.__version__)
    print("Pandas version:", pd.__version__)
    
    # Setup robust logging
    logger = setup_robust_logging()
    
    try:
        # Load and validate credentials
        credentials = {
            'telegram_token': os.getenv("TELEGRAM_BOT_TOKEN"),
            'telegram_chat_id': os.getenv("TELEGRAM_CHAT_ID"),
            'oanda_account_id': os.getenv("OANDA_ACCOUNT_ID"),
            'oanda_api_key': os.getenv("OANDA_API_KEY")
        }
        
        validate_credentials(credentials)
        logger.info("Credentials validated successfully")
        
        # Create robust bots
        bots = []
        threads = []
        
        for timeframe in ["M5", "M15"]:
            try:
                logger.info(f"Creating {timeframe} robust bot")
                bot = RobustColabTradingBot(timeframe, credentials)
                thread = threading.Thread(target=bot.run, name=f"{timeframe}_Robust_Bot")
                thread.daemon = True
                
                bots.append(bot)
                threads.append(thread)
                
            except Exception as e:
                logger.error(f"Failed to create {timeframe} bot: {str(e)}")
                continue
        
        if not bots:
            raise RuntimeError("No bots created successfully")
        
        # Start bots
        for thread in threads:
            thread.start()
            logger.info(f"Started thread: {thread.name}")
        
        # Start health monitor
        health_thread = threading.Thread(
            target=monitor_bot_health, 
            args=(list(zip(bots, threads)),),
            name="Health_Monitor"
        )
        health_thread.daemon = True
        health_thread.start()
        
        # Main monitoring loop
        logger.info("Main thread entering monitoring loop")
        while True:
            status = []
            for bot, thread in zip(bots, threads):
                status.append(f"{bot.timeframe}: {'ALIVE' if thread.is_alive() else 'DEAD'}")
            
            logger.info(f"Bot status: {', '.join(status)}")
            
            # Log memory usage
            try:
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.debug(f"Memory usage: {memory_mb:.1f} MB")
            except:
                pass
            
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}", exc_info=True)
        
        # Send final alert if possible
        if 'credentials' in locals():
            try:
                send_telegram(
                    f"❌ Robust bot system crashed: {str(e)[:500]}",
                    credentials['telegram_token'],
                    credentials['telegram_chat_id']
                )
            except:
                pass
        
        sys.exit(1)
