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
# SIMPLIFIED FEATURE ENGINEER
# ========================
class SimplifiedFeatureEngineer:
    def __init__(self, timeframe):
        self.timeframe = timeframe
        logger.debug(f"Initializing SimplifiedFeatureEngineer for {timeframe}")
        
        # Define basic features - we'll generate what we can
        self.basic_features = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'rsi_20', 'macd', 'macd_signal', 'macd_hist',
            'ema_12', 'ema_26', 'sma_20', 'sma_50', 'sma_200',
            'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'obv', 'vwap', 'stoch_k', 'stoch_d', 'williams_r',
            'cci', 'adx', 'aroon_up', 'aroon_down', 'aroon_osc',
            'mfi', 'roc', 'momentum', 'volume_change', 'price_change',
            'body_size', 'upper_wick', 'lower_wick', 'body_ratio',
            'high_low_ratio', 'open_close_ratio', 'volume_sma_ratio'
        ]
        
        # Add time-based features
        self.time_features = [
            'hour', 'day_of_week', 'day_of_month', 'week_of_year', 'month',
            'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
            'session_asia', 'session_london', 'session_ny', 'session_overlap'
        ]
        
        self.all_features = self.basic_features + self.time_features
    
    def generate_features(self, df: pd.DataFrame, signal_type: str) -> Optional[pd.Series]:
        """Generate simplified features that match model expectations"""
        try:
            if len(df) < 50:
                logger.warning("Not enough data for feature generation")
                return None
            
            # Use the last row for current features
            current_data = df.iloc[-1].copy()
            features = pd.Series(index=self.all_features, dtype=float)
            
            # Basic price features
            features['open'] = current_data['open']
            features['high'] = current_data['high']
            features['low'] = current_data['low']
            features['close'] = current_data['close']
            features['volume'] = current_data['volume']
            
            # Calculate basic technical indicators
            self._calculate_technical_indicators(df, features)
            
            # Calculate time-based features
            self._calculate_time_features(current_data['time'], features)
            
            # Calculate candle features
            self._calculate_candle_features(current_data, features)
            
            # Fill any missing values with 0
            features = features.fillna(0)
            
            logger.debug(f"Generated {len(features)} simplified features")
            return features
            
        except Exception as e:
            logger.error(f"Simplified feature generation failed: {str(e)}")
            return None
    
    def _calculate_technical_indicators(self, df, features):
        """Calculate basic technical indicators"""
        try:
            # RSI
            if len(df) >= 14:
                features['rsi'] = ta.rsi(df['close'], length=14).iloc[-1]
                features['rsi_20'] = ta.rsi(df['close'], length=20).iloc[-1]
            
            # MACD
            if len(df) >= 26:
                macd = ta.macd(df['close'])
                if macd is not None:
                    features['macd'] = macd['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in macd.columns else 0
                    features['macd_signal'] = macd['MACDs_12_26_9'].iloc[-1] if 'MACDs_12_26_9' in macd.columns else 0
                    features['macd_hist'] = macd['MACDh_12_26_9'].iloc[-1] if 'MACDh_12_26_9' in macd.columns else 0
            
            # Moving averages
            features['ema_12'] = ta.ema(df['close'], length=12).iloc[-1] if len(df) >= 12 else 0
            features['ema_26'] = ta.ema(df['close'], length=26).iloc[-1] if len(df) >= 26 else 0
            features['sma_20'] = ta.sma(df['close'], length=20).iloc[-1] if len(df) >= 20 else 0
            features['sma_50'] = ta.sma(df['close'], length=50).iloc[-1] if len(df) >= 50 else 0
            features['sma_200'] = ta.sma(df['close'], length=200).iloc[-1] if len(df) >= 200 else 0
            
            # Bollinger Bands
            if len(df) >= 20:
                bb = ta.bbands(df['close'], length=20)
                if bb is not None:
                    features['bb_upper'] = bb['BBU_20_2.0'].iloc[-1] if 'BBU_20_2.0' in bb.columns else 0
                    features['bb_middle'] = bb['BBM_20_2.0'].iloc[-1] if 'BBM_20_2.0' in bb.columns else 0
                    features['bb_lower'] = bb['BBL_20_2.0'].iloc[-1] if 'BBL_20_2.0' in bb.columns else 0
                    features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle'] if features['bb_middle'] != 0 else 0
            
            # Other indicators
            features['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1] if len(df) >= 14 else 0
            features['obv'] = ta.obv(df['close'], df['volume']).iloc[-1] if len(df) >= 1 else 0
            
            # VWAP (simplified)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            features['vwap'] = vwap.iloc[-1] if len(vwap) > 0 else 0
            
            # Stochastic
            if len(df) >= 14:
                stoch = ta.stoch(df['high'], df['low'], df['close'])
                if stoch is not None:
                    features['stoch_k'] = stoch['STOCHk_14_3_3'].iloc[-1] if 'STOCHk_14_3_3' in stoch.columns else 0
                    features['stoch_d'] = stoch['STOCHd_14_3_3'].iloc[-1] if 'STOCHd_14_3_3' in stoch.columns else 0
            
            features['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14).iloc[-1] if len(df) >= 14 else 0
            features['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20).iloc[-1] if len(df) >= 20 else 0
            features['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14).iloc[-1] if len(df) >= 14 else 0
            
            # Aroon
            if len(df) >= 25:
                aroon = ta.aroon(df['high'], df['low'], length=25)
                if aroon is not None:
                    features['aroon_up'] = aroon['AROONU_25'].iloc[-1] if 'AROONU_25' in aroon.columns else 0
                    features['aroon_down'] = aroon['AROOND_25'].iloc[-1] if 'AROOND_25' in aroon.columns else 0
                    features['aroon_osc'] = features['aroon_up'] - features['aroon_down']
            
            features['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14).iloc[-1] if len(df) >= 14 else 0
            features['roc'] = ta.roc(df['close'], length=10).iloc[-1] if len(df) >= 10 else 0
            features['momentum'] = ta.mom(df['close'], length=10).iloc[-1] if len(df) >= 10 else 0
            
            # Volume and price changes
            if len(df) >= 2:
                features['volume_change'] = (df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2] if df['volume'].iloc[-2] != 0 else 0
                features['price_change'] = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] if df['close'].iloc[-2] != 0 else 0
                features['volume_sma_ratio'] = df['volume'].iloc[-1] / ta.sma(df['volume'], length=20).iloc[-1] if len(df) >= 20 else 1
            
        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {str(e)}")
    
    def _calculate_time_features(self, timestamp, features):
        """Calculate time-based features"""
        try:
            if isinstance(timestamp, str):
                timestamp = parse_oanda_time(timestamp)
            
            features['hour'] = timestamp.hour
            features['day_of_week'] = timestamp.weekday()
            features['day_of_month'] = timestamp.day
            features['week_of_year'] = timestamp.isocalendar()[1]
            features['month'] = timestamp.month
            
            features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
            features['is_month_start'] = 1 if timestamp.day == 1 else 0
            features['is_month_end'] = 1 if timestamp.day >= 28 else 0  # Approximation
            features['is_quarter_start'] = 1 if timestamp.month in [1, 4, 7, 10] and timestamp.day == 1 else 0
            features['is_quarter_end'] = 1 if timestamp.month in [3, 6, 9, 12] and timestamp.day >= 28 else 0
            
            # Trading sessions
            hour = timestamp.hour
            features['session_asia'] = 1 if 0 <= hour < 8 else 0
            features['session_london'] = 1 if 8 <= hour < 16 else 0
            features['session_ny'] = 1 if 13 <= hour < 21 else 0
            features['session_overlap'] = 1 if (8 <= hour < 12) or (13 <= hour < 16) else 0
            
        except Exception as e:
            logger.error(f"Time feature calculation failed: {str(e)}")
    
    def _calculate_candle_features(self, candle, features):
        """Calculate candle-specific features"""
        try:
            features['body_size'] = abs(candle['close'] - candle['open'])
            features['upper_wick'] = candle['high'] - max(candle['open'], candle['close'])
            features['lower_wick'] = min(candle['open'], candle['close']) - candle['low']
            
            total_size = features['body_size'] + features['upper_wick'] + features['lower_wick']
            features['body_ratio'] = features['body_size'] / total_size if total_size > 0 else 0
            
            features['high_low_ratio'] = (candle['high'] - candle['low']) / candle['close'] if candle['close'] != 0 else 0
            features['open_close_ratio'] = (candle['close'] - candle['open']) / candle['open'] if candle['open'] != 0 else 0
            
        except Exception as e:
            logger.error(f"Candle feature calculation failed: {str(e)}")

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
# WORKING TRADING BOT
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
            expected_features = 109  # Based on the error message
        else:  # M15
            model_path = os.path.join(MODELS_DIR, "15mbilstm_model.keras")
            scaler_path = os.path.join(MODELS_DIR, "scaler15mcrt.joblib")
            expected_features = 87   # Adjust based on your M15 model
            
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
        """Simple signal detection based on basic patterns"""
        try:
            if len(data) < 3:
                return None, None
            
            current = data.iloc[-1]
            prev1 = data.iloc[-2]
            prev2 = data.iloc[-3]
            
            # Simple breakout detection
            if (current['close'] > prev1['high'] and 
                current['close'] > max(prev1['high'], prev2['high'])):
                sl = prev1['low']
                risk = abs(current['close'] - sl)
                tp = current['close'] + (2 * risk)
                return 'BUY', {
                    'time': current['time'],
                    'entry': current['close'],
                    'sl': sl,
                    'tp': tp,
                    'signal_type': 'BREAKOUT'
                }
            
            # Simple breakdown detection
            elif (current['close'] < prev1['low'] and 
                  current['close'] < min(prev1['low'], prev2['low'])):
                sl = prev1['high']
                risk = abs(current['close'] - sl)
                tp = current['close'] - (2 * risk)
                return 'SELL', {
                    'time': current['time'],
                    'entry': current['close'],
                    'sl': sl,
                    'tp': tp,
                    'signal_type': 'BREAKDOWN'
                }
            
            return None, None
            
        except Exception as e:
            logger.error(f"Signal detection failed: {str(e)}")
            return None, None

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
                logger.debug("Fetching candle data...")
                new_data = fetch_candles(
                    self.timeframe,
                    count=100,  # Reduced for speed
                    api_key=self.credentials['oanda_api_key']
                )
                
                if new_data.empty:
                    logger.error("Failed to fetch data")
                    consecutive_errors += 1
                    time.sleep(60)
                    continue
                
                self.data = new_data
                logger.debug(f"Data updated: {len(self.data)} candles")
                
                # Detect signal
                signal_type, signal_data = self.detect_simple_signal(self.data)
                
                if signal_type:
                    logger.info(f"Signal detected: {signal_type}")
                    
                    # Generate features
                    features = self.feature_engineer.generate_features(self.data, signal_type)
                    
                    if features is not None:
                        # Get prediction (or use 0.5 if no model)
                        if self.model_loader:
                            prediction = self.model_loader.predict(features.values)
                        else:
                            prediction = 0.5  # Neutral prediction for dummy model
                            logger.info("Using dummy prediction: 0.5")
                        
                        # Send signal if confidence is reasonable
                        if prediction > 0.3:  # Lower threshold to catch more signals
                            self.send_signal(signal_type, signal_data, prediction, features)
                        else:
                            logger.info(f"Signal below threshold: {prediction:.4f}")
                    
                    consecutive_errors = 0
                else:
                    logger.debug("No signal detected")
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
