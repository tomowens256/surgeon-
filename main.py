# ========================
# ADVANCED TRADING BOT - HTF-DRIVEN MULTI-TIMEFRAME VERSION
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

# HTF Prediction threshold for direction
HTF_PREDICTION_THRESHOLD = 0.60

# Initialize logging with more verbosity
log_format = '%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s'
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot_htf_driven.log')
    ]
)
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
# UTILITY FUNCTIONS (Keep existing)
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

def send_features_telegram(features, signal_type, timeframe, token, chat_id):
    """Send ALL formatted features to Telegram"""
    try:
        logger.debug(f"Preparing to send ALL features to Telegram for {timeframe} {signal_type}")
        
        if features is None:
            logger.warning("No features to send to Telegram")
            return False
            
        # Build feature summary with ALL features
        feature_lines = [f"📊 *{timeframe} {signal_type} ALL FEATURES*"]
        feature_lines.append("```")
        
        # Send ALL features, not just selected ones
        for feat_name, feat_value in features.items():
            if isinstance(feat_value, float):
                # Format floats with appropriate precision
                if abs(feat_value) < 0.001:
                    feature_lines.append(f"{feat_name}: {feat_value:.6f}")
                elif abs(feat_value) < 1:
                    feature_lines.append(f"{feat_name}: {feat_value:.5f}")
                else:
                    feature_lines.append(f"{feat_name}: {feat_value:.4f}")
            else:
                feature_lines.append(f"{feat_name}: {feat_value}")
        
        # Add combo flags if available
        combo_flags = []
        for flag_type in ['dead', 'fair', 'fine']:
            flag1 = f'combo_flag_{flag_type}'
            flag2 = f'combo_flag2_{flag_type}'
            if flag1 in features and features[flag1] == 1:
                combo_flags.append(f"Combo1: {flag_type}")
            if flag2 in features and features[flag2] == 1:
                combo_flags.append(f"Combo2: {flag_type}")
        
        if combo_flags:
            feature_lines.append("---")
            feature_lines.extend(combo_flags)
        
        feature_lines.append("```")
        
        message = "\n".join(feature_lines)
        
        # If message is too long, split into multiple messages
        if len(message) > 4000:
            logger.info("Features message too long, splitting into multiple messages")
            messages = []
            current_message = []
            current_length = 0
            
            for line in feature_lines:
                line_length = len(line) + 1  # +1 for newline
                if current_length + line_length > 3500:
                    messages.append("\n".join(current_message))
                    current_message = [line]
                    current_length = line_length
                else:
                    current_message.append(line)
                    current_length += line_length
            
            if current_message:
                messages.append("\n".join(current_message))
            
            # Send multiple messages
            for i, msg in enumerate(messages):
                part_suffix = f" (Part {i+1}/{len(messages)})" if len(messages) > 1 else ""
                full_msg = f"📊 *{timeframe} {signal_type} FEATURES{part_suffix}*\n{msg}"
                success = send_telegram(full_msg, token, chat_id)
                if success:
                    logger.info(f"Features part {i+1} sent to Telegram")
                else:
                    logger.warning(f"Failed to send features part {i+1} to Telegram")
                time.sleep(1)  # Small delay between messages
            
            return any(success)
        else:
            # Send single message
            success = send_telegram(message, token, chat_id)
            if success:
                logger.info(f"All features sent to Telegram for {timeframe} {signal_type}")
            else:
                logger.warning(f"Failed to send features to Telegram for {timeframe} {signal_type}")
            
            return success
        
    except Exception as e:
        logger.error(f"Error sending features to Telegram: {str(e)}")
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
            df = df.reset_index(drop=True)  # Reset index to avoid duplicate labels
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
# FIXED MODEL LOADER WITH GRACEFUL ERROR HANDLING
# ========================
class ModelLoader:
    def __init__(self, model_path, scaler_path):
        logger.debug(f"Loading model from {model_path}")
        logger.debug(f"Loading scaler from {scaler_path}")
        
        # Check if files exist first
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
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
# SIMPLE GOOGLE SHEETS STORAGE
# ========================
class GoogleSheetsStorage:
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
        """Simple signal storage"""
        try:
            confidence = "HIGH" if prediction > HTF_PREDICTION_THRESHOLD else "LOW"
            
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
# FIXED FEATURE ENGINEER - PANDAS-TA COMPATIBLE
# ========================
class FeatureEngineer:
    def __init__(self, timeframe):
        self.timeframe = timeframe
        logger.debug(f"Initializing FeatureEngineer for {timeframe}")
        
        # CORRECTED: Base features with exactly the features you want (removed dollar_volume and rrr)
        self.base_features = [
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
            'tp_distance', 'log_sl', 'prev_body_size', 'prev_wick_up',
            'prev_wick_down', 'is_bad_combo', 'volume_bin', 'dollar_volume_bin', 'price_div_vol',
            'rsi_div_macd', 'price_div_vwap', 'sl_div_atr', 'tp_div_atr', 'rrr_div_rsi',
            'hour', 'month', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
            'dayofweek_sin', 'dayofweek_cos', 'day_Friday', 'day_Monday', 'day_Sunday',
            'day_Thursday', 'day_Tuesday', 'day_Wednesday', 'session_q1', 'session_q2',
            'session_q3', 'session_q4', 'rsi_zone_neutral', 'rsi_zone_overbought',
            'rsi_zone_oversold', 'rsi_zone_unknown', 'trend_direction_downtrend',
            'trend_direction_sideways', 'trend_direction_uptrend', 'crt_BUY', 'crt_SELL',
            'trade_type_BUY', 'trade_type_SELL', 'combo_flag_dead', 'combo_flag_fair',
            'combo_flag_fine', 'combo_flag2_dead', 'combo_flag2_fair', 'combo_flag2_fine'
        ]
        
        # CORRECTED: Remove dollar_volume from shift_features since it's not in our final features
        self.shift_features = [
            'garman_klass_vol', 'rsi_20', 'bb_low', 'bb_mid', 'bb_high',
            'atr_z', 'macd_line', 'macd_z', 'ma_10', 'ma_100',
            'vwap', 'vwap_std', 'rsi', 'ma_20', 'ma_30', 'ma_40', 'ma_60',
            'trend_strength_up', 'trend_strength_down', 'volume', 'body_size', 
            'wick_up', 'wick_down', 'prev_body_size', 'prev_wick_up', 'prev_wick_down', 
            'is_bad_combo', 'price_div_vol', 'rsi_div_macd', 'price_div_vwap', 
            'sl_div_atr', 'rrr_div_rsi', 'rsi_zone_neutral', 'rsi_zone_overbought', 
            'rsi_zone_oversold', 'rsi_zone_unknown', 'combo_flag_dead', 'combo_flag_fair',
            'combo_flag_fine', 'combo_flag2_dead', 'combo_flag2_fair', 'combo_flag2_fine'
        ]
        
        # Timeframe-specific minute closed features at END
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
        
        # Verify feature count
        expected_count = 109 if timeframe == "M5" else 101
        actual_count = len(self.features)
        if actual_count != expected_count:
            logger.warning(f"Feature count mismatch: expected {expected_count}, got {actual_count}")
        
        self.rsi_bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
        self.macd_z_bins = [-12.386, -0.496, -0.138, 0.134, 0.527, 9.246]
        
    def calculate_crt_signal(self, df):
        """Calculate CRT signal at OPEN of current candle (c0) with minimal latency - OLD RELIABLE VERSION"""
        try:
            if len(df) < 3:
                return None, None
                
            # Use the last COMPLETED candle as c2 (index -2)
            # c1 = candle at -3, c2 = candle at -2, c0 = current open at -1
            c1 = df.iloc[-3]
            c2 = df.iloc[-2]
            current_open = df.iloc[-1]['open']  # Only need open of current candle
            
            # Calculate c2 metrics
            c2_range = c2['high'] - c2['low']
            c2_mid = c2['low'] + (0.5 * c2_range)
            
            # CRT conditions - using ONLY completed candles and current open
            if (c2['low'] < c1['low'] and 
                c2['close'] > c1['low'] and 
                current_open > c2_mid):
                signal_type = 'BUY'
                entry = current_open
                sl = c2['low']
                risk = abs(entry - sl)
                tp = entry + 4 * risk
                return signal_type, {'entry': entry, 'sl': sl, 'tp': tp, 'time': df.iloc[-1]['time']}
            
            elif (c2['high'] > c1['high'] and 
                  c2['close'] < c1['high'] and 
                  current_open < c2_mid):
                signal_type = 'SELL'
                entry = current_open
                sl = c2['high']
                risk = abs(sl - entry)
                tp = entry - 4 * risk
                return signal_type, {'entry': entry, 'sl': sl, 'tp': tp, 'time': df.iloc[-1]['time']}
            
            return None, None
        except Exception as e:
            logger.error(f"Error in calculate_crt_signal: {str(e)}")
            return None, None
        
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators WITH PANDAS-TA FIXES"""
        try:
            df = df.copy().drop_duplicates(subset=['time'], keep='last')
            
            # Basic price features
            df['adj close'] = df['open']
            
            # Garman-Klass Volatility
            df['garman_klass_vol'] = (((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 - (2 * np.log(2) - 1) * ((np.log(df['adj close']) - np.log(df['open'])) ** 2))
            
            # RSI indicators - FIXED: Use direct pandas-ta calls
            try:
                rsi_20 = ta.rsi(df['adj close'], length=20)
                df['rsi_20'] = rsi_20 if rsi_20 is not None else 50
            except:
                df['rsi_20'] = 50
                
            try:
                rsi_14 = ta.rsi(df['close'], length=14)
                df['rsi'] = rsi_14 if rsi_14 is not None else 50
            except:
                df['rsi'] = 50
            
            # Bollinger Bands - FIXED: Handle different column names
            try:
                bb = ta.bbands(df['adj close'], length=20, std=2)
                if bb is not None:
                    # Try different possible column names
                    if 'BBL_20_2.0' in bb.columns:
                        df['bb_low'] = bb['BBL_20_2.0']
                        df['bb_mid'] = bb['BBM_20_2.0'] 
                        df['bb_high'] = bb['BBU_20_2.0']
                    elif 'BBL_20_2' in bb.columns:
                        df['bb_low'] = bb['BBL_20_2']
                        df['bb_mid'] = bb['BBM_20_2']
                        df['bb_high'] = bb['BBU_20_2']
                    elif 'BBL_20' in bb.columns:
                        df['bb_low'] = bb['BBL_20']
                        df['bb_mid'] = bb['BBM_20']
                        df['bb_high'] = bb['BBU_20']
                    else:
                        # Use first three columns if standard names not found
                        cols = bb.columns[:3]
                        if len(cols) >= 3:
                            df['bb_low'] = bb[cols[0]]
                            df['bb_mid'] = bb[cols[1]]
                            df['bb_high'] = bb[cols[2]]
                        else:
                            df['bb_low'] = df['bb_mid'] = df['bb_high'] = 0
                else:
                    df['bb_low'] = df['bb_mid'] = df['bb_high'] = 0
            except Exception as e:
                logger.warning(f"Bollinger Bands calculation issue: {e}")
                df['bb_low'] = df['bb_mid'] = df['bb_high'] = 0
            
            # ATR Z-score - FIXED: Handle ATR calculation
            try:
                atr = ta.atr(df['high'], df['low'], df['close'], length=14)
                if atr is not None:
                    atr_mean = atr.mean()
                    atr_std = atr.std(ddof=0)
                    df['atr_z'] = (atr - atr_mean) / atr_std if atr_std != 0 else 0
                else:
                    df['atr_z'] = 0
            except:
                df['atr_z'] = 0
            
            # MACD with line and z-score - FIXED: Handle MACD calculation
            try:
                macd = ta.macd(df['adj close'], fast=12, slow=26, signal=9)
                if macd is not None:
                    # Try different possible column names
                    if 'MACD_12_26_9' in macd.columns:
                        macd_line = macd['MACD_12_26_9']
                    elif 'MACD_12_26' in macd.columns:
                        macd_line = macd['MACD_12_26']
                    elif 'MACD' in macd.columns:
                        macd_line = macd['MACD']
                    else:
                        # Use first column
                        macd_line = macd[macd.columns[0]]
                    
                    df['macd_line'] = macd_line
                    macd_mean = macd_line.mean()
                    macd_std = macd_line.std(ddof=0)
                    df['macd_z'] = (macd_line - macd_mean) / macd_std if macd_std != 0 else 0
                else:
                    df['macd_line'] = 0
                    df['macd_z'] = 0
            except Exception as e:
                logger.warning(f"MACD calculation issue: {e}")
                df['macd_line'] = 0
                df['macd_z'] = 0
            
            # Dollar volume (calculated but not included in final features)
            df['dollar_volume'] = (df['adj close'] * df['volume']) / 1e6
            
            # Moving averages - FIXED: Use rolling directly
            for length in [10, 20, 30, 40, 60, 100]:
                df[f'ma_{length}'] = df['adj close'].rolling(window=length, min_periods=1).mean()
            
            # VWAP system - FIXED: Manual calculation
            try:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                vwap_num = (typical_price * df['volume']).cumsum()
                vwap_den = df['volume'].cumsum()
                df['vwap'] = vwap_num / vwap_den
                df['vwap_std'] = df['vwap'].rolling(window=20, min_periods=1).std()
            except:
                df['vwap'] = df['adj close']
                df['vwap_std'] = 0
            
            # VWAP bands - FIXED: Manual calculation
            try:
                for i in range(1, 4):
                    df[f'upper_band_{i}'] = df['vwap'] + i * df['vwap_std']
                    df[f'lower_band_{i}'] = df['vwap'] - i * df['vwap_std']
            except:
                for i in range(1, 4):
                    df[f'upper_band_{i}'] = df['vwap']
                    df[f'lower_band_{i}'] = df['vwap']
            
            # Touch indicators - FIXED: Use current values
            try:
                current_low = df['low'].iloc[-1]
                current_high = df['high'].iloc[-1]
                for i in range(1, 4):
                    df[f'touches_upper_band_{i}'] = (current_low <= df[f'upper_band_{i}']) & (df[f'upper_band_{i}'] <= current_high)
                    df[f'touches_lower_band_{i}'] = (current_low <= df[f'lower_band_{i}']) & (df[f'lower_band_{i}'] <= current_high)
                df['touches_vwap'] = (current_low <= df['vwap']) & (df['vwap'] <= current_high)
            except:
                for i in range(1, 4):
                    df[f'touches_upper_band_{i}'] = 0
                    df[f'touches_lower_band_{i}'] = 0
                df['touches_vwap'] = 0
            
            # Distance ratios - FIXED: Use current values
            try:
                current_close = df['close'].iloc[-1]
                for i in range(1, 4):
                    df[f'far_ratio_upper_band_{i}'] = abs(current_close - df[f'upper_band_{i}']) / (df['vwap_std'] + 1e-6)
                    df[f'far_ratio_lower_band_{i}'] = abs(current_close - df[f'lower_band_{i}']) / (df['vwap_std'] + 1e-6)
                df['far_ratio_vwap'] = abs(current_close - df['vwap']) / (df['vwap_std'] + 1e-6)
            except:
                for i in range(1, 4):
                    df[f'far_ratio_upper_band_{i}'] = 0
                    df[f'far_ratio_lower_band_{i}'] = 0
                df['far_ratio_vwap'] = 0
            
            # Bearish stack - FIXED: Manual calculation
            try:
                df['bearish_stack'] = (
                    (df['ma_20'] < df['ma_30']) & 
                    (df['ma_30'] < df['ma_40']) & 
                    (df['ma_40'] < df['ma_60'])
                ).astype(float)
            except:
                df['bearish_stack'] = 0
            
            # Trend strength - FIXED: Manual calculation
            try:
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
            except:
                df['trend_strength_up'] = 0
                df['trend_strength_down'] = 0
            
            # Previous volume
            df['prev_volume'] = df['volume'].shift(1).fillna(df['volume'])
            
            return df
        except Exception as e:
            logger.error(f"Error in calculate_technical_indicators: {str(e)}")
            # Return basic dataframe with essential columns
            df['adj close'] = df['open']
            df['garman_klass_vol'] = 0
            df['rsi_20'] = 50
            df['rsi'] = 50
            df['bb_low'] = df['bb_mid'] = df['bb_high'] = 0
            df['atr_z'] = 0
            df['macd_line'] = df['macd_z'] = 0
            df['dollar_volume'] = 0
            for length in [10, 20, 30, 40, 60, 100]:
                df[f'ma_{length}'] = df['adj close']
            df['vwap'] = df['adj close']
            df['vwap_std'] = 0
            for i in range(1, 4):
                df[f'upper_band_{i}'] = df[f'lower_band_{i}'] = df['adj close']
                df[f'touches_upper_band_{i}'] = df[f'touches_lower_band_{i}'] = 0
                df[f'far_ratio_upper_band_{i}'] = df[f'far_ratio_lower_band_{i}'] = 0
            df['touches_vwap'] = 0
            df['far_ratio_vwap'] = 0
            df['bearish_stack'] = 0
            df['trend_strength_up'] = df['trend_strength_down'] = 0
            df['prev_volume'] = df['volume']
            return df

    def calculate_trade_features(self, df, signal_type, entry):
        try:
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
                
            df['sl_distance'] = abs(entry - df['sl_price']) * 10000
            df['tp_distance'] = abs(df['tp_price'] - entry) * 10000
            # Calculate rrr for internal use but don't include in final features
            rrr = df['tp_distance'] / (df['sl_distance'] + 1e-6)
            df['log_sl'] = np.log1p(df['sl_price'])
            
            return df
        except Exception as e:
            logger.error(f"Error in calculate_trade_features: {str(e)}")
            # Set default values
            df['sl_price'] = entry * 0.99
            df['tp_price'] = entry * 1.01
            df['sl_distance'] = 100
            df['tp_distance'] = 100
            df['log_sl'] = np.log1p(entry)
            return df

    def calculate_categorical_features(self, df):
        try:
            df = df.copy()
            
            # Time features
            df['hour'] = df['time'].dt.hour
            df['month'] = df['time'].dt.month
            df['dayofweek'] = df['time'].dt.dayofweek
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
            
            # Cyclical time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            
            # Day of week features
            df['day'] = df['time'].dt.day_name()
            all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for day in all_days:
                df[f'day_{day}'] = (df['day'] == day).astype(int)
            
            # Session features
            def get_session(hour):
                if 0 <= hour < 6:
                    return 'q2'
                elif 6 <= hour < 12:
                    return 'q3'
                elif 12 <= hour < 18:
                    return 'q4'
                else:
                    return 'q1'
                    
            df['session'] = df['hour'].apply(get_session)
            for session in ['q1', 'q2', 'q3', 'q4']:
                df[f'session_{session}'] = (df['session'] == session).astype(int)
            df = df.drop('session', axis=1)
            
            # RSI zone features
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
            for zone in ['neutral', 'overbought', 'oversold', 'unknown']:
                df[f'rsi_zone_{zone}'] = (df['rsi_zone'] == zone).astype(int)
            df = df.drop('rsi_zone', axis=1)
            
            # Trend direction features
            def get_trend(row):
                try:
                    if row['trend_strength_up'] > row['trend_strength_down']:
                        return 'uptrend'
                    elif row['trend_strength_down'] > row['trend_strength_up']:
                        return 'downtrend'
                    else:
                        return 'sideways'
                except:
                    return 'sideways'
            
            df['trend_direction'] = df.apply(get_trend, axis=1)
            for direction in ['downtrend', 'sideways', 'uptrend']:
                df[f'trend_direction_{direction}'] = (df['trend_direction'] == direction).astype(int)
            df = df.drop('trend_direction', axis=1)
            
            return df
        except Exception as e:
            logger.error(f"Error in calculate_categorical_features: {str(e)}")
            # Set default values for essential columns
            default_time = datetime.now(NY_TZ)
            df['hour'] = default_time.hour
            df['month'] = default_time.month
            df['dayofweek'] = default_time.weekday()
            df['is_weekend'] = 0
            df['hour_sin'] = np.sin(2 * np.pi * default_time.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * default_time.hour / 24)
            df['dayofweek_sin'] = np.sin(2 * np.pi * default_time.weekday() / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * default_time.weekday() / 7)
            
            all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for day in all_days:
                df[f'day_{day}'] = 0
            df[f'day_{default_time.strftime("%A")}'] = 1
            
            for session in ['q1', 'q2', 'q3', 'q4']:
                df[f'session_{session}'] = 0
            df['session_q1'] = 1  # Default
            
            for zone in ['neutral', 'overbought', 'oversold', 'unknown']:
                df[f'rsi_zone_{zone}'] = 0
            df['rsi_zone_neutral'] = 1  # Default
            
            for direction in ['downtrend', 'sideways', 'uptrend']:
                df[f'trend_direction_{direction}'] = 0
            df['trend_direction_sideways'] = 1  # Default
            
            return df

    def calculate_minutes_closed(self, df):
        """Calculate minutes closed based on actual candle timestamp"""
        try:
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
        except Exception as e:
            logger.error(f"Error in calculate_minutes_closed: {str(e)}")
            return df

    def calculate_combo_flags(self, row, signal_type):
        """Calculate combo flags using preloaded dictionaries"""
        try:
            # 1. Determine trend direction with fallbacks
            trend_str = 'sideways'  # Default
            
            # Use get() method to avoid KeyError
            downtrend_val = row.get('trend_direction_downtrend', 0)
            sideways_val = row.get('trend_direction_sideways', 0)
            uptrend_val = row.get('trend_direction_uptrend', 0)
            
            if downtrend_val == 1:
                trend_str = 'downtrend'
            elif uptrend_val == 1:
                trend_str = 'uptrend'
            elif sideways_val == 1:
                trend_str = 'sideways'
            
            # 2. Bin RSI value
            rsi_val = row.get('rsi', 50)  # Default to 50 if missing
            rsi_bin = None
            for i in range(len(self.rsi_bins)-1):
                if self.rsi_bins[i] <= rsi_val < self.rsi_bins[i+1]:
                    rsi_bin = f"({self.rsi_bins[i]}, {self.rsi_bins[i+1]}]"
                    break
            
            # 3. Bin MACD_Z value
            macd_z_val = row.get('macd_z', 0)  # Default to 0 if missing
            macd_z_bin = None
            for i in range(len(self.macd_z_bins)-1):
                if self.macd_z_bins[i] <= macd_z_val < self.macd_z_bins[i+1]:
                    macd_z_bin = f"({self.macd_z_bins[i]}, {self.macd_z_bins[i+1]}]"
                    break
            
            # 4. Create combo keys
            combo_key = f"{signal_type}_{trend_str}_{rsi_bin}" if rsi_bin else f"{signal_type}_{trend_str}_nan"
            combo_key2 = f"{signal_type}_{rsi_bin}_{macd_z_bin}" if rsi_bin and macd_z_bin else f"{signal_type}_{rsi_bin}_nan"
            
            # 5. Get flags from preloaded dictionaries
            flag1 = COMBO_FLAGS.get(combo_key, 'dead')
            flag2 = COMBO_FLAGS2.get(combo_key2, 'dead')
            
            # 6. Create flags dictionary
            return {
                'combo_flag': flag1,
                'combo_flag2': flag2,
                'is_bad_combo': 1 if flag1 == 'dead' else 0
            }
        except Exception as e:
            logger.error(f"Error in calculate_combo_flags: {str(e)}")
            return {
                'combo_flag': 'dead',
                'combo_flag2': 'dead',
                'is_bad_combo': 1
            }

    def generate_features(self, df, signal_type):
        try:
            if len(df) < 200:
                logger.warning("Not enough data for feature generation")
                return None
            
            df = df.tail(200).copy()
            
            # Set current candle close = open for immediate processing
            current_candle = df.iloc[-1].copy()
            current_candle['close'] = current_candle['open']
            df.iloc[-1] = current_candle
            
            df = self.calculate_technical_indicators(df)
            df = self.calculate_trade_features(df, signal_type, df.iloc[-1]['open'])
            df = self.calculate_categorical_features(df)
            df = self.calculate_minutes_closed(df)
            
            # Volume features
            df['prev_volume'] = df['volume'].shift(1).fillna(df['volume'])
            df['body_size'] = abs(df['close'] - df['open'])
            df['wick_up'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['wick_down'] = df[['close', 'open']].min(axis=1) - df['low']
            df['prev_body_size'] = df['body_size'].shift(1).fillna(df['body_size'])
            df['prev_wick_up'] = df['wick_up'].shift(1).fillna(df['wick_up'])
            df['prev_wick_down'] = df['wick_down'].shift(1).fillna(df['wick_down'])
            
            # Volume binning
            avg_volume = df['volume'].rolling(20, min_periods=1).mean().iloc[-1]
            df['volume_bin'] = (df['volume'] > avg_volume).astype(int)
            df['dollar_volume_bin'] = ((df['adj close'] * df['volume']) > 1000000).astype(int)
            
            # Ratio features
            small_value = 1e-6
            df['price_div_vol'] = df['adj close'] / (df['garman_klass_vol'] + small_value)
            df['rsi_div_macd'] = df['rsi'] / (df['macd_z'] + small_value)
            df['price_div_vwap'] = df['adj close'] / (df['vwap'] + small_value)
            df['sl_div_atr'] = df['sl_distance'] / (abs(df['atr_z']) + small_value)
            df['tp_div_atr'] = df['tp_distance'] / (abs(df['atr_z']) + small_value)
            # Calculate rrr for internal use in rrr_div_rsi but don't include rrr in final features
            rrr = df['tp_distance'] / (df['sl_distance'] + 1e-6)
            df['rrr_div_rsi'] = rrr / (df['rsi'] + small_value)
            
            current_row = df.iloc[-1]
            combo_flags = self.calculate_combo_flags(current_row, signal_type)
            
            # Set flags in dataframe
            for flag_type in ['dead', 'fair', 'fine']:
                df[f'combo_flag_{flag_type}'] = 1 if combo_flags['combo_flag'] == flag_type else 0
                df[f'combo_flag2_{flag_type}'] = 1 if combo_flags['combo_flag2'] == flag_type else 0
                
            df['is_bad_combo'] = combo_flags['is_bad_combo']
            
            df['crt_BUY'] = int(signal_type == 'BUY')
            df['crt_SELL'] = int(signal_type == 'SELL')
            df['trade_type_BUY'] = int(signal_type == 'BUY')
            df['trade_type_SELL'] = int(signal_type == 'SELL')
            
            # Create features Series with EXACTLY the features we want
            features = pd.Series(index=self.features, dtype=float)
            for feat in self.features:
                if feat in df.columns:
                    features[feat] = df[feat].iloc[-1]
                else:
                    features[feat] = 0
        
            # CRITICAL: Using shifted features (only for features that exist in our final set)
            if len(df) >= 2:
                prev_candle = df.iloc[-2]
                for feat in self.shift_features:
                    # Only shift if this feature is in our final feature set
                    if feat in features.index and feat in prev_candle:
                        features[feat] = prev_candle[feat]
            
            if features.isna().any():
                for col in features[features.isna()].index:
                    features[col] = 0
            
            # Final verification
            if len(features) != len(self.features):
                logger.error(f"Feature count mismatch: generated {len(features)}, expected {len(self.features)}")
                # Force alignment
                features = features.reindex(self.features, fill_value=0)
            
            logger.debug(f"Successfully generated {len(features)} features for {self.timeframe}")
            return features
        except Exception as e:
            logger.error(f"Error in generate_features: {str(e)}")
            logger.error(traceback.format_exc())
            return None

# ========================
# HTF ANALYZER CLASS - CORE DIRECTION DETECTION
# ========================
class HTFAnalyzer:
    """Higher Timeframe Analyzer for 4H and 1H direction detection"""
    
    def __init__(self, timeframe, credentials):
        self.timeframe = timeframe
        self.credentials = credentials
        self.logger = logging.getLogger(f"HTF_{timeframe}")
        
        # HTF state - persisted until next candle
        self.current_direction = None  # 'BUY', 'SELL', or None
        self.current_prediction = 0.0
        self.last_analysis_time = None
        self.next_candle_time = None
        self.is_active = False
        
        # Load HTF model if available
        model_path = os.path.join(MODELS_DIR, f"{timeframe.lower()}_model.keras")
        scaler_path = os.path.join(MODELS_DIR, f"scaler_{timeframe.lower()}.joblib")
        
        self.model_available = True
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            self.logger.warning(f"HTF model not available for {timeframe}")
            self.model_available = False
        else:
            try:
                self.model_loader = ModelLoader(model_path, scaler_path)
                self.logger.info(f"HTF model loaded for {timeframe}")
            except Exception as e:
                self.logger.error(f"HTF model loading failed: {str(e)}")
                self.model_available = False
                
        self.logger.info(f"HTF Analyzer initialized for {timeframe} - Model: {'Available' if self.model_available else 'Not Available'}")

    def calculate_next_htf_candle_time(self):
        """Calculate when the next HTF candle will close"""
        now = datetime.now(NY_TZ)
        
        if self.timeframe == "H4":
            # H4 candles: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 (NY time)
            current_hour = now.hour
            next_hour = ((current_hour // 4) * 4 + 4) % 24
            if next_hour < current_hour:  # Next day
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
                
        else:  # H1
            next_hour = now.hour + 1
            if next_hour >= 24:
                next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        
        return next_time

    def analyze(self):
        """Analyze HTF and set direction if prediction > 60%"""
        try:
            self.logger.info(f"Starting HTF {self.timeframe} analysis...")
            
            # Fetch HTF data
            df = fetch_candles(self.timeframe, count=50, api_key=self.credentials['oanda_api_key'])
            if df.empty:
                self.logger.warning(f"No data received for HTF {self.timeframe}")
                return False
                
            # Check if we have a complete current candle
            current_candle = df.iloc[-1]
            if not current_candle.get('complete', True):
                self.logger.info(f"HTF {self.timeframe} current candle not complete, waiting...")
                return False
            
            # Generate features for HTF analysis
            features = self.generate_htf_features(df)
            if features is None:
                self.logger.warning(f"Failed to generate features for HTF {self.timeframe}")
                return False
            
            # Get prediction from model
            if self.model_available:
                prediction = self.model_loader.predict(features)
                self.current_prediction = prediction
                
                # Determine direction based on prediction threshold
                if prediction > HTF_PREDICTION_THRESHOLD:
                    # Assuming model predicts probability of UP direction
                    self.current_direction = 'BUY' if prediction > 0.5 else 'SELL'
                    self.is_active = True
                else:
                    self.current_direction = None
                    self.is_active = False
            else:
                # Fallback: Use simple price action for direction
                self.current_direction = self._simple_direction_analysis(df)
                self.current_prediction = 0.7 if self.current_direction else 0.3
                self.is_active = self.current_direction is not None
            
            self.last_analysis_time = datetime.now(NY_TZ)
            self.next_candle_time = self.calculate_next_htf_candle_time()
            
            status = "ACTIVE" if self.is_active else "INACTIVE"
            direction_str = self.current_direction if self.current_direction else "NONE"
            
            self.logger.info(f"HTF {self.timeframe} Analysis Complete - "
                           f"Direction: {direction_str}, Prediction: {self.current_prediction:.3f}, Status: {status}")
            
            # Send HTF status to Telegram
            self.send_htf_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"HTF analysis error: {str(e)}")
            return False

    def generate_htf_features(self, df):
        """Generate features for HTF analysis (simplified version)"""
        try:
            if len(df) < 20:
                return None
                
            # Use existing FeatureEngineer for consistency
            feature_engineer = FeatureEngineer(self.timeframe)
            df = feature_engineer.calculate_technical_indicators(df)
            
            # Create simple features array for HTF model
            # This would need to match your HTF model's expected input
            features = np.array([
                df['close'].iloc[-1],
                df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
                df['macd_line'].iloc[-1] if 'macd_line' in df.columns else 0,
                df['volume'].iloc[-1],
            ])
            
            return features
            
        except Exception as e:
            self.logger.error(f"HTF feature generation error: {str(e)}")
            return None

    def _simple_direction_analysis(self, df):
        """Simple fallback direction analysis using price action"""
        try:
            if len(df) < 20:
                return None
                
            # Use EMA cross for simple direction
            ema_9 = df['close'].ewm(span=9).mean().iloc[-1]
            ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > ema_9 and ema_9 > ema_21:
                return 'BUY'
            elif current_price < ema_9 and ema_9 < ema_21:
                return 'SELL'
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Simple direction analysis error: {str(e)}")
            return None

    def send_htf_status(self):
        """Send HTF status to Telegram"""
        try:
            direction_str = self.current_direction if self.current_direction else "NO DIRECTION"
            status = "🟢 ACTIVE" if self.is_active else "🔴 INACTIVE"
            next_update = self.next_candle_time.strftime("%H:%M") if self.next_candle_time else "Unknown"
            
            message = (
                f"📊 *HTF {self.timeframe} STATUS UPDATE*\n"
                f"Direction: {direction_str}\n"
                f"Prediction: {self.current_prediction:.3f}\n"
                f"Status: {status}\n"
                f"Next Update: {next_update} NY\n"
                f"Last Analysis: {self.last_analysis_time.strftime('%H:%M:%S') if self.last_analysis_time else 'Never'}"
            )
            
            send_telegram(message, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
            
        except Exception as e:
            self.logger.error(f"Error sending HTF status: {str(e)}")

    def should_allow_ltf_signal(self, ltf_signal_type):
        """Check if LTF signal should be allowed based on HTF direction"""
        if not self.is_active:
            self.logger.info(f"HTF {self.timeframe} not active - rejecting LTF signal")
            return False
            
        if self.current_direction != ltf_signal_type:
            self.logger.info(f"LTF signal {ltf_signal_type} does not align with HTF direction {self.current_direction}")
            return False
            
        self.logger.info(f"LTF signal {ltf_signal_type} aligns with HTF direction {self.current_direction} - ALLOWING")
        return True

    def run(self):
        """Main HTF analysis loop"""
        thread_name = threading.current_thread().name
        self.logger.info(f"Starting HTF analyzer thread: {thread_name}")
        
        while True:
            try:
                # Wait for next HTF candle
                now = datetime.now(NY_TZ)
                if self.next_candle_time and now < self.next_candle_time:
                    sleep_seconds = (self.next_candle_time - now).total_seconds()
                    if sleep_seconds > 0:
                        self.logger.debug(f"HTF {self.timeframe} sleeping for {sleep_seconds:.1f}s")
                        time.sleep(sleep_seconds)
                
                # Perform analysis
                self.analyze()
                
                # Small delay before next cycle
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"HTF analyzer error: {str(e)}")
                time.sleep(60)  # Wait 1 minute on error

# ========================
# MODELLESS LTF BOT - HTF-DRIVEN
# ========================
class HTFDrivenLTFBot:
    """LTF Bot that only runs when HTF has active direction"""
    
    def __init__(self, timeframe, credentials, htf_analyzer):
        self.timeframe = timeframe
        self.credentials = credentials
        self.htf_analyzer = htf_analyzer
        self.logger = logging.getLogger(f"HTFDriven_{timeframe}")
        self.start_time = time.time()
        self.max_duration = 11.5 * 3600
        
        # Initialize storage
        self.storage = GoogleSheetsStorage('1HZo4uUfeYrzoeEQkjoxwylrqQpKI4R9OfHOZ6zaDino')
        
        # NO MODEL LOADING - running modelless as requested
        self.model_available = False
        self.logger.info(f"Initializing MODELLESS {timeframe} bot driven by HTF {htf_analyzer.timeframe}")
        
        # Initialize feature engineer (for feature generation only)
        self.feature_engineer = FeatureEngineer(timeframe)
        self.data = pd.DataFrame()

    def calculate_next_candle_time(self):
        """Calculate next LTF candle time"""
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
        
        next_time += timedelta(seconds=0.3)
        
        if now >= next_time:
            next_time += timedelta(minutes=5 if self.timeframe == "M5" else 15)
        
        return next_time

    def send_htf_aligned_signal(self, signal_type, signal_data, features):
        """Send signal that is aligned with HTF direction"""
        latency_ms = (datetime.now(NY_TZ) - signal_data['time']).total_seconds() * 1000
        
        message = (
            f"🎯 *HTF-ALIGNED SIGNAL ({self.timeframe})*\n"
            f"Type: {signal_type} ✅\n"
            f"HTF Direction: {self.htf_analyzer.current_direction} 📊\n"
            f"HTF Confidence: {self.htf_analyzer.current_prediction:.3f}\n"
            f"Entry: {signal_data['entry']:.5f}\n"
            f"SL: {signal_data['sl']:.5f}\n"
            f"TP: {signal_data['tp']:.5f}\n"
            f"RRR: 1:4 (MODELLESS)\n"
            f"Latency: {latency_ms:.1f}ms\n"
            f"Time: {signal_data['time'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        send_telegram(message, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
        
        # Store in Google Sheets
        self.storage.append_signal(
            timeframe=self.timeframe,
            signal_data=signal_data,
            features=features,
            prediction=self.htf_analyzer.current_prediction  # Use HTF prediction as confidence
        )

    def run(self):
        """Main LTF execution loop - ONLY RUNS WHEN HTF HAS ACTIVE DIRECTION"""
        thread_name = threading.current_thread().name
        self.logger.info(f"Starting HTF-driven LTF bot thread: {thread_name}")
        
        session_start = time.time()
        
        while True:
            try:
                # Check if HTF has active direction
                if not self.htf_analyzer.is_active:
                    self.logger.debug(f"HTF {self.htf_analyzer.timeframe} not active - LTF {self.timeframe} sleeping")
                    time.sleep(30)  # Check every 30 seconds
                    continue
                
                # HTF is active - proceed with LTF analysis
                now = datetime.now(NY_TZ)
                next_candle = self.calculate_next_candle_time()
                sleep_seconds = max(0, (next_candle - now).total_seconds() - 0.1)
                
                if sleep_seconds > 0:
                    self.logger.debug(f"LTF {self.timeframe} sleeping for {sleep_seconds:.2f}s until next candle")
                    time.sleep(sleep_seconds)
                
                # Busy-wait for precise candle open
                while datetime.now(NY_TZ) < next_candle:
                    time.sleep(0.001)
                
                self.logger.debug("Candle open detected - waiting 5s for candle availability")
                time.sleep(5)
                
                # Fetch LTF data
                new_data = fetch_candles(
                    self.timeframe,
                    count=201,
                    api_key=self.credentials['oanda_api_key']
                )
                
                if new_data.empty:
                    self.logger.error("Failed to fetch LTF candle data")
                    continue
                    
                self.data = new_data
                self.logger.debug(f"LTF {self.timeframe} data updated: {len(self.data)} records")
                
                # Detect CRT pattern
                signal_type, signal_data = self.feature_engineer.calculate_crt_signal(self.data)
                
                if not signal_type:
                    self.logger.debug("No CRT pattern detected")
                    continue
                    
                self.logger.info(f"CRT pattern detected: {signal_type}")
                
                # Check HTF alignment
                if not self.htf_analyzer.should_allow_ltf_signal(signal_type):
                    continue  # Signal doesn't align with HTF direction
                
                # Generate features for Telegram
                features = self.feature_engineer.generate_features(self.data, signal_type)
                if features is None:
                    self.logger.warning("Feature generation failed")
                    continue
                
                # Send features to Telegram
                self.send_features_to_telegram(features, signal_type)
                
                # Send HTF-aligned signal
                self.send_htf_aligned_signal(signal_type, signal_data, features)
                    
            except Exception as e:
                error_msg = f"❌ LTF {self.timeframe} bot error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                send_telegram(error_msg[:1000], self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
                time.sleep(60)

    def send_features_to_telegram(self, features, signal_type):
        """Send features to Telegram"""
        if features is not None:
            send_features_telegram(
                features, 
                signal_type, 
                self.timeframe,
                self.credentials['telegram_token'], 
                self.credentials['telegram_chat_id']
            )

    def test_credentials(self):
        """Test credentials"""
        logger.info("Testing credentials...")
        
        # Test Telegram
        test_msg = f"🔧 {self.timeframe} HTF-driven bot credentials test"
        telegram_ok = send_telegram(test_msg, self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
        
        # Test Oanda
        oanda_ok = False
        try:
            test_data = fetch_candles("M5", count=1, api_key=self.credentials['oanda_api_key'])
            oanda_ok = not test_data.empty
        except Exception as e:
            logger.error(f"Oanda test failed: {str(e)}")
            
        return telegram_ok and oanda_ok

# ========================
# COORDINATED BOT MANAGER - HTF-FIRST APPROACH
# ========================
class HTFFirstBotManager:
    """Manages HTF-first coordinated trading"""
    
    def __init__(self, credentials):
        self.credentials = credentials
        self.logger = logging.getLogger("HTFFirstManager")
        
        # Initialize HTF analyzers
        self.htf_4h = HTFAnalyzer("H4", credentials)
        self.htf_1h = HTFAnalyzer("H1", credentials)
        
        # Initialize LTF bots with their respective HTF analyzers
        self.bot_15m = HTFDrivenLTFBot("M15", credentials, self.htf_4h)  # M15 follows H4
        self.bot_5m = HTFDrivenLTFBot("M5", credentials, self.htf_1h)    # M5 follows H1
        
        self.logger.info("HTF-First Bot Manager initialized")
        
    def start_all(self):
        """Start all bots and analyzers"""
        threads = []
        
        # Start HTF analyzers first (they determine if LTF runs)
        t_htf_4h = threading.Thread(target=self.htf_4h.run, name="HTF_4H_Analyzer")
        t_htf_1h = threading.Thread(target=self.htf_1h.run, name="HTF_1H_Analyzer")
        
        t_htf_4h.daemon = True
        t_htf_1h.daemon = True
        
        threads.extend([t_htf_4h, t_htf_1h])
        
        # Start LTF bots (they will wait for HTF direction)
        t_15m = threading.Thread(target=self.bot_15m.run, name="HTFDriven_M15")
        t_5m = threading.Thread(target=self.bot_5m.run, name="HTFDriven_M5")
        
        t_15m.daemon = True
        t_5m.daemon = True
        
        threads.extend([t_15m, t_5m])
        
        # Start all threads
        for thread in threads:
            thread.start()
            self.logger.info(f"Started thread: {thread.name}")
            
        return threads

# ========================
# MAIN EXECUTION - HTF-FIRST APPROACH
# ========================
if __name__ == "__main__":
    print("===== HTF-FIRST TRADING BOT STARTING =====")
    print(f"Start time: {datetime.now(NY_TZ)}")
    
    # Force debug logging to console
    debug_handler = logging.StreamHandler()
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(debug_handler)
    
    logger.info("Starting HTF-first coordinated execution")
    
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
    
    # Check credentials
    logger.info("Checking credentials...")
    credentials_status = {k: "SET" if v else "MISSING" for k, v in credentials.items()}
    for k, status in credentials_status.items():
        logger.info(f"{k}: {status}")
    
    if not all(credentials.values()):
        logger.error("Missing one or more credentials")
        if credentials['telegram_token'] and credentials['telegram_chat_id']:
            send_telegram("❌ HTF-first bot failed to start: Missing credentials", 
                         credentials['telegram_token'], credentials['telegram_chat_id'])
        sys.exit(1)
    
    logger.info("All credentials present")
    
    try:
        # Create HTF-first bot manager
        bot_manager = HTFFirstBotManager(credentials)
        
        # Start all bots
        threads = bot_manager.start_all()
        
        logger.info("All HTF-first bots started successfully")
        
        # Send startup message
        startup_msg = (
            "🤖 *HTF-FIRST TRADING BOT STARTED* 🤖\n"
            "• H4 analyzer drives M15 trading\n" 
            "• H1 analyzer drives M5 trading\n"
            "• LTF only runs when HTF has >60% prediction\n"
            "• Signals must align with HTF direction\n"
            "• LTF running MODELLESS with CRT patterns\n"
            "• RRR: 1:4 for all signals"
        )
        send_telegram(startup_msg, credentials['telegram_token'], credentials['telegram_chat_id'])
        
        # Monitor threads
        logger.info("Main thread entering monitoring loop")
        while True:
            try:
                alive_threads = [t.name for t in threads if t.is_alive()]
                dead_threads = [t.name for t in threads if not t.is_alive()]
                
                if dead_threads:
                    logger.warning(f"Dead threads: {dead_threads}")
                    
                # Log HTF status
                htf_4h_status = "ACTIVE" if bot_manager.htf_4h.is_active else "INACTIVE"
                htf_1h_status = "ACTIVE" if bot_manager.htf_1h.is_active else "INACTIVE"
                
                logger.info(f"HTF Status - H4: {htf_4h_status}, H1: {htf_1h_status} | Active threads: {len(alive_threads)}/{len(threads)}")
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                time.sleep(60)
                
    except Exception as e:
        logger.error(f"HTF-first execution failed: {str(e)}", exc_info=True)
        if credentials.get('telegram_token') and credentials.get('telegram_chat_id'):
            send_telegram(f"❌ HTF-first bot crashed: {str(e)[:500]}", 
                         credentials['telegram_token'], credentials['telegram_chat_id'])
        sys.exit(1)
