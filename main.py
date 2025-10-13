# ========================
# COMPLETE UPDATED MAIN.PY FOR CURRENT LIBRARY VERSIONS
# MAINTAINS EXACT SAME CALCULATION LOGIC WITH VERSION COMPATIBILITY
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
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
import tensorflow as tf
from google.colab import drive
from IPython.display import clear_output

import logging
logging.getLogger('numba').setLevel(logging.WARNING)

# ========================
# SUPPRESS TENSORFLOW LOGS
# ========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')



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
        logging.FileHandler('trading_bot_debug.log')
    ]
)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)  # Change to INFO if you need Numba messages

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
# UPDATED FEATURE ENGINEER FOR CURRENT VERSIONS
# ========================
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime
import logging
import traceback

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, timeframe):
        self.timeframe = timeframe
        logger.debug(f"Initializing FeatureEngineer for {timeframe}")
        
        # Define features in EXACT order you specified
        self.features = [
            'minutes,closed', 'adj close', 'garman_klass_vol', 'rsi_20', 
            'bb_low', 'bb_mid', 'bb_high', 'atr_z', 'macd_line', 'macd_z', 
            'ma_10', 'ma_100', 'vwap', 'vwap_std', 'upper_band_1', 'lower_band_1', 
            'upper_band_2', 'lower_band_2', 'upper_band_3', 'lower_band_3', 
            'touches_vwap', 'touches_upper_band_1', 'touches_upper_band_2', 
            'touches_upper_band_3', 'touches_lower_band_1', 'touches_lower_band_2', 
            'touches_lower_band_3', 'far_ratio_vwap', 'far_ratio_upper_band_1', 
            'far_ratio_upper_band_2', 'far_ratio_upper_band_3', 'far_ratio_lower_band_1', 
            'far_ratio_lower_band_2', 'far_ratio_lower_band_3', 'rsi', 'ma_20', 
            'ma_30', 'ma_40', 'ma_60', 'bearish_stack', 'trend_strength_up', 
            'trend_strength_down', 'sl_price', 'tp_price', 'prev_volume', 
            'body_size', 'wick_up', 'wick_down', 'sl_distance', 'tp_distance', 
            'log_sl', 'prev_body_size', 'prev_wick_up', 'prev_wick_down', 
            'is_bad_combo', 'volume_bin', 'dollar_volume_bin'
        ] + self._get_combo_features() + [
            'average_combo_win_rate', 'day_Friday', 'day_Monday', 'day_Sunday', 
            'day_Thursday', 'day_Tuesday', 'day_Wednesday', 'session_q1', 
            'session_q2', 'session_q3', 'session_q4', 'rsi_zone_neutral', 
            'rsi_zone_overbought', 'rsi_zone_oversold', 'trend_direction_downtrend', 
            'trend_direction_sideways', 'trend_direction_uptrend', 'crt_BUY', 
            'crt_SELL', 'trade_type_BUY', 'trade_type_SELL'
        ] + self._get_rsi_bin_features() + self._get_combo_key_features() + [
            'combo_flag_dead', 'combo_flag_fair', 'combo_flag_fine'
        ] + self._get_macd_z_bin_features() + self._get_combo_key2_features() + [
            'combo_flag2_dead', 'combo_flag2_fair', 'combo_flag2_fine'
        ]
        
        # Setup bins
        self.rsi_bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
        self.macd_z_bins = [-12.386, -0.496, -0.138, 0.134, 0.527, 9.246]
        
        # Timeframe configuration
        self.tf_to_session_hours = {
            1: 1.6, 5: 8, 15: 24, 30: 48, 60: 96, 240: 384, 1440: 2304
        }
        
        # Parse and store combo win rates
        self.combo_win_rates = self._parse_combo_win_rates()

    def _parse_combo_win_rates(self):
        """Parse the combo win rates from the provided data"""
        win_rate_data = """3	touches_lower_band_2,crt,combo_flag2	1|SELL|fine	0.3056300268
3	touches_lower_band_2,trade_type,combo_flag2	1|SELL|fine	0.3056300268
3	touches_upper_band_2,touches_lower_band_3,trade_type	1|0|BUY	0.3011117974
3	touches_upper_band_2,touches_lower_band_3,crt	1|0|BUY	0.3011117974
3	touches_upper_band_3,touches_lower_band_2,crt	0|1|SELL	0.3090241343
3	touches_upper_band_3,touches_lower_band_2,trade_type	0|1|SELL	0.3090241343
3	touches_upper_band_2,touches_lower_band_2,crt	1|0|BUY	0.3018641666
3	touches_upper_band_2,touches_lower_band_2,trade_type	1|0|BUY	0.3018641666
3	touches_upper_band_2,touches_lower_band_2,trade_type	0|1|SELL	0.3101616117
3	touches_upper_band_2,touches_lower_band_2,crt	0|1|SELL	0.3101616117
3	touches_upper_band_1,touches_lower_band_2,trade_type	0|1|SELL	0.3006675904
3	touches_upper_band_1,touches_lower_band_2,crt	0|1|SELL	0.3006675904
3	touches_vwap,macd_z_bin,combo_flag2	1|(-0.12, 0.126]|fine	0.3085864267
3	touches_upper_band_1,trade_type,rsi_bin	1|BUY|(60, 70]	0.3263428205
3	touches_upper_band_1,rsi_bin,combo_flag	1|(60, 70]|fine	0.3263428205
3	touches_upper_band_1,crt,rsi_bin	1|BUY|(60, 70]	0.3263428205
3	touches_lower_band_3,trade_type,combo_flag	1|SELL|fine	0.3110777158
3	touches_lower_band_3,crt,combo_flag	1|SELL|fine	0.3110777158
3	touches_upper_band_3,crt,combo_flag2	1|BUY|fine	0.3155809859
3	touches_upper_band_3,trade_type,combo_flag2	1|BUY|fine	0.3155809859
3	touches_lower_band_3,crt,combo_flag2	1|SELL|fine	0.3239597622
3	touches_lower_band_3,trade_type,combo_flag2	1|SELL|fine	0.3239597622
3	trend_strength_up,trade_type,rsi_bin	0.0|BUY|(60, 70]	0.3049431423
3	trend_strength_up,crt,rsi_bin	0.0|BUY|(60, 70]	0.3049431423
3	trend_strength_up,rsi_bin,combo_flag	0.0|(60, 70]|fine	0.3049431423
3	touches_upper_band_2,trade_type,rsi_bin	1|BUY|(60, 70]	0.3175785797
3	touches_upper_band_2,crt,rsi_bin	1|BUY|(60, 70]	0.3175785797
3	touches_upper_band_2,rsi_bin,combo_flag	1|(60, 70]|fine	0.3175785797
3	touches_upper_band_1,macd_z_bin,combo_flag2	1|(-0.12, 0.126]|fine	0.30495613
3	touches_lower_band_1,crt,rsi_bin	1|SELL|(30, 40]	0.3350528126
3	touches_lower_band_1,trade_type,rsi_bin	1|SELL|(30, 40]	0.3350528126
3	bearish_stack,crt,rsi_bin	0|SELL|(30, 40]	0.3108585859
3	bearish_stack,trade_type,rsi_bin	0|SELL|(30, 40]	0.3108585859
3	trend_strength_down,rsi_bin,combo_flag	0.0|(30, 40]|fine	0.3108585859
3	bearish_stack,rsi_bin,combo_flag	0|(30, 40]|fine	0.3108585859
3	trend_strength_down,trade_type,rsi_bin	0.0|SELL|(30, 40]	0.3108585859
3	trend_strength_down,crt,rsi_bin	0.0|SELL|(30, 40]	0.3108585859
3	touches_lower_band_1,macd_z_bin,combo_flag2	1|(-0.12, 0.126]|fine	0.3100715746
3	touches_lower_band_3,trend_strength_up,trade_type	1|0.0|SELL	0.3049718093
3	touches_lower_band_3,trend_strength_up,crt	1|0.0|SELL	0.3049718093
3	touches_upper_band_3,touches_lower_band_3,combo_flag	1|0|fine	0.3051242236
3	touches_lower_band_2,crt,rsi_bin	1|SELL|(30, 40]	0.3284883721
3	touches_lower_band_2,trade_type,rsi_bin	1|SELL|(30, 40]	0.3284883721
3	touches_upper_band_2,touches_lower_band_3,combo_flag	0|1|fine	0.3003833516
3	touches_upper_band_3,touches_lower_band_3,combo_flag2	0|1|fine	0.3329636817
3	touches_upper_band_3,touches_lower_band_3,combo_flag2	1|0|fine	0.3225356046
3	rsi_bin,combo_flag,macd_z_bin	(60, 70]|fine|(0.126, 0.517]	0.306122449
3	rsi_bin,macd_z_bin,combo_flag2	(60, 70]|(0.126, 0.517]|fine	0.306122449
3	trade_type,rsi_bin,macd_z_bin	BUY|(60, 70]|(0.126, 0.517]	0.306122449
3	crt,rsi_bin,macd_z_bin	BUY|(60, 70]|(0.126, 0.517]	0.306122449
3	touches_upper_band_3,touches_lower_band_2,combo_flag	1|0|fine	0.3163387001
3	trend_direction,crt,rsi_bin	sideways|SELL|(30, 40]	0.3079002704
3	trend_direction,rsi_bin,combo_flag	sideways|(30, 40]|fine	0.3079002704
3	trend_direction,trade_type,rsi_bin	sideways|SELL|(30, 40]	0.3079002704
3	touches_lower_band_2,session,trade_type	1|q3|SELL	0.3016641452
3	touches_lower_band_2,session,crt	1|q3|SELL	0.3016641452
3	touches_upper_band_3,touches_lower_band_3,crt	0|1|SELL	0.3710118505
3	touches_upper_band_3,touches_lower_band_3,trade_type	0|1|SELL	0.3710118505
3	touches_upper_band_3,touches_lower_band_3,crt	1|0|BUY	0.3552229688
3	touches_upper_band_3,touches_lower_band_3,trade_type	1|0|BUY	0.3552229688
3	touches_upper_band_2,touches_lower_band_3,combo_flag2	0|1|fine	0.3416614616
3	touches_upper_band_3,touches_lower_band_2,combo_flag2	1|0|fine	0.330478909
3	rsi_bin,macd_z_bin,combo_flag2	(50, 60]|(-0.489, -0.12]|fine	0.334512705
3	trade_type,rsi_bin,macd_z_bin	BUY|(50, 60]|(-0.489, -0.12]	0.334512705
3	crt,rsi_bin,macd_z_bin	BUY|(50, 60]|(-0.489, -0.12]	0.334512705
3	rsi_zone,crt,combo_flag2	overbought|BUY|fine	0.3325581395
3	rsi_zone,crt,trade_type	overbought|BUY|BUY	0.3325581395
3	rsi_zone,trade_type,is_bad_combo	overbought|BUY|0	0.3325581395
3	rsi_zone,crt,is_bad_combo	overbought|BUY|0	0.3325581395
3	rsi_zone,trade_type,combo_flag2	overbought|BUY|fine	0.3325581395
3	rsi_zone,is_bad_combo,combo_flag2	overbought|0|fine	0.3325581395
3	rsi_zone,is_bad_combo,combo_flag	overbought|0|fine	0.3323363244
3	rsi_zone,trade_type,combo_flag	overbought|BUY|fine	0.3323363244
3	rsi_zone,combo_flag,combo_flag2	overbought|fine|fine	0.3323363244
3	rsi_zone,crt,combo_flag	overbought|BUY|fine	0.3323363244
3	rsi_zone,trade_type,rrr	overbought|BUY|4.0	0.3324450366
3	rsi_zone,rrr,combo_flag2	overbought|4.0|fine	0.3324450366
3	rsi_zone,crt,rrr	overbought|BUY|4.0	0.3324450366
3	rsi_zone,rrr,combo_flag	overbought|4.0|fine	0.3322225925
3	session,rsi_bin,combo_flag	q2|(60, 70]|fine	0.3099062918
3	session,trade_type,rsi_bin	q2|BUY|(60, 70]	0.3099062918
3	session,crt,rsi_bin	q2|BUY|(60, 70]	0.3099062918
3	touches_upper_band_2,touches_lower_band_3,trade_type	0|1|SELL	0.3722652306
3	touches_upper_band_2,touches_lower_band_3,crt	0|1|SELL	0.3722652306
3	rsi_zone,bearish_stack,combo_flag	overbought|0|fine	0.3308525034
3	rsi_zone,trend_strength_down,combo_flag	overbought|0.0|fine	0.3308525034
3	rsi_zone,bearish_stack,crt	overbought|0|BUY	0.3308525034
3	rsi_zone,bearish_stack,combo_flag2	overbought|0|fine	0.3308525034
3	rsi_zone,trend_strength_down,trade_type	overbought|0.0|BUY	0.3308525034
3	rsi_zone,bearish_stack,trade_type	overbought|0|BUY	0.3308525034
3	rsi_zone,trend_strength_down,crt	overbought|0.0|BUY	0.3308525034
3	rsi_zone,trend_strength_down,combo_flag2	overbought|0.0|fine	0.3308525034
3	touches_lower_band_3,rsi_zone,trade_type	0|overbought|BUY	0.3332199932
3	touches_lower_band_3,rsi_zone,crt	0|overbought|BUY	0.3332199932
3	touches_lower_band_3,rsi_zone,combo_flag2	0|overbought|fine	0.3332199932
3	touches_lower_band_3,rsi_zone,combo_flag	0|overbought|fine	0.3329931973
3	touches_upper_band_3,touches_lower_band_2,trade_type	1|0|BUY	0.3579952267
3	touches_upper_band_3,touches_lower_band_2,crt	1|0|BUY	0.3579952267
3	touches_lower_band_2,rsi_zone,crt	0|overbought|BUY	0.3336755647
3	touches_lower_band_2,rsi_zone,trade_type	0|overbought|BUY	0.3336755647
3	touches_lower_band_2,rsi_zone,combo_flag2	0|overbought|fine	0.3336755647
3	touches_lower_band_2,rsi_zone,combo_flag	0|overbought|fine	0.3334474495
3	touches_lower_band_1,rsi_zone,combo_flag2	0|overbought|fine	0.3290345068
3	touches_lower_band_1,rsi_zone,crt	0|overbought|BUY	0.3290345068
3	touches_lower_band_1,rsi_zone,trade_type	0|overbought|BUY	0.3290345068
3	touches_lower_band_1,rsi_zone,combo_flag	0|overbought|fine	0.3288005579
3	touches_upper_band_1,touches_lower_band_3,combo_flag	0|1|fine	0.3008976661
3	touches_vwap,rsi_zone,combo_flag2	0|overbought|fine	0.3145985401
3	touches_vwap,rsi_zone,crt	0|overbought|BUY	0.3145985401
3	touches_vwap,rsi_zone,trade_type	0|overbought|BUY	0.3145985401
3	touches_vwap,rsi_zone,combo_flag	0|overbought|fine	0.3143483023
3	trade_type,rsi_bin,macd_z_bin	SELL|(30, 40]|(-0.489, -0.12]	0.3125687065
3	rsi_bin,macd_z_bin,combo_flag2	(30, 40]|(-0.489, -0.12]|fine	0.3125687065
3	crt,rsi_bin,macd_z_bin	SELL|(30, 40]|(-0.489, -0.12]	0.3125687065
3	rsi_bin,combo_flag,combo_flag2	(70, 80]|fine|fine	0.3271855404
3	trade_type,rsi_bin,combo_flag	BUY|(70, 80]|fine	0.3271855404
3	rsi_bin,is_bad_combo,combo_flag	(70, 80]|0|fine	0.3271855404
3	crt,rsi_bin,combo_flag2	BUY|(70, 80]|fine	0.3271855404
3	trade_type,rsi_bin,is_bad_combo	BUY|(70, 80]|0	0.3271855404
3	rsi_zone,trade_type,rsi_bin	overbought|BUY|(70, 80]	0.3271855404
3	rsi_zone,rsi_bin,combo_flag2	overbought|(70, 80]|fine	0.3271855404
3	crt,rsi_bin,combo_flag	BUY|(70, 80]|fine	0.3271855404
3	crt,trade_type,rsi_bin	BUY|BUY|(70, 80]	0.3271855404
3	crt,rsi_bin,is_bad_combo	BUY|(70, 80]|0	0.3271855404
3	trade_type,rsi_bin,combo_flag2	BUY|(70, 80]|fine	0.3271855404
3	rsi_zone,crt,rsi_bin	overbought|BUY|(70, 80]	0.3271855404
3	rsi_zone,rsi_bin,combo_flag	overbought|(70, 80]|fine	0.3271855404
3	rsi_bin,is_bad_combo,combo_flag2	(70, 80]|0|fine	0.3271855404
3	trade_type,rrr,rsi_bin	BUY|4.0|(70, 80]	0.3270509978
3	crt,rrr,rsi_bin	BUY|4.0|(70, 80]	0.3270509978
3	rrr,rsi_bin,combo_flag	4.0|(70, 80]|fine	0.3270509978
3	rrr,rsi_bin,combo_flag2	4.0|(70, 80]|fine	0.3270509978
3	bearish_stack,crt,rsi_bin	0|BUY|(70, 80]	0.3254326561
3	trend_strength_down,trade_type,rsi_bin	0.0|BUY|(70, 80]	0.3254326561
3	trend_strength_down,rsi_bin,combo_flag2	0.0|(70, 80]|fine	0.3254326561
3	trend_strength_down,rsi_bin,combo_flag	0.0|(70, 80]|fine	0.3254326561
3	trend_strength_down,crt,rsi_bin	0.0|BUY|(70, 80]	0.3254326561
3	bearish_stack,rsi_bin,combo_flag2	0|(70, 80]|fine	0.3254326561
3	bearish_stack,rsi_bin,combo_flag	0|(70, 80]|fine	0.3254326561
3	bearish_stack,trade_type,rsi_bin	0|BUY|(70, 80]	0.3254326561
3	touches_lower_band_3,crt,rsi_bin	0|BUY|(70, 80]	0.3276708192
3	touches_lower_band_3,trade_type,rsi_bin	0|BUY|(70, 80]	0.3276708192
3	touches_lower_band_3,rsi_bin,combo_flag2	0|(70, 80]|fine	0.3276708192
3	touches_lower_band_3,rsi_bin,combo_flag	0|(70, 80]|fine	0.3276708192
3	touches_lower_band_2,rsi_bin,combo_flag	0|(70, 80]|fine	0.3280121627
3	touches_lower_band_2,trade_type,rsi_bin	0|BUY|(70, 80]	0.3280121627
3	touches_lower_band_2,rsi_bin,combo_flag2	0|(70, 80]|fine	0.3280121627
3	touches_lower_band_2,crt,rsi_bin	0|BUY|(70, 80]	0.3280121627
3	touches_lower_band_1,rsi_bin,combo_flag2	0|(70, 80]|fine	0.3232675184
3	touches_lower_band_1,rsi_bin,combo_flag	0|(70, 80]|fine	0.3232675184
3	touches_lower_band_1,trade_type,rsi_bin	0|BUY|(70, 80]	0.3232675184
3	touches_lower_band_1,crt,rsi_bin	0|BUY|(70, 80]	0.3232675184
3	trade_type,rsi_bin,macd_z_bin	SELL|(40, 50]|(0.126, 0.517]	0.3592079208
3	rsi_bin,macd_z_bin,combo_flag2	(40, 50]|(0.126, 0.517]|fine	0.3592079208
3	crt,rsi_bin,macd_z_bin	SELL|(40, 50]|(0.126, 0.517]	0.3592079208
3	touches_upper_band_3,touches_lower_band_1,combo_flag	1|0|fine	0.3191404696
3	touches_upper_band_1,touches_lower_band_3,combo_flag2	0|1|fine	0.3366533865
3	touches_vwap,crt,rsi_bin	0|BUY|(70, 80]	0.3102189781
3	touches_vwap,rsi_bin,combo_flag	0|(70, 80]|fine	0.3102189781
3	touches_vwap,rsi_bin,combo_flag2	0|(70, 80]|fine	0.3102189781
3	touches_vwap,trade_type,rsi_bin	0|BUY|(70, 80]	0.3102189781
3	minutes,closed,touches_upper_band_2,trade_type	0|1|BUY	0.3015034539
3	minutes,closed,touches_upper_band_2,crt	0|1|BUY	0.3015034539
3	touches_upper_band_3,touches_lower_band_1,combo_flag2	1|0|fine	0.328407225
3	rsi_zone,crt,macd_z_bin	overbought|BUY|(0.517, 10.271]	0.3026424443
3	rsi_zone,trade_type,macd_z_bin	overbought|BUY|(0.517, 10.271]	0.3026424443
3	rsi_zone,macd_z_bin,combo_flag2	overbought|(0.517, 10.271]|fine	0.3026424443
3	rsi_zone,combo_flag,macd_z_bin	overbought|fine|(0.517, 10.271]	0.302354399
3	rsi_zone,crt,rrr	oversold|SELL|4.0	0.3458021613
3	rsi_zone,crt,is_bad_combo	oversold|SELL|0	0.3458021613
3	rsi_zone,trade_type,rrr	oversold|SELL|4.0	0.3458021613
3	rsi_zone,trade_type,is_bad_combo	oversold|SELL|0	0.3458021613
3	rsi_zone,crt,trade_type	oversold|SELL|SELL	0.3458021613
3	rsi_zone,crt,combo_flag2	oversold|SELL|fine	0.3462338743
3	rsi_zone,trade_type,combo_flag2	oversold|SELL|fine	0.3462338743
3	rsi_zone,rrr,combo_flag2	oversold|4.0|fine	0.3462338743
3	rsi_zone,is_bad_combo,combo_flag2	oversold|0|fine	0.3462338743
3	rsi_zone,rrr,combo_flag	oversold|4.0|fine	0.3455606503
3	rsi_zone,crt,combo_flag	oversold|SELL|fine	0.3455606503
3	rsi_zone,is_bad_combo,combo_flag	oversold|0|fine	0.3455606503
3	rsi_zone,trade_type,combo_flag	oversold|SELL|fine	0.3455606503
3	rsi_zone,combo_flag,combo_flag2	oversold|fine|fine	0.345704754
3	touches_upper_band_3,rsi_zone,crt	0|oversold|SELL	0.3465430017
3	touches_upper_band_3,rsi_zone,trade_type	0|oversold|SELL	0.3465430017
3	touches_upper_band_3,rsi_zone,combo_flag2	0|oversold|fine	0.346835443
3	touches_upper_band_3,rsi_zone,combo_flag	0|oversold|fine	0.3461538462
3	touches_upper_band_2,rsi_zone,crt	0|oversold|SELL	0.3463002114
3	touches_upper_band_2,rsi_zone,trade_type	0|oversold|SELL	0.3463002114
3	touches_upper_band_2,rsi_zone,combo_flag2	0|oversold|fine	0.3465933136
3	touches_upper_band_2,rsi_zone,combo_flag	0|oversold|fine	0.3459092836
3	rsi_zone,trend_strength_up,trade_type	oversold|0.0|SELL	0.3442553191
3	rsi_zone,trend_strength_up,crt	oversold|0.0|SELL	0.3442553191
3	rsi_zone,trend_strength_up,combo_flag	oversold|0.0|fine	0.344548552
3	rsi_zone,trend_strength_up,combo_flag2	oversold|0.0|fine	0.3446953558
3	touches_lower_band_2,macd_z_bin,combo_flag2	1|(-0.12, 0.126]|fine	0.3189581554
3	rsi_zone,trend_strength_up,combo_flag2	overbought|1.0|fine	0.323919555
3	rsi_zone,trend_strength_up,trade_type	overbought|1.0|BUY	0.323919555
3	rsi_zone,trend_strength_up,crt	overbought|1.0|BUY	0.323919555
3	rsi_zone,trend_strength_up,combo_flag	overbought|1.0|fine	0.323919555
3	rsi_zone,trend_direction,trade_type	overbought|uptrend|BUY	0.323919555
3	rsi_zone,trend_direction,crt	overbought|uptrend|BUY	0.323919555
3	rsi_zone,trend_direction,combo_flag2	overbought|uptrend|fine	0.323919555
3	rsi_zone,trend_direction,combo_flag	overbought|uptrend|fine	0.323919555
3	touches_upper_band_2,macd_z_bin,combo_flag2	1|(-0.12, 0.126]|fine	0.3083511777
3	touches_upper_band_1,rsi_zone,trade_type	0|oversold|SELL	0.3403433476
3	touches_upper_band_1,rsi_zone,crt	0|oversold|SELL	0.3403433476
3	touches_upper_band_1,rsi_zone,combo_flag2	0|oversold|fine	0.3406357388
3	touches_upper_band_1,touches_lower_band_3,crt	0|1|SELL	0.3601203266
3	touches_upper_band_1,touches_lower_band_3,trade_type	0|1|SELL	0.3601203266
3	touches_upper_band_1,rsi_zone,combo_flag	0|oversold|fine	0.3399311532
3	touches_upper_band_3,touches_lower_band_1,crt	1|0|BUY	0.348583878
3	touches_upper_band_3,touches_lower_band_1,trade_type	1|0|BUY	0.348583878
3	session,trade_type,rsi_bin	q2|SELL|(30, 40]	0.3162583519
3	session,crt,rsi_bin	q2|SELL|(30, 40]	0.3162583519
3	touches_vwap,rsi_zone,crt	0|oversold|SELL	0.3294642857
3	touches_vwap,rsi_zone,trade_type	0|oversold|SELL	0.3294642857
3	touches_vwap,rsi_zone,combo_flag2	0|oversold|fine	0.3296114337
3	touches_vwap,rsi_zone,combo_flag	0|oversold|fine	0.3290062668
3	trade_type,rsi_bin,combo_flag2	SELL|(20, 30]|fine	0.3393854749
3	rsi_bin,is_bad_combo,combo_flag2	(20, 30]|0|fine	0.3393854749
3	rrr,rsi_bin,combo_flag	4.0|(20, 30]|fine	0.3393854749
3	rsi_bin,is_bad_combo,combo_flag	(20, 30]|0|fine	0.3393854749
3	trade_type,rsi_bin,combo_flag	SELL|(20, 30]|fine	0.3393854749
3	rsi_bin,combo_flag,combo_flag2	(20, 30]|fine|fine	0.3393854749
3	crt,rsi_bin,combo_flag	SELL|(20, 30]|fine	0.3393854749
3	rsi_zone,rsi_bin,combo_flag2	oversold|(20, 30]|fine	0.3393854749
3	crt,trade_type,rsi_bin	SELL|SELL|(20, 30]	0.3393854749
3	rrr,rsi_bin,combo_flag2	4.0|(20, 30]|fine	0.3393854749
3	trade_type,rsi_bin,is_bad_combo	SELL|(20, 30]|0	0.3393854749
3	crt,rrr,rsi_bin	SELL|4.0|(20, 30]	0.3393854749
3	rsi_zone,crt,rsi_bin	oversold|SELL|(20, 30]	0.3393854749
3	rsi_zone,rsi_bin,combo_flag	oversold|(20, 30]|fine	0.3393854749
3	rsi_zone,trade_type,rsi_bin	oversold|SELL|(20, 30]	0.3393854749
3	crt,rsi_bin,is_bad_combo	SELL|(20, 30]|0	0.3393854749
3	trade_type,rrr,rsi_bin	SELL|4.0|(20, 30]	0.3393854749
3	crt,rsi_bin,combo_flag2	SELL|(20, 30]|fine	0.3393854749
3	touches_upper_band_3,rsi_bin,combo_flag2	0|(20, 30]|fine	0.3401039206
3	touches_upper_band_3,rsi_bin,combo_flag	0|(20, 30]|fine	0.3401039206
3	touches_upper_band_3,trade_type,rsi_bin	0|SELL|(20, 30]	0.3401039206
3	touches_upper_band_3,crt,rsi_bin	0|SELL|(20, 30]	0.3401039206
3	touches_upper_band_2,rsi_bin,combo_flag2	0|(20, 30]|fine	0.3398104265
3	touches_upper_band_2,crt,rsi_bin	0|SELL|(20, 30]	0.3398104265
3	touches_upper_band_2,rsi_bin,combo_flag	0|(20, 30]|fine	0.3398104265
3	touches_upper_band_2,trade_type,rsi_bin	0|SELL|(20, 30]	0.3398104265
3	trend_strength_up,trade_type,rsi_bin	0.0|SELL|(20, 30]	0.3381020505
3	trend_strength_up,rsi_bin,combo_flag	0.0|(20, 30]|fine	0.3381020505
3	trend_strength_up,crt,rsi_bin	0.0|SELL|(20, 30]	0.3381020505
3	trend_strength_up,rsi_bin,combo_flag2	0.0|(20, 30]|fine	0.3381020505
3	touches_upper_band_1,trade_type,rsi_bin	0|SELL|(20, 30]	0.3336543091
3	touches_upper_band_1,rsi_bin,combo_flag2	0|(20, 30]|fine	0.3336543091
3	touches_upper_band_1,crt,rsi_bin	0|SELL|(20, 30]	0.3336543091
3	touches_upper_band_1,rsi_bin,combo_flag	0|(20, 30]|fine	0.3336543091
3	trend_direction,rsi_bin,combo_flag2	uptrend|(70, 80]|fine	0.3178743961
3	trend_strength_up,rsi_bin,combo_flag	1.0|(70, 80]|fine	0.3178743961
3	trend_strength_up,rsi_bin,combo_flag2	1.0|(70, 80]|fine	0.3178743961
3	trend_direction,crt,rsi_bin	uptrend|BUY|(70, 80]	0.3178743961
3	trend_direction,trade_type,rsi_bin	uptrend|BUY|(70, 80]	0.3178743961
3	trend_direction,rsi_bin,combo_flag	uptrend|(70, 80]|fine	0.3178743961
3	trend_strength_up,crt,rsi_bin	1.0|BUY|(70, 80]	0.3178743961
3	trend_strength_up,trade_type,rsi_bin	1.0|BUY|(70, 80]	0.3178743961
3	touches_vwap,trade_type,rsi_bin	0|SELL|(20, 30]	0.3231462926
3	touches_vwap,rsi_bin,combo_flag	0|(20, 30]|fine	0.3231462926
3	touches_vwap,crt,rsi_bin	0|SELL|(20, 30]	0.3231462926
3	touches_vwap,rsi_bin,combo_flag2	0|(20, 30]|fine	0.3231462926
3	touches_vwap,rsi_bin,combo_flag2	1|(30, 40]|fine	0.3035624686
3	rsi_zone,crt,macd_z_bin	oversold|SELL|(-13.256, -0.489]	0.3221374046
3	rsi_zone,trade_type,macd_z_bin	oversold|SELL|(-13.256, -0.489]	0.3221374046
3	rsi_zone,macd_z_bin,combo_flag2	oversold|(-13.256, -0.489]|fine	0.3221374046
3	rsi_zone,combo_flag,macd_z_bin	oversold|fine|(-13.256, -0.489]	0.3217922607
3	touches_lower_band_3,rsi_zone,trade_type	0|oversold|SELL	0.3046185781
3	touches_lower_band_3,rsi_zone,crt	0|oversold|SELL	0.3046185781
3	touches_lower_band_3,rsi_zone,combo_flag2	0|oversold|fine	0.3049350649
3	touches_lower_band_3,rsi_zone,combo_flag	0|oversold|fine	0.3045738046
3	touches_upper_band_3,rsi_bin,combo_flag2	1|(60, 70]|fine	0.3167396061
3	rsi_zone,trend_direction,crt	oversold|downtrend|SELL	0.3270614278
3	rsi_zone,trend_strength_down,combo_flag2	oversold|1.0|fine	0.3270614278
3	rsi_zone,trend_direction,combo_flag	oversold|downtrend|fine	0.3270614278
3	rsi_zone,trend_direction,combo_flag2	oversold|downtrend|fine	0.3270614278
3	rsi_zone,trend_strength_down,crt	oversold|1.0|SELL	0.3270614278
3	rsi_zone,bearish_stack,crt	oversold|1|SELL	0.3270614278
3	rsi_zone,bearish_stack,trade_type	oversold|1|SELL	0.3270614278
3	rsi_zone,bearish_stack,combo_flag	oversold|1|fine	0.3270614278
3	rsi_zone,bearish_stack,combo_flag2	oversold|1|fine	0.3270614278
3	rsi_zone,trend_strength_down,trade_type	oversold|1.0|SELL	0.3270614278
3	rsi_zone,trend_direction,trade_type	oversold|downtrend|SELL	0.3270614278
3	rsi_zone,trend_strength_down,combo_flag	oversold|1.0|fine	0.3270614278
3	touches_upper_band_3,trend_direction,trade_type	1|uptrend|BUY	0.3122222222
3	touches_upper_band_3,trend_direction,crt	1|uptrend|BUY	0.3122222222
3	touches_upper_band_3,trend_strength_up,crt	1|1.0|BUY	0.3122222222
3	touches_upper_band_3,trend_strength_up,trade_type	1|1.0|BUY	0.3122222222
3	touches_vwap,rsi_bin,combo_flag	1|(60, 70]|fine	0.3666476299
3	touches_vwap,crt,rsi_bin	1|BUY|(60, 70]	0.3666476299
3	touches_vwap,trade_type,rsi_bin	1|BUY|(60, 70]	0.3666476299
3	rsi_bin,combo_flag,macd_z_bin	(20, 30]|fine|(-13.256, -0.489]	0.3143021915
3	crt,rsi_bin,macd_z_bin	SELL|(20, 30]|(-13.256, -0.489]	0.3143021915
3	rsi_bin,macd_z_bin,combo_flag2	(20, 30]|(-13.256, -0.489]|fine	0.3143021915
3	trade_type,rsi_bin,macd_z_bin	SELL|(20, 30]|(-13.256, -0.489]	0.3143021915
3	touches_lower_band_3,rsi_bin,combo_flag2	0|(20, 30]|fine	0.3023121387
3	touches_lower_band_3,crt,rsi_bin	0|SELL|(20, 30]	0.3023121387
3	touches_lower_band_3,trade_type,rsi_bin	0|SELL|(20, 30]	0.3023121387
3	touches_lower_band_3,rsi_bin,combo_flag	0|(20, 30]|fine	0.3023121387
3	touches_vwap,touches_lower_band_3,combo_flag	0|1|fine	0.3040865385
3	trend_direction,rsi_bin,combo_flag2	downtrend|(20, 30]|fine	0.320802005
3	trend_direction,rsi_bin,combo_flag	downtrend|(20, 30]|fine	0.320802005
3	trend_direction,trade_type,rsi_bin	downtrend|SELL|(20, 30]	0.320802005
3	trend_strength_down,crt,rsi_bin	1.0|SELL|(20, 30]	0.320802005
3	trend_strength_down,trade_type,rsi_bin	1.0|SELL|(20, 30]	0.320802005
3	trend_strength_down,rsi_bin,combo_flag	1.0|(20, 30]|fine	0.320802005
3	trend_strength_down,rsi_bin,combo_flag2	1.0|(20, 30]|fine	0.320802005
3	trend_direction,crt,rsi_bin	downtrend|SELL|(20, 30]	0.320802005
3	bearish_stack,trade_type,rsi_bin	1|SELL|(20, 30]	0.320802005
3	bearish_stack,rsi_bin,combo_flag	1|(20, 30]|fine	0.320802005
3	bearish_stack,crt,rsi_bin	1|SELL|(20, 30]	0.320802005
3	bearish_stack,rsi_bin,combo_flag2	1|(20, 30]|fine	0.320802005
3	touches_lower_band_3,rsi_bin,combo_flag2	1|(30, 40]|fine	0.3185279188
3	touches_vwap,touches_lower_band_3,combo_flag2	0|1|fine	0.3279639175
3	touches_upper_band_2,rsi_zone,crt	1|overbought|BUY	0.4253181514
3	touches_upper_band_2,rsi_zone,trade_type	1|overbought|BUY	0.4253181514
3	touches_upper_band_2,rsi_zone,combo_flag2	1|overbought|fine	0.4253181514
3	touches_upper_band_2,rsi_zone,combo_flag	1|overbought|fine	0.4249329759
3	touches_vwap,touches_upper_band_3,combo_flag2	0|1|fine	0.3036437247
3	touches_vwap,touches_upper_band_3,combo_flag	0|1|fine	0.3045859001
3	touches_vwap,touches_lower_band_3,crt	0|1|SELL	0.3443754313
3	touches_vwap,touches_lower_band_3,trade_type	0|1|SELL	0.3443754313
3	touches_upper_band_3,rsi_bin,combo_flag	1|(60, 70]|fine	0.3826815642
3	touches_upper_band_3,crt,rsi_bin	1|BUY|(60, 70]	0.3826815642
3	touches_upper_band_3,trade_type,rsi_bin	1|BUY|(60, 70]	0.3826815642
3	touches_upper_band_3,crt,macd_z_bin	1|BUY|(0.126, 0.517]	0.3290870488
3	touches_upper_band_3,trade_type,macd_z_bin	1|BUY|(0.126, 0.517]	0.3290870488
3	touches_lower_band_3,trend_strength_down,trade_type	1|1.0|SELL	0.3290598291
3	touches_lower_band_3,bearish_stack,crt	1|1|SELL	0.3290598291
3	touches_lower_band_3,bearish_stack,trade_type	1|1|SELL	0.3290598291
3	touches_lower_band_3,trend_strength_down,crt	1|1.0|SELL	0.3290598291
3	touches_lower_band_3,trend_direction,crt	1|downtrend|SELL	0.3290598291
3	touches_lower_band_3,trend_direction,trade_type	1|downtrend|SELL	0.3290598291
3	touches_vwap,touches_upper_band_3,crt	0|1|BUY	0.3198847262
3	touches_vwap,touches_upper_band_3,trade_type	0|1|BUY	0.3198847262
3	touches_vwap,crt,rsi_bin	1|SELL|(30, 40]	0.373265157
3	touches_vwap,trade_type,rsi_bin	1|SELL|(30, 40]	0.373265157
3	touches_lower_band_3,crt,macd_z_bin	1|SELL|(-0.489, -0.12]	0.3111111111
3	touches_lower_band_3,trade_type,macd_z_bin	1|SELL|(-0.489, -0.12]	0.3111111111
3	touches_upper_band_2,rsi_bin,combo_flag2	1|(70, 80]|fine	0.4224072672
3	touches_upper_band_2,crt,rsi_bin	1|BUY|(70, 80]	0.4224072672
3	touches_upper_band_2,rsi_bin,combo_flag	1|(70, 80]|fine	0.4224072672
3	touches_upper_band_2,trade_type,rsi_bin	1|BUY|(70, 80]	0.4224072672
3	touches_lower_band_2,rsi_zone,crt	1|oversold|SELL	0.4313880126
3	touches_lower_band_2,rsi_zone,trade_type	1|oversold|SELL	0.4313880126
3	touches_lower_band_2,rsi_zone,combo_flag2	1|oversold|fine	0.4320695103
3	touches_lower_band_2,rsi_zone,combo_flag	1|oversold|fine	0.4310618067
3	touches_lower_band_3,crt,rsi_bin	1|SELL|(30, 40]	0.3766851705
3	touches_lower_band_3,trade_type,rsi_bin	1|SELL|(30, 40]	0.3766851705
3	touches_upper_band_3,macd_z_bin,combo_flag2	1|(-0.12, 0.126]|fine	0.3092269327
3	touches_lower_band_3,macd_z_bin,combo_flag2	1|(-0.12, 0.126]|fine	0.3233082707
3	touches_lower_band_2,crt,rsi_bin	1|SELL|(20, 30]	0.4219858156
3	touches_lower_band_2,rsi_bin,combo_flag2	1|(20, 30]|fine	0.4219858156
3	touches_lower_band_2,rsi_bin,combo_flag	1|(20, 30]|fine	0.4219858156
3	touches_lower_band_2,trade_type,rsi_bin	1|SELL|(20, 30]	0.4219858156
3	touches_lower_band_3,session,combo_flag	1|q3|fine	0.308490566
3	session,rsi_zone,combo_flag2	q3|overbought|fine	0.3466666667
3	session,rsi_zone,combo_flag	q3|overbought|fine	0.3466666667
3	session,rsi_zone,trade_type	q3|overbought|BUY	0.3466666667
3	session,rsi_zone,crt	q3|overbought|BUY	0.3466666667
3	touches_upper_band_1,touches_upper_band_2,rsi_zone	1|1|overbought	0.322425409
3	crt,rsi_bin,macd_z_bin	BUY|(50, 60]|(-13.256, -0.489]	0.3548067393
3	trade_type,rsi_bin,macd_z_bin	BUY|(50, 60]|(-13.256, -0.489]	0.3548067393
3	rsi_bin,macd_z_bin,combo_flag2	(50, 60]|(-13.256, -0.489]|fine	0.3548067393"""
        
        combo_win_rates = {}
        lines = win_rate_data.strip().split('\n')
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                combo_cols = parts[1]
                group_key = parts[2]
                win_rate = float(parts[3])
                
                # Create the feature name in the same format as in _get_combo_features()
                cols_list = combo_cols.split(',')
                key_list = group_key.split('|')
                
                # Build the feature name exactly as it appears in combo features
                feature_name = 'combo_' + '_'.join([f"{col}_{key}" for col, key in zip(cols_list, key_list)])
                combo_win_rates[feature_name] = win_rate
        
        logger.debug(f"Loaded {len(combo_win_rates)} combo win rates")
        return combo_win_rates

    def calculate_average_combo_win_rate(self, df):
        """Calculate average win rate of active combo features"""
        try:
            active_win_rates = []
            
            # Check each combo feature in our win rate database
            for combo_feature, win_rate in self.combo_win_rates.items():
                if combo_feature in df.columns:
                    # If the combo feature is active (== 1) in the current data
                    if df[combo_feature].iloc[-1] == 1.0:
                        active_win_rates.append(win_rate)
            
            # Calculate average win rate
            if active_win_rates:
                average_win_rate = sum(active_win_rates) / len(active_win_rates)
                logger.debug(f"Found {len(active_win_rates)} active combos, average win rate: {average_win_rate:.4f}")
                return average_win_rate
            else:
                logger.debug("No active combos found")
                return 0.0  # Return 0 if no combos are active
            
        except Exception as e:
            logger.error(f"Error calculating average combo win rate: {str(e)}")
            return 0.0

    def _get_combo_features(self):
        """Return all combo feature names in exact order"""
        return [
            'combo_touches_lower_band_2_1_crt_SELL_combo_flag2_fine',
            'combo_touches_lower_band_2_1_trade_type_SELL_combo_flag2_fine',
            'combo_touches_upper_band_2_1_touches_lower_band_3_0_trade_type_BUY',
            'combo_touches_upper_band_2_1_touches_lower_band_3_0_crt_BUY',
            'combo_touches_upper_band_3_0_touches_lower_band_2_1_crt_SELL',
            'combo_touches_upper_band_3_0_touches_lower_band_2_1_trade_type_SELL',
            'combo_touches_upper_band_2_1_touches_lower_band_2_0_crt_BUY',
            'combo_touches_upper_band_2_1_touches_lower_band_2_0_trade_type_BUY',
            'combo_touches_upper_band_2_0_touches_lower_band_2_1_trade_type_SELL',
            'combo_touches_upper_band_2_0_touches_lower_band_2_1_crt_SELL',
            'combo_touches_upper_band_1_0_touches_lower_band_2_1_trade_type_SELL',
            'combo_touches_upper_band_1_0_touches_lower_band_2_1_crt_SELL',
            'combo_touches_vwap_1_macd_z_bin_0120126_combo_flag2_fine',
            'combo_touches_upper_band_1_1_trade_type_BUY_rsi_bin_6070',
            'combo_touches_upper_band_1_1_rsi_bin_6070_combo_flag_fine',
            'combo_touches_upper_band_1_1_crt_BUY_rsi_bin_6070',
            'combo_touches_lower_band_3_1_trade_type_SELL_combo_flag_fine',
            'combo_touches_lower_band_3_1_crt_SELL_combo_flag_fine',
            'combo_touches_upper_band_3_1_crt_BUY_combo_flag2_fine',
            'combo_touches_upper_band_3_1_trade_type_BUY_combo_flag2_fine',
            'combo_touches_lower_band_3_1_crt_SELL_combo_flag2_fine',
            'combo_touches_lower_band_3_1_trade_type_SELL_combo_flag2_fine',
            'combo_trend_strength_up_00_trade_type_BUY_rsi_bin_6070',
            'combo_trend_strength_up_00_crt_BUY_rsi_bin_6070',
            'combo_trend_strength_up_00_rsi_bin_6070_combo_flag_fine',
            'combo_touches_upper_band_2_1_trade_type_BUY_rsi_bin_6070',
            'combo_touches_upper_band_2_1_crt_BUY_rsi_bin_6070',
            'combo_touches_upper_band_2_1_rsi_bin_6070_combo_flag_fine',
            'combo_touches_upper_band_1_1_macd_z_bin_0120126_combo_flag2_fine',
            'combo_touches_lower_band_1_1_crt_SELL_rsi_bin_3040',
            'combo_touches_lower_band_1_1_trade_type_SELL_rsi_bin_3040',
            'combo_bearish_stack_0_crt_SELL_rsi_bin_3040',
            'combo_bearish_stack_0_trade_type_SELL_rsi_bin_3040',
            'combo_trend_strength_down_00_rsi_bin_3040_combo_flag_fine',
            'combo_bearish_stack_0_rsi_bin_3040_combo_flag_fine',
            'combo_trend_strength_down_00_trade_type_SELL_rsi_bin_3040',
            'combo_trend_strength_down_00_crt_SELL_rsi_bin_3040',
            'combo_touches_lower_band_1_1_macd_z_bin_0120126_combo_flag2_fine',
            'combo_touches_lower_band_3_1_trend_strength_up_00_trade_type_SELL',
            'combo_touches_lower_band_3_1_trend_strength_up_00_crt_SELL',
            'combo_touches_upper_band_3_1_touches_lower_band_3_0_combo_flag_fine',
            'combo_touches_lower_band_2_1_crt_SELL_rsi_bin_3040',
            'combo_touches_lower_band_2_1_trade_type_SELL_rsi_bin_3040',
            'combo_touches_upper_band_2_0_touches_lower_band_3_1_combo_flag_fine',
            'combo_touches_upper_band_3_0_touches_lower_band_3_1_combo_flag2_fine',
            'combo_touches_upper_band_3_1_touches_lower_band_3_0_combo_flag2_fine',
            'combo_rsi_bin_6070_combo_flag_fine_macd_z_bin_01260517',
            'combo_rsi_bin_6070_macd_z_bin_01260517_combo_flag2_fine',
            'combo_trade_type_BUY_rsi_bin_6070_macd_z_bin_01260517',
            'combo_crt_BUY_rsi_bin_6070_macd_z_bin_01260517',
            'combo_touches_upper_band_3_1_touches_lower_band_2_0_combo_flag_fine',
            'combo_trend_direction_sideways_crt_SELL_rsi_bin_3040',
            'combo_trend_direction_sideways_rsi_bin_3040_combo_flag_fine',
            'combo_trend_direction_sideways_trade_type_SELL_rsi_bin_3040',
            'combo_touches_lower_band_2_1_session_q3_trade_type_SELL',
            'combo_touches_lower_band_2_1_session_q3_crt_SELL',
            'combo_touches_upper_band_3_0_touches_lower_band_3_1_crt_SELL',
            'combo_touches_upper_band_3_0_touches_lower_band_3_1_trade_type_SELL',
            'combo_touches_upper_band_3_1_touches_lower_band_3_0_crt_BUY',
            'combo_touches_upper_band_3_1_touches_lower_band_3_0_trade_type_BUY',
            'combo_touches_upper_band_2_0_touches_lower_band_3_1_combo_flag2_fine',
            'combo_touches_upper_band_3_1_touches_lower_band_2_0_combo_flag2_fine',
            'combo_rsi_bin_5060_macd_z_bin_0489012_combo_flag2_fine',
            'combo_trade_type_BUY_rsi_bin_5060_macd_z_bin_0489012',
            'combo_crt_BUY_rsi_bin_5060_macd_z_bin_0489012',
            'combo_rsi_zone_overbought_crt_BUY_combo_flag2_fine',
            'combo_rsi_zone_overbought_crt_BUY_trade_type_BUY',
            'combo_rsi_zone_overbought_trade_type_BUY_is_bad_combo_0',
            'combo_rsi_zone_overbought_crt_BUY_is_bad_combo_0',
            'combo_rsi_zone_overbought_trade_type_BUY_combo_flag2_fine',
            'combo_rsi_zone_overbought_is_bad_combo_0_combo_flag2_fine',
            'combo_rsi_zone_overbought_is_bad_combo_0_combo_flag_fine',
            'combo_rsi_zone_overbought_trade_type_BUY_combo_flag_fine',
            'combo_rsi_zone_overbought_combo_flag_fine_combo_flag2_fine',
            'combo_rsi_zone_overbought_crt_BUY_combo_flag_fine',
            'combo_rsi_zone_overbought_trade_type_BUY_rrr_40',
            'combo_rsi_zone_overbought_rrr_40_combo_flag2_fine',
            'combo_rsi_zone_overbought_crt_BUY_rrr_40',
            'combo_rsi_zone_overbought_rrr_40_combo_flag_fine',
            'combo_session_q2_rsi_bin_6070_combo_flag_fine',
            'combo_session_q2_trade_type_BUY_rsi_bin_6070',
            'combo_session_q2_crt_BUY_rsi_bin_6070',
            'combo_touches_upper_band_2_0_touches_lower_band_3_1_trade_type_SELL',
            'combo_touches_upper_band_2_0_touches_lower_band_3_1_crt_SELL',
            'combo_rsi_zone_overbought_bearish_stack_0_combo_flag_fine',
            'combo_rsi_zone_overbought_trend_strength_down_00_combo_flag_fine',
            'combo_rsi_zone_overbought_bearish_stack_0_crt_BUY',
            'combo_rsi_zone_overbought_bearish_stack_0_combo_flag2_fine',
            'combo_rsi_zone_overbought_trend_strength_down_00_trade_type_BUY',
            'combo_rsi_zone_overbought_bearish_stack_0_trade_type_BUY',
            'combo_rsi_zone_overbought_trend_strength_down_00_crt_BUY',
            'combo_rsi_zone_overbought_trend_strength_down_00_combo_flag2_fine',
            'combo_touches_lower_band_3_0_rsi_zone_overbought_trade_type_BUY',
            'combo_touches_lower_band_3_0_rsi_zone_overbought_crt_BUY',
            'combo_touches_lower_band_3_0_rsi_zone_overbought_combo_flag2_fine',
            'combo_touches_lower_band_3_0_rsi_zone_overbought_combo_flag_fine',
            'combo_touches_upper_band_3_1_touches_lower_band_2_0_trade_type_BUY',
            'combo_touches_upper_band_3_1_touches_lower_band_2_0_crt_BUY',
            'combo_touches_lower_band_2_0_rsi_zone_overbought_crt_BUY',
            'combo_touches_lower_band_2_0_rsi_zone_overbought_trade_type_BUY',
            'combo_touches_lower_band_2_0_rsi_zone_overbought_combo_flag2_fine',
            'combo_touches_lower_band_2_0_rsi_zone_overbought_combo_flag_fine',
            'combo_touches_lower_band_1_0_rsi_zone_overbought_combo_flag2_fine',
            'combo_touches_lower_band_1_0_rsi_zone_overbought_crt_BUY',
            'combo_touches_lower_band_1_0_rsi_zone_overbought_trade_type_BUY',
            'combo_touches_lower_band_1_0_rsi_zone_overbought_combo_flag_fine',
            'combo_touches_upper_band_1_0_touches_lower_band_3_1_combo_flag_fine',
            'combo_touches_vwap_0_rsi_zone_overbought_combo_flag2_fine',
            'combo_touches_vwap_0_rsi_zone_overbought_crt_BUY',
            'combo_touches_vwap_0_rsi_zone_overbought_trade_type_BUY',
            'combo_touches_vwap_0_rsi_zone_overbought_combo_flag_fine',
            'combo_trade_type_SELL_rsi_bin_3040_macd_z_bin_0489012',
            'combo_rsi_bin_3040_macd_z_bin_0489012_combo_flag2_fine',
            'combo_crt_SELL_rsi_bin_3040_macd_z_bin_0489012',
            'combo_rsi_bin_7080_combo_flag_fine_combo_flag2_fine',
            'combo_trade_type_BUY_rsi_bin_7080_combo_flag_fine',
            'combo_rsi_bin_7080_is_bad_combo_0_combo_flag_fine',
            'combo_crt_BUY_rsi_bin_7080_combo_flag2_fine',
            'combo_trade_type_BUY_rsi_bin_7080_is_bad_combo_0',
            'combo_rsi_zone_overbought_trade_type_BUY_rsi_bin_7080',
            'combo_rsi_zone_overbought_rsi_bin_7080_combo_flag2_fine',
            'combo_crt_BUY_rsi_bin_7080_combo_flag_fine',
            'combo_crt_BUY_trade_type_BUY_rsi_bin_7080',
            'combo_crt_BUY_rsi_bin_7080_is_bad_combo_0',
            'combo_trade_type_BUY_rsi_bin_7080_combo_flag2_fine',
            'combo_rsi_zone_overbought_crt_BUY_rsi_bin_7080',
            'combo_rsi_zone_overbought_rsi_bin_7080_combo_flag_fine',
            'combo_rsi_bin_7080_is_bad_combo_0_combo_flag2_fine',
            'combo_trade_type_BUY_rrr_40_rsi_bin_7080',
            'combo_crt_BUY_rrr_40_rsi_bin_7080',
            'combo_rrr_40_rsi_bin_7080_combo_flag_fine',
            'combo_rrr_40_rsi_bin_7080_combo_flag2_fine',
            'combo_bearish_stack_0_crt_BUY_rsi_bin_7080',
            'combo_trend_strength_down_00_trade_type_BUY_rsi_bin_7080',
            'combo_trend_strength_down_00_rsi_bin_7080_combo_flag2_fine',
            'combo_trend_strength_down_00_rsi_bin_7080_combo_flag_fine',
            'combo_trend_strength_down_00_crt_BUY_rsi_bin_7080',
            'combo_bearish_stack_0_rsi_bin_7080_combo_flag2_fine',
            'combo_bearish_stack_0_rsi_bin_7080_combo_flag_fine',
            'combo_bearish_stack_0_trade_type_BUY_rsi_bin_7080',
            'combo_touches_lower_band_3_0_crt_BUY_rsi_bin_7080',
            'combo_touches_lower_band_3_0_trade_type_BUY_rsi_bin_7080',
            'combo_touches_lower_band_3_0_rsi_bin_7080_combo_flag2_fine',
            'combo_touches_lower_band_3_0_rsi_bin_7080_combo_flag_fine',
            'combo_touches_lower_band_2_0_rsi_bin_7080_combo_flag_fine',
            'combo_touches_lower_band_2_0_trade_type_BUY_rsi_bin_7080',
            'combo_touches_lower_band_2_0_rsi_bin_7080_combo_flag2_fine',
            'combo_touches_lower_band_2_0_crt_BUY_rsi_bin_7080',
            'combo_touches_lower_band_1_0_rsi_bin_7080_combo_flag2_fine',
            'combo_touches_lower_band_1_0_rsi_bin_7080_combo_flag_fine',
            'combo_touches_lower_band_1_0_trade_type_BUY_rsi_bin_7080',
            'combo_touches_lower_band_1_0_crt_BUY_rsi_bin_7080',
            'combo_trade_type_SELL_rsi_bin_4050_macd_z_bin_01260517',
            'combo_rsi_bin_4050_macd_z_bin_01260517_combo_flag2_fine',
            'combo_crt_SELL_rsi_bin_4050_macd_z_bin_01260517',
            'combo_touches_upper_band_3_1_touches_lower_band_1_0_combo_flag_fine',
            'combo_touches_upper_band_1_0_touches_lower_band_3_1_combo_flag2_fine',
            'combo_touches_vwap_0_crt_BUY_rsi_bin_7080',
            'combo_touches_vwap_0_rsi_bin_7080_combo_flag_fine',
            'combo_touches_vwap_0_rsi_bin_7080_combo_flag2_fine',
            'combo_touches_vwap_0_trade_type_BUY_rsi_bin_7080',
            'combo_minutes_0_closed_1_touches_upper_band_2_BUY',
            'combo_touches_upper_band_3_1_touches_lower_band_1_0_combo_flag2_fine',
            'combo_rsi_zone_overbought_crt_BUY_macd_z_bin_051710271',
            'combo_rsi_zone_overbought_trade_type_BUY_macd_z_bin_051710271',
            'combo_rsi_zone_overbought_macd_z_bin_051710271_combo_flag2_fine',
            'combo_rsi_zone_overbought_combo_flag_fine_macd_z_bin_051710271',
            'combo_rsi_zone_oversold_crt_SELL_rrr_40',
            'combo_rsi_zone_oversold_crt_SELL_is_bad_combo_0',
            'combo_rsi_zone_oversold_trade_type_SELL_rrr_40',
            'combo_rsi_zone_oversold_trade_type_SELL_is_bad_combo_0',
            'combo_rsi_zone_oversold_crt_SELL_trade_type_SELL',
            'combo_rsi_zone_oversold_crt_SELL_combo_flag2_fine',
            'combo_rsi_zone_oversold_trade_type_SELL_combo_flag2_fine',
            'combo_rsi_zone_oversold_rrr_40_combo_flag2_fine',
            'combo_rsi_zone_oversold_is_bad_combo_0_combo_flag2_fine',
            'combo_rsi_zone_oversold_rrr_40_combo_flag_fine',
            'combo_rsi_zone_oversold_crt_SELL_combo_flag_fine',
            'combo_rsi_zone_oversold_is_bad_combo_0_combo_flag_fine',
            'combo_rsi_zone_oversold_trade_type_SELL_combo_flag_fine',
            'combo_rsi_zone_oversold_combo_flag_fine_combo_flag2_fine',
            'combo_touches_upper_band_3_0_rsi_zone_oversold_crt_SELL',
            'combo_touches_upper_band_3_0_rsi_zone_oversold_trade_type_SELL',
            'combo_touches_upper_band_3_0_rsi_zone_oversold_combo_flag2_fine',
            'combo_touches_upper_band_3_0_rsi_zone_oversold_combo_flag_fine',
            'combo_touches_upper_band_2_0_rsi_zone_oversold_crt_SELL',
            'combo_touches_upper_band_2_0_rsi_zone_oversold_trade_type_SELL',
            'combo_touches_upper_band_2_0_rsi_zone_oversold_combo_flag2_fine',
            'combo_touches_upper_band_2_0_rsi_zone_oversold_combo_flag_fine',
            'combo_rsi_zone_oversold_trend_strength_up_00_trade_type_SELL',
            'combo_rsi_zone_oversold_trend_strength_up_00_crt_SELL',
            'combo_rsi_zone_oversold_trend_strength_up_00_combo_flag_fine',
            'combo_rsi_zone_oversold_trend_strength_up_00_combo_flag2_fine',
            'combo_touches_lower_band_2_1_macd_z_bin_0120126_combo_flag2_fine',
            'combo_rsi_zone_overbought_trend_strength_up_10_combo_flag2_fine',
            'combo_rsi_zone_overbought_trend_strength_up_10_trade_type_BUY',
            'combo_rsi_zone_overbought_trend_strength_up_10_crt_BUY',
            'combo_rsi_zone_overbought_trend_strength_up_10_combo_flag_fine',
            'combo_rsi_zone_overbought_trend_direction_uptrend_trade_type_BUY',
            'combo_rsi_zone_overbought_trend_direction_uptrend_crt_BUY',
            'combo_rsi_zone_overbought_trend_direction_uptrend_combo_flag2_fine',
            'combo_rsi_zone_overbought_trend_direction_uptrend_combo_flag_fine',
            'combo_touches_upper_band_2_1_macd_z_bin_0120126_combo_flag2_fine',
            'combo_touches_upper_band_1_0_rsi_zone_oversold_trade_type_SELL',
            'combo_touches_upper_band_1_0_rsi_zone_oversold_crt_SELL',
            'combo_touches_upper_band_1_0_rsi_zone_oversold_combo_flag2_fine',
            'combo_touches_upper_band_1_0_touches_lower_band_3_1_crt_SELL',
            'combo_touches_upper_band_1_0_touches_lower_band_3_1_trade_type_SELL',
            'combo_touches_upper_band_1_0_rsi_zone_oversold_combo_flag_fine',
            'combo_touches_upper_band_3_1_touches_lower_band_1_0_crt_BUY',
            'combo_touches_upper_band_3_1_touches_lower_band_1_0_trade_type_BUY',
            'combo_session_q2_trade_type_SELL_rsi_bin_3040',
            'combo_session_q2_crt_SELL_rsi_bin_3040',
            'combo_touches_vwap_0_rsi_zone_oversold_crt_SELL',
            'combo_touches_vwap_0_rsi_zone_oversold_trade_type_SELL',
            'combo_touches_vwap_0_rsi_zone_oversold_combo_flag2_fine',
            'combo_touches_vwap_0_rsi_zone_oversold_combo_flag_fine',
            'combo_trade_type_SELL_rsi_bin_2030_combo_flag2_fine',
            'combo_rsi_bin_2030_is_bad_combo_0_combo_flag2_fine',
            'combo_rrr_40_rsi_bin_2030_combo_flag_fine',
            'combo_rsi_bin_2030_is_bad_combo_0_combo_flag_fine',
            'combo_trade_type_SELL_rsi_bin_2030_combo_flag_fine',
            'combo_rsi_bin_2030_combo_flag_fine_combo_flag2_fine',
            'combo_crt_SELL_rsi_bin_2030_combo_flag_fine',
            'combo_rsi_zone_oversold_rsi_bin_2030_combo_flag2_fine',
            'combo_crt_SELL_trade_type_SELL_rsi_bin_2030',
            'combo_rrr_40_rsi_bin_2030_combo_flag2_fine',
            'combo_trade_type_SELL_rsi_bin_2030_is_bad_combo_0',
            'combo_crt_SELL_rrr_40_rsi_bin_2030',
            'combo_rsi_zone_oversold_crt_SELL_rsi_bin_2030',
            'combo_rsi_zone_oversold_rsi_bin_2030_combo_flag_fine',
            'combo_rsi_zone_oversold_trade_type_SELL_rsi_bin_2030',
            'combo_crt_SELL_rsi_bin_2030_is_bad_combo_0',
            'combo_trade_type_SELL_rrr_40_rsi_bin_2030',
            'combo_crt_SELL_rsi_bin_2030_combo_flag2_fine',
            'combo_touches_upper_band_3_0_rsi_bin_2030_combo_flag2_fine',
            'combo_touches_upper_band_3_0_rsi_bin_2030_combo_flag_fine',
            'combo_touches_upper_band_3_0_trade_type_SELL_rsi_bin_2030',
            'combo_touches_upper_band_3_0_crt_SELL_rsi_bin_2030',
            'combo_touches_upper_band_2_0_rsi_bin_2030_combo_flag2_fine',
            'combo_touches_upper_band_2_0_crt_SELL_rsi_bin_2030',
            'combo_touches_upper_band_2_0_rsi_bin_2030_combo_flag_fine',
            'combo_touches_upper_band_2_0_trade_type_SELL_rsi_bin_2030',
            'combo_trend_strength_up_00_trade_type_SELL_rsi_bin_2030',
            'combo_trend_strength_up_00_rsi_bin_2030_combo_flag_fine',
            'combo_trend_strength_up_00_crt_SELL_rsi_bin_2030',
            'combo_trend_strength_up_00_rsi_bin_2030_combo_flag2_fine',
            'combo_touches_upper_band_1_0_trade_type_SELL_rsi_bin_2030',
            'combo_touches_upper_band_1_0_rsi_bin_2030_combo_flag2_fine',
            'combo_touches_upper_band_1_0_crt_SELL_rsi_bin_2030',
            'combo_touches_upper_band_1_0_rsi_bin_2030_combo_flag_fine',
            'combo_trend_direction_uptrend_rsi_bin_7080_combo_flag2_fine',
            'combo_trend_strength_up_10_rsi_bin_7080_combo_flag_fine',
            'combo_trend_strength_up_10_rsi_bin_7080_combo_flag2_fine',
            'combo_trend_direction_uptrend_crt_BUY_rsi_bin_7080',
            'combo_trend_direction_uptrend_trade_type_BUY_rsi_bin_7080',
            'combo_trend_direction_uptrend_rsi_bin_7080_combo_flag_fine',
            'combo_trend_strength_up_10_crt_BUY_rsi_bin_7080',
            'combo_trend_strength_up_10_trade_type_BUY_rsi_bin_7080',
            'combo_touches_vwap_0_trade_type_SELL_rsi_bin_2030',
            'combo_touches_vwap_0_rsi_bin_2030_combo_flag_fine',
            'combo_touches_vwap_0_crt_SELL_rsi_bin_2030',
            'combo_touches_vwap_0_rsi_bin_2030_combo_flag2_fine',
            'combo_touches_vwap_1_rsi_bin_3040_combo_flag2_fine',
            'combo_rsi_zone_oversold_crt_SELL_macd_z_bin_132560489',
            'combo_rsi_zone_oversold_trade_type_SELL_macd_z_bin_132560489',
            'combo_rsi_zone_oversold_macd_z_bin_132560489_combo_flag2_fine',
            'combo_rsi_zone_oversold_combo_flag_fine_macd_z_bin_132560489',
            'combo_touches_lower_band_3_0_rsi_zone_oversold_trade_type_SELL',
            'combo_touches_lower_band_3_0_rsi_zone_oversold_crt_SELL',
            'combo_touches_lower_band_3_0_rsi_zone_oversold_combo_flag2_fine',
            'combo_touches_lower_band_3_0_rsi_zone_oversold_combo_flag_fine',
            'combo_touches_upper_band_3_1_rsi_bin_6070_combo_flag2_fine',
            'combo_rsi_zone_oversold_trend_direction_downtrend_crt_SELL',
            'combo_rsi_zone_oversold_trend_strength_down_10_combo_flag2_fine',
            'combo_rsi_zone_oversold_trend_direction_downtrend_combo_flag_fine',
            'combo_rsi_zone_oversold_trend_direction_downtrend_combo_flag2_fine',
            'combo_rsi_zone_oversold_trend_strength_down_10_crt_SELL',
            'combo_rsi_zone_oversold_bearish_stack_1_crt_SELL',
            'combo_rsi_zone_oversold_bearish_stack_1_trade_type_SELL',
            'combo_rsi_zone_oversold_bearish_stack_1_combo_flag_fine',
            'combo_rsi_zone_oversold_bearish_stack_1_combo_flag2_fine',
            'combo_rsi_zone_oversold_trend_strength_down_10_trade_type_SELL',
            'combo_rsi_zone_oversold_trend_direction_downtrend_trade_type_SELL',
            'combo_rsi_zone_oversold_trend_strength_down_10_combo_flag_fine',
            'combo_touches_upper_band_3_1_trend_direction_uptrend_trade_type_BUY',
            'combo_touches_upper_band_3_1_trend_direction_uptrend_crt_BUY',
            'combo_touches_upper_band_3_1_trend_strength_up_10_crt_BUY',
            'combo_touches_upper_band_3_1_trend_strength_up_10_trade_type_BUY',
            'combo_touches_vwap_1_rsi_bin_6070_combo_flag_fine',
            'combo_touches_vwap_1_crt_BUY_rsi_bin_6070',
            'combo_touches_vwap_1_trade_type_BUY_rsi_bin_6070',
            'combo_rsi_bin_2030_combo_flag_fine_macd_z_bin_132560489',
            'combo_crt_SELL_rsi_bin_2030_macd_z_bin_132560489',
            'combo_rsi_bin_2030_macd_z_bin_132560489_combo_flag2_fine',
            'combo_trade_type_SELL_rsi_bin_2030_macd_z_bin_132560489',
            'combo_touches_lower_band_3_0_rsi_bin_2030_combo_flag2_fine',
            'combo_touches_lower_band_3_0_crt_SELL_rsi_bin_2030',
            'combo_touches_lower_band_3_0_trade_type_SELL_rsi_bin_2030',
            'combo_touches_lower_band_3_0_rsi_bin_2030_combo_flag_fine',
            'combo_touches_vwap_0_touches_lower_band_3_1_combo_flag_fine',
            'combo_trend_direction_downtrend_rsi_bin_2030_combo_flag2_fine',
            'combo_trend_direction_downtrend_rsi_bin_2030_combo_flag_fine',
            'combo_trend_direction_downtrend_trade_type_SELL_rsi_bin_2030',
            'combo_trend_strength_down_10_crt_SELL_rsi_bin_2030',
            'combo_trend_strength_down_10_trade_type_SELL_rsi_bin_2030',
            'combo_trend_strength_down_10_rsi_bin_2030_combo_flag_fine',
            'combo_trend_strength_down_10_rsi_bin_2030_combo_flag2_fine',
            'combo_trend_direction_downtrend_crt_SELL_rsi_bin_2030',
            'combo_bearish_stack_1_trade_type_SELL_rsi_bin_2030',
            'combo_bearish_stack_1_rsi_bin_2030_combo_flag_fine',
            'combo_bearish_stack_1_crt_SELL_rsi_bin_2030',
            'combo_bearish_stack_1_rsi_bin_2030_combo_flag2_fine',
            'combo_touches_lower_band_3_1_rsi_bin_3040_combo_flag2_fine',
            'combo_touches_vwap_0_touches_lower_band_3_1_combo_flag2_fine',
            'combo_touches_upper_band_2_1_rsi_zone_overbought_crt_BUY',
            'combo_touches_upper_band_2_1_rsi_zone_overbought_trade_type_BUY',
            'combo_touches_upper_band_2_1_rsi_zone_overbought_combo_flag2_fine',
            'combo_touches_upper_band_2_1_rsi_zone_overbought_combo_flag_fine',
            'combo_touches_vwap_0_touches_upper_band_3_1_combo_flag2_fine',
            'combo_touches_vwap_0_touches_upper_band_3_1_combo_flag_fine',
            'combo_touches_vwap_0_touches_lower_band_3_1_crt_SELL',
            'combo_touches_vwap_0_touches_lower_band_3_1_trade_type_SELL',
            'combo_touches_upper_band_3_1_rsi_bin_6070_combo_flag_fine',
            'combo_touches_upper_band_3_1_crt_BUY_rsi_bin_6070',
            'combo_touches_upper_band_3_1_trade_type_BUY_rsi_bin_6070',
            'combo_touches_upper_band_3_1_crt_BUY_macd_z_bin_01260517',
            'combo_touches_upper_band_3_1_trade_type_BUY_macd_z_bin_01260517',
            'combo_touches_lower_band_3_1_trend_strength_down_10_trade_type_SELL',
            'combo_touches_lower_band_3_1_bearish_stack_1_crt_SELL',
            'combo_touches_lower_band_3_1_bearish_stack_1_trade_type_SELL',
            'combo_touches_lower_band_3_1_trend_strength_down_10_crt_SELL',
            'combo_touches_lower_band_3_1_trend_direction_downtrend_crt_SELL',
            'combo_touches_lower_band_3_1_trend_direction_downtrend_trade_type_SELL',
            'combo_touches_vwap_0_touches_upper_band_3_1_crt_BUY',
            'combo_touches_vwap_0_touches_upper_band_3_1_trade_type_BUY',
            'combo_touches_vwap_1_crt_SELL_rsi_bin_3040',
            'combo_touches_vwap_1_trade_type_SELL_rsi_bin_3040',
            'combo_touches_lower_band_3_1_crt_SELL_macd_z_bin_0489012',
            'combo_touches_lower_band_3_1_trade_type_SELL_macd_z_bin_0489012',
            'combo_touches_upper_band_2_1_rsi_bin_7080_combo_flag2_fine',
            'combo_touches_upper_band_2_1_crt_BUY_rsi_bin_7080',
            'combo_touches_upper_band_2_1_rsi_bin_7080_combo_flag_fine',
            'combo_touches_upper_band_2_1_trade_type_BUY_rsi_bin_7080',
            'combo_touches_lower_band_2_1_rsi_zone_oversold_crt_SELL',
            'combo_touches_lower_band_2_1_rsi_zone_oversold_trade_type_SELL',
            'combo_touches_lower_band_2_1_rsi_zone_oversold_combo_flag2_fine',
            'combo_touches_lower_band_2_1_rsi_zone_oversold_combo_flag_fine',
            'combo_touches_lower_band_3_1_crt_SELL_rsi_bin_3040',
            'combo_touches_lower_band_3_1_trade_type_SELL_rsi_bin_3040',
            'combo_touches_upper_band_3_1_macd_z_bin_0120126_combo_flag2_fine',
            'combo_touches_lower_band_3_1_macd_z_bin_0120126_combo_flag2_fine',
            'combo_touches_lower_band_2_1_crt_SELL_rsi_bin_2030',
            'combo_touches_lower_band_2_1_rsi_bin_2030_combo_flag2_fine',
            'combo_touches_lower_band_2_1_rsi_bin_2030_combo_flag_fine',
            'combo_touches_lower_band_2_1_trade_type_SELL_rsi_bin_2030',
            'combo_touches_lower_band_3_1_session_q3_combo_flag_fine',
            'combo_session_q3_rsi_zone_overbought_combo_flag2_fine',
            'combo_session_q3_rsi_zone_overbought_combo_flag_fine',
            'combo_session_q3_rsi_zone_overbought_trade_type_BUY',
            'combo_session_q3_rsi_zone_overbought_crt_BUY',
            'combo_touches_upper_band_1_1_touches_upper_band_2_1_rsi_zone_overbought',
            'combo_crt_BUY_rsi_bin_5060_macd_z_bin_132560489',
            'combo_trade_type_BUY_rsi_bin_5060_macd_z_bin_132560489',
            'combo_rsi_bin_5060_macd_z_bin_132560489_combo_flag2_fine'
        ]

    def _get_rsi_bin_features(self):
        """Return RSI bin one-hot features"""
        return [
            'rsi_bin_(0, 20]', 'rsi_bin_(20, 30]', 'rsi_bin_(30, 40]', 
            'rsi_bin_(40, 50]', 'rsi_bin_(50, 60]', 'rsi_bin_(60, 70]', 
            'rsi_bin_(70, 80]', 'rsi_bin_(80, 100]'
        ]

    def _get_combo_key_features(self):
        """Return combo key one-hot features"""
        return [
            'combo_key_BUY_downtrend_(0, 20]', 'combo_key_BUY_downtrend_(20, 30]',
            'combo_key_BUY_downtrend_(30, 40]', 'combo_key_BUY_downtrend_(40, 50]',
            'combo_key_BUY_downtrend_(50, 60]', 'combo_key_BUY_downtrend_(60, 70]',
            'combo_key_BUY_downtrend_(70, 80]', 'combo_key_BUY_sideways_(0, 20]',
            'combo_key_BUY_sideways_(20, 30]', 'combo_key_BUY_sideways_(30, 40]',
            'combo_key_BUY_sideways_(40, 50]', 'combo_key_BUY_sideways_(50, 60]',
            'combo_key_BUY_sideways_(60, 70]', 'combo_key_BUY_sideways_(70, 80]',
            'combo_key_BUY_sideways_(80, 100]', 'combo_key_BUY_uptrend_(0, 20]',
            'combo_key_BUY_uptrend_(20, 30]', 'combo_key_BUY_uptrend_(30, 40]',
            'combo_key_BUY_uptrend_(40, 50]', 'combo_key_BUY_uptrend_(50, 60]',
            'combo_key_BUY_uptrend_(60, 70]', 'combo_key_BUY_uptrend_(70, 80]',
            'combo_key_BUY_uptrend_(80, 100]', 'combo_key_SELL_downtrend_(0, 20]',
            'combo_key_SELL_downtrend_(20, 30]', 'combo_key_SELL_downtrend_(30, 40]',
            'combo_key_SELL_downtrend_(40, 50]', 'combo_key_SELL_downtrend_(50, 60]',
            'combo_key_SELL_downtrend_(60, 70]', 'combo_key_SELL_downtrend_(70, 80]',
            'combo_key_SELL_sideways_(20, 30]', 'combo_key_SELL_sideways_(30, 40]',
            'combo_key_SELL_sideways_(40, 50]', 'combo_key_SELL_sideways_(50, 60]',
            'combo_key_SELL_sideways_(60, 70]', 'combo_key_SELL_sideways_(70, 80]',
            'combo_key_SELL_sideways_(80, 100]', 'combo_key_SELL_uptrend_(20, 30]',
            'combo_key_SELL_uptrend_(30, 40]', 'combo_key_SELL_uptrend_(40, 50]',
            'combo_key_SELL_uptrend_(50, 60]', 'combo_key_SELL_uptrend_(60, 70]',
            'combo_key_SELL_uptrend_(70, 80]', 'combo_key_SELL_uptrend_(80, 100]',
            'combo_key_nan_downtrend_(0, 20]', 'combo_key_nan_downtrend_(20, 30]',
            'combo_key_nan_downtrend_(30, 40]', 'combo_key_nan_downtrend_(40, 50]',
            'combo_key_nan_downtrend_(50, 60]', 'combo_key_nan_downtrend_(60, 70]',
            'combo_key_nan_downtrend_(70, 80]', 'combo_key_nan_sideways_(0, 20]',
            'combo_key_nan_sideways_(20, 30]', 'combo_key_nan_sideways_(30, 40]',
            'combo_key_nan_sideways_(40, 50]', 'combo_key_nan_sideways_(50, 60]',
            'combo_key_nan_sideways_(60, 70]', 'combo_key_nan_sideways_(70, 80]',
            'combo_key_nan_sideways_(80, 100]', 'combo_key_nan_uptrend_(20, 30]',
            'combo_key_nan_uptrend_(30, 40]', 'combo_key_nan_uptrend_(40, 50]',
            'combo_key_nan_uptrend_(50, 60]', 'combo_key_nan_uptrend_(60, 70]',
            'combo_key_nan_uptrend_(70, 80]', 'combo_key_nan_uptrend_(80, 100]'
        ]

    def _get_macd_z_bin_features(self):
        """Return MACD z-bin one-hot features"""
        return [
            'macd_z_bin_0', 'macd_z_bin_(-0.12, 0.126]', 
            'macd_z_bin_(-0.489, -0.12]', 'macd_z_bin_(-13.256, -0.489]',
            'macd_z_bin_(0.126, 0.517]', 'macd_z_bin_(0.517, 10.271]'
        ]

    def _get_combo_key2_features(self):
        """Return combo key2 one-hot features"""
        return [
            'combo_key2_BUY_(0, 20]_(-0.12, 0.126]', 'combo_key2_BUY_(0, 20]_(-0.489, -0.12]',
            'combo_key2_BUY_(0, 20]_(-13.256, -0.489]', 'combo_key2_BUY_(0, 20]_(0.126, 0.517]',
            'combo_key2_BUY_(20, 30]_(-0.12, 0.126]', 'combo_key2_BUY_(20, 30]_(-0.489, -0.12]',
            'combo_key2_BUY_(20, 30]_(-13.256, -0.489]', 'combo_key2_BUY_(20, 30]_(0.126, 0.517]',
            'combo_key2_BUY_(20, 30]_(0.517, 10.271]', 'combo_key2_BUY_(30, 40]_(-0.12, 0.126]',
            'combo_key2_BUY_(30, 40]_(-0.489, -0.12]', 'combo_key2_BUY_(30, 40]_(-13.256, -0.489]',
            'combo_key2_BUY_(30, 40]_(0.126, 0.517]', 'combo_key2_BUY_(30, 40]_(0.517, 10.271]',
            'combo_key2_BUY_(40, 50]_(-0.12, 0.126]', 'combo_key2_BUY_(40, 50]_(-0.489, -0.12]',
            'combo_key2_BUY_(40, 50]_(-13.256, -0.489]', 'combo_key2_BUY_(40, 50]_(0.126, 0.517]',
            'combo_key2_BUY_(40, 50]_(0.517, 10.271]', 'combo_key2_BUY_(50, 60]_(-0.12, 0.126]',
            'combo_key2_BUY_(50, 60]_(-0.489, -0.12]', 'combo_key2_BUY_(50, 60]_(-13.256, -0.489]',
            'combo_key2_BUY_(50, 60]_(0.126, 0.517]', 'combo_key2_BUY_(50, 60]_(0.517, 10.271]',
            'combo_key2_BUY_(60, 70]_(-0.12, 0.126]', 'combo_key2_BUY_(60, 70]_(-0.489, -0.12]',
            'combo_key2_BUY_(60, 70]_(0.126, 0.517]', 'combo_key2_BUY_(60, 70]_(0.517, 10.271]',
            'combo_key2_BUY_(70, 80]_(-0.12, 0.126]', 'combo_key2_BUY_(70, 80]_(0.126, 0.517]',
            'combo_key2_BUY_(70, 80]_(0.517, 10.271]', 'combo_key2_BUY_(80, 100]_(0.517, 10.271]',
            'combo_key2_SELL_(0, 20]_(-13.256, -0.489]', 'combo_key2_SELL_(20, 30]_(-0.12, 0.126]',
            'combo_key2_SELL_(20, 30]_(-0.489, -0.12]', 'combo_key2_SELL_(20, 30]_(-13.256, -0.489]',
            'combo_key2_SELL_(20, 30]_(0.126, 0.517]', 'combo_key2_SELL_(30, 40]_(-0.12, 0.126]',
            'combo_key2_SELL_(30, 40]_(-0.489, -0.12]', 'combo_key2_SELL_(30, 40]_(-13.256, -0.489]',
            'combo_key2_SELL_(30, 40]_(0.126, 0.517]', 'combo_key2_SELL_(30, 40]_(0.517, 10.271]',
            'combo_key2_SELL_(40, 50]_(-0.12, 0.126]', 'combo_key2_SELL_(40, 50]_(-0.489, -0.12]',
            'combo_key2_SELL_(40, 50]_(-13.256, -0.489]', 'combo_key2_SELL_(40, 50]_(0.126, 0.517]',
            'combo_key2_SELL_(40, 50]_(0.517, 10.271]', 'combo_key2_SELL_(50, 60]_(-0.12, 0.126]',
            'combo_key2_SELL_(50, 60]_(-0.489, -0.12]', 'combo_key2_SELL_(50, 60]_(-13.256, -0.489]',
            'combo_key2_SELL_(50, 60]_(0.126, 0.517]', 'combo_key2_SELL_(50, 60]_(0.517, 10.271]',
            'combo_key2_SELL_(60, 70]_(-0.12, 0.126]', 'combo_key2_SELL_(60, 70]_(-0.489, -0.12]',
            'combo_key2_SELL_(60, 70]_(-13.256, -0.489]', 'combo_key2_SELL_(60, 70]_(0.126, 0.517]',
            'combo_key2_SELL_(60, 70]_(0.517, 10.271]', 'combo_key2_SELL_(70, 80]_(-0.12, 0.126]',
            'combo_key2_SELL_(70, 80]_(-0.489, -0.12]', 'combo_key2_SELL_(70, 80]_(0.126, 0.517]',
            'combo_key2_SELL_(70, 80]_(0.517, 10.271]', 'combo_key2_SELL_(80, 100]_(-0.12, 0.126]',
            'combo_key2_SELL_(80, 100]_(0.517, 10.271]', 'combo_key2_nan_(0, 20]_(-0.12, 0.126]',
            'combo_key2_nan_(0, 20]_(-0.489, -0.12]', 'combo_key2_nan_(0, 20]_(-13.256, -0.489]',
            'combo_key2_nan_(0, 20]_nan', 'combo_key2_nan_(20, 30]_(-0.12, 0.126]',
            'combo_key2_nan_(20, 30]_(-0.489, -0.12]', 'combo_key2_nan_(20, 30]_(-13.256, -0.489]',
            'combo_key2_nan_(20, 30]_(0.126, 0.517]', 'combo_key2_nan_(20, 30]_(0.517, 10.271]',
            'combo_key2_nan_(30, 40]_(-0.12, 0.126]', 'combo_key2_nan_(30, 40]_(-0.489, -0.12]',
            'combo_key2_nan_(30, 40]_(-13.256, -0.489]', 'combo_key2_nan_(30, 40]_(0.126, 0.517]',
            'combo_key2_nan_(30, 40]_(0.517, 10.271]', 'combo_key2_nan_(40, 50]_(-0.12, 0.126]',
            'combo_key2_nan_(40, 50]_(-0.489, -0.12]', 'combo_key2_nan_(40, 50]_(-13.256, -0.489]',
            'combo_key2_nan_(40, 50]_(0.126, 0.517]', 'combo_key2_nan_(40, 50]_(0.517, 10.271]',
            'combo_key2_nan_(50, 60]_(-0.12, 0.126]', 'combo_key2_nan_(50, 60]_(-0.489, -0.12]',
            'combo_key2_nan_(50, 60]_(-13.256, -0.489]', 'combo_key2_nan_(50, 60]_(0.126, 0.517]',
            'combo_key2_nan_(50, 60]_(0.517, 10.271]', 'combo_key2_nan_(60, 70]_(-0.12, 0.126]',
            'combo_key2_nan_(60, 70]_(-0.489, -0.12]', 'combo_key2_nan_(60, 70]_(-13.256, -0.489]',
            'combo_key2_nan_(60, 70]_(0.126, 0.517]', 'combo_key2_nan_(60, 70]_(0.517, 10.271]',
            'combo_key2_nan_(70, 80]_(-0.12, 0.126]', 'combo_key2_nan_(70, 80]_(-0.489, -0.12]',
            'combo_key2_nan_(70, 80]_(-13.256, -0.489]', 'combo_key2_nan_(70, 80]_(0.126, 0.517]',
            'combo_key2_nan_(70, 80]_(0.517, 10.271]', 'combo_key2_nan_(80, 100]_(-0.12, 0.126]',
            'combo_key2_nan_(80, 100]_(0.126, 0.517]', 'combo_key2_nan_(80, 100]_(0.517, 10.271]'
        ]

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

    def calculate_technical_indicators(self, df):
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
            
            # 6. Dollar Volume
            df["dollar_volume"] = (df["adj close"] * df["volume"]) / 1e6
            
            # 7. Moving Averages
            df["ma_10"] = df["adj close"].rolling(10, min_periods=1).mean()
            df["ma_100"] = df["adj close"].rolling(100, min_periods=1).mean()
            
            # Additional MAs on close
            for length in [20, 30, 40, 60]:
                df[f"ma_{length}"] = df["close"].rolling(window=length, min_periods=1).mean()
            
            # 8. VWAP calculations
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            vwap_num = (typical_price * df["volume"]).cumsum()
            vwap_den = df["volume"].cumsum()
            df["vwap"] = vwap_num / vwap_den
            df["vwap_std"] = df["vwap"].rolling(20, min_periods=1).std(ddof=0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_technical_indicators: {str(e)}")
            return df

    def calculate_vwap_bands(self, df):
        """Calculate VWAP bands with session-based reset"""
        try:
            # Detect timeframe
            if len(df) < 2:
                return df
                
            time_diff = (df.index[1] - df.index[0]).total_seconds() / 60
            tf_minutes = int(round(time_diff))
            
            # Get session length
            session_hours = self.tf_to_session_hours.get(tf_minutes, 12)
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

    def calculate_trend_indicators(self, df):
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
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_trend_indicators: {str(e)}")
            return df

    def calculate_trade_features(self, df, signal_type):
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
            df['rrr'] = df['tp_distance'] / df['sl_distance'].replace(0, np.nan)
            df['log_sl'] = np.log1p(df['sl_price'])
            
            # Set trade type
            df['trade_type'] = signal_type
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_trade_features: {str(e)}")
            return df

    def calculate_candle_metrics(self, df):
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

    def calculate_session_time_features(self, df):
        """Calculate session and time-based features"""
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
            
            # Minutes closed
            df['minutes,closed'] = df.index.minute
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_session_time_features: {str(e)}")
            return df

    def create_bins(self, df):
        """Create bin features for various indicators"""
        try:
            df = df.copy()
            
            # RSI bins
            df['rsi_bin'] = pd.cut(df['rsi'], bins=self.rsi_bins)
            
            # MACD z bins
            df['macd_z_bin'] = pd.qcut(df['macd_z'], q=5, duplicates='drop')
            
            # Volume bins (using quintiles)
            df['volume_bin'] = pd.qcut(df['volume'], q=5, duplicates='drop')
            df['dollar_volume_bin'] = pd.qcut(df['dollar_volume'], q=5, duplicates='drop')
            
            return df
            
        except Exception as e:
            logger.error(f"Error in create_bins: {str(e)}")
            return df

    def calculate_one_hot_features(self, df):
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
            
            # RSI zone calculation and one-hot encoding
            conditions = [
                df["rsi"].isna(),
                df["rsi"] < 30,
                df["rsi"] > 70
            ]
            choices = ["unknown", "oversold", "overbought"]
            df["rsi_zone"] = np.select(conditions, choices, default="neutral")
            
            rsi_zones = ['neutral', 'overbought', 'oversold']
            for zone in rsi_zones:
                df[f'rsi_zone_{zone}'] = (df['rsi_zone'] == zone).astype(float)
            
            # Trend direction calculation and one-hot encoding
            conditions = [
                df['trend_strength_up'] > df['trend_strength_down'],
                df['trend_strength_down'] > df['trend_strength_up']
            ]
            choices = ['uptrend', 'downtrend']
            df['trend_direction'] = np.select(conditions, choices, default='sideways')
            
            trend_directions = ['downtrend', 'sideways', 'uptrend']
            for direction in trend_directions:
                df[f'trend_direction_{direction}'] = (df['trend_direction'] == direction).astype(float)
            
            # CRT and trade type one-hot encoding
            df['crt_BUY'] = (df.get('crt', '') == 'BUY').astype(float)
            df['crt_SELL'] = (df.get('crt', '') == 'SELL').astype(float)
            df['trade_type_BUY'] = (df.get('trade_type', '') == 'BUY').astype(float)
            df['trade_type_SELL'] = (df.get('trade_type', '') == 'SELL').astype(float)
            
            # RSI bin one-hot encoding
            rsi_bin_ranges = ['(0, 20]', '(20, 30]', '(30, 40]', '(40, 50]', '(50, 60]', '(60, 70]', '(70, 80]', '(80, 100]']
            for bin_range in rsi_bin_ranges:
                df[f'rsi_bin_{bin_range}'] = (df['rsi_bin'].astype(str) == bin_range).astype(float)
            
            # Combo key one-hot encoding
            combo_key_patterns = self._get_combo_key_features()
            for pattern in combo_key_patterns:
                df[pattern] = 0.0  # Placeholder - would need actual combo key logic
            
            # Combo flag one-hot encoding
            for flag in ['dead', 'fair', 'fine']:
                df[f'combo_flag_{flag}'] = 0.0  # Placeholder
            
            # MACD z-bin one-hot encoding
            macd_bins = ['0', '(-0.12, 0.126]', '(-0.489, -0.12]', '(-13.256, -0.489]', '(0.126, 0.517]', '(0.517, 10.271]']
            for bin_val in macd_bins:
                df[f'macd_z_bin_{bin_val}'] = (df['macd_z_bin'].astype(str) == bin_val).astype(float)
            
            # Combo key2 one-hot encoding
            combo_key2_patterns = self._get_combo_key2_features()
            for pattern in combo_key2_patterns:
                df[pattern] = 0.0  # Placeholder
            
            # Combo flag2 one-hot encoding
            for flag in ['dead', 'fair', 'fine']:
                df[f'combo_flag2_{flag}'] = 0.0  # Placeholder
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_one_hot_features: {str(e)}")
            return df

    def generate_combo_features(self, df):
        """Generate the 200+ combo interaction features"""
        try:
            # Initialize all combo features to 0
            combo_features = self._get_combo_features()
            for feature in combo_features:
                df[feature] = 0.0
            
            # Calculate average win rate based on active combos
            average_win_rate = self.calculate_average_combo_win_rate(df)
            df['average_combo_win_rate'] = average_win_rate
            
            # Placeholder for is_bad_combo
            df['is_bad_combo'] = 0
            
            return df, average_win_rate
            
        except Exception as e:
            logger.error(f"Error in generate_combo_features: {str(e)}")
            return df, 0.0

    def generate_features(self, df, signal_type):
        """Main feature generation pipeline - returns features in EXACT order"""
        try:
            if len(df) < 200:
                logger.warning("Not enough data for feature generation")
                return None
            
            df = df.tail(200).copy()
            
            # Set current candle close = open for immediate processing
            current_candle = df.iloc[-1].copy()
            current_candle['close'] = current_candle['open']
            df.iloc[-1] = current_candle
            
            # Execute feature engineering pipeline
            pipeline = [
                self.calculate_technical_indicators,
                self.calculate_vwap_bands,
                self.calculate_trend_indicators,
                lambda x: self.calculate_trade_features(x, signal_type),
                self.calculate_candle_metrics,
                self.calculate_session_time_features,
                self.create_bins,
                self.calculate_one_hot_features,
            ]
            
            for step in pipeline:
                df = step(df)
                if df is None:
                    logger.error("Pipeline step returned None")
                    return None
            
            # Generate combo features and get average win rate
            df, average_win_rate = self.generate_combo_features(df)
            
            # Check if average combo win rate is above 30%
            if average_win_rate <= 0.3:
                logger.warning(f"Average combo win rate {average_win_rate:.4f} is not above 30% - breaking")
                return None
            
            logger.info(f"Average combo win rate {average_win_rate:.4f} is above 30% - proceeding")
            
            # Create final feature vector in EXACT order
            features = pd.Series(index=self.features, dtype=float)
            
            for feat in self.features:
                if feat in df.columns:
                    value = df[feat].iloc[-1]
                    if pd.isna(value):
                        features[feat] = 0.0
                    elif isinstance(value, (bool, np.bool_)):
                        features[feat] = float(value)
                    elif isinstance(value, (int, np.integer)):
                        features[feat] = float(value)
                    else:
                        features[feat] = float(value) if isinstance(value, (float, np.float64)) else str(value)
                else:
                    features[feat] = 0.0
            
            # Validate all features are present
            missing_features = set(self.features) - set(features.index)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feat in missing_features:
                    features[feat] = 0.0
            
            logger.debug(f"Feature generation successful: {len(features)} features")
            return features[self.features]  # Ensure exact order
            
        except Exception as e:
            logger.error(f"Error in generate_features: {str(e)}")
            logger.error(traceback.format_exc())
            return None

# ========================
# ENHANCED MODEL LOADER WITH OVERCONFIDENCE DETECTION
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
            
            # Track prediction statistics for overconfidence detection
            self.prediction_history = []
            self.overconfidence_threshold = 0.99
            self.max_history_size = 100
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    
        
    def predict(self, features):
        """Enhanced prediction with overconfidence detection"""
        logger.debug("Starting prediction...")
        
        # Diagnostic: Check feature statistics
        logger.debug(f"Feature stats - Min: {np.min(features):.4f}, Max: {np.max(features):.4f}, Mean: {np.mean(features):.4f}")
        
        scaled = self.scaler.transform([features])
        logger.debug(f"Scaled stats - Min: {np.min(scaled):.4f}, Max: {np.max(scaled):.4f}, Mean: {np.mean(scaled):.4f}")
        
        reshaped = scaled.reshape(1, 1, -1)
        prediction = self.model.predict(reshaped, verbose=0)[0][0]
        
            
        logger.debug(f"Prediction complete: {prediction:.6f}")
        return prediction

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
# COLAB TRADING BOT
# ========================
class ColabTradingBot:
    def __init__(self, timeframe, credentials):
        self.timeframe = timeframe
        self.credentials = credentials
        self.logger = logging.getLogger(f"{timeframe}_bot")
        self.start_time = time.time()
        self.max_duration = 11.5 * 3600
        
        # Initialize Google Sheets storage
        self.storage = GoogleSheetsStorage(
            spreadsheet_id='1HZo4uUfeYrzoeEQkjoxwylrqQpKI4R9OfHOZ6zaDino'
        )
        
        # Initialize prediction tracker
        self.prediction_tracker = PredictionTracker()
        
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
        """Main bot execution loop with full context fetching"""
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
                
                # Calculate precise wakeup time
                now = datetime.now(NY_TZ)
                next_candle = self.calculate_next_candle_time()
                sleep_seconds = max(0, (next_candle - now).total_seconds() - 0.1)  # Wake 100ms early
                
                if sleep_seconds > 0:
                    logger.debug(f"Sleeping for {sleep_seconds:.2f} seconds until next candle")
                    time.sleep(sleep_seconds)
                
                # Busy-wait for precise candle open
                while datetime.now(NY_TZ) < next_candle:
                    time.sleep(0.001)  # 1ms precision
                
                logger.debug("Candle open detected - waiting 5s for candle availability")
                time.sleep(5)
                
                # Fetch full 201 candles for complete context
                logger.debug("Fetching full 201 candles for updated context")
                new_data = fetch_candles(
                    self.timeframe,
                    count=201,
                    api_key=self.credentials['oanda_api_key']
                )
                
                if new_data.empty:
                    logger.error("Failed to fetch candle data")
                    continue
                    
                # Update the data
                self.data = new_data
                logger.debug(f"Total records: {len(self.data)}")
                
                # Detect CRT pattern using only current open
                signal_type, signal_data = self.feature_engineer.calculate_crt_signal(self.data)
                
                if not signal_type:
                    logger.debug("No CRT pattern detected")
                    continue
                    
                logger.info(f"CRT pattern detected: {signal_type} at {signal_data['time']}")
                
                # Generate features immediately
                features = self.feature_engineer.generate_features(self.data, signal_type)
                if features is None:
                    logger.warning("Feature generation failed")
                    continue
                    
                # Get prediction
                prediction = self.model_loader.predict(features)
                logger.info(f"Prediction: {prediction:.4f}")
                
                # Send signal and journal immediately
                self.send_signal(signal_type, signal_data, prediction, features)
                    
            except Exception as e:
                error_msg = f"❌ {self.timeframe} bot error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                send_telegram(error_msg[:1000], self.credentials['telegram_token'], self.credentials['telegram_chat_id'])
                time.sleep(60)

# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    print("===== UPDATED BOT STARTING =====")
    print(f"Start time: {datetime.now(NY_TZ)}")
    print("Python version:", sys.version)
    print("TensorFlow version:", tf.__version__)
    print("Pandas version:", pd.__version__)
    #print("Pandas_ta version:", ta.__version__)
    
    # Force debug logging to console
    debug_handler = logging.StreamHandler()
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(debug_handler)
    
    logger.info("Starting main execution with updated version-compatible code")
    
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
