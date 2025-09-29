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
class FeatureEngineer:

    def __init__(self, timeframe):

        self.timeframe = timeframe

        logger.debug(f"Initializing FeatureEngineer for {timeframe}")



        # Base features in exact order

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

        self.rsi_bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]

        self.macd_z_bins = [-12.386, -0.496, -0.138, 0.134, 0.527, 9.246]



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



    def calculate_crt_signal(self, df):

        """Calculate CRT signal at OPEN of current candle (c0) with minimal latency"""

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

        """Calculate technical indicators WITHOUT volume imputation"""

        try:

            df = df.copy().drop_duplicates(subset=['time'], keep='last')

            # REMOVED VOLUME ESTIMATION - Using shifted features instead

            df['adj close'] = df['open']

            df['garman_klass_vol'] = (((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 -(2 * np.log(2) - 1) * ((np.log(df['adj close']) - np.log(df['open'])) ** 2))

            df['rsi_20'] = ta.rsi(df['adj close'], length=20)

            df['rsi'] = ta.rsi(df['close'], length=14)



            bb = ta.bbands(close=np.log1p(df['adj close']), length=20)

            # Normalize names in case pandas-ta version dropped the ".0"
            bb = bb.rename(columns={
                "BBL_20_2": "BBL_20_2.0",
                "BBM_20_2": "BBM_20_2.0",
                "BBU_20_2": "BBU_20_2.0"
            })
            
            df['bb_low']  = bb['BBL_20_2.0']
            df['bb_mid']  = bb['BBM_20_2.0']
            df['bb_high'] = bb['BBU_20_2.0']




            atr = ta.atr(df['high'], df['low'], df['close'], length=14)

            df['atr_z'] = (atr - atr.mean()) / atr.std()



            macd = ta.macd(close=df['adj close'], fast=12, slow=26, signal=9)

            if "MACD_12_26_9" in macd.columns:
                df['macd_z'] = (macd['MACD_12_26_9'] - macd['MACD_12_26_9'].mean()) / macd['MACD_12_26_9'].std()
            else:
                logger.warning("MACD_12_26_9 not found, filling macd_z with NaN")
                df['macd_z'] = np.nan



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

        except Exception as e:

            logger.error(f"Error in calculate_technical_indicators: {str(e)}")

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



            df['sl_distance'] = abs(entry - df['sl_price']) * 10

            df['tp_distance'] = abs(df['tp_price'] - entry) * 10

            df['rrr'] = df['tp_distance'] / df['sl_distance'].replace(0, np.nan)

            df['log_sl'] = np.log1p(df['sl_price'])



            return df

        except Exception as e:

            logger.error(f"Error in calculate_trade_features: {str(e)}")

            return df



    def calculate_categorical_features(self, df):

        try:

            df = df.copy()



            # Day of week features

            df['day'] = df['time'].dt.day_name()

            all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']

            for day in all_days:

                df[f'day_{day}'] = 0

            today = datetime.now(NY_TZ).strftime('%A')

            df[f'day_{today}'] = 1



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



            df['session'] = df['time'].dt.hour.apply(get_session)

            session_dummies = pd.get_dummies(df['session'], prefix='session')



            # Ensure all session columns exist

            expected_session_cols = ['session_q1', 'session_q2', 'session_q3', 'session_q4']

            for col in expected_session_cols:

                if col not in session_dummies.columns:

                    session_dummies[col] = 0



            df = pd.concat([df, session_dummies], axis=1)

            df.drop(['session'], axis=1, inplace=True)



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

            rsi_dummies = pd.get_dummies(df['rsi_zone'], prefix='rsi_zone')



            # Ensure all RSI zone columns exist

            expected_rsi_cols = ['rsi_zone_oversold', 'rsi_zone_overbought', 'rsi_zone_neutral', 'rsi_zone_unknown']

            for col in expected_rsi_cols:

                if col not in rsi_dummies.columns:

                    rsi_dummies[col] = 0



            df = pd.concat([df, rsi_dummies], axis=1)

            df.drop(['rsi_zone'], axis=1, inplace=True)



            # Trend strength features

            def is_bullish_stack(row):

                try:

                    return int(row['ma_20'] > row['ma_30'] > row['ma_40'] > row['ma_60'])

                except:

                    return 0



            def is_bearish_stack(row):

                try:

                    return int(row['ma_20'] < row['ma_30'] < row['ma_40'] < row['ma_60'])

                except:

                    return 0



            df['trend_strength_up'] = df.apply(is_bullish_stack, axis=1).astype(float)

            df['trend_strength_down'] = df.apply(is_bearish_stack, axis=1).astype(float)



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

            trend_dummies = pd.get_dummies(df['trend_direction'], prefix='trend_direction')



            # Ensure all trend direction columns exist

            expected_trend_cols = ['trend_direction_downtrend', 'trend_direction_sideways', 'trend_direction_uptrend']

            for col in expected_trend_cols:

                if col not in trend_dummies.columns:

                    trend_dummies[col] = 0



            df = pd.concat([df, trend_dummies], axis=1)

            df.drop(['trend_direction'], axis=1, inplace=True)



            return df

        except Exception as e:

            logger.error(f"Error in calculate_categorical_features: {str(e)}")

            # Return a dataframe with at least the expected columns

            expected_cols = self.base_features

            for col in expected_cols:

                if col not in df.columns:

                    df[col] = 0

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



            # Use get() method to avoid KeyError - THIS IS THE CRITICAL FIX

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

            # Return default values on error

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



            features = pd.Series(index=self.features, dtype=float)
            logger.error(print(features))#for debug bro

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
            logger.error(print(df.filter(like="BB").columns))



            return features

        except Exception as e:

            logger.error(f"Error in generate_features: {str(e)}")

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
    
    def detect_overconfidence_pattern(self, prediction):
        """Detect if model is showing overconfidence patterns"""
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > self.max_history_size:
            self.prediction_history.pop(0)
            
        if len(self.prediction_history) < 10:
            return False
            
        # Check if recent predictions are consistently overconfident
        recent_overconfident = sum(p > self.overconfidence_threshold for p in self.prediction_history[-10:])
        return recent_overconfident >= 8  # 80% of recent predictions overconfident
        
    def predict(self, features):
        """Enhanced prediction with overconfidence detection"""
        logger.debug("Starting prediction...")
        
        # Diagnostic: Check feature statistics
        logger.debug(f"Feature stats - Min: {np.min(features):.4f}, Max: {np.max(features):.4f}, Mean: {np.mean(features):.4f}")
        
        scaled = self.scaler.transform([features])
        logger.debug(f"Scaled stats - Min: {np.min(scaled):.4f}, Max: {np.max(scaled):.4f}, Mean: {np.mean(scaled):.4f}")
        
        reshaped = scaled.reshape(1, 1, -1)
        prediction = self.model.predict(reshaped, verbose=0)[0][0]
        
        # Overconfidence detection and correction
        if prediction > 0.99:
            logger.warning(f"⚠️ Suspicious prediction: {prediction:.6f}")
            
            # Get raw model output before activation for analysis
            try:
                # For models with custom layers, we may need to access intermediate outputs
                raw_output = self.model(reshaped, training=False).numpy()[0][0]
                logger.warning(f"Raw model output: {raw_output:.6f}")
                
                # Apply conservative correction for extreme overconfidence
                if prediction > 0.995:
                    correction_factor = 0.9  # Reduce extreme confidence
                    corrected_prediction = prediction * correction_factor
                    logger.warning(f"Applying overconfidence correction: {prediction:.6f} -> {corrected_prediction:.6f}")
                    prediction = corrected_prediction
                    
            except Exception as e:
                logger.error(f"Error in overconfidence analysis: {str(e)}")
        
        # Check for pattern of overconfidence
        if self.detect_overconfidence_pattern(prediction):
            logger.error("🚨 PATTERN DETECTED: Model showing consistent overconfidence behavior!")
            
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
