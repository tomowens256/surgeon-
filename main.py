import os
import time
import sys
import threading
import logging
import pytz
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
import requests
import re
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
from collections import deque
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments

# ========================
# CONSTANTS & CONFIG
# ========================
NY_TZ = pytz.timezone("America/New_York")
MODEL_PATH = os.getenv("MODEL_PATH", "./ml_models")
FEATURES = [
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
    'combo_flag2_dead', 'combo_flag2_fair', 'combo_flag2_fine',
    'minutes,closed_0', 'minutes,closed_15', 'minutes,closed_30', 'minutes,closed_45'
]

# Oanda configuration
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
API_KEY = os.getenv("OANDA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Only needed instrument and timeframe
INSTRUMENT = "XAU_USD"
TIMEFRAME = "M15"

# Global variables
GLOBAL_LOCK = threading.Lock()
CRT_SIGNAL_COUNT = 0
LAST_SIGNAL_TIME = 0
SIGNALS = deque(maxlen=100)
TRADE_JOURNAL = deque(maxlen=50)
PERF_CACHE = {"updated": 0, "data": None}

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize Oanda API
oanda_api = API(access_token=API_KEY, environment="practice")

# Initialize Flask app
app = Flask(__name__)

# ========================
# UTILITY FUNCTIONS
# ========================
def parse_oanda_time(time_str):
    """Parse Oanda's timestamp with variable fractional seconds"""
    if '.' in time_str and len(time_str.split('.')[1]) > 7:
        time_str = re.sub(r'\.(\d{6})\d+', r'.\1', time_str)
    return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.utc).astimezone(NY_TZ)

def send_telegram(message):
    """Send formatted message to Telegram with detailed error handling"""
    if len(message) > 4000:
        message = message[:4000] + "... [TRUNCATED]"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }, timeout=10)
        
        logging.info(f"Telegram response: {response.status_code} - {response.text}")
        
        if response.status_code != 200:
            logging.error(f"Telegram error: {response.status_code} - {response.text}")
            return False
            
        if not response.json().get('ok'):
            logging.error(f"Telegram API error: {response.json()}")
            return False
            
        return True
    except Exception as e:
        logging.exception(f"Telegram connection failed")
        return False

def fetch_candles():
    """Fetch candles for XAU_USD M15 only"""
    params = {
        "granularity": TIMEFRAME,
        "count": 200,
        "price": "BA"
    }
    try:
        request = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
        response = oanda_api.request(request)
        candles = response.get('candles', [])
        
        data = []
        for candle in candles:
            if candle['complete']:
                data.append({
                    'time': parse_oanda_time(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })
        return pd.DataFrame(data)
    except V20Error as e:
        logging.error(f"Oanda error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.exception(f"Failed to fetch candles")
        return pd.DataFrame()

# ========================
# MODEL VALIDATION SYSTEM
# ========================
class ModelValidator:
    def __init__(self):
        self.models = []
        self.scaler = None
        self.loaded = False
        
    def load_resources(self):
        if self.loaded:
            return True
            
        try:
            scaler_path = os.path.join(MODEL_PATH, 'scaler_oversample.joblib')
            self.scaler = joblib.load(scaler_path)
            
            model_files = [
                'model_f1_0.0000_20250719_090727.keras',
                'model_f1_0.0000_20250719_092134.keras',
                'model_f1_0.0000_20250719_093712.keras',
                'model_f1_0.0000_20250719_095056.keras',
                'model_f1_0.0000_20250719_100411.keras',
                'model_f1_0.0000_20250719_102457.keras',
                'model_f1_0.0000_20250719_104011.keras',
                'model_f1_0.0000_20250719_110914.keras'
            ]
            
            for model_file in model_files:
                model_path = os.path.join(MODEL_PATH, model_file)
                model = load_model(model_path)
                self.models.append(model)
                
            self.loaded = True
            logging.info("ML models loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            return False
    
    def validate(self, features):
        if not self.load_resources():
            return False
            
        try:
            if isinstance(features, pd.Series):
                features = features[FEATURES].values.reshape(1, -1)
            
            scaled = self.scaler.transform(features)
            reshaped = scaled.reshape(scaled.shape[0], 1, scaled.shape[1])
            
            predictions = []
            for model in self.models:
                pred = model.predict(reshaped, verbose=0).flatten()
                predictions.append(pred)
            
            avg_prob = np.mean(predictions, axis=0)[0]
            return avg_prob >= 0.55
        except Exception as e:
            logging.error(f"Validation error: {e}")
            return False

# ========================
# FEATURE ENGINEER
# ========================
class FeatureEngineer:
    def __init__(self):
        self.history_size = 200
        
    def transform(self, df_history, signal_type, minutes_closed):
        try:
            features = df_history.iloc[-1].copy()
            
            features['trade_type_BUY'] = 1 if signal_type == 'BUY' else 0
            features['trade_type_SELL'] = 1 if signal_type == 'SELL' else 0
            features['crt_BUY'] = 1 if features.get('crt') == 'BUY' else 0
            features['crt_SELL'] = 1 if features.get('crt') == 'SELL' else 0
            
            minute_col = f'minutes,closed_{minutes_closed}'
            for col in ['minutes,closed_0', 'minutes,closed_15', 
                        'minutes,closed_30', 'minutes,closed_45']:
                features[col] = 1 if col == minute_col else 0
            
            for feature in FEATURES:
                if feature not in features:
                    features[feature] = 0
                    
            return features[FEATURES]
        except Exception as e:
            logging.error(f"Feature engineering error: {e}")
            return None

# ========================
# CANDLE SCHEDULER
# ========================
class CandleScheduler(threading.Thread):
    def __init__(self, timeframe=15):
        super().__init__(daemon=True)
        self.timeframe = timeframe
        self.callback = None
        self.active = True
        self.next_candle = None
        self.minutes_closed = 0
        self.event = threading.Event()
    
    def register_callback(self, callback):
        self.callback = callback
        
    def calculate_next_candle(self):
        now = datetime.now(NY_TZ)
        current_minute = now.minute
        remainder = current_minute % self.timeframe
        
        if remainder == 0:
            next_candle = now.replace(second=0, microsecond=0) + timedelta(minutes=self.timeframe)
        else:
            next_minute = current_minute - remainder + self.timeframe
            if next_minute >= 60:
                next_candle = now.replace(hour=now.hour+1, minute=0, second=0, microsecond=0)
            else:
                next_candle = now.replace(minute=next_minute, second=0, microsecond=0)
        
        return next_candle
    
    def calculate_minutes_closed(self):
        now = datetime.now(NY_TZ)
        if self.next_candle:
            elapsed = (now - (self.next_candle - timedelta(minutes=self.timeframe))).total_seconds() / 60
            if elapsed < 15: return 15
            elif elapsed < 30: return 30
            elif elapsed < 45: return 45
        return 0
    
    def run(self):
        while self.active:
            try:
                self.next_candle = self.calculate_next_candle()
                now = datetime.now(NY_TZ)
                sleep_seconds = (self.next_candle - now).total_seconds()
                
                if sleep_seconds > 0:
                    logging.info(f"Sleeping {sleep_seconds:.1f}s until next candle")
                    time.sleep(sleep_seconds)
                
                start_time = time.time()
                self.minutes_closed = self.calculate_minutes_closed()
                
                if self.callback:
                    self.callback(self.minutes_closed)
                
                processing_time = time.time() - start_time
                if processing_time < 30:
                    time.sleep(30 - processing_time)
            except Exception as e:
                logging.error(f"Scheduler error: {e}")
                time.sleep(60)

# ========================
# TRADING DETECTOR
# ========================
class TradingDetector:
    def __init__(self):
        self.data = pd.DataFrame()
        self.validator = ModelValidator()
        self.feature_engineer = FeatureEngineer()
        self.scheduler = CandleScheduler(timeframe=15)
        self.pending_signals = []
        self.scheduler.register_callback(self.process_pending_signals)
        self.scheduler.start()
        
    def process_pending_signals(self, minutes_closed):
        if not self.pending_signals or self.data.empty:
            return
            
        try:
            df_history = self.data.tail(self.feature_engineer.history_size)
            if len(df_history) < self.feature_engineer.history_size:
                return
                
            for signal in list(self.pending_signals):
                features = self.feature_engineer.transform(
                    df_history, 
                    signal['signal'], 
                    minutes_closed
                )
                
                if features is None:
                    continue
                
                if self.validator.validate(features):
                    self.trigger_alert(signal)
                    self.pending_signals.remove(signal)
        except Exception as e:
            logging.error(f"Signal processing error: {e}")
    
    def trigger_alert(self, signal_info):
        global CRT_SIGNAL_COUNT, LAST_SIGNAL_TIME
        
        with GLOBAL_LOCK:
            CRT_SIGNAL_COUNT += 1
            LAST_SIGNAL_TIME = time.time()
        
        alert_time = signal_info['time'].astimezone(NY_TZ)
        send_telegram(
            f"🚀 *VALIDATED CRT* XAU/USD {signal_info['signal']}\n"
            f"Timeframe: M15\n"
            f"Time: {alert_time.strftime('%Y-%m-%d %H:%M')} NY\n"
            f"RSI Zone: {signal_info['rsi_zone']}\n"
            f"Confidence: High"
        )
        
        with GLOBAL_LOCK:
            SIGNALS.append({
                "time": time.time(),
                "pair": "XAU_USD",
                "timeframe": "M15",
                "signal": signal_info['signal'],
                "outcome": "pending",
                "rrr": None
            })
    
    def update_data(self, df_new):
        if self.data.empty:
            self.data = df_new
        else:
            df_combined = pd.concat([self.data, df_new])
            df_combined = df_combined.drop_duplicates(subset=['time'], keep='last')
            self.data = df_combined.sort_values('time').reset_index(drop=True)
        
        self.check_signals()
    
    def check_signals(self):
        if len(self.data) < 10:
            return
            
        # PLACEHOLDER: ADD YOUR ACTUAL SIGNAL DETECTION LOGIC HERE
        # This should be replaced with your CRT detection algorithm
        last_row = self.data.iloc[-1]
        
        # Example signal detection (simplified for demonstration)
        # Calculate RSI
        rsi = ta.rsi(self.data['close'], length=14)
        if len(rsi) == 0:
            return
            
        current_rsi = rsi.iloc[-1]
        
        # Determine RSI zone
        if current_rsi > 70:
            rsi_zone = "overbought"
        elif current_rsi < 30:
            rsi_zone = "oversold"
        else:
            rsi_zone = "neutral"
            
        # Detect potential signals based on RSI zone
        if rsi_zone == "overbought":
            signal_type = "SELL"
        elif rsi_zone == "oversold":
            signal_type = "BUY"
        else:
            return
            
        # Add signal to pending queue
        signal_info = {
            'signal': signal_type,
            'time': last_row['time'],
            'rsi_zone': rsi_zone
        }
        
        self.pending_signals.append(signal_info)
        logging.info(f"Signal queued for validation: {signal_type}")

# ========================
# FLASK UI ROUTES
# ========================
@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/journal')
def journal():
    return render_template('journal.html')

@app.route('/metrics')
def metrics():
    return jsonify(calculate_performance_metrics())

@app.route('/signals')
def signals():
    with GLOBAL_LOCK:
        return jsonify(list(SIGNALS)[-20:])

@app.route('/journal/entries')
def journal_entries():
    with GLOBAL_LOCK:
        return jsonify(list(TRADE_JOURNAL))

@app.route('/journal/add', methods=['POST'])
def add_entry():
    data = request.json
    add_journal_entry(
        data.get('type', 'note'),
        data.get('content', ''),
        data.get('image', None)
    )
    return jsonify({"status": "success"})

# ========================
# SUPPORT FUNCTIONS FOR UI
# ========================
def calculate_performance_metrics():
    global PERF_CACHE
    
    if time.time() - PERF_CACHE["updated"] < 300 and PERF_CACHE["data"]:
        return PERF_CACHE["data"]
    
    with GLOBAL_LOCK:
        recent_signals = list(SIGNALS)[-100:]
        
        if not recent_signals:
            return {
                "win_rate": 0,
                "avg_rrr": 0,
                "hourly_dist": {},
                "asset_dist": {}
            }
        
        wins = sum(1 for s in recent_signals if s.get('outcome') == 'win')
        win_rate = round((wins / len(recent_signals)) * 100, 1) if recent_signals else 0
        
        rrr_values = [s.get('rrr', 0) for s in recent_signals if s.get('rrr') is not None]
        avg_rrr = round(np.mean(rrr_values), 2) if rrr_values else 0
        
        hourly_dist = {}
        for signal in recent_signals:
            hour = datetime.fromtimestamp(signal['time']).hour
            hourly_dist[hour] = hourly_dist.get(hour, 0) + 1
        
        asset_dist = {}
        for signal in recent_signals:
            pair = signal['pair'].split('_')[0]
            asset_dist[pair] = asset_dist.get(pair, 0) + 1
        
        metrics = {
            "win_rate": win_rate,
            "avg_rrr": avg_rrr,
            "hourly_dist": hourly_dist,
            "asset_dist": asset_dist
        }
        
        PERF_CACHE = {"updated": time.time(), "data": metrics}
        return metrics

def add_journal_entry(entry_type, content, image_url=None):
    with GLOBAL_LOCK:
        TRADE_JOURNAL.append({
            "timestamp": time.time(),
            "type": entry_type,
            "content": content,
            "image": image_url
        })

# ========================
# MAIN BOT OPERATION
# ========================
def run_bot():
    send_telegram(f"🚀 *Bot Started*\nInstrument: XAU/USD\nTimeframe: M15\nTime: {datetime.now(NY_TZ)}")
    
    detector = TradingDetector()
    refresh_interval = 300  # 5 minutes
    
    while True:
        try:
            df = fetch_candles()
            if not df.empty:
                detector.update_data(df)
            
            time.sleep(refresh_interval)
        except Exception as e:
            logging.exception("Main loop error")
            time.sleep(60)

if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    app.run(host='0.0.0.0', port=5000)
