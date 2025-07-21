# ========================
# IMPORTS & CONFIGURATION
# ========================
import os
import time
import threading
import logging
import pytz
import numpy as np
import pandas as pd
import pandas_ta
import joblib
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.instruments import InstrumentsCandles
from tensorflow.keras.models import load_model
from collections import deque
import sys
# Initialize Flask
app = Flask(__name__)

# ========================
# GLOBAL CONFIGURATION
# ========================
# Validate environment variables
required_env_vars = ["OANDA_ACCOUNT_ID", "OANDA_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"Missing environment variables: {', '.join(missing_vars)}")
    exit(1)

ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ACCESS_TOKEN = os.getenv("OANDA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MODEL_PATH = os.getenv("MODEL_PATH", "./ml_models")
NY_TZ = pytz.timezone("America/New_York")

# Instruments and timeframes
INSTRUMENTS = ["XAU_USD", "GBP_USD", "NAS100_USD", "SPX500_USD", "US30_USD", "EUR_USD"]
TIMEFRAMES = ["H4", "H2", "H1", "M30", "M15"]
CANDLE_COUNTS = {"H4": 100, "H2": 150, "H1": 200, "M30": 300, "M15": 400}
ALERT_COOLDOWN = {"H4": 3600, "H2": 1800, "H1": 900, "M30": 600, "M15": 300}

# Global state with thread safety
GLOBAL_LOCK = threading.Lock()
LAST_ACTIVITY = time.time()
LAST_SIGNAL_TIME = None
ERROR_COUNT = 0
SIGNAL_COUNT = 0
WATCHDOG_ENABLED = True
HEARTBEAT_INTERVAL = 3600
SSMT_SIGNAL_COUNT = 0
CRT_SIGNAL_COUNT = 0
SIGNALS = deque(maxlen=100)  # Track last 100 signals
TRADE_JOURNAL = deque(maxlen=50)  # Journal entries
PERF_CACHE = {"updated": 0, "data": None}

# Initialize Oanda API
api = API(access_token=ACCESS_TOKEN, environment="practice")

# ========================
# LOGGING SETUP
# ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# ========================
# TELEGRAM NOTIFICATIONS
# ========================
def send_telegram(message):
    """Send formatted message to Telegram"""
    if len(message) > 4000:
        message = message[:4000] + "... [TRUNCATED]"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }, timeout=10)
        response.raise_for_status()
        logging.info(f"Telegram sent: {message[:100]}...")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# ========================
# ML VALIDATION SYSTEM
# ========================
class ModelValidator:
    def __init__(self):
        self.models = []
        self.scaler = None
        self.loaded = False
        self.features = [
            # ... (your feature list from earlier) ...
        ]
        
    def load_resources(self):
        """Lazy loading of models"""
        if self.loaded: 
            return True
            
        try:
            # Load scaler
            scaler_path = os.path.join(MODEL_PATH, 'scaler_oversample.joblib')
            self.scaler = joblib.load(scaler_path)
            
            # Load models
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
        """Validate signal using ensemble prediction"""
        if not self.load_resources():
            return False
            
        try:
            # Convert to numpy array
            if isinstance(features, pd.Series):
                features = features[self.features].values.reshape(1, -1)
            
            # Scale features
            scaled = self.scaler.transform(features)
            reshaped = scaled.reshape(scaled.shape[0], 1, scaled.shape[1])
            
            # Get predictions
            predictions = []
            for model in self.models:
                pred = model.predict(reshaped, verbose=0).flatten()
                predictions.append(pred)
            
            # Average probabilities
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
        """Generate features for current candle"""
        try:
            features = df_history.iloc[-1].copy()
            
            # Add signal-specific features
            features['trade_type_BUY'] = 1 if signal_type == 'BUY' else 0
            features['trade_type_SELL'] = 1 if signal_type == 'SELL' else 0
            features['crt_BUY'] = 1 if features.get('crt') == 'BUY' else 0
            features['crt_SELL'] = 1 if features.get('crt') == 'SELL' else 0
            
            # Add minute feature
            minute_col = f'minutes,closed_{minutes_closed}'
            for col in ['minutes,closed_0', 'minutes,closed_15', 
                        'minutes,closed_30', 'minutes,closed_45']:
                features[col] = 1 if col == minute_col else 0
            
            return features
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
    
    def register_callback(self, callback):
        self.callback = callback
        
    def calculate_next_candle(self):
        """Calculate next candle time"""
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
        """Calculate minutes closed feature"""
        now = datetime.now(NY_TZ)
        if self.next_candle:
            elapsed = (now - (self.next_candle - timedelta(minutes=self.timeframe))).total_seconds() / 60
            if elapsed < 15: return 15
            elif elapsed < 30: return 30
            elif elapsed < 45: return 45
        return 0
    
    def run(self):
        """Main scheduling loop"""
        while self.active:
            try:
                self.next_candle = self.calculate_next_candle()
                now = datetime.now(NY_TZ)
                sleep_seconds = (self.next_candle - now).total_seconds()
                
                if sleep_seconds > 0:
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
        self.data = {pair: {tf: pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close']) 
                     for tf in TIMEFRAMES} for pair in INSTRUMENTS}
        self.last_alert = {pair: {tf: None for tf in TIMEFRAMES} for pair in INSTRUMENTS}
        self.validator = ModelValidator()
        self.feature_engineer = FeatureEngineer()
        self.scheduler = CandleScheduler(timeframe=15)
        self.pending_signals = []
        self.scheduler.register_callback(self.process_pending_signals)
        self.scheduler.start()
        
    def process_pending_signals(self, minutes_closed):
        """Process queued signals at new candle start"""
        if not self.pending_signals:
            return
            
        for signal in self.pending_signals[:]:
            pair, tf, signal_info = signal
            try:
                df_history = self.data[pair][tf].tail(self.feature_engineer.history_size)
                if len(df_history) < self.feature_engineer.history_size:
                    continue
                
                features = self.feature_engineer.transform(
                    df_history, 
                    signal_info['signal'], 
                    minutes_closed
                )
                
                if features is None:
                    continue
                
                if self.validator.validate(features):
                    self.trigger_alert(signal_info)
                    self.pending_signals.remove(signal)
            except Exception as e:
                logging.error(f"Signal processing error: {e}")
    
    def trigger_alert(self, signal_info):
        """Send validated alert"""
        global CRT_SIGNAL_COUNT, LAST_SIGNAL_TIME
        
        with GLOBAL_LOCK:
            CRT_SIGNAL_COUNT += 1
            LAST_SIGNAL_TIME = time.time()
        
        alert_time = signal_info['time'].astimezone(NY_TZ)
        message = (
            f"🚀 *VALIDATED CRT* {signal_info['pair'].replace('_','/')} {signal_info['signal']}\n"
            f"Timeframe: {signal_info['timeframe']}\n"
            f"Time: {alert_time.strftime('%Y-%m-%d %H:%M')} NY\n"
            f"RSI Zone: {signal_info['rsi_zone']}"
        )
        send_telegram(message)
        
        # Add to UI
        SIGNALS.append({
            "time": time.time(),
            "pair": signal_info['pair'],
            "timeframe": signal_info['timeframe'],
            "signal": signal_info['signal'],
            "outcome": "validated"
        })
    
    # ... [REST OF TRADING DETECTOR METHODS FROM EARLIER] ...

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
    # ... [Performance metrics implementation from earlier] ...
    pass

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
    with GLOBAL_LOCK:
        TRADE_JOURNAL.append({
            "timestamp": time.time(),
            "type": data.get('type', 'note'),
            "content": data.get('content', ''),
            "image": data.get('image', None)
        })
    return jsonify({"status": "success"})

# ========================
# BOT OPERATION
# ========================
def run_bot():
    # ... [Bot implementation from earlier with ML integration] ...
    pass

# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    # Start bot in background thread
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Start Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)
