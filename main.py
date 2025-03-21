import ccxt
import time
import pandas as pd
import pandas_ta as ta
import csv
import os
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

# Charger les variables d'environnement
load_dotenv()

# Configuration des logs
logging.basicConfig(level=logging.INFO)

class TradingBot:
    def __init__(self):
        # Configuration de l'API Binance
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'options': {
                'adjustForTimeDifference': True,
                'enableRateLimit': True,
            },
        })
        self.exchange.set_sandbox_mode(True)
# Vérification du solde
try:
    balance = exchange.fetch_balance()
    print(f"Solde disponible : {balance['total']['USDT']} USDT")
except ccxt.AuthenticationError as e:
    print(f"Erreur d'authentification : {e}")
    print("Vérifiez votre clé API et votre secret.")
except ccxt.NetworkError as e:
    print(f"Erreur réseau : {e}")
except ccxt.ExchangeError as e:
    print(f"Erreur d'échange : {e}")
except Exception as e:
    print(f"Erreur inattendue : {e}")
        # Variables de trading
        self.balance = 0.0
        self.max_position_size = 0.1  # 10% du capital
        self.active_trades = []

        try:
            balance = self.exchange.fetch_balance()
            self.balance = balance['total'].get('USDT', 0.0)
            logging.info(f"Solde initial: {self.balance:.2f} USDT")
        except Exception as e:
            logging.error(f"Erreur initialisation: {str(e)}")

        # Configuration stratégie
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        self.timeframe = '1h'
        self.ltf_timeframe = '5m'
        self.asian_session_start = 16
        self.asian_session_end = 22
        self.risk_per_trade = 0.01
        self.max_simultaneous_trades = 1

    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calcule la taille de position sécurisée"""
        try:
            risk_amount = self.balance * self.risk_per_trade
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share <= 0:
                logging.warning("Risque par position invalide")
                return 0
                
            position_size = risk_amount / risk_per_share
            return round(min(position_size, self.max_position_size), 4)
        except Exception as e:
            logging.error(f"Erreur calcul position: {str(e)}")
            return 0

    def execute_trade(self, symbol, action):
        """Exécute un trade avec gestion de risque"""
        if len(self.active_trades) >= self.max_simultaneous_trades:
            return

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            stop_loss = current_price * 0.99 if action == 'buy' else current_price * 1.01
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            if position_size <= 0:
                return

            # Mode TEST/prod
            if os.getenv('ENVIRONMENT') == 'TEST':
                logging.info(f"SIMULATION {action} {position_size} {symbol}")
            else:
                self.exchange.create_market_order(symbol, action, position_size)

            self._log_trade(symbol, action, current_price, position_size, stop_loss)
            self.send_email(f"Trade {action} {symbol}", f"Taille: {position_size}\nPrix: {current_price}")

        except ccxt.InsufficientFunds:
            logging.error("Erreur: Fonds insuffisants")
        except Exception as e:
            logging.error(f"Erreur trade: {str(e)}")

    def _log_trade(self, symbol, action, price, size, stop_loss):
        """Journalise les trades"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'price': price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': price * 1.02 if action == 'buy' else price * 0.98
        }
        self.active_trades.append(trade)
        
        # Sauvegarde CSV
        with open('trades.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade['timestamp'],
                trade['symbol'],
                trade['action'],
                trade['price'],
                trade['size'],
                trade['stop_loss'],
                trade['take_profit']
            ])

    def send_email(self, subject, body):
        """Service de notifications"""
        try:
            msg = MIMEMultipart()
            msg['From'] = os.getenv('EMAIL_ADDRESS')
            msg['To'] = os.getenv('EMAIL_ADDRESS')
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(os.getenv('EMAIL_ADDRESS'), os.getenv('EMAIL_PASSWORD'))
            server.send_message(msg)
            server.quit()
        except Exception as e:
            logging.error(f"Erreur email: {str(e)}")

    def manage_risk(self):
        """Gestion des stops et profits"""
        for trade in list(self.active_trades):
            try:
                ticker = self.exchange.fetch_ticker(trade['symbol'])
                current_price = ticker['last']
                
                if (current_price <= trade['stop_loss'] or 
                    current_price >= trade['take_profit']):
                    self.active_trades.remove(trade)
            except Exception as e:
                logging.error(f"Erreur gestion risque: {str(e)}")

    def main(self):
        """Logique principale"""
        try:
            now = datetime.utcnow()
            if self.asian_session_start <= now.hour < self.asian_session_end:
                for symbol in self.symbols:
                    df = self.fetch_ohlcv(symbol, self.timeframe)
                    if df is not None:
                        ltf_df = self.fetch_ohlcv(symbol, self.ltf_timeframe, 50)
                        action = self.analyze_data(df, ltf_df)
                        if action in ['buy', 'sell']:
                            self.execute_trade(symbol, action)
                self.manage_risk()
        except Exception as e:
            logging.error(f"Erreur main: {str(e)}")

    def fetch_ohlcv(self, symbol, timeframe, limit=100):
        """Récupère les données de marché"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.set_index('timestamp')
        except Exception as e:
            logging.error(f"Erreur OHLCV: {str(e)}")
            return None

    def analyze_data(self, htf_df, ltf_df):
        """Analyse technique"""
        try:
            # Analyse timeframe supérieur
            htf_df['ema50'] = ta.ema(htf_df['close'], 50)
            htf_df['ema200'] = ta.ema(htf_df['close'], 200)
            
            # Analyse timeframe inférieur
            ltf_df['rsi'] = ta.rsi(ltf_df['close'], 14)
            macd = ta.macd(ltf_df['close'])
            
            # Logique de trading
            last_close = ltf_df['close'].iloc[-1]
            
            if last_close > htf_df['ema50'].iloc[-1] and macd['MACD_12_26_9'].iloc[-1] > 0:
                return 'buy'
            elif last_close < htf_df['ema200'].iloc[-1] and macd['MACD_12_26_9'].iloc[-1] < 0:
                return 'sell'
                
            return 'hold'
        except Exception as e:
            logging.error(f"Erreur analyse: {str(e)}")
            return 'hold'

# Serveur Flask
app = Flask(__name__)

@app.route('/')
def status():
    return "Trading Bot Operational 🌟"

def trading_loop():
    bot = TradingBot()
    while True:
        bot.main()
        time.sleep(300)

if __name__ == "__main__":
    threading.Thread(target=trading_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
