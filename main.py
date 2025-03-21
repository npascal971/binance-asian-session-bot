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
            'urls': {
                'api': {
                    'public': 'https://testnet.binance.vision/api',
                    'private': 'https://testnet.binance.vision/api',
                },
            },
            'options': {
                'adjustForTimeDifference': True,
                'enableRateLimit': True,
            },
        })
        self.exchange.set_sandbox_mode(True)

        # Vérification du solde
        try:
            balance = self.exchange.fetch_balance()
            logging.info(f"Solde disponible : {balance['total']['USDT']} USDT")
        except ccxt.AuthenticationError as e:
            logging.error(f"Erreur d'authentification : {e}")
            logging.error("Vérifiez votre clé API et votre secret.")
        except ccxt.NetworkError as e:
            logging.error(f"Erreur réseau : {e}")
        except ccxt.ExchangeError as e:
            logging.error(f"Erreur d'échange : {e}")
        except Exception as e:
            logging.error(f"Erreur inattendue : {e}")

        # Configuration des symboles et paramètres
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'DOGE/USDT', 'DOT/USDT']
        self.timeframe = '1h'
        self.ltf_timeframe = '5m'
        self.asian_session_start = 16  # 16h UTC
        self.asian_session_end = 22    # 22h UTC
        self.risk_per_trade = 0.01
        self.max_simultaneous_trades = 1
        self.active_trades = []

    def send_email(self, subject, body):
        """Envoyer un e-mail de notification."""
        sender_email = os.getenv('EMAIL_ADDRESS')
        receiver_email = os.getenv('EMAIL_ADDRESS')
        password = os.getenv('EMAIL_PASSWORD')

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender_email, password)
            text = message.as_string()
            server.sendmail(sender_email, receiver_email, text)
            server.quit()
            logging.info("E-mail envoyé avec succès")
        except Exception as e:
            logging.error(f"Erreur lors de l'envoi de l'e-mail : {e}")

    def fetch_ohlcv(self, symbol, timeframe, limit=100):
        """Récupérer les données OHLCV pour un symbole donné."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                logging.warning(f"Aucune donnée OHLCV récupérée pour {symbol}")
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            logging.info(f"Données OHLCV pour {symbol} :\n{df}")
            return df
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des données OHLCV pour {symbol} : {e}")
            return None

    def get_asian_range(self, df):
        """Calculer la plage asiatique (high et low)."""
        asian_df = df.between_time(f'{self.asian_session_start}:00', f'{self.asian_session_end}:00')
        asian_high = asian_df['high'].max()
        asian_low = asian_df['low'].min()
        return asian_high, asian_low

    def identify_liquidity_zones(self, df, symbol):
        """Identifier les zones de liquidité."""
        if df.empty:
            logging.warning(f"Aucune donnée disponible pour {symbol}")
            return None
        liquidity_zones = {
            'highs': df['high'].tail(10).tolist(),
            'lows': df['low'].tail(10).tolist(),
        }
        return liquidity_zones

    def check_reversal_setup(self, ltf_df):
        """Vérifier les configurations de retournement."""
        if ltf_df.empty:
            logging.warning("Aucune donnée disponible pour l'analyse technique")
            return 'hold'
        
        ltf_df['rsi'] = ta.rsi(ltf_df['close'], length=14)
        macd = ta.macd(ltf_df['close'], fast=12, slow=26, signal=9)
        ltf_df['macd'] = macd['MACD_12_26_9']
        ltf_df['signal'] = macd['MACDs_12_26_9']

        last_close = ltf_df['close'].iloc[-1]
        prev_close = ltf_df['close'].iloc[-2]
        last_rsi = ltf_df['rsi'].iloc[-1]
        last_macd = ltf_df['macd'].iloc[-1]
        last_signal = ltf_df['signal'].iloc[-1]

        if last_close > prev_close and last_rsi < 45 and last_macd > last_signal:
            return 'buy'
        elif last_close < prev_close and last_rsi > 55 and last_macd < last_signal:
            return 'sell'
        return 'hold'

   def calculate_position_size(self, entry_price, stop_loss_price):
        try:
            balance = self.exchange.fetch_balance()['total']['USDT']
        except Exception as e:
            logging.error(f"Error fetching balance: {e}")
            return 0  # Or handle this error appropriately

        risk_amount = balance * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss_price)
        position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
        # Limit size to avoid over-leveraging
        return min(position_size, self.max_position_size)

    def log_trade(self, symbol, action, entry_price, size, stop_loss, take_profit):
        """Enregistrer les trades dans un fichier CSV."""
        file_exists = os.path.isfile('trades_log.csv')
        with open('trades_log.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Timestamp', 'Symbol', 'Action', 'Entry Price', 'Size', 'Stop Loss', 'Take Profit'])
            writer.writerow([pd.Timestamp.now(), symbol, action, entry_price, size, stop_loss, take_profit])

    def execute_trade(self, symbol, action, balance):
        """Exécuter un trade (achat ou vente)."""
        if len(self.active_trades) >= self.max_simultaneous_trades:
            logging.warning("Trade ignoré - limite de trades simultanés atteinte")
            return

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            stop_loss_price = current_price * (0.99 if action == 'buy' else 1.01)
            take_profit_price = current_price * (1.02 if action == 'buy' else 0.98)
            position_size = self.calculate_position_size(balance, current_price, stop_loss_price)

            if os.getenv('ENVIRONMENT') == 'TEST':
                logging.info(f"Mode TEST - Trade simulé : {action} {symbol}")
                return

            if action == 'buy':
                order = self.exchange.create_market_buy_order(symbol, position_size)
            else:
                order = self.exchange.create_market_sell_order(symbol, position_size)

            self.active_trades.append({
                'symbol': symbol,
                'action': action,
                'entry_price': current_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'size': position_size,
            })
            self.log_trade(symbol, action, current_price, position_size, stop_loss_price, take_profit_price)
            self.send_email(f"Trade exécuté - {symbol}", f"Action: {action}\nPrice: {current_price}")

        except Exception as e:
            logging.error(f"Erreur lors de l'exécution du trade : {e}")

    def manage_active_trades(self):
        """Gérer les trades actifs."""
        for trade in self.active_trades[:]:
            try:
                ticker = self.exchange.fetch_ticker(trade['symbol'])
                current_price = ticker['last']
                if (current_price <= trade['stop_loss'] or 
                    current_price >= trade['take_profit']):
                    self.active_trades.remove(trade)
            except Exception as e:
                logging.error(f"Erreur lors de la gestion du trade : {e}")

    def main(self):
        """Fonction principale du bot."""
        try:
            now = pd.Timestamp.now(tz='UTC')
            logging.info(f"Heure actuelle : {now}")
            
            if self.asian_session_start <= now.hour < self.asian_session_end:
                logging.info("Session asiatique active")
                for symbol in self.symbols:
                    htf_df = self.fetch_ohlcv(symbol, self.timeframe)
                    if htf_df is not None:
                        asian_high, asian_low = self.get_asian_range(htf_df)
                        ltf_df = self.fetch_ohlcv(symbol, self.ltf_timeframe, limit=50)
                        if ltf_df is not None:
                            action = self.check_reversal_setup(ltf_df)
                            if action in ['buy', 'sell']:
                                balance = self.exchange.fetch_balance()['total']['USDT']
                                self.execute_trade(symbol, action, balance)
                self.manage_active_trades()
            else:
                logging.info("Hors de la session asiatique")
                
        except Exception as e:
            logging.error(f"Erreur dans main : {e}")

# Configuration de Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Trading bot is running!"

def main_loop():
    bot = TradingBot()
    while True:
        try:
            bot.main()
            time.sleep(300)  # 5 minutes
        except Exception as e:
            logging.error(f"Erreur critique : {e}")
            time.sleep(60)

if __name__ == "__main__":
    thread = threading.Thread(target=main_loop)
    thread.daemon = True
    thread.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
