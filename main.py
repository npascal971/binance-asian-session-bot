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

    # ... (le reste du code reste inchangé)

# Configuration de Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Trading bot is running!"

def main_loop():
    bot = TradingBot()
    while True:
        try:
            logging.info("🔁 Début de l'itération")
            bot.main()
            logging.info("✅ Fin de l'itération")
        except Exception as e:
            logging.error(f"Erreur dans la boucle principale : {e}")
        time.sleep(60 * 5)

if __name__ == "__main__":
    thread = threading.Thread(target=main_loop)
    thread.daemon = True
    thread.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
