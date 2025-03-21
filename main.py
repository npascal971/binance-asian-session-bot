import ccxt
import time
import pandas as pd
import pandas_ta as ta
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Chargement des variables d'environnement
load_dotenv()

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log'), logging.StreamHandler()]
)

class AsianSessionTrader:
    def __init__(self):
        self.exchange = self.configure_exchange()
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        self.risk_per_trade = 0.02  # 2% du capital
        self.session_data = {}

        # Heures de session (en test 17h00 -> 17h30 UTC)
        self.asian_session = {
            'start': {'hour': 18, 'minute': 0},
            'end': {'hour': 18, 'minute': 30}
        }

        self.update_balance()
        logging.info(f"Configuration session : {self.asian_session}")
        logging.info(f"UTC maintenant : {datetime.utcnow()}")

    def configure_exchange(self):
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True
        })
        exchange.set_sandbox_mode(True)
        return exchange

    def update_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            self.balance = balance['total'].get('USDT', 0)
            logging.info(f"Solde actuel : {self.balance:.2f} USDT")
        except Exception as e:
            logging.error(f"Erreur de solde : {str(e)}")

    def analyze_session(self):
        """Analyse corrigée avec gestion d'erreurs renforcée"""
    try:
        for symbol in self.symbols:
            # Récupération des données avec vérification
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            if not ohlcv or len(ohlcv) < 34:  # 26 périodes pour MACD standard
                logging.warning(f"Données insuffisantes pour {symbol}")
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calcul MACD avec paramètres explicites
            macd = ta.macd(
                close=df['close'],
                fast=12,
                slow=26,
                signal=9
            )
            
            # Vérification complète des données MACD
            if macd is None or macd.empty:
                logging.warning(f"MACD vide pour {symbol}, utilisation de 0")
                macd_value = 0.0
            else:
                macd_value = macd.iloc[-1].get('MACD_12_26_9', 0.0)

            self.session_data[symbol] = {
                'high': df['high'].max(),
                'low': df['low'].min(),
                'vwap': (df['high'].mean() + df['low'].mean() + df['close'].mean()) / 3,
                'macd': macd_value
            }

        logging.info("✅ Analyse terminée avec succès")
        self.send_email("📈 Rapport Technique", self.generate_report())

    except Exception as e:
        logging.error(f"❌ Échec critique de l'analyse : {str(e)}")
        self.send_email("🚨 Erreur d'Analyse", f"Erreur détectée : {str(e)}")

    def execute_post_session_trades(self):
        """Exécute les trades après la session"""
        try:
            for symbol, data in self.session_data.items():
                current_price = self.get_current_price(symbol)

                if current_price is None:
                    continue

                if current_price > data['vwap'] and data['macd'] > 0:
                    self.place_order(symbol, 'buy', current_price)
                elif current_price < data['vwap'] and data['macd'] < 0:
                    self.place_order(symbol, 'sell', current_price)

            self.session_data = {}
            self.update_balance()

        except Exception as e:
            logging.error(f"Erreur d'exécution : {str(e)}")

    def run_cycle(self):
        """Gestion du cycle de trading"""
        while True:
            now = datetime.utcnow()
            start_time = datetime(now.year, now.month, now.day,
                                  self.asian_session['start']['hour'],
                                  self.asian_session['start']['minute'])
            end_time = datetime(now.year, now.month, now.day,
                                self.asian_session['end']['hour'],
                                self.asian_session['end']['minute'])

            if start_time <= now < end_time:
                if not self.session_data:
                    logging.info("🚀 Début de l'analyse...")
                    self.analyze_session()
            elif now >= end_time:
                if self.session_data:
                    logging.info("💡 Exécution des trades...")
                    self.execute_post_session_trades()

            time.sleep(60)

    def place_order(self, symbol, side, price):
        """Passer un ordre de marché"""
        try:
            position_size = (self.balance * self.risk_per_trade) / price
            if os.getenv('ENVIRONMENT') == 'LIVE':
                self.exchange.create_market_order(symbol, side, position_size)
                logging.info(f"Ordre exécuté : {side} {position_size:.4f} {symbol}")
            else:
                logging.info(f"SIMULATION : {side} {position_size:.4f} {symbol}")

            self.send_email(
                "🎯 Nouveau Trade",
                f"**{symbol}** | {side.upper()}\n"
                f"Montant: {position_size:.4f}\n"
                f"Prix: {price:.2f}"
            )

        except Exception as e:
            logging.error(f"Erreur d'ordre : {str(e)}")

    def get_current_price(self, symbol):
        """Obtenir le prix actuel"""
        try:
            return self.exchange.fetch_ticker(symbol)['last']
        except Exception as e:
            logging.error(f"Erreur de prix : {str(e)}")
            return None

    def get_session_start(self):
        """Calcule le timestamp de début de session"""
        return int(datetime(
            datetime.utcnow().year,
            datetime.utcnow().month,
            datetime.utcnow().day,
            self.asian_session['start']['hour'],
            self.asian_session['start']['minute']
        ).timestamp() * 1000)

    def generate_report(self):
        """Génère un rapport d'analyse"""
        report = "📈 **Rapport de Session**\n\n"
        for symbol, data in self.session_data.items():
            report += (
                f"**{symbol}**\n"
                f"- HIGH: {data['high']:.2f}\n"
                f"- LOW: {data['low']:.2f}\n"
                f"- VWAP: {data['vwap']:.2f}\n"
                f"- MACD: {data['macd']:.4f}\n\n"
            )
        return report

    def send_email(self, subject, body):
        """Envoi de notification par email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = os.getenv('EMAIL_ADDRESS')
            msg['To'] = os.getenv('EMAIL_ADDRESS')
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(os.getenv('EMAIL_ADDRESS'), os.getenv('EMAIL_PASSWORD'))
                server.send_message(msg)

        except Exception as e:
            logging.error(f"Erreur email : {str(e)}")

# === FLASK APP POUR MONITORING ===
app = Flask(__name__)

@app.route('/')
def status():
    return """
    <h1>Trading Bot Actif</h1>
    <p>Stratégie : Post-Session Asiatique</p>
    <p>🕒 Analyse à : 17h00 UTC</p>
    """

def run_bot():
    trader = AsianSessionTrader()
    trader.run_cycle()

if __name__ == '__main__':
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)
