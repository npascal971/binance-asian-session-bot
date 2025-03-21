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
        # Configuration API Binance
        self.exchange = self.configure_exchange()
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        self.risk_per_trade = 0.02  # 2% du capital par trade
        self.session_data = {}
        
        # Heures UTC
        self.asian_hours = {
            'start': 15,  # 16h UTC
            'end': 15.5     # 22h UTC
        }
        
        self.update_balance()
def run_cycle(self):
        """Exécute le cycle complet"""
        while True:
            now = datetime.utcnow()
            
            # Pendant la session
            if self.asian_hours['start'] <= now.hour < self.asian_hours['end']:
                if not self.session_data:
                    self.analyze_session()
                    
            # Après la session
            else:
                if self.session_data:
                    self.execute_post_session_trades()
            
            time.sleep(60 * 15)
    def configure_exchange(self):
        """Configure l'API d'échange"""
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True
        })
        exchange.set_sandbox_mode(True)
        return exchange

    def update_balance(self):
        """Met à jour le solde du portefeuille"""
        try:
            balance = self.exchange.fetch_balance()
            self.balance = balance['total'].get('USDT', 0)
            logging.info(f"Solde actuel : {self.balance:.2f} USDT")
        except Exception as e:
            logging.error(f"Erreur de solde : {str(e)}")

    def analyze_session(self):
        """Analyse la session asiatique en cours"""
        try:
            for symbol in self.symbols:
                # Récupération des données
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=self.get_session_start())
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Calcul des indicateurs
                df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
                macd = ta.macd(df['close'])
                
                # Enregistrement des données
                self.session_data[symbol] = {
                    'high': df['high'].max(),
                    'low': df['low'].min(),
                    'vwap': df['vwap'].mean(),
                    'macd': macd.iloc[-1]['MACD_12_26_9']
                }
                
            logging.info("Analyse de session terminée")
            self.send_email("Rapport de Session", self.generate_report())
            
        except Exception as e:
            logging.error(f"Erreur d'analyse : {str(e)}")

    def execute_post_session_trades(self):
        """Exécute les trades après la session"""
        try:
            for symbol, data in self.session_data.items():
                current_price = self.get_current_price(symbol)
                
                # Logique de trading
                if current_price > data['vwap'] and data['macd'] > 0:
                    self.place_order(symbol, 'buy', current_price)
                elif current_price < data['vwap'] and data['macd'] < 0:
                    self.place_order(symbol, 'sell', current_price)
                    
            self.session_data = {}  # Réinitialisation
            self.update_balance()
            
        except Exception as e:
            logging.error(f"Erreur d'exécution : {str(e)}")

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
                "Nouveau Trade", 
                f"{symbol} | {side.upper()}\n"
                f"Montant: {position_size:.4f}\n"
                f"Prix: {price:.2f}"
            )
            
        except Exception as e:
            logging.error(f"Erreur d'ordre : {str(e)}")

    def get_current_price(self, symbol):
        """Obtenir le prix actuel"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logging.error(f"Erreur de prix : {str(e)}")
            return None

    def get_session_start(self):
        """Calcule le timestamp de début de session"""
        now = datetime.utcnow()
        session_start = datetime(now.year, now.month, now.day, self.asian_hours['start'])
        return int(session_start.timestamp() * 1000)

    def generate_report(self):
        """Génère un rapport d'analyse"""
        report = "📊 Rapport de Session Asiatique\n\n"
        for symbol, data in self.session_data.items():
            report += (
                f"🔹 {symbol}\n"
                f"HIGH: {data['high']:.2f}\n"
                f"LOW: {data['low']:.2f}\n"
                f"VWAP: {data['vwap']:.2f}\n"
                f"MACD: {data['macd']:.4f}\n\n"
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



# Configuration Flask
app = Flask(__name__)

@app.route('/')
def status():
    return """
    <h1>Trading Bot Actif</h1>
    <p>Stratégie : Post-Session Asiatique</p>
    <p>🕒 Prochaine analyse : 16h00 UTC</p>
    """

def run_bot():
    trader = AsianSessionTrader()
    trader.run_cycle()

if __name__ == '__main__':
    # Lancement du bot dans un thread séparé
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Configuration du serveur Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)
