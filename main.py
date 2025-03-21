import ccxt
import time
import pandas as pd
import pandas_ta as ta
import os
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Charger les variables d'environnement
load_dotenv()

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

        # Configuration des symboles et paramètres
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'DOGE/USDT', 'DOT/USDT']
        self.timeframe = '1h'
        self.ltf_timeframe = '5m'
        self.asian_session_start = 1  # 00h UTC (modifié pour tester)
        self.asian_session_end = 2   # 23h UTC (modifié pour tester)
        self.risk_per_trade = 0.01
        self.max_simultaneous_trades = 1
        self.active_trades = []
        self.asian_session_data = {symbol: {'high': None, 'low': None} for symbol in self.symbols}

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
            print("E-mail envoyé avec succès")
        except Exception as e:
            print(f"Erreur lors de l'envoi de l'e-mail : {e}")

    def fetch_ohlcv(self, symbol, timeframe, limit=100):
        """Récupérer les données OHLCV pour un symbole donné."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                print(f"Aucune donnée OHLCV récupérée pour {symbol}")
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            print(f"Données OHLCV pour {symbol} :\n{df.tail()}")
            return df
        except Exception as e:
            print(f"Erreur lors de la récupération des données OHLCV pour {symbol} : {e}")
            return None

    def record_asian_session_data(self, symbol, df):
        """Enregistrer les hauts et les bas de la session asiatique."""
        asian_df = df.between_time(f'{self.asian_session_start}:00', f'{self.asian_session_end}:00')
        if not asian_df.empty:
            self.asian_session_data[symbol]['high'] = asian_df['high'].max()
            self.asian_session_data[symbol]['low'] = asian_df['low'].min()
            print(f"{symbol} - Session asiatique enregistrée : High={self.asian_session_data[symbol]['high']}, Low={self.asian_session_data[symbol]['low']}")
        else:
            print(f"{symbol} - Aucune donnée trouvée pour la session asiatique (asian_df est vide)")

    def check_reversal_setup(self, ltf_df, asian_high, asian_low):
        """Vérifier les configurations de retournement."""
        if ltf_df.empty:
            print("Aucune donnée disponible pour l'analyse technique")
            return 'hold'
        
        ltf_df['rsi'] = ta.rsi(ltf_df['close'], length=14)
        macd = ta.macd(ltf_df['close'], fast=12, slow=26, signal=9)
        ltf_df['macd'] = macd['MACD_12_26_9']
        ltf_df['signal'] = macd['MACDs_12_26_9']

        last_close = ltf_df['close'].iloc[-1]
        last_rsi = ltf_df['rsi'].iloc[-1]
        last_macd = ltf_df['macd'].iloc[-1]
        last_signal = ltf_df['signal'].iloc[-1]

        print(f"Dernières valeurs - Close: {last_close}, RSI: {last_rsi}, MACD: {last_macd}, Signal: {last_signal}")

        # Conditions de trading basées sur les niveaux de la session asiatique
        if last_close > asian_high and last_rsi < 45 and last_macd > last_signal:
            print(f"Signal d'achat détecté")
            return 'buy'
        elif last_close < asian_low and last_rsi > 55 and last_macd < last_signal:
            print(f"Signal de vente détecté")
            return 'sell'
        return 'hold'

    def calculate_position_size(self, balance, entry_price, stop_loss_price):
        """Calculer la taille de la position en fonction du risque."""
        risk_amount = balance * self.risk_per_trade
        stop_loss_distance = abs(entry_price - stop_loss_price)
        if stop_loss_distance == 0:
            print("Erreur : stop_loss_distance = 0")
            return 0
        position_size = risk_amount / stop_loss_distance
        return round(position_size, 6)

    def execute_trade(self, symbol, action, balance):
        """Exécuter un trade (achat ou vente)."""
        if len(self.active_trades) >= self.max_simultaneous_trades:
            print(f"Trade ignoré - max atteint")
            return

        # Vérifier le solde avant d'exécuter un trade
        if balance <= 0:
            print(f"Solde insuffisant pour exécuter un trade. Solde actuel : {balance} USDT")
            return

        ticker = self.exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        stop_loss_price = current_price * (0.99 if action == 'buy' else 1.01)
        take_profit_price = current_price * (1.02 if action == 'buy' else 0.98)
        position_size = self.calculate_position_size(balance, current_price, stop_loss_price)

        if os.getenv('ENVIRONMENT') == 'TEST':
            print(f"Mode TEST - Trade simulé : {action} {symbol} avec un solde de {balance} USDT")
            print(f"Details: Entry Price={current_price}, Size={position_size}, Stop Loss={stop_loss_price}, Take Profit={take_profit_price}")
            return

        try:
            if action == 'buy':
                order = self.exchange.create_market_buy_order(symbol, position_size)
            else:
                order = self.exchange.create_market_sell_order(symbol, position_size)
            print(f"Ordre exécuté : {order}")
            self.active_trades.append({
                'symbol': symbol,
                'action': action,
                'entry_price': current_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'size': position_size,
            })
            subject = f"Trade exécuté - {symbol}"
            body = f"""
            Un trade a été exécuté :
            - Symbol: {symbol}
            - Action: {action}
            - Entry Price: {current_price}
            - Size: {position_size}
            - Stop Loss: {stop_loss_price}
            - Take Profit: {take_profit_price}
            """
            self.send_email(subject, body)
        except Exception as e:
            print(f"Erreur trade : {e}")

    def manage_active_trades(self):
        """Gérer les trades actifs (fermeture si conditions remplies)."""
        for trade in self.active_trades[:]:
            symbol = trade['symbol']
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            if trade['action'] == 'buy':
                if current_price <= trade['stop_loss'] or current_price >= trade['take_profit']:
                    print(f"Trade {symbol} fermé (achat)")
                    self.active_trades.remove(trade)
            elif trade['action'] == 'sell':
                if current_price >= trade['stop_loss'] or current_price <= trade['take_profit']:
                    print(f"Trade {symbol} fermé (vente)")
                    self.active_trades.remove(trade)

    def main(self):
        """Fonction principale du bot."""
        try:
            now = pd.Timestamp.now(tz='UTC')
            print(f"🕐 Heure actuelle : {now}")

            # 🔍 Récupérer et afficher le solde du compte test
            balance = self.exchange.fetch_balance()['total']['USDT']
            print(f"💰 Solde du compte test : {balance:.2f} USDT")

            # ✅ Si session asiatique en cours
            if self.asian_session_start <= now.hour or now.hour < self.asian_session_end:
                print("🌏 Session asiatique en cours - Enregistrement des données...")
                for symbol in self.symbols:
                    htf_df = self.fetch_ohlcv(symbol, self.timeframe)
                    if htf_df is not None:
                        self.record_asian_session_data(symbol, htf_df)

            # ✅ Après session asiatique, on cherche les signaux
            elif now.hour >= self.asian_session_end:
                print("✅ Session asiatique terminée - Analyse des données...")
                for symbol in self.symbols:
                    asian_high = self.asian_session_data[symbol]['high']
                    asian_low = self.asian_session_data[symbol]['low']
                    print(f"{symbol} - 📊 High: {asian_high}, Low: {asian_low}")

                    if asian_high is not None and asian_low is not None:
                        ltf_df = self.fetch_ohlcv(symbol, self.ltf_timeframe, limit=50)
                        if ltf_df is not None:
                            action = self.check_reversal_setup(ltf_df, asian_high, asian_low)
                            print(f"{symbol} - ⚡ Signal détecté : {action}")

                            if action == 'hold':
                                print(f"{symbol} - ⏸ Aucun trade car conditions RSI/MACD non remplies ou prix pas en dehors des bornes asiatiques.")
                            elif action in ['buy', 'sell']:
                                self.execute_trade(symbol, action, balance)
                        else:
                            print(f"{symbol} - ⚠️ Données LTF non disponibles")
                    else:
                        print(f"{symbol} - ⚠️ Données asiatiques non valides ou non encore enregistrées")

            # 🔁 Gérer les trades ouverts
            self.manage_active_trades()

        except Exception as e:
            print(f"❌ Erreur dans main: {e}")

        return "✅ Script exécuté avec succès"


# Configuration de Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Trading bot is running!"

def main_loop():
    bot = TradingBot()
    while True:
        try:
            print("🔁 Début de l'itération")
            bot.main()
            print("✅ Fin de l'itération")
        except Exception as e:
            print(f"Erreur dans la boucle principale : {e}")
        time.sleep(60 * 5)

if __name__ == "__main__":
    thread = threading.Thread(target=main_loop)
    thread.daemon = True
    thread.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
