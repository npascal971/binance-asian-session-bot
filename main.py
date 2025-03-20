import ccxt
import time
import pandas as pd
import pandas_ta as ta
import csv
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Charger les variables d'environnement
load_dotenv()

# Configuration de l'API Binance
exchange = ccxt.binance({
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

exchange.set_sandbox_mode(True)

# Configuration des symboles et paramètres
symbols = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'DOGE/USDT', 'DOT/USDT'
]
timeframe = '1h'
ltf_timeframe = '5m'
asian_session_start = 22  # 22h UTC (début de la session asiatique)
asian_session_end = 6     # 6h UTC (fin de la session asiatique)
risk_per_trade = 0.01
max_simultaneous_trades = 1
active_trades = []

# Dictionnaire pour stocker les données de la session asiatique
asian_session_data = {symbol: {'high': None, 'low': None} for symbol in symbols}

# Fonction pour envoyer un e-mail
def send_email(subject, body):
    sender_email = os.getenv('EMAIL_ADDRESS')  # Votre adresse e-mail
    receiver_email = os.getenv('EMAIL_ADDRESS')  # Adresse du destinataire
    password = os.getenv('EMAIL_PASSWORD')  # Mot de passe de l'e-mail

    # Création du message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Ajout du corps du message
    message.attach(MIMEText(body, "plain"))

    try:
        # Connexion au serveur SMTP de Gmail
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Sécurisation de la connexion
        server.login(sender_email, password)  # Authentification
        text = message.as_string()  # Conversion du message en texte
        server.sendmail(sender_email, receiver_email, text)  # Envoi de l'e-mail
        server.quit()  # Fermeture de la connexion
        print("E-mail envoyé avec succès")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'e-mail : {e}")

# Fonction pour récupérer les données OHLCV
def fetch_ohlcv(symbol, timeframe, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
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

# Fonction pour enregistrer les hauts et les bas de la session asiatique
def record_asian_session_data(symbol, df):
    asian_df = df.between_time(f'{asian_session_start}:00', f'{asian_session_end}:00')
    if not asian_df.empty:
        asian_session_data[symbol]['high'] = asian_df['high'].max()
        asian_session_data[symbol]['low'] = asian_df['low'].min()
        print(f"{symbol} - Session asiatique enregistrée : High={asian_session_data[symbol]['high']}, Low={asian_session_data[symbol]['low']}")

# Fonction pour vérifier les configurations de retournement
print(f"{symbol} - Action détectée : {action}")
def check_reversal_setup(ltf_df, asian_high, asian_low):
    if ltf_df.empty:
        print("Aucune donnée disponible pour l'analyse technique")
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

    print(f"Dernières valeurs - Close: {last_close}, RSI: {last_rsi}, MACD: {last_macd}, Signal: {last_signal}")

    # Conditions de trading basées sur les niveaux de la session asiatique
    if last_close > asian_high and last_rsi < 45 and last_macd > last_signal:
        print(f"Signal d'achat détecté")
        return 'buy'
    elif last_close < asian_low and last_rsi > 55 and last_macd < last_signal:
        print(f"Signal de vente détecté")
        return 'sell'
    return 'hold'

# Fonction pour calculer la taille de position
def calculate_position_size(balance, entry_price, stop_loss_price):
    risk_amount = balance * risk_per_trade  # Combien tu risques sur le trade (1% ici)
    stop_loss_distance = abs(entry_price - stop_loss_price)
    if stop_loss_distance == 0:
        print("Erreur : stop_loss_distance = 0")
        return 0
    position_size = risk_amount / stop_loss_distance
    return round(position_size, 6)  # Tu peux ajuster le nombre de décimales selon l'asset


# Fonction pour exécuter un trade
def execute_trade(symbol, action, balance):
    global active_trades
    if len(active_trades) >= max_simultaneous_trades:
        print(f"Trade ignoré - max atteint")
        return
    ticker = exchange.fetch_ticker(symbol)
    current_price = ticker['last']
    stop_loss_price = current_price * (0.99 if action == 'buy' else 1.01)
    take_profit_price = current_price * (1.02 if action == 'buy' else 0.98)
    position_size = calculate_position_size(balance, current_price, stop_loss_price)
    try:
        if action == 'buy':
            order = exchange.create_market_buy_order(symbol, position_size)
        else:
            order = exchange.create_market_sell_order(symbol, position_size)
        print(f"Ordre exécuté : {order}")
        active_trades.append({
            'symbol': symbol,
            'action': action,
            'entry_price': current_price,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'size': position_size,
        })
        log_trade(symbol, action, current_price, position_size, stop_loss_price, take_profit_price)
        
        # Envoyer une notification par e-mail
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
        send_email(subject, body)
    except Exception as e:
        print(f"Erreur trade : {e}")

# Fonction pour gérer les trades actifs
def manage_active_trades():
    global active_trades
    for trade in active_trades[:]:  # Utilisez une copie de la liste pour éviter des modifications pendant l'itération
        symbol = trade['symbol']
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Vérifier les conditions de fermeture du trade
        if trade['action'] == 'buy':
            if current_price <= trade['stop_loss'] or current_price >= trade['take_profit']:
                print(f"Trade {symbol} fermé (achat)")
                active_trades.remove(trade)
        elif trade['action'] == 'sell':
            if current_price >= trade['stop_loss'] or current_price <= trade['take_profit']:
                print(f"Trade {symbol} fermé (vente)")
                active_trades.remove(trade)
execute_trade('BTC/USDT', 'buy', 100)

# Fonction principale
def main():
    try:
        now = pd.Timestamp.now(tz='UTC')
        print(f"Heure actuelle : {now}")
        
        # Si nous sommes pendant la session asiatique, enregistrez les données
        if asian_session_start <= now.hour < asian_session_end:
            print("Session asiatique en cours - Enregistrement des données...")
            for symbol in symbols:
                htf_df = fetch_ohlcv(symbol, timeframe)
                if htf_df is not None:
                    record_asian_session_data(symbol, htf_df)
        
        # Si nous sommes après la session asiatique, analysez et prenez des trades
elif now.hour >= asian_session_end:
    print("Session asiatique terminée - Analyse des données...")
    for symbol in symbols:
        asian_high = asian_session_data[symbol]['high']
        asian_low = asian_session_data[symbol]['low']
        print(f"{symbol} - Asian high: {asian_high}, Asian low: {asian_low}")
        if asian_high is not None and asian_low is not None:
            ltf_df = fetch_ohlcv(symbol, ltf_timeframe, limit=50)
            if ltf_df is not None:
                action = check_reversal_setup(ltf_df, asian_high, asian_low)
                print(f"Action pour {symbol} : {action}")
                if action in ['buy', 'sell']:
                    balance = exchange.fetch_balance()['total']['USDT']
                    execute_trade(symbol, action, balance)

        
        # Gérer les trades actifs
        manage_active_trades()
    except Exception as e:
        print(f"Erreur dans main: {e}")
    return "Script exécuté avec succès"

# Configuration de Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Trading bot is running!"

# Boucle principale
def main_loop():
    while True:
        print("Début de l'itération")
        main()
        print("Fin de l'itération")
        time.sleep(60 * 5)  # Attendre 5 minutes avant la prochaine itération

# Démarrer le bot
if __name__ == "__main__":
    thread = threading.Thread(target=main_loop)
    thread.daemon = True
    thread.start()

    app.run(host="0.0.0.0", port=10000, debug=True)
