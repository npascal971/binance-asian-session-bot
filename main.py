
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
asian_session_start = 0
asian_session_end = 23
risk_per_trade = 0.01
max_simultaneous_trades = 1
active_trades = []
asian_session_data = {symbol: {'high': None, 'low': None} for symbol in symbols}

def send_email(subject, body):
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

def fetch_ohlcv(symbol, timeframe, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            print(f"Aucune donnée OHLCV récupérée pour {symbol}")
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Erreur lors de la récupération des données OHLCV pour {symbol} : {e}")
        return None

def record_asian_session_data(symbol, df):
    asian_df = df.between_time(f'{asian_session_start}:00', f'{asian_session_end}:00')
    if not asian_df.empty:
        asian_session_data[symbol]['high'] = asian_df['high'].max()
        asian_session_data[symbol]['low'] = asian_df['low'].min()
        print(f"{symbol} - Session asiatique enregistrée : High={asian_session_data[symbol]['high']}, Low={asian_session_data[symbol]['low']}")

def check_reversal_setup(ltf_df, asian_high, asian_low):
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

    if last_close > asian_high and last_rsi < 45 and last_macd > last_signal:
        return 'buy'
    elif last_close < asian_low and last_rsi > 55 and last_macd < last_signal:
        return 'sell'
    return 'hold'

def calculate_position_size(balance, entry_price, stop_loss_price):
    risk_amount = balance * risk_per_trade
    stop_loss_distance = abs(entry_price - stop_loss_price)
    if stop_loss_distance == 0:
        print("Erreur : stop_loss_distance = 0")
        return 0
    position_size = risk_amount / stop_loss_distance
    return round(position_size, 6)

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

    if os.getenv('ENVIRONMENT') == 'TEST':
        print(f"Mode TEST - Trade simulé : {action} {symbol} avec un solde de {balance} USDT")
        print(f"Details: Entry Price={current_price}, Size={position_size}, Stop Loss={stop_loss_price}, Take Profit={take_profit_price}")
        return

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

def manage_active_trades():
    global active_trades
    for trade in active_trades[:]:
        symbol = trade['symbol']
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        if trade['action'] == 'buy':
            if current_price <= trade['stop_loss'] or current_price >= trade['take_profit']:
                print(f"Trade {symbol} fermé (achat)")
                active_trades.remove(trade)
        elif trade['action'] == 'sell':
            if current_price >= trade['stop_loss'] or current_price <= trade['take_profit']:
                print(f"Trade {symbol} fermé (vente)")
                active_trades.remove(trade)

def main():
    try:
        now = pd.Timestamp.now(tz='UTC')
        print(f"Heure actuelle : {now}")
        balance = exchange.fetch_balance()['total']['USDT']
        print(f"💰 Solde du compte test : {balance:.2f} USDT")

        if asian_session_start <= now.hour or now.hour < asian_session_end:
            print("🌏 Session asiatique en cours - Enregistrement des données...")
            for symbol in symbols:
                htf_df = fetch_ohlcv(symbol, timeframe)
                if htf_df is not None:
                    record_asian_session_data(symbol, htf_df)

        elif now.hour >= asian_session_end:
            print("✅ Session asiatique terminée - Analyse des données...")
            for symbol in symbols:
                asian_high = asian_session_data[symbol]['high']
                asian_low = asian_session_data[symbol]['low']
                print(f"{symbol} - 📊 High: {asian_high}, Low: {asian_low}")
                if asian_high is not None and asian_low is not None:
                    ltf_df = fetch_ohlcv(symbol, ltf_timeframe, limit=50)
                    if ltf_df is not None:
                        action = check_reversal_setup(ltf_df, asian_high, asian_low)
                        print(f"{symbol} - ⚡ Signal détecté : {action}")
                        if action in ['buy', 'sell']:
                            execute_trade(symbol, action, balance)
        manage_active_trades()
    except Exception as e:
        print(f"❌ Erreur dans main: {e}")
    return "Script exécuté avec succès"

app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Trading bot is running!"

def main_loop():
    while True:
        try:
            print("🔁 Début de l'itération")
            main()
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
