from flask import Flask
import threading
import time
import ccxt
import pandas as pd
import pandas_ta as ta
import csv
import os
from datetime import datetime
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Initialisation de Flask
app = Flask(__name__)

# Configuration de l'exchange Binance Testnet
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

# Paramètres
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'DOGE/USDT', 'DOT/USDT']
timeframe = '1h'
ltf_timeframe = '5m'
asian_session_start = 18
asian_session_end = 6
risk_per_trade = 0.01
max_simultaneous_trades = 1
active_trades = []

# Fonctions
def fetch_ohlcv(symbol, timeframe, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        app.logger.error(f"Erreur OHLCV {symbol} : {e}")
        return None

def get_asian_range(df):
    asian_df = df.between_time(f'{asian_session_start}:00', f'{asian_session_end}:00')
    return asian_df['high'].max(), asian_df['low'].min()

def identify_liquidity_zones(df, symbol):
    return {'highs': df['high'].tail(10).tolist(), 'lows': df['low'].tail(10).tolist()}

def check_reversal_setup(ltf_df):
    ltf_df['rsi'] = ta.rsi(ltf_df['close'], length=14)
    macd = ta.macd(ltf_df['close'], fast=12, slow=26, signal=9)
    ltf_df['macd'] = macd['MACD_12_26_9']
    ltf_df['signal'] = macd['MACDs_12_26_9']
    last_close = ltf_df['close'].iloc[-1]
    prev_close = ltf_df['close'].iloc[-2]
    last_rsi = ltf_df['rsi'].iloc[-1]
    last_macd = ltf_df['macd'].iloc[-1]
    last_signal = ltf_df['signal'].iloc[-1]
    if last_close > prev_close and last_rsi < 40 and last_macd > last_signal:
        return 'buy'
    elif last_close < prev_close and last_rsi > 60 and last_macd < last_signal:
        return 'sell'
    return 'hold'

def calculate_position_size(balance, entry_price, stop_loss_price):
    risk_amount = balance * risk_per_trade
    distance = abs(entry_price - stop_loss_price)
    return risk_amount / distance if distance > 0 else 0

def log_trade(symbol, action, entry_price, size, stop_loss, take_profit):
    file_exists = os.path.isfile('trades_log.csv')
    with open('trades_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Symbol', 'Action', 'Entry Price', 'Size', 'Stop Loss', 'Take Profit'])
        writer.writerow([pd.Timestamp.now(), symbol, action, entry_price, size, stop_loss, take_profit])

def execute_trade(symbol, action, balance):
    global active_trades
    if len(active_trades) >= max_simultaneous_trades:
        app.logger.info("Limite de trades atteinte")
        return
    ticker = exchange.fetch_ticker(symbol)
    current_price = ticker['last']
    stop_loss_price = current_price * (0.99 if action == 'buy' else 1.01)
    take_profit_price = current_price * (1.02 if action == 'buy' else 0.98)
    position_size = calculate_position_size(balance, current_price, stop_loss_price)
    try:
        if action == 'buy':
            exchange.create_market_buy_order(symbol, position_size)
        else:
            exchange.create_market_sell_order(symbol, position_size)
        active_trades.append({
            'symbol': symbol,
            'action': action,
            'entry_price': current_price,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'size': position_size,
        })
        log_trade(symbol, action, current_price, position_size, stop_loss_price, take_profit_price)
        app.logger.info(f"Trade exécuté : {symbol} {action} {position_size} @ {current_price}")
    except Exception as e:
        app.logger.error(f"Erreur trade : {e}")

def manage_active_trades():
    global active_trades
    for trade in active_trades[:]:
        symbol = trade['symbol']
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        if trade['action'] == 'buy':
            if current_price <= trade['stop_loss'] or current_price >= trade['take_profit']:
                active_trades.remove(trade)
                app.logger.info(f"Trade fermé (BUY) : {symbol}")
        elif trade['action'] == 'sell':
            if current_price >= trade['stop_loss'] or current_price <= trade['take_profit']:
                active_trades.remove(trade)
                app.logger.info(f"Trade fermé (SELL) : {symbol}")

def main():
    logs = []
    try:
        now = pd.Timestamp.now(tz='UTC')
        app.logger.info(f"Heure actuelle : {now}")
        logs.append(f"Heure actuelle : {now}")

        if now.hour >= asian_session_start or now.hour < asian_session_end:
            logs.append("Session asiatique en cours")
            app.logger.info("Session asiatique en cours")
            for symbol in symbols:
                logs.append(f"Traitement du symbole : {symbol}")
                app.logger.info(f"Traitement du symbole : {symbol}")

                htf_df = fetch_ohlcv(symbol, timeframe)
                if htf_df is not None:
                    asian_high, asian_low = get_asian_range(htf_df)
                    logs.append(f"{symbol} - Asian High: {asian_high}, Asian Low: {asian_low}")
                    app.logger.info(f"{symbol} - Asian High: {asian_high}, Asian Low: {asian_low}")

                    liquidity_zones = identify_liquidity_zones(htf_df, symbol)
                    logs.append(f"{symbol} - Liquidity Zones: {liquidity_zones}")
                    app.logger.info(f"{symbol} - Liquidity Zones: {liquidity_zones}")

                    ltf_df = fetch_ohlcv(symbol, ltf_timeframe, limit=50)
                    if ltf_df is not None:
                        action = check_reversal_setup(ltf_df)
                        logs.append(f"{symbol} - Action recommandée : {action}")
                        app.logger.info(f"{symbol} - Action recommandée : {action}")
                        if action in ['buy', 'sell']:
                            balance = exchange.fetch_balance()['total']['USDT']
                            logs.append(f"{symbol} - Solde disponible : {balance}")
                            app.logger.info(f"{symbol} - Solde disponible : {balance}")
                            execute_trade(symbol, action, balance)
                        else:
                            logs.append(f"{symbol} - Aucun setup valide")
                            app.logger.info(f"{symbol} - Aucun setup valide")
                    else:
                        logs.append(f"{symbol} - Données LTF indisponibles")
                        app.logger.info(f"{symbol} - Données LTF indisponibles")
                else:
                    logs.append(f"{symbol} - Données HTF indisponibles")
                    app.logger.info(f"{symbol} - Données HTF indisponibles")
        else:
            logs.append("En dehors de la session asiatique")
            app.logger.info("En dehors de la session asiatique")

        manage_active_trades()
    except Exception as e:
        logs.append(f"Erreur main: {e}")
        app.logger.error(f"Erreur main: {e}")
    return logs

# Boucle de fond
def main_loop():
    while True:
        main()
        time.sleep(300)

@app.route('/')
def home():
    logs = main()
    return "<br>".join(logs)

if __name__ == '__main__':
    thread = threading.Thread(target=main_loop)
    thread.daemon = True
    thread.start()

    app.run(host='0.0.0.0', port=10000)
