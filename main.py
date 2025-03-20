import ccxt
import time
import pandas as pd
import pandas_ta as ta
import csv
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask
from threading import Thread

# Charger les variables d'environnement
load_dotenv()

# Initialiser l'API Binance testnet
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

try:
    balance = exchange.fetch_balance()
    print(f"Solde disponible : {balance['total']['USDT']} USDT")
except Exception as e:
    print(f"Erreur : {e}")

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
            print(f"Aucune donnée OHLCV récupérée pour {symbol}")
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Erreur lors de la récupération OHLCV de {symbol} : {e}")
        return None

def get_asian_range(df):
    asian_df = df.between_time(f'{asian_session_start}:00', f'{asian_session_end}:00')
    asian_high = asian_df['high'].max()
    asian_low = asian_df['low'].min()
    return asian_high, asian_low

def identify_liquidity_zones(df, symbol):
    liquidity_zones = {
        'highs': df['high'].tail(10).tolist(),
        'lows': df['low'].tail(10).tolist(),
    }
    return liquidity_zones

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
    else:
        return 'hold'

def calculate_position_size(balance, entry_price, stop_loss_price):
    risk_amount = balance * risk_per_trade
    distance = abs(entry_price - stop_loss_price)
    return risk_amount / distance if distance != 0 else 0

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
        print(f"Max trades simultanés atteints.")
        return

    ticker = exchange.fetch_ticker(symbol)
    current_price = ticker['last']
    stop_loss_pct = 0.01
    take_profit_pct = 0.02

    stop_loss = current_price * (1 - stop_loss_pct) if action == 'buy' else current_price * (1 + stop_loss_pct)
    take_profit = current_price * (1 + take_profit_pct) if action == 'buy' else current_price * (1 - take_profit_pct)
    size = calculate_position_size(balance, current_price, stop_loss)

    try:
        if action == 'buy':
            order = exchange.create_market_buy_order(symbol, size)
        elif action == 'sell':
            order = exchange.create_market_sell_order(symbol, size)
        else:
            return

        active_trades.append({
            'symbol': symbol,
            'action': action,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': size,
        })
        log_trade(symbol, action, current_price, size, stop_loss, take_profit)
        print(f"{action.capitalize()} exécuté pour {symbol}")

    except Exception as e:
        print(f"Erreur lors du trade : {e}")

def manage_active_trades():
    global active_trades
    for trade in active_trades[:]:
        symbol = trade['symbol']
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']

        if trade['action'] == 'buy' and (price <= trade['stop_loss'] or price >= trade['take_profit']):
            print(f"Trade BUY clôturé {symbol} à {price}")
            active_trades.remove(trade)

        elif trade['action'] == 'sell' and (price >= trade['stop_loss'] or price <= trade['take_profit']):
            print(f"Trade SELL clôturé {symbol} à {price}")
            active_trades.remove(trade)

def main():
    try:
        now = pd.Timestamp.now(tz='UTC')
        if now.hour >= asian_session_start or now.hour < asian_session_end:
            for symbol in symbols:
                htf_df = fetch_ohlcv(symbol, timeframe)
                if htf_df is not None:
                    asian_high, asian_low = get_asian_range(htf_df)
                    identify_liquidity_zones(htf_df, symbol)
                    ltf_df = fetch_ohlcv(symbol, ltf_timeframe, limit=50)
                    if ltf_df is not None:
                        action = check_reversal_setup(ltf_df)
                        if action in ['buy', 'sell']:
                            balance = exchange.fetch_balance()['total']['USDT']
                            execute_trade(symbol, action, balance)
        manage_active_trades()
    except Exception as e:
        print(f"Erreur dans main : {e}")

# === FLASK SERVER POUR RENDER ===
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot crypto is running 🚀"

def run_flask():
    app.run(host='0.0.0.0', port=10000)

if __name__ == "__main__":
    Thread(target=run_flask).start()

    while True:
        main()
        time.sleep(60 * 5)
