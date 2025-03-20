import ccxt
import time
import pandas as pd
import pandas_ta as ta
import csv
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

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
symbols = [
    'BTC/USDT',  # Bitcoin contre USDT
    'ETH/USDT',  # Ethereum contre USDT
    'BNB/USDT',  # Binance Coin contre USDT
    'ADA/USDT',  # Cardano contre USDT
    'SOL/USDT',  # Solana contre USDT
    'DOGE/USDT', # Dogecoin contre USDT
    'DOT/USDT'   # Polkadot contre USDT
]
timeframe = '1h'
ltf_timeframe = '5m'
asian_session_start = 18  # 18h (6 PM)
asian_session_end = 6     # 6h (6 AM)
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
        print(f"Données OHLCV pour {symbol} :\n{df.tail()}")
        return df
    except ccxt.NetworkError as e:
        print(f"Erreur réseau lors de la récupération des données OHLCV pour {symbol} : {e}")
        return None
    except ccxt.ExchangeError as e:
        print(f"Erreur d'échange lors de la récupération des données OHLCV pour {symbol} : {e}")
        return None
    except Exception as e:
        print(f"Erreur inattendue lors de la récupération des données OHLCV pour {symbol} : {e}")
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
    print(f"Zones de liquidité pour {symbol} : {liquidity_zones}")
    return liquidity_zones

def check_reversal_setup(ltf_df):
    # Calcul du RSI et du MACD avec pandas_ta
    ltf_df['rsi'] = ta.rsi(ltf_df['close'], length=14)
    macd = ta.macd(ltf_df['close'], fast=12, slow=26, signal=9)
    ltf_df['macd'] = macd['MACD_12_26_9']
    ltf_df['signal'] = macd['MACDs_12_26_9']
    
    # Dernières valeurs
    last_close = ltf_df['close'].iloc[-1]
    prev_close = ltf_df['close'].iloc[-2]
    last_rsi = ltf_df['rsi'].iloc[-1]
    last_macd = ltf_df['macd'].iloc[-1]
    last_signal = ltf_df['signal'].iloc[-1]
    
    print(f"Dernières valeurs - Close: {last_close}, RSI: {last_rsi}, MACD: {last_macd}, Signal: {last_signal}")
    
    # Conditions de retournement
    if last_close > prev_close and last_rsi < 40 and last_macd > last_signal:
        print(f"Signal d'achat détecté pour {symbol}")
        return 'buy'
    elif last_close < prev_close and last_rsi > 60 and last_macd < last_signal:
        print(f"Signal de vente détecté pour {symbol}")
        return 'sell'
    else:
        print("Aucun signal de trading détecté")
        return 'hold'

def calculate_position_size(balance, entry_price, stop_loss_price):
    risk_amount = balance * risk_per_trade
    distance = abs(entry_price - stop_loss_price)
    return risk_amount / distance

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
        print(f"Trade non exécuté : nombre maximum de trades simultanés atteint ({max_simultaneous_trades}).")
        return
    
    ticker = exchange.fetch_ticker(symbol)
    current_price = ticker['last']
    
    stop_loss_percentage = 0.01
    take_profit_percentage = 0.02
    stop_loss_price = current_price * (1 - stop_loss_percentage) if action == 'buy' else current_price * (1 + stop_loss_percentage)
    take_profit_price = current_price * (1 + take_profit_percentage) if action == 'buy' else current_price * (1 - take_profit_percentage)
    
    position_size = calculate_position_size(balance, current_price, stop_loss_price)
    
    print(f"Tentative d'exécution d'un trade {action} sur {symbol} avec un solde de {balance} USDT")
    
    try:
        if action == 'buy':
            order = exchange.create_market_buy_order(symbol, position_size)
            print(f"Achat effectué sur {symbol} : {order}")
        elif action == 'sell':
            order = exchange.create_market_sell_order(symbol, position_size)
            print(f"Vente effectuée sur {symbol} : {order}")
        
        active_trades.append({
            'symbol': symbol,
            'action': action,
            'entry_price': current_price,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'size': position_size,
        })
        
        log_trade(symbol, action, current_price, position_size, stop_loss_price, take_profit_price)
    except Exception as e:
        print(f"Erreur lors de l'exécution du trade : {e}")

def manage_active_trades():
    global active_trades
    for trade in active_trades[:]:
        symbol = trade['symbol']
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        if trade['action'] == 'buy':
            if current_price <= trade['stop_loss'] or current_price >= trade['take_profit']:
                print(f"Fermeture du trade {trade['action']} sur {symbol} à {current_price}")
                active_trades.remove(trade)
        elif trade['action'] == 'sell':
            if current_price >= trade['stop_loss'] or current_price <= trade['take_profit']:
                print(f"Fermeture du trade {trade['action']} sur {symbol} à {current_price}")
                active_trades.remove(trade)

def main():
    try:
        now = pd.Timestamp.now(tz='UTC')
        print(f"Heure actuelle : {now}")
        
        if now.hour >= asian_session_start or now.hour < asian_session_end:
            print("Session asiatique en cours")
            for symbol in symbols:
                print(f"Traitement du symbole : {symbol}")
                htf_df = fetch_ohlcv(symbol, timeframe)
                if htf_df is not None:
                    asian_high, asian_low = get_asian_range(htf_df)
                    print(f"{symbol} - Asian High: {asian_high}, Asian Low: {asian_low}")

                    liquidity_zones = identify_liquidity_zones(htf_df, symbol)
                    print(f"{symbol} - Liquidity Zones:", liquidity_zones)

                    ltf_df = fetch_ohlcv(symbol, ltf_timeframe, limit=50)
                    if ltf_df is not None:
                        action = check_reversal_setup(ltf_df)
                        print(f"Action recommandée : {action}")

                        if action in ['buy', 'sell']:
                            balance = exchange.fetch_balance()['total']['USDT']
                            print(f"Solde disponible pour {symbol} : {balance} USDT")
                            execute_trade(symbol, action, balance)
        else:
            print("En dehors de la session asiatique")

        # Gestion des trades actifs
        manage_active_trades()

    except Exception as e:
        print(f"Erreur inattendue : {e}")

    return "Script exécuté avec succès"

# Exécution du script
if __name__ == "__main__":
    while True:
        main()
        time.sleep(60 * 5)  # Attendre 5 minutes avant la prochaine itération
