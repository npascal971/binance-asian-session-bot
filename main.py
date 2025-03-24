import os
import time
import logging
from datetime import datetime, timedelta, time as dtime
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.trades as trades

load_dotenv()

# Configuration API OANDA
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

client = oandapyV20.API(access_token=OANDA_API_KEY)

# Paramètres de trading
PAIRS = ["XAU_USD", "EUR_USD", "GBP_JPY", "BTC_USD", "ETH_USD"]
RISK_PERCENTAGE = 1  # 1% du capital
TRAILING_ACTIVATION_THRESHOLD_PIPS = 20
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 3.0
SESSION_START = dtime(7, 0)
SESSION_END = dtime(23, 0)
RISK_AMOUNT_CAP = 100  # $100 max de risque par trade
CRYPTO_PAIRS = ["BTC_USD", "ETH_USD"]

# Valeur des pips et unités minimales par instrument
PIP_VALUES = {
    "EUR_USD": 0.0001,
    "GBP_JPY": 0.01,
    "XAU_USD": 0.01,
    "BTC_USD": 1,
    "ETH_USD": 1
}
MIN_UNITS = {
    "EUR_USD": 1000,
    "GBP_JPY": 1000,
    "XAU_USD": 1,
    "BTC_USD": 0.001,
    "ETH_USD": 0.001
}

# Configuration logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('oanda_trading_bot.log')
    ]
)
logger = logging.getLogger()

SIMULATION_MODE = True  # Mettre à False pour trading réel
trade_history = []
active_trades = set()

def check_active_trades():
    """Vérifie les trades actuellement ouverts avec OANDA"""
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        response = client.request(r)
        current_trades = {t['instrument'] for t in response['trades']}
        
        global active_trades
        active_trades = current_trades
        
        logger.info(f"Trades actifs: {current_trades}")
        return current_trades
    except Exception as e:
        logger.error(f"Erreur vérification trades: {e}")
        return set()

def get_account_balance():
    """Récupère le solde du compte OANDA"""
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
        client.request(r)
        return float(r.response["account"]["balance"])
    except Exception as e:
        logger.error(f"Erreur récupération solde: {e}")
        return 0

def calculate_position_size(account_balance, entry_price, stop_loss_price, pair):
    """Calcule la taille de position avec gestion précise du risque"""
    # Montant à risquer (1% du solde, max RISK_AMOUNT_CAP)
    risk_amount = min(account_balance * (RISK_PERCENTAGE / 100), RISK_AMOUNT_CAP)
    pip_size = PIP_VALUES.get(pair, 0.0001)
    
    # Calcul de la distance en pips
    distance_pips = abs(entry_price - stop_loss_price) / pip_size
    
    # Valeur d'un pip pour 1 unité
    if pair in CRYPTO_PAIRS or pair == "XAU_USD":
        pip_value = pip_size * 1  # Pour cryptos et or
    else:
        pip_value = pip_size * 100000  # Pour Forex
    
    # Calcul des unités
    if distance_pips == 0:
        logger.error("Distance SL nulle - trade annulé")
        return 0
    
    units = (risk_amount / (distance_pips * pip_value))
    min_unit = MIN_UNITS.get(pair, 1000)
    
    # Arrondir selon le type d'instrument
    if pair in CRYPTO_PAIRS:
        units = round(units, 6)
    elif pair == "XAU_USD":
        units = round(units, 2)
    else:
        units = round(units)
    
    # Vérification unités minimales
    if units < min_unit:
        logger.warning(f"Unités ({units}) < minimum ({min_unit}) - trade annulé")
        return 0
    
    logger.info(
        f"Risk Management:\n"
        f"  Solde: ${account_balance:.2f}\n"
        f"  Risque: {RISK_PERCENTAGE}% = ${risk_amount:.2f}\n"
        f"  Distance SL: {distance_pips:.1f} pips\n"
        f"  Unités calculées: {units}\n"
        f"  Risque réel: ${distance_pips * units * pip_value:.2f}"
    )
    return units

def should_open_trade(pair, rsi, macd, macd_signal, breakout_detected):
    """Détermine si les conditions pour ouvrir un trade sont remplies"""
    signal_detected = False
    reason = []

    if rsi > 70:
        signal_detected = True
        reason.append("RSI > 70 : signal de VENTE")
    elif rsi < 30:
        signal_detected = True
        reason.append("RSI < 30 : signal d'ACHAT")

    if macd > macd_signal:
        signal_detected = True
        reason.append("MACD > Signal : ACHAT")
    elif macd < macd_signal:
        signal_detected = True
        reason.append("MACD < Signal : VENTE")

    if breakout_detected:
        signal_detected = True
        reason.append("Breakout détecté")

    if signal_detected:
        logger.info(f"Signal détecté pour {pair}: {', '.join(reason)}")
    else:
        logger.info(f"Aucun signal pour {pair}")

    return signal_detected

def place_trade(pair, direction, entry_price, stop_price, atr, account_balance):
    """Exécute un trade avec gestion précise du risque"""
    if pair in active_trades:
        logger.info(f"Trade actif existant sur {pair}")
        return None

    units = calculate_position_size(account_balance, entry_price, stop_price, pair)
    if units <= 0:
        return None

    # Calcul TP
    if direction == "buy":
        take_profit = round(entry_price + (ATR_MULTIPLIER_TP * atr), 5)
    else:
        take_profit = round(entry_price - (ATR_MULTIPLIER_TP * atr), 5)

    logger.info(
        f"\n💎 NOUVEAU TRADE 💎\n"
        f"Paire: {pair}\n"
        f"Direction: {direction.upper()}\n"
        f"Entry: {entry_price:.5f}\n"
        f"Stop: {stop_price:.5f}\n"
        f"TP: {take_profit:.5f}\n"
        f"Unités: {units}\n"
        f"Risque: {RISK_PERCENTAGE}% du solde"
    )

    trade_info = {
        "pair": pair,
        "direction": direction,
        "entry": entry_price,
        "stop": stop_price,
        "tp": take_profit,
        "units": units,
        "time": datetime.utcnow().isoformat()
    }

    if not SIMULATION_MODE:
        try:
            order_data = {
                "order": {
                    "instrument": pair,
                    "units": str(int(units)) if direction == "buy" else str(-int(units)),
                    "type": "MARKET",
                    "positionFill": "DEFAULT",
                    "stopLossOnFill": {
                        "price": f"{stop_price:.5f}",
                        "timeInForce": "GTC"
                    },
                    "takeProfitOnFill": {
                        "price": f"{take_profit:.5f}",
                        "timeInForce": "GTC"
                    }
                }
            }

            r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
            response = client.request(r)

            if 'orderFillTransaction' in response:
                trade_id = response['orderFillTransaction']['id']
                logger.info(f"Trade exécuté! ID: {trade_id}")
                trade_info['id'] = trade_id
                active_trades.add(pair)
                trade_history.append(trade_info)
                return trade_id
            else:
                logger.error(f"Erreur OANDA: {response}")
                return None

        except Exception as e:
            logger.error(f"Erreur création ordre: {e}")
            return None
    else:
        trade_info['id'] = "SIMULATION"
        active_trades.add(pair)
        trade_history.append(trade_info)
        logger.info("Mode simulation - Trade non envoyé")
        return "SIMULATION"

def analyze_pair(pair):
    """Analyse une paire et exécute les trades si conditions remplies"""
    logger.info(f"Analyse de {pair}...")
    
    try:
        params = {"granularity": "M5", "count": 50, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        candles = client.request(r)['candles']

        closes = [float(c['mid']['c']) for c in candles if c['complete']]
        highs = [float(c['mid']['h']) for c in candles if c['complete']]
        lows = [float(c['mid']['l']) for c in candles if c['complete']]

        if len(closes) < 20:
            logger.warning("Données insuffisantes")
            return

        # Calcul indicateurs
        df = pd.DataFrame({'close': closes, 'high': highs, 'low': lows})
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Dernières valeurs
        last_close = closes[-1]
        last_rsi = rsi.iloc[-1]
        last_macd = macd.iloc[-1]
        last_signal = signal.iloc[-1]
        atr = np.mean([h - l for h, l in zip(highs[-14:], lows[-14:])])
        
        # Détection signal
        breakout_up = last_close > max(closes[-10:-1])
        breakout_down = last_close < min(closes[-10:-1])
        
        if should_open_trade(pair, last_rsi, last_macd, last_signal, breakout_up or breakout_down):
            entry = last_close
            if breakout_up:
                stop = entry - (ATR_MULTIPLIER_SL * atr)
                place_trade(pair, "buy", entry, stop, atr, get_account_balance())
            elif breakout_down:
                stop = entry + (ATR_MULTIPLIER_SL * atr)
                place_trade(pair, "sell", entry, stop, atr, get_account_balance())

    except Exception as e:
        logger.error(f"Erreur analyse {pair}: {e}")

if __name__ == "__main__":
    logger.info("=== Démarrage du Bot OANDA ===")
    
    try:
        balance = get_account_balance()
        logger.info(f"Solde initial: ${balance:.2f}")
        check_active_trades()
    except Exception as e:
        logger.error(f"Erreur initialisation: {e}")
        exit(1)

    while True:
        now = datetime.utcnow().time()
        
        if SESSION_START <= now <= SESSION_END:
            logger.info("--- Session active ---")
            check_active_trades()
            
            for pair in PAIRS:
                analyze_pair(pair)
                time.sleep(1)
            
            time.sleep(60)
        else:
            logger.info("Session inactive - Attente...")
            time.sleep(300)
