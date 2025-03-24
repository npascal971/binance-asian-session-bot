import os
import time
import logging
from datetime import datetime, timedelta, time as dtime
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
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
RISK_PERCENTAGE = 1
TRAILING_ACTIVATION_THRESHOLD_PIPS = 20
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 3.0
SESSION_START = dtime(7, 0)
SESSION_END = dtime(21, 0)
RETEST_TOLERANCE_PIPS = 10
RETEST_ZONE_RANGE = RETEST_TOLERANCE_PIPS * 0.0001
RISK_AMOUNT_CAP = 100

# Configuration logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

SIMULATION_MODE = True

trade_history = []
CRYPTO_PAIRS = ["BTC_USD", "ETH_USD"]
active_trades = set()

# Détection de signaux techniques
def should_open_trade(pair, rsi, macd, macd_signal, breakout_detected):
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
        reason.append("MACD croise au-dessus du signal : signal d'ACHAT")
    elif macd < macd_signal:
        signal_detected = True
        reason.append("MACD croise en dessous du signal : signal de VENTE")

    if breakout_detected:
        signal_detected = True
        reason.append("Breakout détecté sur le range asiatique")

    if signal_detected:
        logger.info(f"💡 Signal détecté pour {pair} → Raisons: {', '.join(reason)}")
    else:
        logger.info(f"🔍 Aucun signal détecté pour {pair}")

    return signal_detected

# Calcul de la taille de position ajustée au prix de l'instrument
def calculate_position_size(account_balance, entry_price, stop_loss_price, pair):
    risk_amount = min(account_balance * (RISK_PERCENTAGE / 100), RISK_AMOUNT_CAP)
    pip_value = abs(entry_price - stop_loss_price)
    if pip_value == 0:
        return 0
    units = risk_amount / pip_value
    if pair in CRYPTO_PAIRS or pair == "XAU_USD":
        units = units / entry_price
    return int(units)

# Ouverture du trade
def place_trade(pair, direction, entry_price, stop_price, atr, account_balance):
    if pair in active_trades:
        logger.info(f"🚫 Trade déjà actif sur {pair}, aucun nouveau trade ne sera ouvert.")
        return None

    units = calculate_position_size(account_balance, entry_price, stop_price, pair)
    take_profit_price = entry_price + ATR_MULTIPLIER_TP * atr if direction == "buy" else entry_price - ATR_MULTIPLIER_TP * atr

    logger.info(f"💖 Nouveau trade exécuté 💖 {pair} | Direction: {direction} | Entrée: {entry_price} | SL: {stop_price} | TP: {take_profit_price} | Unités: {units}")
    trade_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "pair": pair,
        "direction": direction,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "take_profit": take_profit_price,
        "units": units
    }
    trade_history.append(trade_info)
    active_trades.add(pair)

    if not SIMULATION_MODE:
        order_data = {
            "order": {
                "instrument": pair,
                "units": str(units if direction == "buy" else -units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": str(stop_price)},
                "takeProfitOnFill": {"price": str(take_profit_price)},
                "trailingStopLossOnFill": {"distance": str(TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001)}
            }
        }
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        client.request(r)
        logger.info(f"✔️ Trade envoyé à OANDA. ID de commande: {r.response['orderCreateTransaction']['id']}")
    return "SIMULATED_TRADE_ID" if SIMULATION_MODE else r.response['orderCreateTransaction']['id']

# Balance du compte
def get_account_balance():
    r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
    client.request(r)
    return float(r.response["account"]["balance"])

# Exemple d'intégration dans une boucle d'analyse
def analyze_pair(pair):
    logger.info(f"🔍 Analyse de la paire {pair}...")
    try:
        params = {"granularity": "M5", "count": 50, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        client.request(r)
        candles = r.response['candles']

        closes = [float(c['mid']['c']) for c in candles if c['complete']]
        highs = [float(c['mid']['h']) for c in candles if c['complete']]
        lows = [float(c['mid']['l']) for c in candles if c['complete']]

        if len(closes) < 26:
            logger.warning("Pas assez de données pour le calcul technique.")
            return

        close_series = pd.Series(closes)

        delta = close_series.diff().dropna()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1]

        ema12 = close_series.ewm(span=12, adjust=False).mean()
        ema26 = close_series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        latest_macd = macd_line.iloc[-1]
        latest_signal = signal_line.iloc[-1]

        breakout_up = closes[-1] > max(closes[-11:-1])
        breakout_down = closes[-1] < min(closes[-11:-1])
        breakout_detected = breakout_up or breakout_down

        logger.info(f"🌠 RSI: {latest_rsi:.2f} | MACD: {latest_macd:.4f} | Signal MACD: {latest_signal:.4f} | Breakout: {breakout_detected}")

        if should_open_trade(pair, latest_rsi, latest_macd, latest_signal, breakout_detected):
            logger.info(f"🚀 Trade potentiel détecté sur {pair} selon les critères techniques ou breakout.")
            entry_price = closes[-1]
            atr = np.mean([h - l for h, l in zip(highs[-14:], lows[-14:])])
            stop_price = entry_price - ATR_MULTIPLIER_SL * atr if breakout_up else entry_price + ATR_MULTIPLIER_SL * atr
            direction = "buy" if breakout_up else "sell"
            account_balance = get_account_balance()
            place_trade(pair, direction, entry_price, stop_price, atr, account_balance)
        else:
            logger.info("📉 Pas de conditions suffisantes pour ouvrir un trade.")

    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de {pair} : {e}")

if __name__ == "__main__":
    logger.info("🚀 Démarrage du bot de trading Asian Session...")
    while True:
        now = datetime.utcnow().time()
        if SESSION_START <= now <= SESSION_END:
            logger.info("⏱ Session active - Analyse des paires...")
            for pair in PAIRS:
                analyze_pair(pair)
        else:
            logger.info("🛑 Session de trading inactive. En attente...")
        time.sleep(60)
