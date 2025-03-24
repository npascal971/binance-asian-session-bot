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
PAIR = "XAU_USD"
RISK_PERCENTAGE = 1
TRAILING_ACTIVATION_THRESHOLD_PIPS = 20
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 3.0
SESSION_START = dtime(7, 0)
RETEST_TOLERANCE_PIPS = 10
RETEST_ZONE_RANGE = RETEST_TOLERANCE_PIPS * 0.0001

# Configuration logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def send_email(subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_ADDRESS
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

def get_account_balance():
    r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
    client.request(r)
    return float(r.response["account"]["balance"])

def get_candles(pair, count=100, granularity="H1"):
    params = {"count": count, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)
    return r.response["candles"]

def compute_atr(candles, period=14):
    highs = [float(c["mid"]["h"]) for c in candles if c["complete"]]
    lows = [float(c["mid"]["l"]) for c in candles if c["complete"]]
    closes = [float(c["mid"]["c"]) for c in candles if c["complete"]]
    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) for i in range(1, len(highs))]
    return sum(tr[-period:]) / period

def compute_rsi(prices, period=14):
    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices, short=12, long=26, signal=9):
    prices = pd.Series(prices)
    ema_short = prices.ewm(span=short, adjust=False).mean()
    ema_long = prices.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def detect_asian_range_breakout(candles):
    asian_high = max(float(c["mid"]["h"]) for c in candles[0:7])
    asian_low = min(float(c["mid"]["l"]) for c in candles[0:7])
    breakout_candle = candles[8]
    high = float(breakout_candle["mid"]["h"])
    low = float(breakout_candle["mid"]["l"])
    if high > asian_high:
        return "BUY", asian_high
    elif low < asian_low:
        return "SELL", asian_low
    return None, None

def generate_trade_signal(candles):
    closes = [float(c["mid"]["c"]) for c in candles if c["complete"]]
    rsi = compute_rsi(closes)
    macd, signal = compute_macd(closes)
    if rsi.iloc[-1] > 55 and macd.iloc[-1] > signal.iloc[-1]:
        return "BUY"
    elif rsi.iloc[-1] < 45 and macd.iloc[-1] < signal.iloc[-1]:
        return "SELL"
    return None

def calculate_position_size(balance, entry_price, stop_price):
    risk_amount = balance * (RISK_PERCENTAGE / 100)
    stop_distance = abs(entry_price - stop_price)
    if stop_distance == 0:
        return 0
    units = int(risk_amount / stop_distance)
    return units

def detect_reversal_structure(candles, direction):
    closes = [float(c["mid"]["c"]) for c in candles if c["complete"]]
    macd, signal = compute_macd(closes)
    rsi = compute_rsi(closes)
    if direction == "BUY" and rsi.iloc[-1] > 50 and macd.iloc[-1] > signal.iloc[-1]:
        return True
    elif direction == "SELL" and rsi.iloc[-1] < 50 and macd.iloc[-1] < signal.iloc[-1]:
        return True
    return False

def wait_for_retest_and_reversal(pair, breakout_level, direction):
    logger.info("En attente de retest...")
    for _ in range(12):
        candles = get_candles(pair, count=20, granularity="M5")
        last_close = float(candles[-1]["mid"]["c"])
        if direction == "BUY" and abs(last_close - breakout_level) <= RETEST_ZONE_RANGE:
            if detect_reversal_structure(candles, direction):
                logger.info("Retest confirmé avec structure de retournement LTF.")
                return True
        elif direction == "SELL" and abs(last_close - breakout_level) <= RETEST_ZONE_RANGE:
            if detect_reversal_structure(candles, direction):
                logger.info("Retest confirmé avec structure de retournement LTF.")
                return True
        time.sleep(300)
    logger.info("Pas de retest structuré détecté.")
    return False

def place_trade(pair, direction, entry_price, stop_price, atr):
    balance = get_account_balance()
    units = calculate_position_size(balance, entry_price, stop_price)
    if direction == "SELL":
        units = -units
    sl_distance = atr * ATR_MULTIPLIER_SL
    tp_distance = atr * ATR_MULTIPLIER_TP
    trailing_stop_distance = max(sl_distance, TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001)
    take_profit_price = entry_price + tp_distance if direction == "BUY" else entry_price - tp_distance
    data = {
        "order": {
            "type": "MARKET",
            "instrument": pair,
            "units": str(units),
            "trailingStopLossOnFill": {"distance": f"{trailing_stop_distance:.5f}"},
            "takeProfitOnFill": {"price": f"{take_profit_price:.5f}"}
        }
    }
    r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=data)
    client.request(r)
    logger.info(f"Trade exécuté : {direction} {units} unités de {pair} | SL dynamique: {trailing_stop_distance:.5f}, TP: {take_profit_price:.5f}")

def monitor_open_trades():
    r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
    client.request(r)
    for trade in r.response["trades"]:
        unrealized_pl = float(trade["unrealizedPL"])
        sl = trade.get("trailingStopLoss")
        tp = trade.get("takeProfit")
        logger.info(f"Trade actif {trade['instrument']} | PnL latent: {unrealized_pl:.2f} USD | SL: {sl.get('distance', 'N/A')} | TP: {tp.get('price', 'N/A')}")
        if trade["state"] == "CLOSED":
            send_email("Trade fermé", f"Trade fermé avec PnL: {unrealized_pl:.2f} USD")

if __name__ == "__main__":
    logger.info("Démarrage du bot de trading Asian Session...")
    while True:
        now = datetime.utcnow().time()
        if now >= SESSION_START:
            try:
                candles = get_candles(PAIR)
                direction, trigger_price = detect_asian_range_breakout(candles)
                signal = generate_trade_signal(candles)
                if direction and signal == direction:
                    logger.info(f"Cassure détectée ! Direction: {direction} au niveau de prix {trigger_price}")
                    if wait_for_retest_and_reversal(PAIR, trigger_price, direction):
                        entry_price = float(candles[-1]["mid"]["c"])
                        stop_price = trigger_price
                        atr = compute_atr(candles)
                        place_trade(PAIR, direction, entry_price, stop_price, atr)
                    else:
                        logger.info("Aucune structure de retournement détectée après retest.")
                else:
                    logger.info("Aucune cassure ou signal technique contradictoire.")
                monitor_open_trades()
            except Exception as e:
                logger.error(f"Erreur dans le système: {e}")
        time.sleep(60)
