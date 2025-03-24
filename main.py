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
PAIRS = ["XAU_USD", "EUR_USD", "GBP_JPY"]
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


def notify_trade_trigger(direction, entry_price):
    subject = "Signal de Trade Détecté - Asian Session Bot"
    body = f"Un signal de trade a été détecté.\nDirection: {direction}\nPrix d'entrée: {entry_price:.2f}\nVeuillez vérifier la stratégie."
    send_email(subject, body)


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


def log_trailing_stop_update(trade_id, new_sl):
    logger.info(f"🔄 Mise à jour du stop loss (Trailing Stop) pour le trade {trade_id} : Nouveau SL = {new_sl}")


def calculate_trade_units(entry_price, stop_loss_price, balance):
    risk_amount = min(balance * RISK_PERCENTAGE / 100, RISK_AMOUNT_CAP)
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit == 0:
        return 0
    units = risk_amount / risk_per_unit
    logger.info(f"🔢 Calcul des unités à trader: {units:.2f} (pour un risque de {risk_amount:.2f} USD)")
    return int(units)


def monitor_open_trades():
    r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
    client.request(r)
    open_trades = r.response.get("trades", [])
    if not open_trades:
        logger.info("Aucun trade actif en ce moment.")
    for trade in open_trades:
        trade_id = trade["id"]
        instrument = trade["instrument"]
        price = trade["price"]
        open_time = trade["openTime"]
        sl = trade.get("stopLossOrder", {}).get("price", "Non défini")
        logger.info(f"✅ Trade actif - ID: {trade_id}, Instrument: {instrument}, Prix: {price}, SL: {sl}, Ouvert depuis: {open_time}")
        if float(price) - float(sl) > TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001:
            new_sl = float(price) - TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001
            log_trailing_stop_update(trade_id, new_sl)
            logger.info(f"🚨 Trailing Stop mis à jour pour le trade {trade_id} sur {instrument}. Nouveau SL: {new_sl}")


def get_candles(pair):
    params = {"granularity": "M5", "count": 100, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)
    return r.response["candles"]


def detect_asian_range_breakout(candles):
    return None, None


def generate_trade_signal(candles):
    closes = [float(candle["mid"]["c"]) for candle in candles]
    rsi = compute_rsi(closes)
    macd_line, macd_signal = compute_macd(closes)
    logger.info(f"🔠 RSI: {rsi[-1]:.2f} | MACD: {macd_line[-1]:.4f} | Signal MACD: {macd_signal[-1]:.4f}")
    logger.info("🧠 Analyse technique complétée - Aucun retournement significatif si divergence absente.")
    return None


def compute_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0).values


def compute_macd(prices, fast=12, slow=26, signal=9):
    exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.values, signal_line.values


def compute_atr(candles):
    return 0.001


def wait_for_retest_and_reversal(pair, trigger_price, direction):
    return True


def place_trade(pair, direction, entry_price, stop_price, atr, units):
    logger.info(f"💖 Nouveau trade exécuté 💖 {pair} | Direction: {direction} | Entrée: {entry_price} | SL: {stop_price} | Unités: {units}")
    trade_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "pair": pair,
        "direction": direction,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "units": units
    }
    trade_history.append(trade_info)
    return "TRADE_ID_123"


if __name__ == "__main__":
    logger.info("Démarrage du bot de trading Asian Session...")
    while True:
        now = datetime.utcnow().time()
        logger.info(f"\n===== Nouvelle boucle d'analyse à {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC =====")
        logger.info(f"Heure actuelle UTC: {now.strftime('%H:%M:%S')}")
        balance = get_account_balance()
        logger.info(f"Solde actuel du compte: {balance:.2f} USD")

        if SESSION_START <= now <= SESSION_END:
            logger.info("Période de trading active. Lancement de l'analyse...")
            for pair in PAIRS:
                logger.info(f"\n🔍 Analyse de la paire {pair}...")
                try:
                    candles = get_candles(pair)
                    direction, trigger_price = detect_asian_range_breakout(candles)
                    signal = generate_trade_signal(candles)
                    if direction and signal == direction:
                        logger.info(f"Cassure détectée ! Direction: {direction} au niveau de prix {trigger_price}")
                        notify_trade_trigger(direction, trigger_price)
                        if wait_for_retest_and_reversal(pair, trigger_price, direction):
                            entry_price = float(candles[-1]["mid"]["c"])
                            stop_price = trigger_price
                            atr = compute_atr(candles)
                            units = calculate_trade_units(entry_price, stop_price, balance)
                            trade_id = place_trade(pair, direction, entry_price, stop_price, atr, units)
                        else:
                            logger.info("Aucune structure de retournement détectée après retest.")
                    else:
                        logger.info("Aucune cassure ou signal technique contradictoire.")
                    monitor_open_trades()
                except Exception as e:
                    logger.error(f"Erreur dans le système pour la paire {pair}: {e}")
                    send_email("Erreur dans le système de trading", f"Une erreur est survenue pour {pair} : {e}")
        else:
            logger.info("Hors session de trading. Aucune analyse effectuée.")

        time.sleep(60)
