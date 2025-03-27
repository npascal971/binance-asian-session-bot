import os
import time
import logging
from datetime import datetime, time as dtime
import numpy as np
import pandas as pd
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
from oandapyV20 import API

# Configuration API OANDA
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

client = API(access_token=OANDA_API_KEY, environment="practice")

# Paramètres de trading
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
CRYPTO_PAIRS = ["BTC_USD", "ETH_USD"]
RISK_PERCENTAGE = 1  # 1% du capital
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 2.0
ASIAN_SESSION_START = dtime(0, 0)  # 00h00 UTC
ASIAN_SESSION_END = dtime(6, 0)  # 06h00 UTC
LONDON_SESSION_START = dtime(7, 0)  # 07h00 UTC
NY_SESSION_END = dtime(16, 30)  # 16h30 UTC
MAX_RISK_USD = 100  # $100 max de risque par trade
MIN_CRYPTO_UNITS = 0.001  # Unités minimales pour les cryptos

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

# Variables globales
asian_ranges = {}
active_trades = set()
trade_history = []

def get_candles(pair, start_time, end_time=None, granularity="M15", count=None):
    """Récupère les bougies pour une plage horaire spécifique."""
    params = {"granularity": granularity, "price": "M"}
    if count:
        params["count"] = count
    if start_time and end_time:
        params["from"] = start_time.isoformat() + "Z"
        params["to"] = end_time.isoformat() + "Z"
    try:
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        candles = client.request(r)['candles']
        return candles
    except Exception as e:
        logger.error(f"❌ Erreur récupération candles {pair}: {str(e)}")
        return []

def fetch_historical_asian_range(pair):
    """Récupère le range asiatique historique pour une paire."""
    now = datetime.utcnow()
    today = now.date()
    start_time = datetime.combine(today, ASIAN_SESSION_START)
    end_time = datetime.combine(today, ASIAN_SESSION_END)
    try:
        candles = get_candles(pair, start_time, end_time)
        highs = [float(c['mid']['h']) for c in candles if 'mid' in c]
        lows = [float(c['mid']['l']) for c in candles if 'mid' in c]
        if not highs or not lows:
            logger.warning(f"⚠️ Données insuffisantes pour {pair}")
            return None
        return {"high": max(highs), "low": min(lows)}
    except Exception as e:
        logger.error(f"❌ Erreur calcul range {pair}: {str(e)}")
        return None

def analyze_asian_session():
    """Analyse la session asiatique pour calculer le range."""
    global asian_ranges
    logger.info("🌏 LANCEMENT ANALYSE ASIATIQUE AVANCÉE")
    pairs = PAIRS + CRYPTO_PAIRS
    success_count = 0
    max_retries = 3
    retry_delay = 60

    for attempt in range(1, max_retries + 1):
        logger.info(f"🔁 Tentative {attempt}/{max_retries}")
        for pair in pairs:
            if pair in asian_ranges:
                continue
            try:
                historical_range = fetch_historical_asian_range(pair)
                if historical_range:
                    asian_ranges[pair] = historical_range
                    success_count += 1
                    logger.info(f"✅ Range asiatique calculé pour {pair}: {historical_range}")
            except Exception as e:
                logger.error(f"❌ Erreur analyse asiatique {pair} (tentative {attempt}): {str(e)}")
        if success_count >= len(pairs) // 2:  # Au moins 50% de réussite
            logger.info("🌍 ANALYSE ASIATIQUE TERMINÉE AVEC SUCCÈS")
            break
        time.sleep(retry_delay)
    else:
        logger.warning("⚠️ Échec complet de l'analyse asiatique après plusieurs tentatives")

def calculate_rsi(pair, period=14):
    """Calcule le RSI pour une paire donnée."""
    candles = get_candles(pair, LONDON_SESSION_START, NY_SESSION_END, count=period * 2)
    closes = [float(c['mid']['c']) for c in candles if 'mid' in c]
    if len(closes) < period + 1:
        logger.warning(f"⚠️ Données insuffisantes pour RSI ({len(closes)} points)")
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(pair):
    """Calcule le MACD pour une paire donnée."""
    candles = get_candles(pair, LONDON_SESSION_START, NY_SESSION_END, count=200)
    closes = [float(c['mid']['c']) for c in candles if 'mid' in c]
    if len(closes) < 200:
        logger.warning(f"⚠️ Données insuffisantes pour MACD ({len(closes)} points)")
        return None, None
    series = pd.Series(closes)
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line.iloc[-1] - signal_line.iloc[-1]
    signal = "BUY" if macd_line.iloc[-1] > signal_line.iloc[-1] else "SELL"
    logger.info(f"📊 MACD calculé pour {pair} - Signal: {signal}, Histogramme: {histogram:.5f}")
    return signal, histogram

def place_trade(pair, direction, entry_price, stop_loss, take_profit):
    """Place un trade avec trailing stop."""
    try:
        units = calculate_position_size(pair, entry_price, stop_loss)
        order_data = {
            "order": {
                "instrument": pair,
                "units": str(units) if direction == "buy" else str(-units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": f"{stop_loss:.5f}", "timeInForce": "GTC"},
                "takeProfitOnFill": {"price": f"{take_profit:.5f}", "timeInForce": "GTC"}
            }
        }
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        response = client.request(r)
        trade_id = response['orderCreateTransaction']['id']
        active_trades.add(pair)
        trade_history.append({
            "pair": pair,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trade_id": trade_id
        })
        logger.info(f"✅ Trade exécuté! ID: {trade_id}")
        return trade_id
    except Exception as e:
        logger.error(f"❌ Erreur création ordre: {str(e)}")
        return None

def check_tp_sl():
    """Vérifie si les stops ou take-profits sont touchés."""
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        open_trades = client.request(r).get('trades', [])
        for trade in open_trades:
            trade_id = trade['id']
            instrument = trade['instrument']
            current_units = float(trade['currentUnits'])
            price = float(trade['price'])
            stop_loss = float(trade['stopLossOrder']['price'])
            take_profit = float(trade['takeProfitOrder']['price'])
            current_price = get_current_price(instrument)
            if current_price <= stop_loss or current_price >= take_profit:
                close_trade(trade_id)
                send_email(f"Trade {instrument} fermé", f"SL/TP touché pour {instrument}")
    except Exception as e:
        logger.error(f"❌ Erreur vérification TP/SL: {str(e)}")

def send_email(subject, body):
    """Envoie un email."""
    import smtplib
    from email.mime.text import MIMEText
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_ADDRESS
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, msg.as_string())
        logger.info(f"📧 Email envoyé: {subject}")
    except Exception as e:
        logger.error(f"❌ Erreur envoi email: {str(e)}")

def main_loop():
    """Boucle principale du bot."""
    while True:
        try:
            now = datetime.utcnow()
            current_time = now.time()
            logger.info(f"⏳ Heure actuelle: {current_time}")

            # Session asiatique
            if ASIAN_SESSION_START <= current_time < ASIAN_SESSION_END:
                logger.info("🌏 SESSION ASIATIQUE EN COURS")
                analyze_asian_session()
            # Session Londres/NY
            elif LONDON_SESSION_START <= current_time <= NY_SESSION_END:
                logger.info("🌍 SESSION LONDRES/NY EN COURS")
                for pair in PAIRS + CRYPTO_PAIRS:
                    if pair not in active_trades:
                        analyze_pair(pair)
            # Vérification des stops et take-profits
            check_tp_sl()
            logger.info("⏰ Pause avant le prochain cycle...")
            time.sleep(60)
        except Exception as e:
            logger.error(f"💥 ERREUR GRAVE: {str(e)}", exc_info=True)
            time.sleep(300)

if __name__ == "__main__":
    logger.info("🚀 DÉMARRAGE DU BOT DE TRADING 🚀")
    main_loop()
