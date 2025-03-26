import os
import time
import logging
from datetime import datetime, timedelta, time as dtime
from email.message import EmailMessage
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing
import requests
import pytz

# Chargement des variables d'environnement
load_dotenv()

# Configuration API OANDA
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
client = oandapyV20.API(access_token=OANDA_API_KEY, environment="practice")

# Paramètres de trading
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
CRYPTO_PAIRS = ["BTC_USD", "ETH_USD"]
RISK_PERCENTAGE = 1  # 1% du capital
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 2.0
ASIAN_SESSION_START = dtime(0, 0)    # 00h00 UTC
ASIAN_SESSION_END = dtime(6, 0)      # 06h00 UTC
LONDON_SESSION_START = dtime(7, 0)   # 07h00 UTC
NY_SESSION_END = dtime(16, 30)       # 16h30 UTC
MAX_RISK_USD = 100  # $100 max de risque par trade
MIN_CRYPTO_UNITS = 0.001  # Unités minimales pour les cryptos

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("trading.log")]
)
logger = logging.getLogger()

# Variables globales
asian_ranges = {}
active_trades = {}


def send_email_notification(trade_info, hit_type):
    """Envoie une notification email pour TP/SL atteint."""
    try:
        emoji = "💰" if hit_type == "TP" else "🛑"
        subject = f"{emoji} {trade_info['pair']} {hit_type} ATTEINT {emoji}"
        profit_loss = (trade_info['tp'] - trade_info['entry']) * trade_info['units']
        if trade_info['direction'] == 'sell':
            profit_loss = -profit_loss
        result = f"PROFIT: ${profit_loss:.2f} ✅" if hit_type == "TP" else f"LOSS: ${abs(profit_loss):.2f} ❌"
        body = f"""
        {emoji * 3} {hit_type} ATTEINT {emoji * 3}
        Paire: {trade_info['pair']}
        Direction: {trade_info['direction'].upper()}
        Entrée: {trade_info['entry']:.5f}
        {hit_type}: {trade_info[hit_type.lower()]:.5f}
        Unités: {trade_info['units']}
        {result}
        Heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_ADDRESS
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info(f"📧 Notification {hit_type} envoyée!")
    except Exception as e:
        logger.error(f"❌ Erreur envoi email: {str(e)}")


def calculate_position_size(pair, account_balance, entry_price, stop_loss):
    """Calcule la taille de position avec gestion de risque stricte."""
    pip_location = get_instrument_details(pair)['pip_location']
    risk_amount = min(account_balance * (RISK_PERCENTAGE / 100), MAX_RISK_USD)
    pip_value = 10 ** pip_location
    distance_pips = abs(entry_price - stop_loss) / pip_value
    units = risk_amount / distance_pips
    return round(units, 2)

def is_price_in_valid_range(current_price, asian_range, buffer=0.0002):
    """
    Vérifie si le prix actuel est dans la plage valide définie par le range asiatique.
    
    Args:
        current_price (float): Le prix actuel de la paire.
        asian_range (dict): Dictionnaire contenant les clés 'high' et 'low' pour le range asiatique.
        buffer (float): Une marge de sécurité pour éviter les faux signaux (en pips ou en unités).
    
    Returns:
        bool: True si le prix est dans la plage valide, False sinon.
    """
    if not asian_range or "high" not in asian_range or "low" not in asian_range:
        logger.warning("⚠️ Range asiatique invalide ou manquant")
        return False
    
    # Appliquer un buffer pour éviter les faux signaux près des bords
    lower_bound = asian_range["low"] - buffer
    upper_bound = asian_range["high"] + buffer
    
    # Vérifier si le prix actuel est dans la plage avec le buffer
    if lower_bound <= current_price <= upper_bound:
        logger.info(f"✅ Prix {current_price:.5f} dans la plage valide ({lower_bound:.5f} - {upper_bound:.5f})")
        return True
    else:
        logger.info(f"❌ Prix {current_price:.5f} hors de la plage valide ({lower_bound:.5f} - {upper_bound:.5f})")
        return False

def get_instrument_details(pair):
    """Retourne les spécifications de l'instrument."""
    specs = {
        "EUR_USD": {"pip": 0.0001, "pip_location": -4},
        "GBP_USD": {"pip": 0.0001, "pip_location": -4},
        "USD_JPY": {"pip": 0.01, "pip_location": -2},
        "XAU_USD": {"pip": 0.01, "pip_location": -2},
        "BTC_USD": {"pip": 1, "pip_location": 0},
        "ETH_USD": {"pip": 0.1, "pip_location": -1},
    }
    return specs.get(pair, {"pip": 0.0001, "pip_location": -4})


def analyze_asian_session():
    """Analyse la session asiatique pour calculer le range."""
    global asian_ranges
    for pair in PAIRS:
        try:
            candles = get_candles(pair, ASIAN_SESSION_START, ASIAN_SESSION_END)
            highs = [float(c['mid']['h']) for c in candles]
            lows = [float(c['mid']['l']) for c in candles]
            asian_ranges[pair] = {"high": max(highs), "low": min(lows)}
            logger.info(f"📊 Range asiatique {pair}: {min(lows):.5f} - {max(highs):.5f}")
        except Exception as e:
            logger.error(f"❌ Erreur analyse asiatique {pair}: {str(e)}")


def get_candles(pair, start_time, end_time=None):
    """Récupère les bougies pour une plage horaire spécifique."""
    now = datetime.utcnow()
    end_date = datetime.combine(now.date(), end_time) if end_time else now
    start_date = datetime.combine(now.date(), start_time)
    end_date = min(end_date, now)  # Prevent future dates
    
    params = {
        "granularity": "M15",
        "from": start_date.isoformat() + "Z",
        "to": end_date.isoformat() + "Z",
        "price": "M"
    }
    try:
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        candles = client.request(r)['candles']
        return candles
    except Exception as e:
        logger.error(f"❌ Erreur récupération candles {pair}: {str(e)}")
        return []


def check_high_impact_events():
    """Vérifie les événements macro à haut impact."""
    logger.info("🔍 Vérification des événements macro...")
    # Implémentez ici l'appel à une API économique ou calendrier économique.
    return False


def analyze_pair(pair):
    """Analyse une paire pour détecter des opportunités de trading."""
    if check_high_impact_events():
        logger.warning("⚠️ Événement macro majeur - Pause de 5 min")
        return
    
    asian_range = asian_ranges.get(pair)
    if not asian_range:
        logger.warning(f"⚠️ Aucun range asiatique disponible pour {pair}")
        return
    
    current_price = get_current_price(pair)
    if not is_price_in_valid_range(current_price, asian_range):
        logger.info(f"❌ Prix hors range valide pour {pair}")
        return
    
    rsi = calculate_rsi(pair)
    macd_signal = calculate_macd(pair)
    logger.info(f"📊 Analyse {pair} - RSI: {rsi:.2f}, MACD: {macd_signal}")
    
    if rsi < 30 and macd_signal == "BUY":
        place_trade(pair, "buy", current_price, asian_range["low"], asian_range["high"])
    elif rsi > 70 and macd_signal == "SELL":
        place_trade(pair, "sell", current_price, asian_range["high"], asian_range["low"])

def place_trade(pair, direction, entry_price, stop_loss, take_profit):
    """Place un trade avec trailing SL/TP."""
    account_balance = get_account_balance()
    units = calculate_position_size(pair, account_balance, entry_price, stop_loss)
    if units <= 0:
        logger.warning(f"❌ Impossible de placer le trade {pair} - Taille de position invalide")
        return
    trade_info = {
        "pair": pair,
        "direction": direction,
        "entry": entry_price,
        "stop": stop_loss,
        "tp": take_profit,
        "units": units,
    }
    active_trades[pair] = trade_info
    logger.info(f"🚀 NOUVEAU TRADE {direction.upper()} {pair} - Entry: {entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")


def check_tp_sl():
    """Vérifie si TP/SL est atteint pour les trades actifs."""
    for pair, trade in list(active_trades.items()):
        current_price = get_current_price(pair)
        if trade["direction"] == "buy":
            if current_price >= trade["tp"]:
                send_email_notification(trade, "TP")
                del active_trades[pair]
            elif current_price <= trade["stop"]:
                send_email_notification(trade, "SL")
                del active_trades[pair]
        elif trade["direction"] == "sell":
            if current_price <= trade["tp"]:
                send_email_notification(trade, "TP")
                del active_trades[pair]
            elif current_price >= trade["stop"]:
                send_email_notification(trade, "SL")
                del active_trades[pair]


def get_current_price(pair):
    """Récupère le prix actuel d'une paire."""
    params = {"instruments": pair}
    r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params=params)
    response = client.request(r)
    return float(response['prices'][0]['bids'][0]['price'])


def calculate_rsi(pair, period=14):
    candles = get_candles(pair, LONDON_SESSION_START, NY_SESSION_END)
    closes = [float(c['mid']['c']) for c in candles]
    
    if len(closes) < period + 1:
        logger.warning(f"⚠️ Données insuffisantes pour RSI ({len(closes)} points)")
        return None
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period]) or 1e-10
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(pair):
    candles = get_candles(pair, LONDON_SESSION_START, NY_SESSION_END)
    closes = [float(c['mid']['c']) for c in candles]
    
    if len(closes) < 26:
        logger.warning(f"⚠️ Données insuffisantes pour MACD ({len(closes)} points)")
        return "NEUTRAL"
    
    ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    return "BUY" if macd_line.iloc[-1] > signal_line.iloc[-1] else "SELL"

def fetch_historical_asian_range(pair):
    """Récupère le range asiatique historique pour une paire."""
    now = datetime.utcnow()
    today = now.date()
    start_time = datetime.combine(today, ASIAN_SESSION_START)
    end_time = datetime.combine(today, ASIAN_SESSION_END)
    
    # Si la session asiatique est aujourd'hui mais déjà terminée
    if now > end_time:
        try:
            candles = get_candles(pair, ASIAN_SESSION_START, ASIAN_SESSION_END)
            if not candles:
                logger.warning(f"⚠️ Aucune donnée historique pour {pair}")
                return None
            highs = [float(c['mid']['h']) for c in candles if 'mid' in c]
            lows = [float(c['mid']['l']) for c in candles if 'mid' in c]
            if not highs or not lows:
                logger.warning(f"⚠️ Données insuffisantes pour {pair}")
                return None
            return {"high": max(highs), "low": min(lows)}
        except Exception as e:
            logger.error(f"❌ Erreur récupération range asiatique historique {pair}: {str(e)}")
            return None
    return None

def main_loop():
    """Boucle principale du bot."""
    while True:
        now = datetime.utcnow()
        current_time = now.time()
        if ASIAN_SESSION_START <= current_time < ASIAN_SESSION_END:
            logger.info("🌏 SESSION ASIATIQUE EN COURS")
            analyze_asian_session()
        elif LONDON_SESSION_START <= current_time <= NY_SESSION_END:
            logger.info("🏙️ SESSION LONDRES/NY EN COURS")
            for pair in PAIRS:
                analyze_pair(pair)
        check_tp_sl()
        time.sleep(60)


if __name__ == "__main__":
    logger.info("✨ DÉMARRAGE DU BOT DE TRADING ✨")
    
    # Initialisation des données asiatiques
    for pair in PAIRS:
        if pair not in asian_ranges:
            logger.info(f"🔍 Récupération des données asiatiques historiques pour {pair}...")
            historical_range = fetch_historical_asian_range(pair)
            if historical_range:
                asian_ranges[pair] = historical_range
                logger.info(f"📊 Range asiatique historique {pair}: {historical_range['low']:.5f} - {historical_range['high']:.5f}")
            else:
                logger.warning(f"⚠️ Échec récupération range asiatique pour {pair}")
    
    main_loop()
