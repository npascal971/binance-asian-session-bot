import os
import logging
from datetime import datetime, timedelta
import pytz
import time
import pandas as pd
import numpy as np
from functools import lru_cache, wraps
import oandapyV20
from oandapyV20.endpoints import instruments, accounts, trades, orders
from oandapyV20.endpoints import pricing

# Configuration des logs
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("trading.log")],
)
logger = logging.getLogger()

# Variables d'environnement
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Paramètres globaux
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
CRYPTO_PAIRS = ["BTC_USD", "ETH_USD"]
RISK_PERCENTAGE = 1  # 1% du capital
MAX_RISK_USD = 100  # Risque maximal par trade
RISK_REWARD_RATIO = 1.5
MIN_CONFLUENCE_SCORE = 2
SIMULATION_MODE = True
UTC = pytz.UTC

# Spécifications des instruments
INSTRUMENT_SPECS = {
    "EUR_USD": {"pip": 0.0001, "min_units": 1000, "precision": 0, "margin_rate": 0.03},
    "GBP_USD": {"pip": 0.0001, "min_units": 1000, "precision": 0, "margin_rate": 0.05},
    "USD_JPY": {"pip": 0.01, "min_units": 1000, "precision": 0, "margin_rate": 0.04},
    "XAU_USD": {"pip": 0.01, "min_units": 1, "precision": 2, "margin_rate": 0.02},
}

# Sessions horaires
ASIAN_SESSION_START = datetime.strptime("00:00", "%H:%M").time()
ASIAN_SESSION_END = datetime.strptime("08:00", "%H:%M").time()
LONDON_SESSION_START = datetime.strptime("08:00", "%H:%M").time()
NY_SESSION_END = datetime.strptime("23:00", "%H:%M").time()

# Connexion à l'API OANDA
client = oandapyV20.API(access_token=OANDA_API_KEY, environment="practice")

# Fonctions utilitaires
def get_candles(pair, granularity="M15", count=None, from_time=None, to_time=None):
    """Récupère les bougies avec des options supplémentaires."""
    params = {"granularity": granularity, "price": "M"}
    if count:
        params["count"] = count
    if from_time and to_time:
        params["from"] = from_time.isoformat() + "Z"
        params["to"] = to_time.isoformat() + "Z"
    try:
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        candles = client.request(r)['candles']
        return candles
    except Exception as e:
        logger.error(f"❌ Erreur récupération candles {pair}: {str(e)}")
        return []

def calculate_session_range(pairs, session_start, session_end):
    """Calcule les plages de prix pour une session donnée."""
    ranges = {}
    for pair in pairs:
        candles = get_candles(pair, session_start, session_end)
        if not candles:
            logger.warning(f"⚠️ Aucune donnée pour {pair} - Plage ignorée")
            continue
        highs = [float(c["mid"]["h"]) for c in candles]
        lows = [float(c["mid"]["l"]) for c in candles]
        ranges[pair] = {"high": max(highs), "low": min(lows)}
        logger.info(f"🌍 Range calculé pour {pair}: {min(lows):.5f} - {max(highs):.5f}")
    return ranges

def get_candles(pair, granularity="M15", count=None, from_time=None, to_time=None):
    """Récupère les bougies avec des options supplémentaires."""
    params = {"granularity": granularity, "price": "M"}
    if count:
        params["count"] = count
    if from_time and to_time:
        params["from"] = from_time.isoformat() + "Z"
        params["to"] = to_time.isoformat() + "Z"
    try:
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        candles = client.request(r)['candles']
        return candles
    except Exception as e:
        logger.error(f"❌ Erreur récupération candles {pair}: {str(e)}")
        return []

def safe_execute(func, *args, error_message="Erreur inconnue"):
    """Exécute une fonction avec gestion des erreurs."""
    try:
        return func(*args)
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}")
        return None

def is_price_in_valid_range(current_price, range_to_use, buffer=0.0002):
    """
    Vérifie si le prix actuel est dans la plage valide définie par le range asiatique.
    
    Args:
        current_price (float): Le prix actuel de la paire.
        range_to_use (dict): Dictionnaire contenant les clés 'high' et 'low' pour le range asiatique.
        buffer (float): Une marge de sécurité pour éviter les faux signaux (en pips ou en unités).
    
    Returns:
        bool: True si le prix est dans la plage valide, False sinon.
    """
    try:
        if not range_to_use or "high" not in range_to_use or "low" not in range_to_use:
            logger.warning("⚠️ Range invalide ou manquant")
            return False

        lower_bound = range_to_use["low"] - buffer
        upper_bound = range_to_use["high"] + buffer

        if lower_bound <= current_price <= upper_bound:
            logger.info(f"✅ Prix {current_price:.5f} dans la plage valide ({lower_bound:.5f} - {upper_bound:.5f})")
            return True
        else:
            logger.info(f"❌ Prix {current_price:.5f} hors de la plage valide ({lower_bound:.5f} - {upper_bound:.5f})")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur validation du range: {str(e)}")
        return False



def calculate_rsi(closes, period=14):
    """
    Calcule l'indice de force relative (RSI) à partir des prix de clôture.
    
    Args:
        closes (list): Liste des prix de clôture (float).
        period (int): Période utilisée pour le calcul du RSI (par défaut 14).
    
    Returns:
        float: Valeur du RSI si calculée avec succès, sinon None.
    """
    try:
        # Vérification que nous avons suffisamment de données
        if len(closes) < period + 1:
            logger.warning(f"⚠️ Données insuffisantes pour RSI ({len(closes)} points)")
            return None

        # Calcul des variations quotidiennes
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)  # Gains (positifs)
        losses = np.where(deltas < 0, -deltas, 0)  # Pertes (positifs)

        # Moyenne des gains et pertes initiaux
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        # Éviter une division par zéro
        if avg_loss == 0:
            return 100

        # Calcul du RS (Relative Strength)
        rs = avg_gain / avg_loss

        # Calcul du RSI
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)

    except Exception as e:
        logger.error(f"❌ Erreur lors du calcul du RSI: {str(e)}")
        return None



def calculate_macd(closes, fast=12, slow=26, signal=9):
    """
    Calcule le MACD (Moving Average Convergence Divergence) à partir des prix de clôture.
    
    Args:
        closes (list): Liste des prix de clôture (float).
        fast (int): Période de la moyenne mobile exponentielle rapide (par défaut 12).
        slow (int): Période de la moyenne mobile exponentielle lente (par défaut 26).
        signal (int): Période de la ligne de signal (par défaut 9).
    
    Returns:
        tuple: 
            - macd_line (float): Valeur actuelle de la ligne MACD.
            - signal_line (float): Valeur actuelle de la ligne de signal.
            - histogram (float): Différence entre la ligne MACD et la ligne de signal.
    """
    try:
        # Vérification que nous avons suffisamment de données
        if len(closes) < max(fast, slow, signal):
            logger.warning(f"⚠️ Données insuffisantes pour MACD ({len(closes)} points)")
            return None, None, None

        # Conversion en série pandas pour les calculs
        series = pd.Series(closes)

        # Calcul des moyennes mobiles exponentielles
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()

        # Calcul de la ligne MACD
        macd_line = ema_fast - ema_slow

        # Calcul de la ligne de signal
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # Calcul de l'histogramme
        histogram = macd_line.iloc[-1] - signal_line.iloc[-1]

        # Récupération des valeurs actuelles
        macd_value = float(macd_line.iloc[-1])
        signal_value = float(signal_line.iloc[-1])

        # Logs détaillés
        logger.info(f"📊 MACD calculé - Ligne MACD: {macd_value:.5f}, Ligne Signal: {signal_value:.5f}, Histogramme: {histogram:.5f}")

        return macd_value, signal_value, histogram

    except Exception as e:
        logger.error(f"❌ Erreur lors du calcul du MACD: {str(e)}")
        return None, None, None

def calculate_ema(pair, period=200):
    """
    Calcule l'EMA (Exponential Moving Average) pour une paire donnée.
    
    Args:
        pair (str): La paire de devises (ex: "EUR_USD").
        period (int): Période de l'EMA (par défaut 200).
    
    Returns:
        float: Valeur de l'EMA si calculée avec succès, sinon None.
    """
    try:
        # Récupération des données historiques
        candles = get_candles(pair, granularity="H4", count=period * 2)  # Récupère suffisamment de bougies
        if not candles or len(candles) < period:
            logger.warning(f"EMA{period}: Données insuffisantes ({len(candles) if candles else 0}/{period} bougies)")
            return None

        # Extraction des prix de clôture
        closes = []
        for candle in candles:
            try:
                if 'mid' in candle and 'c' in candle['mid']:
                    closes.append(float(candle['mid']['c']))
            except (KeyError, TypeError, ValueError) as e:
                logger.debug(f"Bougie ignorée - {str(e)}")
                continue

        if len(closes) < period:
            logger.warning(f"EMA{period}: Trop de valeurs invalides ({len(closes)}/{period} valides)")
            return None

        # Calcul de l'EMA avec pandas
        series = pd.Series(closes)
        ema = series.ewm(span=period, adjust=False).mean()

        # Vérification du résultat final
        if pd.isna(ema.iloc[-1]):
            logger.warning(f"EMA{period}: Calcul invalide (valeur NaN)")
            return None

        return float(ema.iloc[-1])

    except Exception as e:
        logger.error(f"❌ Erreur EMA{period} pour {pair}: {str(e)}")
        return None

def analyze_session(session_type, pairs):
    """Analyse une session (asiatique ou européenne)."""
    global asian_ranges, european_ranges
    session_ranges = {}

    if session_type == "ASIE":
        session_ranges = safe_execute(
            calculate_session_range,
            pairs,
            ASIAN_SESSION_START,
            ASIAN_SESSION_END,
            error_message="❌ Erreur analyse session asiatique",
        )
        asian_ranges = session_ranges
    elif session_type == "EUROPE":
        session_ranges = safe_execute(
            calculate_session_range,
            pairs,
            LONDON_SESSION_START,
            NY_SESSION_END,
            error_message="❌ Erreur analyse session européenne",
        )
        european_ranges = session_ranges

    if session_ranges:
        logger.info(f"✅ Session {session_type} analysée avec succès")
    else:
        logger.warning(f"⚠️ Échec analyse session {session_type}")

def get_current_price(pair):
    """
    Récupère le prix actuel d'une paire de devises via l'API OANDA.

    Args:
        pair (str): La paire de devises (ex: "EUR_USD").

    Returns:
        float: Le prix actuel de la paire si récupéré avec succès, sinon None.
    """
    try:
        # Préparation de la requête pour récupérer le prix
        params = {"instruments": pair}
        r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params=params)

        # Exécution de la requête
        response = client.request(r)

        # Extraction du prix bid (prix d'achat)
        if "prices" in response and len(response["prices"]) > 0:
            price = float(response["prices"][0]["bids"][0]["price"])
            logger.info(f"✅ Prix actuel récupéré pour {pair}: {price:.5f}")
            return price
        else:
            logger.warning(f"⚠️ Aucun prix disponible pour {pair}")
            return None

    except Exception as e:
        logger.error(f"❌ Erreur récupération prix pour {pair}: {str(e)}")
        return None

def check_active_trades():
    """
    Vérifie les trades actuellement ouverts sur le compte OANDA.
    
    Returns:
        set: Ensemble des paires avec des trades actifs (ex: {"EUR_USD", "GBP_USD"}).
    """
    try:
        # Requête pour récupérer les trades ouverts
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        response = client.request(r)

        # Extraction des paires actives
        open_trades = response.get("trades", [])
        active_pairs = {trade["instrument"] for trade in open_trades}

        # Logs pour suivre les trades actifs
        if active_pairs:
            logger.info(f"📊 Trades actifs détectés: {', '.join(active_pairs)}")
        else:
            logger.info("📊 Aucun trade actif détecté")

        return active_pairs

    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des trades actifs: {str(e)}")
        return set()

def analyze_pair(pair, range_to_use):
    """Analyse une paire pour détecter des opportunités de trading."""
    try:
        logger.info(f"🔍 Début analyse approfondie pour {pair}")

        # 1. Vérification si le prix est dans la plage valide
        current_price = get_current_price(pair)
        if not is_price_in_valid_range(current_price, range_to_use):
            logger.info(f"❌ Prix hors range valide pour {pair}")
            return

        # 2. Calcul des indicateurs techniques
        rsi = calculate_rsi(pair)
        macd_signal = calculate_macd(pair)
        ema200 = calculate_ema(pair)  # Calcul de l'EMA200

        # Logs détaillés
        logger.info(f"📊 Analyse {pair} - Prix: {current_price:.5f}, Range: {range_to_use['low']:.5f} - {range_to_use['high']:.5f}")
        logger.info(f"📈 RSI: {rsi:.2f}, MACD Signal: {macd_signal}, EMA200: {ema200:.5f}")

        # 3. Décision de placement de trade avec confirmation EMA200
        if ema200 and current_price > ema200 and rsi < 40 and macd_signal == "BUY":
            place_trade(pair, "buy", current_price, range_to_use["low"], range_to_use["high"])
            logger.info(f"🎯 Configuration haussière confirmée par EMA200 pour {pair}")
        elif ema200 and current_price < ema200 and rsi > 60 and macd_signal == "SELL":
            place_trade(pair, "sell", current_price, range_to_use["high"], range_to_use["low"])
            logger.info(f"🎯 Configuration baissière confirmée par EMA200 pour {pair}")
        else:
            logger.debug(f"❌ Conditions non remplies pour {pair} - RSI: {rsi}, MACD: {macd_signal}, EMA200: {ema200}")

    except Exception as e:
        logger.error(f"❌ Erreur analyse {pair}: {str(e)}")

def place_trade(pair, direction, entry_price, stop_loss, take_profit):
    """Place un trade avec trailing SL/TP."""
    global SIMULATION_MODE
    account_balance = get_account_balance()
    units = calculate_position_size(pair, account_balance, entry_price, stop_loss)

    if units <= 0:
        logger.warning(f"⚠️ Impossible de placer le trade {pair} - Taille de position invalide")
        return

    logger.info(
        f"""🚀 NOUVEAU TRADE {'ACHAT' if direction == 'buy' else 'VENTE'} 🚀
• Paire: {pair}
• Direction: {direction.upper()}
• Entrée: {entry_price:.5f}
• Stop: {stop_loss:.5f}
• TP: {take_profit:.5f}
• Unités: {units}
• Risque: ${min(account_balance * (RISK_PERCENTAGE / 100), MAX_RISK_USD):.2f}"""
    )

    if SIMULATION_MODE:
        logger.info("🧪 Mode simulation - Trade non envoyé")
        return "SIMULATION"

    try:
        order_data = {
            "order": {
                "instrument": pair,
                "units": str(units) if direction == "buy" else str(-units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": f"{stop_loss:.5f}", "timeInForce": "GTC"},
                "takeProfitOnFill": {"price": f"{take_profit:.5f}", "timeInForce": "GTC"},
            }
        }
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        response = client.request(r)
        if "orderFillTransaction" in response:
            trade_id = response["orderFillTransaction"]["id"]
            logger.info(f"✅ Trade exécuté! ID: {trade_id}")
            return trade_id
    except Exception as e:
        logger.error(f"❌ Erreur création ordre: {str(e)}")
        return None

def main_loop():
    while True:
        try:
            now = datetime.utcnow()
            current_time = now.time()
            logger.info(f"⏳ Heure actuelle: {current_time}")

            # Vérification des trades actifs
            active_trades = check_active_trades()
            logger.info(f"📊 Trades actifs: {len(active_trades)}")

            # Limite globale de 1 trade maximum
            if len(active_trades) >= 1:
                logger.info("⚠️ Limite de 1 trade atteinte - Attente...")
                time.sleep(60)
                continue

            # Détermination de la session active
            if ASIAN_SESSION_START <= current_time < ASIAN_SESSION_END:
                analyze_session("ASIE", PAIRS)
            elif LONDON_SESSION_START <= current_time <= NY_SESSION_END:
                analyze_session("EUROPE", PAIRS)
            else:
                logger.info("⚠️ Hors plage horaire définie")
                continue

            # Analyse des paires
            for pair in PAIRS:
                range_to_use = asian_ranges.get(pair) or european_ranges.get(pair)
                if range_to_use:
                    analyze_pair(pair, range_to_use)
                else:
                    logger.warning(f"⚠️ Aucun range disponible pour {pair} - Analyse ignorée")

            # Pause avant le prochain cycle
            logger.info("⏰ Pause avant le prochain cycle...")
            time.sleep(60)

        except Exception as e:
            logger.error(f"💥 ERREUR GRAVE: {str(e)}", exc_info=True)

# Exécution principale
if __name__ == "__main__":
    asian_ranges = {}
    european_ranges = {}
    main_loop()
