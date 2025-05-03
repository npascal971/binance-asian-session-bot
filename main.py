import os
import time
import logging
from datetime import datetime, timedelta, time as dtime
from dotenv import load_dotenv
from oandapyV20.endpoints import positions
import numpy as np
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.trades as trades
from oandapyV20.endpoints import pricing
from email.mime.text import MIMEText
import smtplib 
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np

load_dotenv()

# Configuration API OANDA
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

client = oandapyV20.API(access_token=OANDA_API_KEY)
# Ajouter dans les paramètres globaux
CACHE_TTL = 60  # 5 minutes
MAX_ZONES = 5
# Paramètres de trading
PAIRS = [
    "XAU_USD", "XAG_USD",  # Métaux précieux
    "EUR_USD", "GBP_USD", "USD_CHF", "AUD_USD", "NZD_USD",  # Majors
    "GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY", "CAD_JPY"  # Crosses et JPY  
]

# Ajouter en haut avec les autres variables globales
CLOSES_HISTORY = {}  # Dictionnaire pour stocker l'historique par paire
HISTORY_LENGTH = 200  # Nombre de bougies à conserver
RISK_PERCENTAGE = 1
TRAILING_ACTIVATION_THRESHOLD_PIPS = 20
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 3.0
SESSION_START = dtime(1, 0) #7
SESSION_END = dtime(23, 59)
RETEST_TOLERANCE_PIPS = 15
RETEST_ZONE_RANGE = RETEST_TOLERANCE_PIPS * 0.0001
RISK_AMOUNT_CAP = 100
CRYPTO_PAIRS = ["BTC_USD", "ETH_USD"]
# Seuil pour détecter une pin bar (ratio entre la taille des mèches et le corps)
PIN_BAR_RATIO_THRESHOLD = 2.5  # Exemple : une mèche doit être au moins 3 fois plus grande que le corps

# Configuration logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger()

SIMULATION_MODE = True  # Mettre à True pour tester sans exécuter de vrais trades
CONFIRMATION_REQUIRED = {
    "XAU_USD": 2,  # Nombre de confirmations requises pour XAU_USD
    "EUR_USD": 1,  # Nombre de confirmations requises pour EUR_USD
    "DEFAULT": 1   # Valeur par défaut pour les autres paires
}
trade_history = []
active_trades = set()


class DataCache:
    def __init__(self):
        self.cache = {}
    
    def get(self, key):
        item = self.cache.get(key)
        if item and (time.time() < item['expiry']):
            return item['data']
        return None

    def set(self, key, data, ttl=60):
        self.cache[key] = {
            'data': data,
            'expiry': time.time() + ttl  # Nouveau paramètre TTL
        }
# Initialisation globale du cache
cache = DataCache()


def check_active_trades():
    """Désactivée"""
    pass

def get_account_balance():
    """Récupère le solde du compte OANDA"""
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
        client.request(r)
        return float(r.response["account"]["balance"])
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du solde: {e}")
        return 0
        balance = float(r.response["account"]["balance"])
        logger.debug(f"Solde du compte récupéré: {balance}")
        return balance

def check_momentum(self, pair, zone):
    """Vérifie l'alignement du momentum avec la zone"""
    closes = self.get_closes(pair, 'H1', 14)
    if len(closes) < 14:
        return False
    
    # Calcul de la pente des prix
    trend = np.polyfit(range(len(closes)), closes, 1)[0]
    
    # Vérification de la cohérence directionnelle
    if (zone > closes[-1] and trend > 0) or (zone < closes[-1] and trend < 0):
        return True
    return False

def confirm_signal(pair, direction):
    """Vérification finale avant exécution"""
    try:
        # 1. Vérification du spread
        candles = get_candles(pair, 'M5', 1)
        ask = float(candles[-1]['ask']['c'])
        bid = float(candles[-1]['bid']['c'])
        spread = ask - bid
        
        if spread > 0.0005:  # Spread trop élevé
            return False

        # 2. Volumes récents
        volume_candles = get_candles(pair, 'M15', 5)
        volumes = [float(c['volume']) for c in volume_candles]
        if volumes[-1] < np.mean(volumes[:-1]):
            return False

        # 3. Alignment price/zone
        price = get_current_price(pair)
        raw_zones = analyze_htf(pair, params)
        zones = [process_zone(z, 'ob') for z in raw_zones]
        valid_zones = [z for z in zones if isinstance(z, (list, tuple)) and len(z) > 0]
        return any(abs(price - z[0]) < 0.0002 for z in valid_zones)


    except Exception as e:
        logger.error(f"Erreur confirmation: {e}")
        return False
    
    return (
        last_volume > avg_volume * 1.2 and 
        spread < 0.0002 and 
        is_price_approaching(pivot_levels[pair])
    )

def calculate_dynamic_levels(pair, entry_price, direction):
    """Calcule SL/TP avec Bollinger Bands et EMA50 (Nouvelle version)"""
    closes_m15 = get_closes(pair, "M15", 50)
    atr_value = calculate_atr_for_pair(pair)
    
    # Bollinger Bands M15
    bb_upper, bb_lower = calculate_bollinger_bands(closes_m15)
    
    # EMA50 M15
    ema50 = pd.Series(closes_m15).ewm(span=50, adjust=False).mean().iloc[-1]
    
    # Calcul de base SL
    if direction == "SELL":
        base_sl = entry_price + (PAIR_SETTINGS[pair]["atr_multiplier_sl"] * atr_value)
        # Ajustement Bollinger
        if not np.isnan(bb_upper) and base_sl < bb_upper + 0.2 * (bb_upper - bb_lower):
            base_sl = bb_upper + 0.5 * atr_value
    else:
        base_sl = entry_price - (PAIR_SETTINGS[pair]["atr_multiplier_sl"] * atr_value)
        if not np.isnan(bb_lower) and base_sl > bb_lower - 0.2 * (bb_upper - bb_lower):
            base_sl = bb_lower - 0.5 * atr_value
    
    # Vérification EMA50
    current_price = get_closes(pair, "M15", 1)[-1]
    if ((direction == "SELL" and current_price > ema50) or 
        (direction == "BUY" and current_price < ema50)):
        base_sl *= 1.25
    
    # Calcul TP
    tp_distance = abs(entry_price - base_sl) * PAIR_SETTINGS[pair]["atr_multiplier_tp"]
    take_profit = entry_price - tp_distance if direction == "SELL" else entry_price + tp_distance
    
    return round(base_sl, 2), round(take_profit, 2)

def is_ranging(pair, timeframe="H1", threshold=0.5):
    """Determine if a pair is in a ranging market using ADX"""
    try:
        # Get ADX value (already have this calculation in your code)
        params = {"granularity": timeframe, "count": 20, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        candles = client.request(r)["candles"]
        
        highs = [float(c["mid"]["h"]) for c in candles if c["complete"]]
        lows = [float(c["mid"]["l"]) for c in candles if c["complete"]]
        closes = [float(c["mid"]["c"]) for c in candles if c["complete"]]
        
        if len(closes) < 14:  # Need at least 14 periods for ADX
            return False
            
        adx_value = calculate_adx(highs, lows, closes)
        return adx_value < threshold  # Market is ranging if ADX below threshold
        
    except Exception as e:
        logger.error(f"Error checking ranging market for {pair}: {e}")
        return False  # Default to False if error occurs

def calculate_bollinger_bands(closes, window=20, num_std=2):
    """Version sécurisée avec gestion des données manquantes"""
    try:
        if len(closes) < window or pd.Series(closes).isnull().any():
            logger.debug(f"Données insuffisantes pour BB: {len(closes)}/{window}")
            if closes:
                last_close = closes[-1]
                return (last_close, last_close)  # Retourne le dernier prix connu
            return (0, 0)
            
        series = pd.Series(closes).dropna()
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        
        upper = rolling_mean + (rolling_std * num_std)
        lower = rolling_mean - (rolling_std * num_std)
        
        return (upper.iloc[-1], lower.iloc[-1])
    except Exception as e:
        logger.error(f"Erreur Bollinger Bands: {str(e)}")
        return (closes[-1], closes[-1]) if closes else (0, 0)
    
def get_current_price(pair):
    """Récupère le prix actuel via l'endpoint de pricing"""
    try:
        params = {"instruments": pair}
        r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params=params)
        response = client.request(r)
        
        if 'prices' not in response or not response['prices']:
            logger.warning(f"Aucune donnée de prix pour {pair}")
            return None
            
        price_info = response['prices'][0]
        bid = float(price_info['bids'][0]['price'])
        ask = float(price_info['asks'][0]['price'])
        return (bid + ask) / 2  # Prix moyen
        
    except (KeyError, IndexError) as e:
        logger.error(f"Structure de données invalide pour {pair}: {e}")
    except oandapyV20.exceptions.V20Error as e:
        logger.error(f"Erreur OANDA {pair} [{e.code}]: {e.msg}")
    except Exception as e:
        logger.error(f"Erreur inattendue pour {pair}: {str(e)}")
    
    return None
def calculate_atr(highs, lows, closes, period=14):
    try:
        if len(highs) < period or len(lows) < period or len(closes) < period:
            logger.warning(f"Données insuffisantes pour ATR ({len(highs)}/{period})")
            return 0.0

        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)

        atr = sum(true_ranges[:period]) / period
        
        for i in range(period, len(true_ranges)):
            atr = (atr * (period - 1) + true_ranges[i]) / period

        return round(atr, 5)  # ➔ toujours 5 décimales par défaut
    except Exception as e:
        logger.error(f"Erreur calcul ATR: {str(e)}")
        return 0.0


def send_trade_alert(pair, direction, entry_price, stop_price, take_profit, reasons):
    """Envoie une alerte par email au lieu d'exécuter un trade."""
    subject = f"🚨 Signal {direction.upper()} détecté sur {pair}"
    body = f"""Nouveau signal de trading détecté !
📌 Paire : {pair}
📈 Direction : {direction.upper()}
💰 Prix d'entrée : {entry_price:.5f}
🎯 Take Profit : {take_profit:.5f}
🛡️ Stop Loss : {stop_price:.5f}
📊 Raisons du signal :
- {chr(10).join(reasons)}
⚠️ Ceci est une alerte informative - Aucun trade n'a été exécuté automatiquement."""
    send_email(subject, body)

def send_email(subject, body):
    """Envoie un e-mail avec les signaux"""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_ADDRESS

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, msg.as_string())
        logger.info(f"E-mail envoyé avec succès: {subject}")
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'e-mail: {e}")
# Remplacer is_trend_aligned() par :
def is_trend_aligned(pair, direction):
    alignments = 0
    try:
        for tf in ['H1', 'M5']:
            closes = get_closes(pair, tf, 20)
            if len(closes) < 15:
                continue
                
            fast_span = 9 if "_JPY" not in pair else 5
            slow_span = 21 if "_JPY" not in pair else 13
            
            ema_fast = pd.Series(closes).ewm(span=fast_span).mean()
            ema_slow = pd.Series(closes).ewm(span=slow_span).mean()
            
            # Vérification pente + croisement
            slope = ema_fast.iloc[-1] - ema_fast.iloc[-5]
            condition = (
                (ema_fast.iloc[-1] > ema_slow.iloc[-1]) & (slope > 0)
                if direction == "buy" else 
                (ema_fast.iloc[-1] < ema_slow.iloc[-1]) & (slope < 0)
            )
            
            if condition:
                alignments += 1
                logger.debug(f"Alignement {direction} confirmé sur {tf}")
                
        return alignments >= 1  # Au moins 1 TF aligné
        
    except Exception as e:
        logger.error(f"Erreur tendance {pair}: {str(e)}")
        return False

def is_price_approaching(price, zone, pair, threshold_pips=None):
    """
    Version simplifiée maintenant que get_current_price() retourne un float
    """
    try:
        # 1. Validation des paramètres d'entrée
        if price is None or zone is None:
            logger.warning(f"Paramètre manquant pour {pair}: price={price}, zone={zone}")
            return False
            
        # 2. Normalisation de la zone
        if isinstance(zone, (tuple, list)):
            if len(zone) == 2:
                zone_min, zone_max = float(zone[0]), float(zone[1])
            else:
                logger.error(f"Format de zone invalide pour {pair}: {zone}")
                return False
        elif isinstance(zone, dict):
            if 'high' in zone and 'low' in zone:
                zone_min, zone_max = float(zone['low']), float(zone['high'])
            else:
                logger.error(f"Dictionnaire de zone invalide pour {pair}: {zone}")
                return False
        else:
            logger.error(f"Type de zone non supporté pour {pair}: {type(zone)}")
            return False

        # 3. Détermination du seuil
        threshold_pips = threshold_pips or (15 if "_JPY" in pair else 10)
        threshold = threshold_pips * (0.01 if "_JPY" in pair else 0.0001)
        
        # 4. Vérification de la proximité
        return (zone_min - threshold) <= price <= (zone_max + threshold)
        
    except Exception as e:
        logger.error(f"Erreur is_price_approaching pour {pair}: {str(e)}")
        return False

def dynamic_sl_tp(atr, direction, risk_reward=1.5, min_sl_multiplier=1.8):
    """Gestion dynamique avec paramètres par paire"""
    settings = PAIR_SETTINGS.get(pair, PAIR_SETTINGS["DEFAULT"])
    
    sl = atr * settings.get("atr_multiplier_sl", 1.5)
    tp = atr * settings.get("atr_multiplier_tp", 3.0)
    
    return (sl, tp) if direction == "buy" else (-sl, -tp)

def get_closes(pair, timeframe, count):
    """Récupère les prix de clôture"""
    params = {"granularity": timeframe, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    return [float(c['mid']['c']) for c in client.request(r)['candles'] if c['complete']]

def is_trend_aligned(pair, direction):
    ema_values = []
    for tf in ['H1', 'M5']:
        closes = get_closes(pair, tf, 50)
        if len(closes) < 50: continue
        
        ema_fast = pd.Series(closes).ewm(span=20).mean().iloc[-1]
        ema_slow = pd.Series(closes).ewm(span=50).mean().iloc[-1]
        
        if direction == "buy":
            ema_values.append(ema_fast > ema_slow)
        else:
            ema_values.append(ema_fast < ema_slow)
    
    return sum(ema_values) >= 1  # Au moins 1 TF aligné

def get_asian_session_range(pair):
    """Récupère le high et le low de la session asiatique"""
    # Définir les heures de début et de fin de la session asiatique
    asian_start_time = dtime(23, 0)  # 23:00 UTC
    asian_end_time = dtime(7, 0)     # 07:00 UTC

    # Obtenir la date et l'heure actuelles en UTC
    now = datetime.utcnow()

    # Calculer la date de début et de fin de la session asiatique
    if now.time() < asian_end_time:
        # Si nous sommes avant 07:00 UTC, la session asiatique correspond à la veille
        asian_start_date =  (now - timedelta(days=1)).date()
        asian_end_date = now.date()
    else:
        # Sinon, la session asiatique correspond à aujourd'hui  *********** faire l'inverse juste en dessous ligne 96 pour 97 et 97 pour 96
        asian_start_date = (now + timedelta(days=-1)).date()
        asian_end_date = now.date()

    # Créer les objets datetime complets pour le début et la fin
    asian_start = datetime.combine(asian_start_date, asian_start_time).isoformat() + "Z"
    
    # Limiter asian_end à l'heure actuelle si nécessaire
    if now.time() < asian_end_time:
        # Si nous sommes dans la session asiatique actuelle, limiter asian_end à now
        asian_end = now.isoformat() + "Z"
    else:
        # Sinon, utiliser la fin normale de la session asiatique
        asian_end = datetime.combine(asian_end_date, asian_end_time).isoformat() + "Z"

    # Logs des timestamps calculés
    logger.debug(f"Timestamps calculés pour {pair}: from={asian_start}, to={asian_end}")

    # Paramètres de la requête API
    params = {
        "granularity": "M5",
        "from": asian_start,
        "to": asian_end,
        "price": "M"
    }

    # Logs des paramètres de la requête API
    logger.debug(f"Requête API pour {pair}: URL=https://api-fxpractice.oanda.com/v3/instruments/{pair}/candles, Params={params}")

    # Effectuer la requête API
    try:
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        response = client.request(r)

        # Logs de la réponse API reçue
        logger.debug(f"Réponse API reçue pour {pair}: {response}")

        # Extraire les données des bougies
        candles = response['candles']
        highs = [float(c['mid']['h']) for c in candles if c['complete']]
        lows = [float(c['mid']['l']) for c in candles if c['complete']]

        # Vérifier si des données valides ont été reçues
        if not highs or not lows:
            logger.warning(f"Aucune donnée valide disponible pour le range asiatique de {pair}.")
            return None, None

        # Calculer le high et le low de la session asiatique
        asian_high = max(highs)
        asian_low = min(lows)

        # Logs du range asiatique calculé
        logger.info(f"Range asiatique pour {pair}: High={asian_high}, Low={asian_low}")
        return asian_high, asian_low
        if asian_high is not None and asian_low is not None:
            return (asian_high, asian_low)  # Toujours retourner un tuple
        return None, None

    except Exception as e:
        # Logs en cas d'erreur lors de la récupération des données
        logger.error(f"Erreur lors de la récupération du range asiatique pour {pair}: {e}")
        return None, None
# Définir le seuil de ratio pour les pin bars
PIN_BAR_RATIO_THRESHOLD = 3.0  # Exemple : une mèche doit être au moins 3 fois plus grande que le corps

PAIR_SETTINGS = {
    "XAU_USD": {
        "min_atr": 0.5,
        "rsi_overbought": 68,
        "rsi_oversold": 32,
        "pin_bar_ratio": 2.0,
        "atr_multiplier_sl": 2.5,  # Nouveau
        "atr_multiplier_tp": 4.0,   # Nouveau
        "min_volume_ratio": 1.3,
        "bollinger_margin": 0.02,  # 2% de marge des Bandes
        "volume_multiplier": 1.2      # Nouveau
    },
    "USD_JPY": {
        "atr_multiplier_sl": 2.0,  # Augmenté de 1.8
        "atr_multiplier_tp": 3.5,  # Augmenté de 3.2 
        "min_sl_distance": 0.2,    # 20 pips
        "spread_max": 0.03,
        "rsi_overbought": 75,
        "rsi_oversold": 25
    },
    "EUR_USD": {
        "min_atr": 0.0003,  # Réduit de 0.0005
        "rsi_overbought": 68,
        "rsi_oversold": 32,
        "pin_bar_ratio": 2.5
    },
    "GBP_JPY": {
        "min_atr": 0.025,  # Réduit de 0.03
        "rsi_overbought": 72,  # Ajusté pour les paires JPY
        "rsi_oversold": 28,
        "pin_bar_ratio": 2.3
    },
    "DEFAULT": {
        "min_atr": 0.4,  # Réduit de 0.5
        "rsi_overbought": 68,
        "rsi_oversold": 32,
        "pin_bar_ratio": 2.5
    },
    "CAD_JPY": {
        "min_atr": 0.15,  # Seuil relevé
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "pin_bar_ratio": 3.0,
        "volume_multiplier": 1.5  # Nécessite 50% plus de volume que la moyenne
    },    
    "XAG_USD": {  # Silver
        "min_atr": 0.1,  # Augmenté de 0.1 → 0.3 (3$ de volatilité)
        "rsi_overbought": 72,  # Rehaussé de 68 à 72
        "rsi_oversold": 28,    # Baissé de 32 à 28
        "pin_bar_ratio": 3.0,  # Plus strict (2.0 → 3.0)
        "atr_multiplier_sl": 2.0,  # Nouveau paramètre spécifique
        "atr_multiplier_tp": 4.0   # RR 2:1
    }
}

    
def detect_pin_bars(candles, pair=None):
    settings = PAIR_SETTINGS.get(pair, PAIR_SETTINGS["DEFAULT"])
    pin_bars = []
    for candle in candles:
        volume = float(candle['volume'])
        avg_volume = np.mean([float(c['volume']) for c in candles[-20:]])
        
        # Rejet des pin bars à faible volume
        if volume < avg_volume * 0.7:
            continue
        try:
            if not candle['complete']:
                continue

            o = float(candle['mid']['o'])
            h = float(candle['mid']['h'])
            l = float(candle['mid']['l'])
            c = float(candle['mid']['c'])

            body_size = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l

            if body_size == 0:
                continue  # Bougie doji = on skip

            ratio = max(upper_wick, lower_wick) / body_size
            ratio_threshold = settings.get("pin_bar_ratio", PIN_BAR_RATIO_THRESHOLD)

            # Nouveau filtre : on veut que la mèche soit au moins 2x plus grande que l'autre mèche
            wick_ratio = max(upper_wick, lower_wick) / (min(upper_wick, lower_wick) + 1e-6)

            if ratio >= ratio_threshold and wick_ratio > 2.0:
                direction = "bullish" if lower_wick > upper_wick else "bearish"
                pin_bars.append({
                    "type": direction,
                    "ratio": round(ratio, 1),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c
                })

        except Exception as e:
            logger.error(f"Erreur analyse bougie {pair}: {str(e)}")
            continue

    return pin_bars

def validate_signal(pair, signal):
    """Version améliorée avec les nouvelles validations"""
    try:
        # 1. Vérifications existantes
        weekly_atr = calculate_atr_for_pair(pair, "D", 14)
        if signal.get('atr', 0) < weekly_atr * 0.15:
            logger.warning(f"Volatilité actuelle trop faible pour {pair}")
            return False

        if not check_breakout(pair, signal['entry']):
            logger.info(f"Aucune cassure confirmée pour {pair}")
            return False

        if is_ranging(pair):
            logger.info(f"Marché en range pour {pair}")
            return False

        # 2. Nouveaux contrôles techniques
        closes = get_closes(pair, "H1", 50)
        current_price = get_current_price(pair)
        
        # Vérification Bollinger Bands
        bb_upper, bb_lower = calculate_bollinger_bands(closes)
        if current_price > bb_upper * 0.98 or current_price < bb_lower * 1.02:
            logger.warning(f"Prix trop proche des Bandes Bollinger - Rejet")
            return False

        # Alignement EMA50
        ema50 = pd.Series(closes).ewm(span=50).mean().iloc[-1]
        if (signal['direction'] == "buy" and current_price < ema50) or \
           (signal['direction'] == "sell" and current_price > ema50):
            logger.warning("Désalignement avec EMA50 H1")
            return False

        # 3. Confirmation multi-timeframe
        tf_confirmations = 0
        for tf in ["M15", "H1", "H4"]:
            tf_closes = get_closes(pair, tf, 50)
            if len(tf_closes) < 50: continue
            
            ema_fast = pd.Series(tf_closes).ewm(span=20).mean().iloc[-1]
            ema_slow = pd.Series(tf_closes).ewm(span=50).mean().iloc[-1]
            
            if signal['direction'] == "buy" and ema_fast > ema_slow:
                tf_confirmations += 1
            elif signal['direction'] == "sell" and ema_fast < ema_slow:
                tf_confirmations += 1

        if tf_confirmations < 2:
            logger.warning(f"Confirmations multi-TF insuffisantes ({tf_confirmations}/3)")
            return False

        # 4. Analyse volumétrique
        volume_data = get_volume_analysis(pair)  # À implémenter
        if volume_data['current'] < volume_data['average'] * 1.2:
            logger.warning("Volume insuffisant")
            return False

        # 5. Validations existantes pour paires JPY
        jpy_pairs = ["CAD_JPY", "USD_JPY", "GBP_JPY"]
        if pair in jpy_pairs:
            settings = PAIR_SETTINGS.get(pair, PAIR_SETTINGS["DEFAULT"])
            if signal['atr'] < settings["min_atr"] * 1.2:
                logger.info(f"ATR trop faible pour {pair} ({signal['atr']})")
                return False

        # 6. Validation ADX existante
        highs = [float(c['mid']['h']) for c in candles]
        lows = [float(c['mid']['l']) for c in candles]
        closes = [float(c['mid']['c']) for c in candles]
        adx = calculate_adx(highs, lows, closes)
        
        min_adx = 20 if pair in CRYPTO_PAIRS else 25
        if adx < min_adx:
            logger.warning(f"Tendance trop faible (ADX={adx:.1f} < {min_adx})")
            return False

        return True

    except Exception as e:
        logger.error(f"Erreur validation signal {pair}: {str(e)}")
        return False

def get_volume_analysis(pair):
    """Analyse les volumes sur les dernières 20 bouches H1"""
    try:
        candles = fetch_candles(pair, "H1", 20)
        volumes = [float(c['volume']) for c in candles if c['complete']]
        
        return {
            'current': volumes[-1] if volumes else 0,
            'average': np.mean(volumes[:-1]) if len(volumes) > 1 else 0,
            'max': max(volumes) if volumes else 0
        }
    except Exception as e:
        logger.error(f"Erreur analyse volume {pair}: {str(e)}")
        return {'current': 0, 'average': 0}

def is_strong_trend(pair, direction):
    """Vérifie l'alignement sur M15/H1/H4 avec force"""
    timeframes = ['M15', 'H1', 'H4']
    alignments = 0
    
    for tf in timeframes:
        try:
            params = {"granularity": tf, "count": 50}
            candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params))['candles']
            closes = [float(c['mid']['c']) for c in candles if c['complete']]
            
            # Vérifie la pente des EMAs
            ema20 = pd.Series(closes).ewm(span=20).mean()
            ema50 = pd.Series(closes).ewm(span=50).mean()
            
            if direction == "sell":
                if closes[-1] < ema20.iloc[-1] < ema50.iloc[-1]:
                    alignments += 1
            else:
                if closes[-1] > ema20.iloc[-1] > ema50.iloc[-1]:
                    alignments += 1
                    
        except Exception as e:
            logger.error(f"Error checking trend on {tf}: {e}")
    
    return alignments >= 2  # Au moins 2/3 timeframes alignés

def analyze_gold():
    """Analyse technique avancée spécifique à l'or (XAU_USD)"""
    pair = "XAU_USD"
    params = {"granularity": "H1", "count": 100, "price": "M"}
    logger.info(f"Analyse approfondie de {pair}...")

    try:
        # 1. Récupération des données multi-timeframe
        candles_h1 = fetch_candles(pair, params["granularity"], params)  
        candles_m15 = fetch_candles(pair, "M15", {"count": 50})           

        # 2. Calcul des indicateurs techniques
        closes_h1 = [float(c['mid']['c']) for c in candles_h1 if c['complete']]
        highs_h1 = [float(c['mid']['h']) for c in candles_h1 if c['complete']]
        lows_h1 = [float(c['mid']['l']) for c in candles_h1 if c['complete']]

        # 3. Détection des niveaux clés
        asian_high, asian_low = get_asian_session_range(pair)
        support = min(lows_h1[-20:])
        resistance = max(highs_h1[-20:])

        # 4. Configuration spécifique à l'or
        GOLD_SETTINGS = {
            "rsi_overbought": 65,
            "rsi_oversold": 35,
            "atr_multiplier_sl": 1.8,
            "atr_multiplier_tp": 3.2
        }

        # 5. Analyse des conditions
        current_price = get_current_price(pair)
        rsi_h1 = calculate_rsi(closes_h1[-14:])
        rsi_m15 = calculate_rsi([float(c['mid']['c']) for c in candles_m15][-14:])
        atr_h1 = calculate_atr(highs_h1, lows_h1, closes_h1)

        # 6. Scénarios de trading
        scenarios = []
        if current_price <= asian_high * 1.005 and rsi_h1 < 35:
            scenarios.append({...})  # Signal d'achat

        if current_price >= asian_low * 0.995 and rsi_h1 > 65:
            scenarios.append({...})  # Signal de vente

        # 7. Envoi d'alerte
        if scenarios:
            send_gold_alert(pair, scenarios, {...})

    except Exception as e:
        logger.error(f"Erreur analyse OR: {str(e)}")

def validate_gold_signal(scenario, atr):
    """Valide les paramètres de risque pour l'or"""
    risk = abs(scenario['entry'] - scenario['sl'])
    reward = abs(scenario['tp'] - scenario['entry'])
    return (reward / risk) >= 2.0 and risk <= (atr * 2)

def send_gold_alert(pair, scenarios, indicators):
    """Envoie une alerte détaillée avec analyse technique"""
    subject = f"🔥 {pair} - {len(scenarios)} scénario(s) détecté(s)"
    
    analysis = f"""
📊 **Analyse Technique Avancée - {pair}**
-----------------------------------------
💰 Prix Actuel : {indicators['price']:.2f}
⚡ Volatilité (ATR H1) : {indicators['atr']:.2f}
📉 RSI H1 : {indicators['rsi_h1']:.1f} | RSI M15 : {indicators['rsi_m15']:.1f}

📌 **Niveaux Clés :**
- Support : {indicators['support']:.2f}
- Résistance : {indicators['resistance']:.2f}
- Bollinger Bands (H1) : [{indicators['bb_lower']:.2f} - {indicators['bb_upper']:.2f}]

🔍 **Scénarios Valides ({len(scenarios)}) :**
"""
    for i, scenario in enumerate(scenarios, 1):
        analysis += f"""
🎯 **Scénario {i} - {scenario['direction']}**
➤ Raison : {scenario['reason']}
➤ Entrée : {scenario['entry']:.2f}
➤ Take Profit : {scenario['tp']:.2f} (+{(scenario['tp']-scenario['entry']):.2f})
➤ Stop Loss : {scenario['sl']:.2f} (Risque: {(scenario['entry']-scenario['sl']):.2f})
➤ Risk/Reward : {(abs(scenario['tp']-scenario['entry'])/abs(scenario['entry']-scenario['sl'])):.1f}:1
-----------------------------------------
"""

    analysis += "\n⚠️ **Recommendation :** Surveillez les niveaux clés et confirmez avec les volumes !"
    
    send_email(subject, analysis)

def check_rsi_conditions(pair):
    """Vérifie les conditions RSI sur H1 et M15"""
    try:
        # Récupère RSI H1
        params_h1 = {"granularity": "H1", "count": 14, "price": "M"}
        candles_h1 = client.request(instruments.InstrumentsCandles(instrument=pair, params=params_h1))["candles"]
        closes_h1 = [float(c["mid"]["c"]) for c in candles_h1 if c["complete"]]
        rsi_h1 = calculate_rsi(closes_h1[-14:]) if len(closes_h1) >= 14 else 50

        # Récupère RSI M15
        params_m15 = {"granularity": "M15", "count": 14, "price": "M"}
        candles_m15 = client.request(instruments.InstrumentsCandles(instrument=pair, params=params_m15))["candles"]
        closes_m15 = [float(c["mid"]["c"]) for c in candles_m15 if c["complete"]]
        rsi_m15 = calculate_rsi(closes_m15[-14:]) if len(closes_m15) >= 14 else 50

        return {
            "h1": rsi_h1,
            "m15": rsi_m15,
            "buy_signal": rsi_h1 > 40 and rsi_m15 > 50,
            "sell_signal": rsi_h1 < 60 and rsi_m15 < 50
        }
    except Exception as e:
        logger.error(f"Erreur calcul RSI multi-TF: {e}")
        return {"h1": 50, "m15": 50, "buy_signal": False, "sell_signal": False}

def calculate_atr_for_pair(pair, timeframe="H1", period=14):
    """Calcule l'ATR pour une paire donnée avec gestion des erreurs améliorée"""
    try:
        params = {
            "granularity": timeframe,  # Utilisation du paramètre timeframe
            "count": period * 2, 
            "price": "M",
            "smooth": True
        }
        
        logger.debug(f"Récupération ATR pour {pair} avec params: {params}")
        
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        candles = client.request(r)["candles"]
        
        complete_candles = [c for c in candles if c['complete']]
        if not complete_candles:
            logger.warning(f"Aucune bougie complète pour {pair}")
            return 0.0
            
        highs = [float(c['mid']['h']) for c in complete_candles]
        lows = [float(c['mid']['l']) for c in complete_candles]
        closes = [float(c['mid']['c']) for c in complete_candles]
        
        if len(highs) < period:
            logger.warning(f"Données insuffisantes ({len(highs)}/{period}) pour {pair}")
            return 0.0
            
        atr_value = calculate_atr(highs, lows, closes, period)
        
        precision = 3 if pair in ["XAU_USD", "XAG_USD"] else (2 if "_JPY" in pair else 5)
        
        return round(atr_value, precision)
        
    except Exception as e:
        logger.error(f"Erreur critique dans calculate_atr_for_pair({pair}): {str(e)}")
        return 0.0

def detect_engulfing_patterns(candles):
    """Détecte des engulfing patterns dans une série de bougies"""
    engulfing_patterns = []
    for i in range(1, len(candles)):
        prev_open = float(candles[i - 1]['mid']['o'])
        prev_close = float(candles[i - 1]['mid']['c'])
        current_open = float(candles[i]['mid']['o'])
        current_close = float(candles[i]['mid']['c'])

        # Bullish engulfing
        if current_close > current_open and prev_close < prev_open and \
           current_open <= prev_close and current_close >= prev_open:
            engulfing_patterns.append(("Bullish Engulfing", i))

        # Bearish engulfing
        elif current_close < current_open and prev_close > prev_open and \
             current_open >= prev_close and current_close <= prev_open:
            engulfing_patterns.append(("Bearish Engulfing", i))
    
    return engulfing_patterns

def fetch_candles(pair, timeframe, params=None):
    """Récupère les bougies avec validation renforcée du format de réponse"""
    try:
        if params is None:
            params = {"granularity": timeframe, "count": 200}
        elif not isinstance(params, dict):
            raise TypeError("Les paramètres doivent être un dictionnaire")

        # Envoi de la requête API
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        response = client.request(r)

        # Validation approfondie de la réponse
        if not isinstance(response, dict):
            logger.error(f"Réponse API invalide pour {pair}: Type {type(response)}")
            return []

        if 'candles' not in response:
            logger.error(f"Clé 'candles' manquante pour {pair}: {response.keys()}")
            return []

        candles = response['candles']
        if not isinstance(candles, list):
            logger.error(f"Format de bougies invalide pour {pair}: {type(candles)}")
            return []

        # Filtrage des bougies complètes
        valid_candles = []
        for c in candles:
            if isinstance(c, dict) and c.get('complete', False):
                valid_candles.append(c)
                
        return valid_candles

    except oandapyV20.exceptions.V20Error as e:
        logger.error(f"Erreur API OANDA ({pair}): {e.code} {e.msg}")
        return []
    except Exception as e:
        logger.error(f"Erreur critique fetch_candles: {str(e)}")
        return []

def update_closed_trades():
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        response = client.request(r)
        current_trades = {t['instrument'] for t in response.get('trades', [])}
        
        # Retirer les paires qui ne sont plus actives
        closed_pairs = active_trades - current_trades
        if closed_pairs:
            logger.info(f"❌ Trades fermés détectés: {closed_pairs}")
            active_trades.clear()
            active_trades.update(current_trades)
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour des trades fermés: {e}")


def check_breakout(pair, price, window=5):
    """
    Vérifie si le prix actuel réalise un breakout par rapport aux bougies précédentes.
    
    Args:
        pair (str): La paire de trading (ex: "XAU_USD")
        price (float): Le prix actuel
        window (int): Nombre de bougies à analyser

    Returns:
        str: "breakout_up" ou "breakout_down", None si pas de breakout
    """
    try:
        # Paramètres pour récupérer les bougies
        params = {"granularity": "M15", "count": window}
        
        # Récupère les bougies via la fonction fetch_candles
        candles = fetch_candles(pair, params["granularity"], params)
        
        if not candles or len(candles) < window:
            raise ValueError(f"Pas assez de données pour {pair}")
        
        highs = [float(candle['mid']['h']) for candle in candles if candle['complete']]
        lows  = [float(candle['mid']['l']) for candle in candles if candle['complete']]

        # Détecte breakout haussier ou baissier
        if price > max(highs[:-1]):
            return "breakout_up"
        elif price < min(lows[:-1]):
            return "breakout_down"
        else:
            return None
    
    except Exception as e:
        print(f"[ERREUR] check_breakout({pair}, {price}): {e}")
        return None


def analyze_htf(pair, params=None):
    """Analyse des Order Blocks haute timeframe avec validation de type stricte"""
    try:
        # ... (code existant)
        
        ob_zones = []
        for i in range(2, len(candles)):
            # ... (logique existante de détection OB)
            
            # Validation stricte du type de sortie
            if isinstance(zone_high, (int, float)) and isinstance(zone_low, (int, float)):
                ob_zones.append((round(zone_high, 5), round(zone_low, 5)))
            else:
                logger.warning(f"Format de zone invalide ignoré pour {pair}")

        return ob_zones[-3:] if ob_zones else [(0.0, 0.0)]  # Tuple par défaut

    except Exception as e:
        logger.error(f"Erreur analyse_htf: {str(e)}")
        return [(0.0, 0.0)]  # Retourne un tuple vide sécurisé

def process_zone(zone, zone_type):
    """Conversion robuste en tuple avec fallback sécurisé"""
    try:
        # Gestion des numpy types
        if hasattr(zone, 'item'):
            zone = zone.item()
            
        # Conversion des singles valeurs en range
        if isinstance(zone, (int, float)):
            spread = 0.00015 if "_JPY" not in pair else 0.015
            return (round(zone - spread, 5), round(zone + spread, 5))
            
        # Conversion des dicts
        if isinstance(zone, dict):
            return (float(zone.get('low', 0)), float(zone.get('high', 0)))
            
        # Validation finale
        if isinstance(zone, (list, tuple)) and len(zone) >= 2:
            return (round(float(zone[0]), 5), round(float(zone[1]), 5))

        logger.warning(f"Structure de zone inconnue: {type(zone)}")
        return (0.0, 0.0)  # Fallback sécurisé
        
    except Exception as e:
        logger.error(f"Erreur process_zone: {str(e)}")
        return (0.0, 0.0)
def detect_ltf_patterns(candles, pairs):
    """Détecte des patterns sur des timeframes basses (pin bars, engulfing patterns)"""
    patterns_detected = []

    # Détection des pin bars avec le paramètre pair
    pin_bars = detect_pin_bars(candles, pair)
    if pin_bars:
        patterns_detected.extend(("Pin Bar", i) for i in range(len(candles)) if i < len(pin_bars))    

    # Pin bar detection
    for i in range(1, len(candles)):
        body = abs(float(candles[i]['mid']['c']) - float(candles[i]['mid']['o']))
        wick_top = float(candles[i]['mid']['h']) - max(float(candles[i]['mid']['c']), float(candles[i]['mid']['o']))
        wick_bottom = min(float(candles[i]['mid']['c']), float(candles[i]['mid']['o'])) - float(candles[i]['mid']['l'])
        if wick_top > 2 * body or wick_bottom > 2 * body:
            patterns_detected.append(("Pin Bar", i))

    # Engulfing pattern detection
    for i in range(1, len(candles)):
        prev_body = abs(float(candles[i - 1]['mid']['c']) - float(candles[i - 1]['mid']['o']))
        current_body = abs(float(candles[i]['mid']['c']) - float(candles[i]['mid']['o']))
        if current_body > prev_body:
            if (float(candles[i]['mid']['c']) > float(candles[i]['mid']['o']) and
                float(candles[i]['mid']['c']) > float(candles[i - 1]['mid']['h'])):
                patterns_detected.append(("Bullish Engulfing", i))
            elif (float(candles[i]['mid']['c']) < float(candles[i]['mid']['o']) and
                  float(candles[i]['mid']['o']) < float(candles[i - 1]['mid']['l'])):
                patterns_detected.append(("Bearish Engulfing", i))

    return patterns_detected

MIN_DISTANCE = 0.0001  # Distance minimale acceptable

def calculate_position_size(account_balance, entry_price, stop_loss_price, pair):
    """Version with zero division protection"""
    try:
        if None in [account_balance, entry_price, stop_loss_price]:
            return 0
            
        risk_amount = min(account_balance * (RISK_PERCENTAGE / 100), RISK_AMOUNT_CAP)
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            logger.error(f"Risk per unit too small for {pair}: {risk_per_unit}")
            return 0
            
        if "_JPY" in pair:
            return round(risk_amount / risk_per_unit)
        elif pair in ["XAU_USD", "XAG_USD"]:
            return round(risk_amount / risk_per_unit, 2)
        else:
            return round((risk_amount / risk_per_unit) / 10000)
    except Exception as e:
        logger.error(f"Erreur calcul position {pair}: {str(e)}")
        return 0

def process_zone(zone, zone_type):
    """Gère les différents formats de zones"""
    try:
        if isinstance(zone, float):
            return (zone - 0.0001, zone + 0.0001)  # Crée un mini-range
            
        if isinstance(zone, (list, tuple)):
            return (float(min(zone)), float(max(zone)))
            
        if isinstance(zone, dict):
            return (float(zone.get('low', 0)), float(zone.get('high', 0)))
            
        logger.warning(f"Type de zone non supporté: {type(zone)}")
        return (0, 0)
    except Exception as e:
        logger.error(f"Erreur traitement zone: {str(e)}")
        return (0, 0)

def process_pin_bar(pin_bar_data):
    """Traite les données d'une Pin Bar."""
    try:
        # Extraction des valeurs numériques
        high = validate_numeric(pin_bar_data.get("high"), "high")
        low = validate_numeric(pin_bar_data.get("low"), "low")
        close = validate_numeric(pin_bar_data.get("close"), "close")

        if None in [high, low, close]:
            logger.warning("Données Pin Bar invalides. Traitement ignoré.")
            return None

        # Calcul de la taille de la barre
        bar_size = high - low
        tail_size = max(close - low, high - close)

        # Vérification si c'est une vraie Pin Bar
        if tail_size > bar_size * 0.5:
            return {"type": "Pin Bar", "size": bar_size}
        else:
            logger.info("Pin Bar détectée mais trop petite pour être considérée.")
            return None

    except Exception as e:
        logger.error(f"Erreur lors du traitement de la Pin Bar: {e}")
        return None

def update_stop_loss(trade_id, new_stop_loss):
    """Met à jour le Stop Loss d'une position existante"""
    try:
        data = {
            "order": {
                "stopLoss": {
                    "price": "{0:.5f}".format(new_stop_loss),
                    "timeInForce": "GTC"
                }
            }
        }
        r = orders.OrderReplace(accountID=OANDA_ACCOUNT_ID, orderSpecifier=trade_id, data=data)
        response = client.request(r)
        if 'orderCancelTransaction' in response:
            logger.info(f"Stop loss mis à jour: {new_stop_loss}")
        else:
            logger.error(f"Échec mise à jour SL: {response}")
    except Exception as e:
        logger.error(f"Erreur mise à jour SL: {e}")

def should_open_trade(pair, rsi, macd, macd_signal, breakout_detected, price, key_zones, atr, candles):
    """Détermine si les conditions pour ouvrir un trade sont remplies"""
    try:
        # Initialisations
        direction = None
        reasons = []
        signals = {
            "rsi": False,
            "macd": False,
            "price_action": False,
            "breakout": False,
            "zone": False
        }

        # 1. Validations de base
        if any(v is None for v in [rsi, macd, macd_signal]):
            logger.error(f"Indicateurs manquants pour {pair}")
            return False

        settings = PAIR_SETTINGS.get(pair, PAIR_SETTINGS["DEFAULT"])
        if atr < settings["min_atr"]:
            logger.info(f"Volatilité trop faible pour {pair} (ATR={atr:.2f})")
            return False

        # 2. Vérification des RSI multi-timeframe (uniquement pour XAU_USD)
        if pair == "XAU_USD":
            rsi_conditions = check_rsi_conditions(pair)
            reasons.append(f"RSI H1: {rsi_conditions['h1']:.1f}, M15: {rsi_conditions['m15']:.1f}")

        # 3. Détection des signaux
        # Vérification des zones clés
        for raw_zone in key_zones:
            zone = process_zone(raw_zone, 'ob')
            if abs(price - zone[0]) <= RETEST_ZONE_RANGE:
                signals["zone"] = True
                reasons.append("Prix dans zone clé")
                break
        
        # RSI
        if rsi < settings["rsi_oversold"] and closes[-1] < closes[-3]:
            signals["rsi"] = True
            reasons.append(f"RSI {rsi:.1f} < {settings['rsi_oversold']} (suracheté)")
        elif rsi > settings["rsi_overbought"]:
            signals["rsi"] = True
            reasons.append(f"RSI {rsi:.1f} > {settings['rsi_overbought']} (survendu)")

        # MACD
        macd_crossover = (macd > macd_signal and macd_signal > 0) or (macd < macd_signal and macd_signal < 0)
        if macd_crossover:
            signals["macd"] = True
            reasons.append("Croisement MACD confirmé")

        # Breakout
        if breakout_detected and atr > settings["min_atr"] * 1.5:
            signals["breakout"] = True
            reasons.append("Breakout fort détecté")

        # Price Action
        pin_bars = detect_pin_bars(candles, pair)  # Ajout du paramètre pair
        engulfing_patterns = detect_engulfing_patterns(candles)
        
        if pin_bars:
            signals["price_action"] = True
            reasons.append(f"Pin Bar détectée (ratio: {pin_bars[-1]['ratio']}x)")
        if engulfing_patterns:
            signals["price_action"] = True
            reasons.append("Engulfing Pattern détecté")

        # 4. Validation des signaux
        required = CONFIRMATION_REQUIRED.get(pair, 1)
        if sum(signals.values()) < required:
            logger.info(f"Signaux insuffisants ({sum(signals.values())}/{required})")
            return False

        # 5. Détermination de la direction
        bullish_signals = sum([
            signals["rsi"] and rsi < settings["rsi_oversold"],
            signals["macd"] and macd > macd_signal,
            signals["price_action"]
        ])
        
        bearish_signals = sum([
            signals["rsi"] and rsi > settings["rsi_overbought"],
            signals["macd"] and macd <= macd_signal,
            signals["price_action"]
        ])

        if bullish_signals >= bearish_signals:
            direction = "buy"
        else:
            direction = "sell"

        # 6. Validations finales
        if is_ranging(pair) and not breakout_detected:
            logger.warning("Marché en range")
            return False

        if direction and not is_trend_aligned(pair, direction):
            logger.warning("Désalignement des tendances HTF")
            return False
        if not check_momentum(pair):
            logger.info(f"Momentum défavorable pour {pair}")
            return False
        # Avant la vérification des zones
        if not isinstance(zones, list):
            logger.warning("Format de zones invalide")
            return False
        if not confirm_signal(pair, direction):
            logger.info(f"Confirmation échouée pour {pair}")
            return False

        if direction and any([signals["breakout"], signals["price_action"], signals["zone"]]):
            logger.info(f"✅ Signal {direction.upper()} confirmé - Raisons: {', '.join(reasons)}")
            return direction

        logger.info("❌ Signaux contradictoires")
        logger.info(f"Analyse {pair} - RSI: {rsi}, ATR: {atr}, Trend aligned: {is_trend_aligned(pair, direction)}")
        logger.info(f"Signaux détectés: {signals}")
        return False
        

    except Exception as e:
        logger.error(f"Erreur dans should_open_trade pour {pair}: {str(e)}")
        return False

def validate_spread(pair):
    bid = get_current_price(pair, 'bid')
    ask = get_current_price(pair, 'ask')
    spread = ask - bid
    
    if "_JPY" in pair:
        return spread <= 0.03  # 3 pips max
    elif "XAU" in pair:
        return spread <= 0.5   # 50 cents max
    else:
        return spread <= 0.0003

def detect_reversal_patterns(candles):
    """Détecte des patterns de retournement (pin bars, engulfings)"""
    reversal_patterns = []

    for i in range(1, len(candles)):
        open_price = float(candles[i]['mid']['o'])
        close_price = float(candles[i]['mid']['c'])
        high_price = float(candles[i]['mid']['h'])
        low_price = float(candles[i]['mid']['l'])

        # Pin bar
        body = abs(open_price - close_price)
        wick = high_price - low_price
        if wick > 3 * body:
            reversal_patterns.append("Pin Bar")

        # Engulfing
        prev_close = float(candles[i - 1]['mid']['c'])
        prev_open = float(candles[i - 1]['mid']['o'])
        if close_price > prev_open and open_price < prev_close:
            reversal_patterns.append("Bullish Engulfing")
        elif close_price < prev_open and open_price > prev_close:
            reversal_patterns.append("Bearish Engulfing")

    return reversal_patterns

# Configuration des distances minimales par paire
MIN_TRAILING_STOP_LOSS_DISTANCE = {
    "XAU_USD": 0.5,  # Valeur minimale pour XAU/USD
    "XAG_USD": 0.1,  # Valeur minimale pour XAG/USD
    "EUR_USD": 0.0005,  # Valeur minimale pour EUR/USD
    "GBP_JPY": 0.05,  # Valeur minimale pour GBP/JPY
    "USD_JPY": 0.05,  # Valeur minimale pour USD/JPY
    "DEFAULT": 0.0005  # Valeur par défaut pour les autres paires
}

logger = logging.getLogger(__name__)

def validate_trailing_stop_loss_distance(pair, distance):
    """Valide la distance du Trailing Stop Loss."""
   
    min_distance = MIN_TRAILING_STOP_LOSS_DISTANCE.get(pair, MIN_TRAILING_STOP_LOSS_DISTANCE["DEFAULT"])
    if distance < min_distance:
        logger.warning(f"Distance Trailing Stop Loss ({distance}) inférieure à la valeur minimale autorisée ({min_distance}). Ajustement automatique.")
        return min_distance
    return distance

def place_trade(*args, **kwargs):
    """Désactivée"""
    logger.info("Fonction désactivée - Mode alerte email seulement")

    # 1. Vérifier si un trade est déjà actif sur cette paire
    if pair in active_trades:
        logger.info(f"🚫 Trade déjà actif sur {pair}. Aucun nouveau trade ouvert.")
        return None
    if None in [entry_price, stop_loss_price, direction, atr, account_balance]:
        logger.error("Paramètres manquants pour le trade")
        return None

    # 2. Conversion spécifique pour certaines paires (exemple : GBP_JPY)
    PAIR_SETTINGS = {
        "XAU_USD": {"decimal": 2, "min_distance": 0.5},
        "XAG_USD": {"decimal": 2, "min_distance": 0.1},
        "EUR_USD": {"decimal": 5, "min_distance": 0.0005},
        "GBP_JPY": {"decimal": 3, "min_distance": 0.05},
        "USD_JPY": {"decimal": 3, "min_distance": 0.05},
        "DEFAULT": {"decimal": 5, "min_distance": 0.0005}  # Valeur par défaut
    }
    settings = PAIR_SETTINGS.get(pair, PAIR_SETTINGS["DEFAULT"])

    # Appliquer l'arrondi spécifique à chaque paire
    entry_price = round(entry_price, settings["decimal"])
    stop_loss_price = round(stop_loss_price, settings["decimal"])

    # 3. Calcul de la taille de position
    units = calculate_position_size(account_balance, entry_price, stop_loss_price, pair)
    if units <= 0:
        logger.error(f"❌ Taille de position invalide ({units}). Aucun trade exécuté.")
        return None

    # 4. Calcul des niveaux de Stop Loss et Take Profit
    take_profit_price = round(
        entry_price + (ATR_MULTIPLIER_TP * atr if direction == "buy" else -ATR_MULTIPLIER_TP * atr),
        settings["decimal"]
    )

    # Validation de la distance minimale pour le Stop Loss
    min_distance = settings["min_distance"]
    if abs(entry_price - stop_loss_price) < min_distance:
        logger.warning(f"Distance SL trop faible (<{min_distance}), ajustement automatique.")
        stop_loss_price = round(
            entry_price - (min_distance if direction == "buy" else -min_distance),
            settings["decimal"]
        )

    # 5. Validation de la distance initiale du Trailing Stop Loss
    trailing_stop_loss_distance = TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001
    validated_trailing_distance = validate_trailing_stop_loss_distance(pair, trailing_stop_loss_distance)

    # 6. Préparation des données pour l'ordre
    order_data = {
        "order": {
            "instrument": pair,
            "units": str(int(units)) if direction == "buy" else str(-int(units)),
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {
                "price": f"{stop_loss_price:.{settings['decimal']}f}"
            },
            "takeProfitOnFill": {
                "price": f"{take_profit_price:.{settings['decimal']}f}"
            },
            "trailingStopLossOnFill": {
                "distance": f"{validated_trailing_distance:.5f}",
                "timeInForce": "GTC"
            }
        }
    }

    # Journalisation des détails du trade
    logger.info(f"""
    📈 SIGNAL CONFIRMÉ - {pair} {direction.upper()} ✅
    ░ Entrée: {entry_price:.5f}
    ░ SL: {stop_price:.5f} (Distance: {abs(entry_price-stop_price):.2f} pips)
    ░ TP: {take_profit:.5f} (RR: {(take_profit-entry_price)/(entry_price-stop_price):.1f}:1)
    ░ ATR H1: {atr_h1:.5f}
    ░ Alignement tendances: {is_trend_aligned(pair, direction)}
    ░ Régime marché: {'Trending' if not is_ranging(pair) else 'Range'}
    """)
    # 7. Exécution en mode réel
    if not SIMULATION_MODE:
        try:
            r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
            response = client.request(r)
            
            if 'orderCreateTransaction' in response:
                trade_id = response['orderCreateTransaction']['id']
                logger.info(f"✅ Trade exécuté (ID: {trade_id})")
                return trade_id
            else:
                logger.error(f"❌ Réponse OANDA anormale: {response}")
                return None

        except oandapyV20.exceptions.V20Error as e:
            error_details = e.msg if hasattr(e, "msg") else str(e)
            logger.error(f"Erreur OANDA: {error_details}")
            
            # Si l'erreur est liée à la distance minimale
            if "TRAILING_STOP_LOSS_ON_FILL_PRICE_DISTANCE_MINIMUM_NOT_MET" in error_details:
                logger.warning("La distance minimale du Trailing Stop Loss n'est pas respectée. Réessayer avec une distance ajustée.")
                adjusted_distance = validate_trailing_stop_loss_distance(pair, validated_trailing_distance * 2)
                order_data["order"]["trailingStopLossOnFill"]["distance"] = f"{adjusted_distance:.5f}"
                
                try:
                    response = client.request(orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data))
                    logger.info(f"Trade exécuté après ajustement: {response}")
                    return response['orderCreateTransaction']['id']
                except Exception as e:
                    logger.error(f"Échec de l'exécution après ajustement: {e}")
                    return None

    # 8. Mode simulation
    else:
        trade_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "pair": pair,
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_loss_price,
            "take_profit": take_profit_price,
            "units": units,
            "atr": atr
        }
        trade_history.append(trade_info)
        active_trades.add(pair)
        logger.info("📊 Trade simulé (non exécuté)")
        return f"SIM_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
def validate_numeric(value, name):
    """Valide qu'une valeur est numérique."""
    try:
        return float(value)
    except ValueError:
        logger.error(f"❌ Erreur de formatage {name}: '{value}' n'est pas un nombre.")
        return None
def calculate_adx(highs, lows, closes, window=14):
    """Calcule l'Average Directional Index (ADX)"""
    try:
        # Calcul des mouvements directionnels
        plus_dm = []
        minus_dm = []
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)

        # Calcul des True Ranges
        tr = [max(highs[i], closes[i-1]) - min(lows[i], closes[i-1]) for i in range(1, len(highs))]
        
        # Calcul des indicateurs directionnels
        plus_di = 100 * (pd.Series(plus_dm).rolling(window).sum() / pd.Series(tr).rolling(window).sum()).fillna(0)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window).sum() / pd.Series(tr).rolling(window).sum()).fillna(0)
        
        # Calcul de l'ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.rolling(window).mean().iloc[-1]
        
        return round(adx, 2)
    except Exception as e:
        logger.error(f"Erreur calcul ADX: {str(e)}")
        return 0

def get_atr(pair, timeframe="H1", period=14):
    """Get the ATR for a given pair and timeframe"""
    try:
        params = {"granularity": timeframe, "count": period*2, "price": "M"}
        candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params))["candles"]
        highs = [float(c["mid"]["h"]) for c in candles if c["complete"]]
        lows = [float(c["mid"]["l"]) for c in candles if c["complete"]]
        closes = [float(c["mid"]["c"]) for c in candles if c["complete"]]
        
        if not highs or not lows or not closes:
            logger.warning(f"No complete candles for {pair} {timeframe}")
            return 0.0
            
        return calculate_atr(highs, lows, closes, period)
    except Exception as e:
        logger.error(f"Error getting ATR for {pair}: {str(e)}")
        return 0.0


def detect_liquidity_zones(prices, bandwidth=0.5):
    """Detect liquidity zones using KDE"""
    try:
        kde = gaussian_kde(prices, bw_method=bandwidth)
        x = np.linspace(min(prices), max(prices), 100)
        density = kde(x)
        peaks = np.argwhere(density == np.max(density)).flatten()
        return x[peaks[0]] if len(peaks) > 0 else np.median(prices)
    except Exception as e:
        logger.error(f"Error detecting liquidity zones: {e}")
        return np.median(prices)

def calculate_sl(entry_price, direction, atr_h1):
    """Calculate stop loss based on ATR"""
    if atr_h1 <= 0:
        logger.error("Invalid ATR value for SL calculation")
        return None
    
    if direction.upper() == "BUY":
        return entry_price - (1.5 * atr_h1)
    else:
        return entry_price + (1.5 * atr_h1)

def calculate_vwap(closes, volumes):
    """Calcule le Volume Weighted Average Price"""
    try:
        cumulative_pv = np.cumsum([c * v for c, v in zip(closes, volumes)])
        cumulative_volume = np.cumsum(volumes)
        return cumulative_pv[-1] / cumulative_volume[-1]
    except ZeroDivisionError:
        return closes[-1]
    except Exception as e:
        logger.error(f"Erreur calcul VWAP: {str(e)}")
        return 0

def update_trailing_stop(*args, **kwargs):
    """Désactivée"""
    pass


# RSI robuste avec gestion des divisions par zéro
def calculate_rsi(prices, window=14):
    """RSI calculation with zero division protection"""
    if len(prices) < window + 1:
        return 50  # Default neutral value
    
    deltas = np.diff(prices)
    seed = deltas[:window]
    
    # Handle division by zero cases
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    
    if down == 0:
        return 100.0 if up != 0 else 50.0
    
    rs = up/down
    return 100.0 - (100.0/(1.0 + rs))

# ... (conservons les imports et configurations existants)

class LiquidityHunter:
    def __init__(self):
        """Initialisation avec valeurs par défaut"""
        self.liquidity_zones = {}  # Stocke les zones de liquidité par paire
        self.session_ranges = {}   # Stocke les ranges de session
        self.price_cache = {}      # Cache des derniers prix
        self.last_update = {}      # Timestamp de dernière mise à jour
        self.pip_values = {        # Valeur d'un pip par paire
            "XAU_USD": 0.1,
            "XAG_USD": 0.1,
            "CAD_JPY": 0.01,
            "USD_JPY": 0.01,
            "DEFAULT": 0.0001
        }

    def _get_volume_data(self, pair):
        """Récupère les données de volume historiques"""
        try:
            params = {"granularity": "H1", "count": 20}
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            candles = client.request(r)["candles"]
            
            volumes = [float(c['volume']) for c in candles if c['complete']]
            
            return {
                "last": volumes[-1] if volumes else 0,
                "average": np.mean(volumes) if volumes else 0,
                "max": np.max(volumes) if volumes else 0,
                "min": np.min(volumes) if volumes else 0
            }
        except Exception as e:
            logger.error(f"Erreur récupération volume {pair}: {str(e)}")
            return {"last": 0, "average": 0, "max": 0, "min": 0}
            
    def _is_london_session(self):
        """Vérifie si c'est la session londonienne (8h-17h CET)"""
        now = datetime.utcnow()
        return (6 <= now.hour < 15)  # 8h-17h CET en UTC

    def _is_new_york_session(self):
        """Vérifie si c'est la session new-yorkaise (13h-22h CET)"""
        now = datetime.utcnow()
        return (12 <= now.hour < 21)

    def get_pip_value(self, pair):
        """Retourne la valeur d'un pip pour la paire"""
        return self.pip_values.get(pair, self.pip_values["DEFAULT"])

    def get_cached_price(self, pair, cache_duration=5):
        """Version optimisée avec gestion des paires spécifiques"""
        try:
            # Vérification du cache
            current_time = time.time()
            cached_data_valid = (
                pair in self.price_cache and 
                pair in self.last_update and
                current_time - self.last_update[pair] < cache_duration
            )
            
            if cached_data_valid:
                return self.price_cache[pair]
                
            # Récupération du prix
            price = get_current_price(pair)
            if price is not None:
                self.price_cache[pair] = price
                self.last_update[pair] = current_time
                logger.debug(f"Cache mis à jour pour {pair}: {price}")
                
            return price
            
        except Exception as e:
            logger.error(f"Erreur critique get_cached_price {pair}: {str(e)}")
            return None

    def update_asian_range(self, pair):
        """Met à jour le range asiatique avec une gestion plus robuste"""
        asian_high, asian_low = get_asian_session_range(pair)
        if asian_high and asian_low:
            self.session_ranges[pair] = {
                'high': asian_high,
                'low': asian_low,
                'mid': (asian_high + asian_low) / 2
            }
            return True
        return False
    
    def analyze_htf_liquidity(self, pair):
        """Analyse des zones de liquidité haute timeframe"""
        try:
            # Récupération des données avec timeout
            params = {"granularity": "H4", "count": 100, "price": "M"}
            candles = fetch_candles(pair, "H4", params)
        
            if not isinstance(candles, list):
                logger.error(f"Type de bougies invalide pour {pair}: {type(candles)}")
                return False

            # Validation des données
            if len(candles) < 50:
                logger.warning(f"Données insuffisantes pour {pair} ({len(candles)} bougies)")
                return False

            # Traitement des volumes
            volumes = []
            closes = []
            for c in candles:
                if isinstance(c, dict) and c.get('complete', False):
                    try:
                        volumes.append(float(c['volume']))
                        closes.append(float(c['mid']['c']))
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Erreur traitement bougie: {e}")

            # Détection des zones de volume
            high_volume_zones = [
                closes[i] 
                for i in range(len(volumes)) 
                if volumes[i] > np.mean(volumes) * 1.5
            ]

            # Mise à jour du cache
            self.liquidity_zones[pair] = {
                'volume_zones': high_volume_zones,
                'last_update': datetime.utcnow()
            }

            return True

        except Exception as e:
            logger.error(f"Erreur analyse liquidité HTF {pair}: {str(e)}")
            return False
    
    def _is_price_near_zone(self, price, zone, pair):
        """Vérifie si le prix est proche d'une zone."""
        try:
            # Vérification du type de zone
            if isinstance(zone, (tuple, list)):
                zone_min, zone_max = float(min(zone)), float(max(zone))
            elif isinstance(zone, dict):
                zone_min, zone_max = float(zone.get('low')), float(zone.get('high'))
            else:
                logger.error(f"Type de zone invalide pour {pair}: {type(zone)}")
                return False

            # Calcul du seuil dynamique
            atr = calculate_atr_for_pair(pair)
            base_pips = 15 if "_JPY" in pair else 10
            dynamic_threshold = min(base_pips, atr * 0.33)  # Max 1/3 de l'ATR
            pip_value = 0.01 if "_JPY" in pair else 0.0001
            threshold = dynamic_threshold * pip_value

            # Vérification de la proximité
            return (zone_min - threshold) <= price <= (zone_max + threshold)

        except Exception as e:
            logger.error(f"Erreur _is_price_near_zone pour {pair}: {e}")
            return False

    def _calculate_zone_weight(self, price, zone):
        """Calcule un poids pour une zone en fonction de sa proximité avec le prix."""
        if isinstance(zone, (list, tuple)):
            distance = min(abs(price - z) for z in zone)
        else:
            distance = abs(price - zone)
        # Plus la zone est proche, plus le poids est élevé
        return max(0, 100 - distance * 1000)  # Ajustez le facteur selon vos besoins

    def find_best_opportunity(self, pair):
        current_price = self.get_cached_price(pair)
        if current_price is None:
            return None

        zones = self.liquidity_zones.get(pair, {})
        session = self.session_ranges.get(pair, {})
        opportunities = []

        # Priorité 1: FVG
        for fvg in zones.get("fvg", []):
            weight = self._calculate_zone_weight(current_price, fvg)
            if weight > 50:  # Seuil minimum
                opportunities.append({"zone": fvg, "type": "fvg", "weight": weight})

        # Priorité 2: Order Blocks
        for ob in zones.get("ob", []):
            weight = self._calculate_zone_weight(current_price, ob)
            if weight > 50:
                opportunities.append({"zone": ob, "type": "ob", "weight": weight})

        # Priorité 3: Niveaux de session asiatique
        for level in ["high", "low", "mid"]:
            zone = session.get(level)
            if zone:
                weight = self._calculate_zone_weight(current_price, zone)
                if weight > 50:
                    opportunities.append({"zone": zone, "type": "session", "weight": weight})

        # Trier les opportunités par poids
        opportunities.sort(key=lambda x: x["weight"], reverse=True)

        # Retourner la meilleure opportunité
        if opportunities:
            best = opportunities[0]
            return self._prepare_trade(pair, current_price, best["zone"], best["type"])

        return None
    
    def _confirm_zone(self, pair, zone, zone_type):
        try:
            # Récupère les données M5
            params = {"granularity": "M5", "count": 20, "price": "M"}
            candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params))['candles']
        
            # Vérifie les patterns de prix
            patterns = detect_ltf_patterns(candles, pair)  # Ajout du paramètre pair
            pin_bars = detect_pin_bars(candles, pair)
        
            # Vérifie le momentum
            rsi = calculate_rsi([float(c['mid']['c']) for c in candles if c['complete']])
            atr = calculate_atr_for_pair(pair)
        
            # Conditions de confirmation selon le type de zone
            if zone_type in ['fvg', 'ob']:
                score = 0
                if any(p[0] in ['Pin Bar', 'Engulfing'] for p in patterns):
                    score += 20
                if rsi > 40:
                    score += 15
                volume_data = self._get_volume_data(pair)
                if volume_data["last"] > volume_data["max"] * 0.8:
                    score += 20
                elif volume_data["last"] < volume_data["min"] * 1.2:
                    score -= 15
                vwap = calculate_vwap(pair)
                if vwap and abs(current_price - vwap) < atr * 0.5:
                    score += 15
                return score >= 50  # Minimum pour confirmation
        
            return False
        except Exception as e:
            logger.error(f"Erreur dans _confirm_zone pour {pair}: {str(e)}")
            return False
    
      
    
    def _prepare_trade(self, pair, price, zone, zone_type):
    
        try:
            # Gestion explicite des types de zones
            price = float(price)
            if isinstance(zone, (list, tuple)):
                zone = [float(z) for z in zone]
                # Pour les FVG (zone de type [high, low])
                zone_high = float(zone[0])
                zone_low = float(zone[1])
                direction = 'buy' if price < zone_low else 'sell'  # Stratégie de breakout
            elif isinstance(zone, dict):
                # Pour les Order Blocks (structure complexe)
                zone_value = float(zone['price'])
                direction = 'buy' if price > zone_value else 'sell'
            else:
                # Pour les niveaux simples
                zone_value = float(zone)
                direction = 'buy' if price < zone_value else 'sell'

            # Calculs sécurisés
            atr = float(calculate_atr_for_pair(pair))
        
            if atr <= 0:
                raise ValueError(f"ATR invalide: {atr}")

            # Calcul SL/TP avec vérification de type
            stop_loss = (
                price - 1.5 * atr if direction == 'buy' 
                else price + 1.5 * atr
            )
            take_profit = (
                price + 3 * atr if direction == 'buy'
                else price - 3 * atr
            )

            return {
                'pair': pair,
                'direction': direction,
                'entry': float(price),
                'sl': float(stop_loss),
                'tp': float(take_profit),
                'zone_type': zone_type,
                'confidence': self._calculate_confidence(pair, price, zone_type, zone, direction),
                'atr': atr
            }

        except (TypeError, ValueError, IndexError) as e:
            logger.error(f"Erreur préparation trade {pair}: {str(e)}")
            logger.debug(f"Détails erreur: price={price} | zone={zone} | type={type(zone)}")
            return None
    
   

    def _calculate_confidence(self, pair, price, zone_type, zone, direction):
        try:
            score = 0
            atr = calculate_atr_for_pair(pair)

            # 1. Validation des données de base
            candles = fetch_candles(pair, "M5", {"granularity": "M5", "count": 20})
            if not candles:
                logger.warning(f"Aucune bougie disponible pour {pair}")
                return 0

            # 2. Calcul sécurisé des closes
            closes = []
            for c in candles:
                if c['complete'] and 'mid' in c and 'c' in c['mid']:
                    try:
                        closes.append(float(c['mid']['c']))
                    except (ValueError, TypeError):
                        continue

            # 3. Calcul Bollinger Bands protégé
            bb_upper, bb_lower = calculate_bollinger_bands(closes)
            bb_percentage = 0.5  # Valeur neutre par défaut
        
            if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
                try:
                    bb_percentage = (price - bb_lower) / (bb_upper - bb_lower)
                except ZeroDivisionError:
                    logger.warning("Bollinger Bands collapse (division par zéro)")

            # 4. Logique de score révisée
            if bb_percentage > 0.8:
                score -= 25
            elif bb_percentage < 0.2:
                score += 20

            # 5. Vérification de proximité
            is_near = False
            try:
                is_near = self._is_price_near_zone(price, zone, pair)
            except Exception as e:
                logger.error(f"Erreur vérification proximité: {str(e)}")
        
            if is_near:
                score += 20

            # 6. Détection Pin Bars sécurisée
            pin_bars = []
            try:
                pin_bars = detect_pin_bars(candles, pair) or []
            except Exception as e:
                logger.error(f"Erreur détection Pin Bars: {str(e)}")

            if len(pin_bars) >= 1:
                score += 10
                if any(p.get('ratio', 0) > 2 for p in pin_bars):
                    score += 5

            # 7. Calcul RSI protégé
            rsi = 50  # Valeur neutre
            try:
                if len(closes) >= 14:  # Minimum pour RSI
                    rsi = calculate_rsi(closes[-14:])
            except Exception as e:
                logger.error(f"Erreur calcul RSI: {str(e)}")

            # 8. Logique RSI révisée
            if zone_type == "fvg" and rsi > 55:
                score += 10
            elif zone_type == "ob" and 45 < rsi < 55:
                score += 5

            # 9. Gestion volume sécurisée
            volume_data = {"last": 0, "average": 1}  # Valeurs par défaut
            try:
                volume_data = self._get_volume_data(pair) or volume_data
            except Exception as e:
                logger.error(f"Erreur données volume: {str(e)}")

            try:
                if volume_data["last"] > volume_data.get("average", 1) * 1.2:
                    score += 15
                elif volume_data["last"] > volume_data.get("average", 1):
                    score += 5
            except KeyError:
                logger.warning("Clés manquantes dans volume_data")

            # 10. Vérification finale tendance
            try:
                zone_ref = zone[0] if isinstance(zone, (list, tuple)) and len(zone) > 0 else price
                current_direction = "buy" if price < zone_ref else "sell"
                if is_trend_aligned(pair, current_direction):
                    score += 20
            except (IndexError, TypeError) as e:
                logger.error(f"Erreur détermination direction: {str(e)}")

            # 11. Limites et logging
            final_score = max(0, min(100, score))
            logger.debug(f"Confiance finale {pair}: {final_score}%")
        
            return final_score

        except Exception as e:
            logger.error(f"ERREUR CRITIQUE confiance {pair}: {str(e)}", exc_info=True)
            return 0
def initialize_pair_data(pair):
    """Initialisation sécurisée des données par paire"""
    try:
        # Vérification du type de réponse
        params = {"granularity": "H1", "count": HISTORY_LENGTH}
        candles = fetch_candles(pair, "H1", params)
        
        if not isinstance(candles, list):
            raise TypeError(f"Type inattendu pour candles: {type(candles)}")

        # Traitement des closes
        valid_closes = []
        for c in candles:
            if isinstance(c, dict) and 'mid' in c:
                try:
                    valid_closes.append(float(c['mid']['c']))
                except (KeyError, ValueError):
                    continue

        CLOSES_HISTORY[pair] = valid_closes[-HISTORY_LENGTH:]
        logger.info(f"Données initialisées pour {pair}: {len(valid_closes)} closes valides")

    except Exception as e:
        logger.error(f"Erreur initialisation {pair}: {str(e)}")
        CLOSES_HISTORY[pair] = []

    def calculate_vwap(pair, timeframe="H1", period=20):
        """Calcule le VWAP pour une paire donnée."""
        try:
            params = {"granularity": timeframe, "count": period * 2, "price": "M"}
            candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params))["candles"]
            closes = [float(c["mid"]["c"]) for c in candles if c["complete"]]
            volumes = [float(c["volume"]) for c in candles if c["complete"]]
            if not closes or not volumes:
                return None
            vwap = sum(c * v for c, v in zip(closes, volumes)) / sum(volumes)
            return round(vwap, 5)
        except Exception as e:
            logger.error(f"Erreur lors du calcul du VWAP pour {pair}: {e}")
            return None
    
    def _get_volume_data(self, pair):
        """Récupère les volumes récents (H1) pour une paire."""
        try:
            params = {"granularity": "H1", "count": 50}
            candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params))["candles"]
            volumes = [float(c["volume"]) for c in candles if c["complete"]]
            return {
                "last": volumes[-1] if volumes else 0,
                "average": np.mean(volumes) if volumes else 0,
                "max": np.max(volumes),
                "min": np.min(volumes)
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des volumes pour {pair}: {e}")
            return {"last": 0, "average": 0}

    def _get_zone_price(self, zone, zone_type):
        """Extract relevant price from zone data"""
        try:
            if zone_type == "fvg":
                return (zone[0] + zone[1]) / 2  # FVG midpoint
            elif zone_type == "ob":
                return zone[1]  # Use high of bearish OB / low of bullish OB
            else:  # Session levels
                return float(zone)
        except (TypeError, IndexError) as e:
            logger.error(f"Zone price error ({zone_type}): {str(e)}")
            return None

    def check_breakout(pair, price, window=3):
        """Vérifie une cassure récente sur un timeframe réduit"""
        try:
            params = {"granularity": "M15", "count": window}
            candles = fetch_candles(pair, params)
            highs = [float(c['mid']['h']) for c in candles]
            lows = [float(c['mid']['l']) for c in candles]

            # Cassure haussière
            if price > max(highs[:-1]):
                return "bullish_breakout"
            # Cassure baissière
            elif price < min(lows[:-1]):
                return "bearish_breakout"
            return None
        
        except Exception as e:
            logger.error(f"Erreur check_breakout: {str(e)}")
            return None
def clean_historical_data():
    """Nettoie les données historiques toutes les heures"""
    global CLOSES_HISTORY
    for pair in list(CLOSES_HISTORY.keys()):
        if len(CLOSES_HISTORY[pair]) > HISTORY_LENGTH * 1.2:
            CLOSES_HISTORY[pair] = CLOSES_HISTORY[pair][-HISTORY_LENGTH:]

def check_rsi_divergence(prices, rsi_values, lookback=14, min_trend_strength=0.1, min_rsi=30, max_rsi=70):
    """
    Détecte les divergences RSI avec validation de la tendance et filtrage avancé.
    
    Args:
        prices (list): Liste des prix de clôture (plus récents en dernier)
        rsi_values (list): Liste des valeurs RSI correspondantes
        lookback (int): Période d'analyse (nombre de bougies)
        min_trend_strength (float): Pente minimale pour considérer une tendance
        min_rsi (int): Filtre les divergences en zone neutre
        max_rsi (int): Filtre les divergences en zone neutre
    
    Returns:
        str: 'bearish', 'bullish' ou None
    """
    
   
           
    # Validation des données
    if len(prices) < lookback or len(rsi_values) < lookback:
        logger.warning("Données insuffisantes pour détecter une divergence")
        return None
    
    # Extraction des données récentes
    prices = prices[-lookback:]
    rsi = rsi_values[-lookback:]
    
    # Calcul de la pente des prix
    price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
    rsi_trend = np.polyfit(range(len(rsi)), rsi, 1)[0]
    
    # Détection des pics/creux avec gestion des plateaux
    def find_extremes(data, window=3, mode='high'):
        extremes = []
        for i in range(window, len(data)-window):
            if mode == 'high':
                if data[i] == max(data[i-window:i+window+1]):
                    extremes.append(i)
            else:
                if data[i] == min(data[i-window:i+window+1]):
                    extremes.append(i)
        return extremes
    
    # Détection des pics (highs) et creux (lows)
    price_highs = find_extremes(prices, mode='high')
    price_lows = find_extremes(prices, mode='low')
    rsi_highs = find_extremes(rsi, mode='high')
    rsi_lows = find_extremes(rsi, mode='low')
    
    # Vérification du nombre de points nécessaires
    if len(price_highs) < 2 or len(rsi_highs) < 2 or len(price_lows) < 2 or len(rsi_lows) < 2:
        return None
    
    # Derniers points significatifs
    last_price_high = prices[price_highs[-1]]
    prev_price_high = prices[price_highs[-2]]
    last_rsi_high = rsi[rsi_highs[-1]]
    prev_rsi_high = rsi[rsi_highs[-2]]
    
    last_price_low = prices[price_lows[-1]]
    prev_price_low = prices[price_lows[-2]]
    last_rsi_low = rsi[rsi_lows[-1]]
    prev_rsi_low = rsi[rsi_lows[-2]]
    
    # Divergence baissière
    bearish_conditions = [
        last_price_high > prev_price_high,  # Prix fait un plus haut
        last_rsi_high < prev_rsi_high,      # RSI fait un plus bas
        abs(price_trend) > min_trend_strength,  # Tendance significative
        last_rsi_high > max_rsi,            # Zone de sur-achat
        rsi[-1] < prev_rsi_high             # Confirmation momentum baissier
    ]
    
    # Divergence haussière
    bullish_conditions = [
        last_price_low < prev_price_low,    # Prix fait un plus bas
        last_rsi_low > prev_rsi_low,        # RSI fait un plus haut
        abs(price_trend) > min_trend_strength,  # Tendance significative
        last_rsi_low < min_rsi,             # Zone de sur-vente
        rsi[-1] > prev_rsi_low              # Confirmation momentum haussier
    ]
    
    # Validation des conditions
    if all(bearish_conditions):
        return 'bearish'
    elif all(bullish_conditions):
        return 'bullish'
    
    return None

def analyze_pair(pair):
    """Nouvelle version focalisée sur les liquidités"""
    logger.info(f"\n=== Analyse détaillée pour {pair} ===")
    
    global CLOSES_HISTORY
    
    # Initialiser l'historique si nécessaire
    if pair not in CLOSES_HISTORY:
        CLOSES_HISTORY[pair] = []

    try:
        # Récupération des données historiques
        params = {"granularity": "H1", "count": HISTORY_LENGTH, "price": "M"}
        candles = fetch_candles(pair, params["granularity"], params)

        # Filtrage des bougies valides
        valid_closes = []
        for c in candles:
            if isinstance(c, dict) and 'mid' in c and 'c' in c['mid'] and c.get('complete', False):
                valid_closes.append(float(c['mid']['c']))
            else:
                logger.warning(f"Bougie mal formatée ignorée pour {pair}: {c}")

        # Mise à jour de l'historique
        CLOSES_HISTORY[pair] = (CLOSES_HISTORY[pair] + valid_closes)[-HISTORY_LENGTH:]

        # Vérifier la qualité des données
        if len(CLOSES_HISTORY[pair]) < 50:
            logger.warning(f"Données insuffisantes pour {pair} ({len(CLOSES_HISTORY[pair])} bougies)")
            return

        # Calcul des RSI glissants
        rsi_values = [calculate_rsi(CLOSES_HISTORY[pair][i-14:i]) for i in range(14, len(CLOSES_HISTORY[pair]))]
        divergence = check_rsi_divergence(CLOSES_HISTORY[pair][-len(rsi_values):], rsi_values)

    except Exception as e:
        logger.error(f"Erreur initialisation données {pair}: {str(e)}")
        return

    # Vérifiez le prix actuel
    price = get_current_price(pair)
    if price is None:
        logger.error(f"Impossible de récupérer le prix pour {pair}")
        return
    logger.info(f"Prix actuel: {price}")
    
    # Vérifiez les indicateurs
    atr = calculate_atr_for_pair(pair)
    logger.info(f"ATR: {atr if atr else 'Erreur de calcul'}")
    
    # Vérifiez les tendances
    logger.info(f"Tendance H1 alignée BUY: {is_trend_aligned(pair, 'buy')}")
    logger.info(f"Tendance H1 alignée SELL: {is_trend_aligned(pair, 'sell')}")    
    
    # Vérification divergence
    if divergence == 'bearish':
        logger.warning("Divergence baissière détectée - annulation signal")
        return

    hunter = LiquidityHunter()

    try:
        # Récupération des données brutes
        r = instruments.InstrumentsCandles(
            instrument=pair,
            params={"granularity": "H1", "count": 50}
        )
        response = client.request(r)

        if not isinstance(response, dict) or 'candles' not in response:
            logger.error(f"Réponse API invalide pour {pair}: {type(response)}")
            logger.debug(f"Contenu de la réponse: {response}")
            return

        candles = response['candles']
        logger.info(f"Données reçues pour {pair}: {len(candles)} bougies")

    except Exception as e:
        logger.error(f"ERREUR CRITIQUE lors de la récupération des bougies: {str(e)}")
        logger.exception(e)
        return

    if not hunter.update_asian_range(pair):
        logger.warning(f"Échec mise à jour range asiatique pour {pair}")
        return
    
    if not hunter.analyze_htf_liquidity(pair):
        logger.warning(f"Échec analyse liquidité HTF pour {pair}")
        return
    
    try:
        opportunity = hunter.find_best_opportunity(pair)
    except KeyError as ke:
        logger.error(f"Erreur de structure dans les données: {ke}")
        return
    except Exception as e:
        logger.error(f"Erreur générique dans find_best_opportunity: {e}")
        return

    if not opportunity:
        logger.info(f"Aucune opportunité trouvée pour {pair}")
        return

    try:
        if opportunity['confidence'] < 50:
            logger.info(f"Confiance insuffisante ({opportunity['confidence']}%)")
            return
            
        if not validate_signal(pair, opportunity):
            logger.info(f"Validation finale échouée pour {pair}")
            return

        current_volume = get_current_volume(pair)
        if current_volume < get_average_volume(pair):
            logger.warning("Volume actuel inférieur à la moyenne")
            return

        send_trade_alert(
            pair=opportunity['pair'],
            direction=opportunity['direction'],
            entry_price=opportunity['entry'],
            stop_price=opportunity['sl'],
            take_profit=opportunity['tp'],
            reasons=[
                f"Type: {opportunity['zone_type'].upper()}",
                f"Confiance: {opportunity['confidence']}%",
                f"ATR: {atr:.5f}",
                f"Alignement tendance: {is_trend_aligned(pair, opportunity['direction'])}"
            ]
        )
        logger.info(f"✅ Signal envoyé pour {pair} ({opportunity['direction'].upper()}) à {opportunity['entry']:.5f}")

    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'alerte: {e}")

# ... (le reste du code main reste similaire mais utilise la nouvelle analyse_pair)

if __name__ == "__main__":
    logger.info("🚀 Démarrage du bot de trading OANDA - Mode Sniper Liquidités...")
    
    # Initialisation du chasseur de liquidités
    liquidity_hunter = LiquidityHunter()
    # Initialiser toutes les paires au lancement
    for pair in PAIRS:
        initialize_pair_data(pair)
        time.sleep(0.5)  # Respect rate limits
    # Vérification initiale de la connexion
    try:
        account_info = accounts.AccountDetails(OANDA_ACCOUNT_ID)
        client.request(account_info)
        logger.info(f"✅ Connecté avec succès au compte OANDA: {OANDA_ACCOUNT_ID}")
        logger.info(f"🔧 Mode simulation: {'ACTIVÉ' if SIMULATION_MODE else 'DÉSACTIVÉ'}")
    except Exception as e:
        logger.error(f"❌ Échec de la connexion à OANDA: {e}")
        exit(1)
    
    while True:
        now = datetime.utcnow().time()
        if SESSION_START <= now <= SESSION_END:
            clean_historical_data()            
            logger.info("⏱ Session active - Chasse aux liquidités en cours...")

            try:
                for pair in sorted(PAIRS, key=lambda x: 0 if x == "XAU_USD" else 1):
                    try:
                        # Mise à jour des données de base
                        liquidity_hunter.update_asian_range(pair)
                        liquidity_hunter.analyze_htf_liquidity(pair)
                        
                        if pair == "XAU_USD":
                            analyze_gold()  # Analyse spécifique pour l'or
                        else:
                            # Utilisation de analyze_pair() qui intègre déjà find_best_opportunity()
                            analyze_pair(pair)
                    except Exception as e:
                        logger.error(f"Erreur analyse {pair}: {str(e)}")
                        continue
                
                # Gestion des trades existants
                update_closed_trades()
                for pair in list(active_trades):
                    try:
                        manage_open_trade(pair)
                    except Exception as e:
                        logger.error(f"Erreur gestion trade {pair}: {e}")
            except Exception as e:
                logger.error(f"Erreur majeure: {str(e)}")
            
            time.sleep(15)
        else:
            logger.info("🛑 Session inactive. Prochaine vérification dans 5 minutes...")
            time.sleep(300)


def manage_open_trade(pair):
    """Gère les trades ouverts avec trailing stop et prise de profits partiels"""
    try:
        # Récupération du prix actuel
        params = {"granularity": "M5", "count": 1, "price": "M"}
        candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params))['candles']
        current_price = float(candles[0]['mid']['c'])
        
        # Récupération des détails de la position
        r = positions.PositionDetails(accountID=OANDA_ACCOUNT_ID, instrument=pair)
        position = client.request(r)
        
        # Vérification de la direction
        if float(position['position']['long']['units']) > 0:
            direction = 'buy'
            entry_price = float(position['position']['long']['averagePrice'])
            current_sl = float(position['position']['long']['stopLossOrder']['price'])
        elif float(position['position']['short']['units']) < 0:
            direction = 'sell'
            entry_price = float(position['position']['short']['averagePrice'])
            current_sl = float(position['position']['short']['stopLossOrder']['price'])
        else:
            return
        
        # Calcul du profit actuel
        profit_pips = (current_price - entry_price) * (1 if direction == 'buy' else -1)
        
        # Gestion du trailing stop
        if profit_pips > TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001:
            new_sl = entry_price + (0.0001 if direction == 'buy' else -0.0001)  # Break-even
            if (direction == 'buy' and new_sl > current_sl) or (direction == 'sell' and new_sl < current_sl):
                update_stop_loss(position['position'][direction]['tradeIDs'][0], new_sl)
        
        # Prise de profit partielle (pour les trades très profitants)
        if profit_pips > 50 * 0.0001:  # +50 pips
            close_partial_position(pair, direction, percentage=0.5)  # Ferme 50% de la position
    
    except Exception as e:
        logger.error(f"Erreur gestion trade {pair}: {e}")
        raise

def close_partial_position(pair, direction, percentage):
    """Ferme un pourcentage de la position"""
    try:
        units = int(float(get_position_size(pair)) * percentage)
        if direction == 'sell':
            units = -units  # Inversion pour les positions short
        
        data = {
            "order": {
                "units": str(units),
                "instrument": pair,
                "type": "MARKET",
                "positionFill": "REDUCE_ONLY"
            }
        }
        
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=data)
        response = client.request(r)
        logger.info(f"Prise de profit partielle sur {pair}: {units} unités")
    
    except Exception as e:
        logger.error(f"Échec prise de profit partielle {pair}: {e}")

def get_position_size(pair):
    """Récupère la taille de la position ouverte"""
    r = positions.PositionDetails(accountID=OANDA_ACCOUNT_ID, instrument=pair)
    position = client.request(r)
    if float(position['position']['long']['units']) > 0:
        return position['position']['long']['units']
    elif float(position['position']['short']['units']) < 0:
        return position['position']['short']['units']
    return 0
