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

# Paramètres de trading
PAIRS = [
    "XAU_USD", "XAG_USD",  # Métaux précieux
    "EUR_USD", "GBP_USD", "USD_CHF", "AUD_USD", "NZD_USD",  # Majors
    "GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY", "CAD_JPY"  # Crosses et JPY  
]


RISK_PERCENTAGE = 1
TRAILING_ACTIVATION_THRESHOLD_PIPS = 20
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 3.0
SESSION_START = dtime(7, 0) #7
SESSION_END = dtime(23, 0)
RETEST_TOLERANCE_PIPS = 10
RETEST_ZONE_RANGE = RETEST_TOLERANCE_PIPS * 0.0001
RISK_AMOUNT_CAP = 100
CRYPTO_PAIRS = ["BTC_USD", "ETH_USD"]
# Seuil pour détecter une pin bar (ratio entre la taille des mèches et le corps)
PIN_BAR_RATIO_THRESHOLD = 3.0  # Exemple : une mèche doit être au moins 3 fois plus grande que le corps

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

SIMULATION_MODE = False  # Mettre à True pour tester sans exécuter de vrais trades
CONFIRMATION_REQUIRED = {
    "XAU_USD": 2,  # Nombre de confirmations requises pour XAU_USD
    "EUR_USD": 1,  # Nombre de confirmations requises pour EUR_USD
    "DEFAULT": 1   # Valeur par défaut pour les autres paires
}
trade_history = []
active_trades = set()

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

def get_current_price(pair, max_retries=3):
    """
    Récupère le prix actuel avec gestion des erreurs et réessais
    Args:
        pair: Paire de devises
        max_retries: Nombre maximum de tentatives
    Returns:
        float: Prix actuel ou None si échec
    """
    for attempt in range(max_retries):
        try:
            params = {
                "granularity": "S5",
                "count": 1,
                "price": "M"
            }
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            response = client.request(r)
            
            if not response.get('candles'):
                raise ValueError("Aucune donnée de prix disponible")
                
            candle = response['candles'][0]
            return float(candle['mid']['c'])
            
        except Exception as e:
            logger.warning(f"Tentative {attempt + 1} échouée pour {pair}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Attente avant réessai
                
    logger.error(f"Échec après {max_retries} tentatives pour {pair}")
    return None

def calculate_atr(highs, lows, closes, period=14):
    """Version améliorée avec gestion des arrondis"""
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
        
        # Smoothing avec Wilder
        for i in range(period, len(true_ranges)):
            atr = (atr * (period - 1) + true_ranges[i]) / period

        # Arrondi adaptatif selon le type de paire
        # Supprimer la vérification incorrecte et utiliser un paramètre supplémentaire
        if pair.endswith('_USD'):  # Pour toutes les paires USD
            return round(atr, 5)
        elif '_JPY' in pair:  # Paires JPY
            return round(atr, 3)
        else:  # Valeur par défaut
            return round(atr, 5)

    except Exception as e:
        logger.error(f"Erreur calcul ATR: {str(e)}")
        return 0.0

def send_trade_alert(pair, direction, entry_price, stop_price, take_profit, reasons):
    """Envoie une alerte par email au lieu d'exécuter un trade"""
    subject = f"🚨 Signal {direction.upper()} détecté sur {pair}"
    body = f"""
Nouveau signal de trading détecté !

📌 Paire : {pair}
📈 Direction : {direction.upper()}
💰 Prix d'entrée : {entry_price:.5f}
🎯 Take Profit : {take_profit:.5f}
🛡️ Stop Loss : {stop_price:.5f}

📊 Raisons du signal :
- {chr(10).join(reasons)}

⚠️ Ceci est une alerte informative - Aucun trade n'a été exécuté automatiquement.
"""

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
def is_trend_aligned(pair, direction):
    timeframes = ["M15", "H1", "H4"]
    trends = []
    
    for tf in timeframes:
        candles = get_candles(pair, tf, 50)  # Implémentez cette fonction
        ema50 = pd.Series([c["mid"]["c"] for c in candles]).ewm(span=50).mean().iloc[-1]
        current_price = float(candles[-1]["mid"]["c"])
        trends.append(current_price > ema50 if direction == "BUY" else current_price < ema50)
    
    return all(trends)

def is_price_approaching(price, zone, pair, threshold_pips=None):
    """
    Version améliorée avec gestion automatique des pips selon le type de paire
    """
    try:
        # Détermine le seuil en pips
        if threshold_pips is None:
            if "_JPY" in pair:
                threshold_pips = 15  # Seuil plus large pour les paires JPY
            elif pair in ["XAU_USD", "XAG_USD"]:
                threshold_pips = 30  # Métaux avec plus de volatilité
            else:
                threshold_pips = 10  # Valeur par défaut
        
        # Conversion pips en valeur de prix
        pip_value = 0.01 if "_JPY" in pair else 0.0001
        threshold = threshold_pips * pip_value
        
        if isinstance(zone, (tuple, list)):
            zone_min, zone_max = min(zone), max(zone)
            return (zone_min - threshold) <= price <= (zone_max + threshold)
        else:
            return abs(price - zone) <= threshold
            
    except Exception as e:
        logger.error(f"Erreur is_price_approaching pour {pair}: {str(e)}")
        return False

def dynamic_sl_tp(atr, direction, risk_reward=1.5, min_sl_multiplier=1.8):
    """Gestion dynamique avec filet de sécurité"""
    base_sl = max(atr * 1.5, atr * min_sl_multiplier)  # Le plus grand des deux
    sl = base_sl * 1.2  # Marge supplémentaire de 20%
    tp = sl * risk_reward
    
    return (sl, tp) if direction == "buy" else (-sl, -tp)

def is_trend_aligned(pair, direction):
    """Vérifie l'alignement sur M15/H1/H4"""
    timeframes = ['M15', 'H1', 'H4']
    aligned = []
    
    for tf in timeframes:
        try:
            # Récupère les 50 dernières bougies
            params = {"granularity": tf, "count": 50}
            candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params))['candles']
            closes = [float(c['mid']['c']) for c in candles if c['complete']]
            
            # Calcule la tendance
            sma_50 = pd.Series(closes).rolling(50).mean().iloc[-1]
            current_price = closes[-1]
            
            if direction == "buy":
                aligned.append(current_price > sma_50)
            else:
                aligned.append(current_price < sma_50)
                
        except Exception as e:
            logger.error(f"Erreur vérification alignement {tf} : {str(e)}")
            aligned.append(False)  # Fail-safe
    
    return sum(aligned) >= 2  # Au moins 2/3 timeframes alignés

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
    except Exception as e:
        # Logs en cas d'erreur lors de la récupération des données
        logger.error(f"Erreur lors de la récupération du range asiatique pour {pair}: {e}")
        return None, None
# Définir le seuil de ratio pour les pin bars
PIN_BAR_RATIO_THRESHOLD = 3.0  # Exemple : une mèche doit être au moins 3 fois plus grande que le corps
PAIR_SETTINGS = {
        "XAU_USD": {"min_atr": 0.8, "rsi_overbought": 65, "rsi_oversold": 35, "pin_bar_ratio": 2.5,  # Moins strict pour détecter plus de pin bars
        "required_confirmations": 2},
        "XAG_USD": {"min_atr": 0.3, "rsi_overbought": 65, "rsi_oversold": 35},
        "EUR_USD": {"min_atr": 0.0005, "rsi_overbought": 65, "rsi_oversold": 35},
        "GBP_JPY": {"min_atr": 0.05, "rsi_overbought": 70, "rsi_oversold": 30},
        "USD_JPY": {"min_atr": 0.05, "rsi_overbought": 70, "rsi_oversold": 30},
        "DEFAULT": {"min_atr": 0.5, "rsi_overbought": 65, "rsi_oversold": 35}
    }
    
def detect_pin_bars(candles, pair=None):
    """Détecte des pin bars dans une série de bougies avec gestion des dojis"""
    settings = PAIR_SETTINGS.get(pair, PAIR_SETTINGS["DEFAULT"])
    pin_bars = []
    
    for candle in candles:
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
            total_range = h - l
            
            # Gestion spéciale des dojis (body_size = 0)
            if body_size == 0:
                # On considère le range total comme la mèche
                if total_range > 0:
                    ratio = 99  # Valeur arbitraire élevée pour doji significatif
                else:
                    continue  # Bougie plate, on ignore
            else:
                ratio = max(upper_wick, lower_wick) / body_size
                
            # Seuil dynamique selon la paire
            ratio_threshold = settings.get("pin_bar_ratio", PIN_BAR_RATIO_THRESHOLD)
            
            if ratio >= ratio_threshold:
                direction = "bullish" if c > o else "bearish"
                pin_bars.append({
                    "type": direction,
                    "ratio": round(ratio, 1),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "body_size": body_size,
                    "is_doji": (body_size == 0)
                })
                
        except Exception as e:
            logger.error(f"Erreur analyse bougie {pair}: {str(e)}")
            continue
    
    return pin_bars

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
    pair = "XAU_USD"
    logger.info(f"Analyse approfondie de {pair}...")
    
    # 1. Récupérer les données
    params = {"granularity": "H1", "count": 100, "price": "M"}
    candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params))['candles']
    
    # 2. Calculer les indicateurs
    closes = [float(c['mid']['c']) for c in candles if c['complete']]
    highs = [float(c['mid']['h']) for c in candles if c['complete']]
    lows = [float(c['mid']['l']) for c in candles if c['complete']]
    
    rsi = calculate_rsi(closes[-14:])
    atr = calculate_atr(highs[-14:], lows[-14:], closes[-14:])
    pin_bars = detect_pin_bars(candles[-6:], pair)  # Dernières 6 bouches
    
    # 3. Vérifier les conditions de votre trade gagnant
    sell_conditions = (
        rsi > 60 and  # RSI élevé
        any(p['type'] == 'bearish' and p['ratio'] > 2 for p in pin_bars) and  # Pin bar baissière
        is_strong_trend(pair, "sell") and  # Tendance baissière
        closes[-1] < max(highs[-5:-1])  # Sous le récent high
    )
    
    if sell_conditions:
        stop_loss = max(highs[-5:]) + atr * 0.5
        take_profit = closes[-1] - atr * 2
        target_zone = (take_profit + closes[-1]) / 2
        
        signal_details = {
            "direction": "sell",
            "current_price": closes[-1],
            "target_zone": target_zone,
            "stop_loss": stop_loss,
            "rsi": rsi,
            "atr": atr,
            "momentum": True,
            "pattern": "Pin Bar Baissière + RSI Élevé"
        }
        
        send_gold_alert(pair, signal_details)

def send_gold_alert(pair, signal_details):
    """Envoie une alerte détaillée pour l'or"""
    subject = f"🚨 ALERTE OR - Signal {signal_details['direction'].upper()} détecté"
    
    body = f"""
🔥 Signal de trading confirmé pour {pair} 🔥

📊 Direction: {signal_details['direction'].upper()}
💰 Prix actuel: {signal_details['current_price']:.2f}
🎯 Zone cible: {signal_details['target_zone']:.2f}
📉 Stop Loss: {signal_details['stop_loss']:.2f}

📈 Indicateurs:
- RSI: {signal_details['rsi']:.2f}
- ATR: {signal_details['atr']:.2f}
- Momentum: {'Fort' if signal_details['momentum'] else 'Faible'}

📌 Pattern détecté: {signal_details['pattern']}
📆 Heure du signal: {datetime.now().strftime('%Y-%m-%d %H:%M')}

⚠️ Ceci est une alerte basée sur votre stratégie gagnante récente.
"""
    send_email(subject, body)

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

def calculate_atr_for_pair(pair, period=14):
    """Calcule l'ATR pour une paire donnée avec gestion des erreurs améliorée"""
    try:
        params = {
            "granularity": "H1", 
            "count": period*2, 
            "price": "M",
            "smooth": True  # Ajout du lissage
        }
        
        # Journalisation de débogage
        logger.debug(f"Récupération ATR pour {pair} avec params: {params}")
        
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        candles = client.request(r)["candles"]
        
        # Extraction avec vérification de complétude
        complete_candles = [c for c in candles if c['complete']]
        if not complete_candles:
            logger.warning(f"Aucune bougie complète pour {pair}")
            return 0.0
            
        highs = [float(c['mid']['h']) for c in complete_candles]
        lows = [float(c['mid']['l']) for c in complete_candles]
        closes = [float(c['mid']['c']) for c in complete_candles]
        
        # Vérification taille des données
        if len(highs) < period:
            logger.warning(f"Données insuffisantes ({len(highs)}/{period}) pour {pair}")
            return 0.0
            
        # Calcul final avec arrondi adaptatif
        atr_value = calculate_atr(highs, lows, closes, period)
        
        # Détermination précision décimale
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

def analyze_htf(pair):
    """Analyse les timeframes élevés pour identifier des zones clés (FVG, OB, etc.)"""
    htf_params = {"granularity": "H4", "count": 50, "price": "M", "smooth": True}
    try:
        r = instruments.InstrumentsCandles(instrument=pair, params=htf_params)
        client.request(r)
        candles = [c for c in r.response['candles'] if c['complete']]  # Filtrage
        # Vérification des données
        if not candles or not all(c['complete'] for c in candles):
            logger.warning(f"Données incomplètes ou invalides pour {pair}.")
            return [], []
        
        closes = [float(c['mid']['c']) for c in candles if c['complete']]
        highs = [float(c['mid']['h']) for c in candles if c['complete']]
        lows = [float(c['mid']['l']) for c in candles if c['complete']]
        
        # Vérifier si au moins deux bougies sont disponibles
        if len(closes) < 2:
            logger.warning(f"Pas assez de données HTF pour {pair}.")
            return [], []
        
        # Calcul des FVG (Fair Value Gaps)
        fvg_zones = []
        for i in range(1, len(candles) - 1):
            if highs[i] < lows[i - 1] and closes[i + 1] > highs[i]:
                fvg_zones.append((highs[i], lows[i - 1]))
            elif lows[i] > highs[i - 1] and closes[i + 1] < lows[i]:
                fvg_zones.append((lows[i], highs[i - 1]))
        
        # Identification des Order Blocks (OB)
        ob_zones = []
        for i in range(len(candles) - 1):
            if closes[i] > closes[i + 1]:  # Bearish candle
                ob_zones.append((lows[i + 1], highs[i]))
            elif closes[i] < closes[i + 1]:  # Bullish candle
                ob_zones.append((lows[i], highs[i + 1]))
        
        logger.info(f"Zones HTF pour {pair}: FVG={fvg_zones}, OB={ob_zones}")
        return fvg_zones, ob_zones
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse HTF pour {pair}: {e}")
        return [], []

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
    """Version corrigée pour GBP_JPY et autres paires JPY"""
    try:
        # Validation des paramètres
        if None in [account_balance, entry_price, stop_loss_price]:
            logger.error("Paramètres manquants pour le calcul des unités")
            return 0

        # Calcul du montant de risque
        risk_amount = min(account_balance * (RISK_PERCENTAGE / 100), RISK_AMOUNT_CAP)
        risk_per_unit = abs(entry_price - stop_loss_price)

        logger.debug(f"Paramètres calcul: RiskAmount={risk_amount:.2f} "
                    f"RiskPerUnit={risk_per_unit:.5f} "
                    f"Pair={pair}")

        if risk_per_unit <= 0:
            logger.error("Distance SL nulle ou négative")
            return 0

        # Conversion spéciale pour les paires JPY
        if "_JPY" in pair:
            units = (risk_amount / risk_per_unit)  # Pas de division supplémentaire pour JPY
            units = round(units)  # Unités entières
        elif pair in ["XAU_USD", "XAG_USD"]:
            units = risk_amount / risk_per_unit
            units = round(units, 2)
        else:  # Forex standard
            units = (risk_amount / risk_per_unit) / 10000
            units = round(units)

        logger.info(f"Unités calculées: {units} (Type: {'JPY' if '_JPY' in pair else 'Standard'})")
        
        return units if units > 0 else 0

    except Exception as e:
        logger.error(f"Erreur calcul position: {str(e)}")
        return 0

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
        for zone in key_zones:
            if abs(price - zone[0]) <= RETEST_ZONE_RANGE or abs(price - zone[1]) <= RETEST_ZONE_RANGE:
                signals["zone"] = True
                reasons.append("Prix dans zone clé")
                break
        
        # RSI
        if rsi < settings["rsi_oversold"]:
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
        required = CONFIRMATION_REQUIRED.get(pair, 2)
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

        if direction and any([signals["breakout"], signals["price_action"], signals["zone"]]):
            logger.info(f"✅ Signal {direction.upper()} confirmé - Raisons: {', '.join(reasons)}")
            return direction

        logger.info("❌ Signaux contradictoires")
        return False

    except Exception as e:
        logger.error(f"Erreur dans should_open_trade pour {pair}: {str(e)}")
        return False

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
    deltas = np.diff(prices)
    seed = deltas[:window]
    
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    
    # Cas particulier initial
    if down == 0:
        return 100.0  # Si aucune perte, RSI = 100
    
    rs = up/down
    rsi = 100.0 - (100.0/(1.0 + rs))
    
    # Calcul pour les valeurs suivantes
    for i in range(window, len(deltas)):
        delta = deltas[i]
        
        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta
            
        up = (up*(window-1) + upval)/window
        down = (down*(window-1) + downval)/window
        
        if down == 0:
            rsi = 100.0  # Évite la division par zéro
        else:
            rs = up/down
            rsi = 100.0 - (100.0/(1.0 + rs))
    
    return rsi


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
        """Analyse approfondie des zones de liquidité HTF"""
        try:
            # Analyse des FVG et Order Blocks
            fvg_zones, ob_zones = analyze_htf(pair)
            
            # Détection des pics de volume comme zones de liquidité
            params = {"granularity": "H4", "count": 100, "price": "M"}
            candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params))['candles']
            volumes = [float(c['volume']) for c in candles if c['complete']]
            closes = [float(c['mid']['c']) for c in candles if c['complete']]
            
            # Détection des zones de volume élevé
            high_volume_zones = []
            avg_volume = np.mean(volumes)
            for i in range(len(volumes)):
                if volumes[i] > avg_volume * 2:  # Volume 2x supérieur à la moyenne
                    high_volume_zones.append(closes[i])
            
            # Utilisation de KDE pour trouver des clusters de liquidité
            if len(closes) > 10:
                liquidity_pools = detect_liquidity_zones(closes)
            else:
                liquidity_pools = []
            
            # Enregistrement des zones
            self.liquidity_zones[pair] = {
                'fvg': fvg_zones,
                'ob': ob_zones,
                'volume': high_volume_zones,
                'kde': liquidity_pools,
                'last_update': datetime.utcnow()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur analyse liquidité HTF {pair}: {e}")
            return False
    
    def _is_price_near_zone(self, price, zone, pair):
        """Méthode interne avec logique avancée"""
        # 1. Détermine le seuil dynamique
        atr = calculate_atr_for_pair(pair)
        base_pips = 15 if "_JPY" in pair else 10
        dynamic_threshold = min(base_pips, atr * 0.33)  # Max 1/3 de l'ATR
        
        # 2. Conversion en valeur absolue
        pip_value = 0.01 if "_JPY" in pair else 0.0001
        threshold = dynamic_threshold * pip_value
        
        # 3. Vérification de la zone
        if isinstance(zone, (tuple, list)):
            return (min(zone) - threshold) <= price <= (max(zone) + threshold)
        return abs(price - zone) <= threshold

    def find_best_opportunity(self, pair):
        current_price = self.get_cached_price(pair)
        if current_price is None:
            return None
            
        zones = self.liquidity_zones[pair]
        session = self.session_ranges[pair]
        
        # Priorité 1: FVG
        for fvg in zones['fvg']:
            if self._is_price_near_zone(current_price, fvg, pair):
                if self._confirm_zone(pair, fvg, 'fvg'):
                    return self._prepare_trade(pair, current_price, fvg, 'fvg')
        
        # Priorité 2: Order Blocks avec volume élevé
        for ob in zones['ob']:
            if is_price_approaching(current_price, ob, threshold_pips=0.001):
                if self._confirm_zone(pair, ob, 'ob'):
                    return self._prepare_trade(pair, current_price, ob, 'ob')
        
        # Priorité 3: Niveaux clés du range asiatique
        for level in ['high', 'low', 'mid']:
            if abs(current_price - session[level]) < 0.001:
                if self._confirm_zone(pair, session[level], 'session'):
                    return self._prepare_trade(pair, current_price, session[level], 'session')
        
        return None
    
    def _confirm_zone(self, pair, zone, zone_type):
        """Confirme la validité d'une zone avec analyse LTF"""
        try:
            # Récupère les données M5
            params = {"granularity": "M5", "count": 20, "price": "M"}
            candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params))['candles']
            
            # Vérifie les patterns de prix
            patterns = detect_ltf_patterns(candles)
            pin_bars = detect_pin_bars(candles, pair)
            
            # Vérifie le momentum
            rsi = calculate_rsi([float(c['mid']['c']) for c in candles if c['complete']])
            atr = calculate_atr_for_pair(pair)
            
            # Conditions de confirmation
            if zone_type in ['fvg', 'ob']:
                return (any(p[0] in ['Pin Bar', 'Engulfing'] for p in patterns)) and rsi > 40
            else:  # Session levels
                return len(pin_bars) > 0 and atr > PAIR_SETTINGS.get(pair, {}).get('min_atr', 0.5)
                
        except Exception as e:
            logger.error(f"Erreur confirmation zone {pair}: {e}")
            return False
    
    def _prepare_trade(self, pair, price, zone, zone_type):
        """Prépare les détails du trade"""
        atr = calculate_atr_for_pair(pair)
        direction = 'buy' if price < zone[0] else 'sell' if isinstance(zone, (list, tuple)) else 'buy' if price < zone else 'sell'
        
        # Calcul des niveaux SL/TP
        if direction == 'buy':
            stop_loss = price - 1.5 * atr
            take_profit = price + 3 * atr
        else:
            stop_loss = price + 1.5 * atr
            take_profit = price - 3 * atr
        
        return {
            'pair': pair,
            'direction': direction,
            'entry': price,
            'sl': stop_loss,
            'tp': take_profit,
            'zone_type': zone_type,
            'confidence': self._calculate_confidence(pair, price, zone_type)
        }
    
    def _calculate_confidence(self, pair, price, zone_type):
        """Calcule un score de confiance pour le trade"""
        # Basé sur la confluence des facteurs
        score = 0
        
        # 1. Alignement avec la tendance HTF
        if is_trend_aligned(pair, 'buy' if price < zone[0] else 'sell' if isinstance(zone, (list, tuple)) else 'buy' if price < zone else 'sell'):
            score += 30
        
        # 2. Force de la zone
        if zone_type == 'fvg':
            score += 25
        elif zone_type == 'ob':
            score += 20
        else:  # Session levels
            score += 15
        
        # 3. Momentum
        rsi = check_rsi_conditions(pair)
        if (rsi['buy_signal'] and direction == 'buy') or (rsi['sell_signal'] and direction == 'sell'):
            score += 20
        
        # 4. Volatilité
        atr = calculate_atr_for_pair(pair)
        if atr > PAIR_SETTINGS.get(pair, {}).get('min_atr', 0.5):
            score += 15
        
        # 5. Volume récent
        params = {"granularity": "M15", "count": 5, "price": "M"}
        candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params))['candles']
        last_volume = float(candles[-1]['volume']) if candles else 0
        avg_volume = np.mean([float(c['volume']) for c in candles[:-1] if c['complete']]) if len(candles) > 1 else 0
        
        if last_volume > avg_volume * 1.5:
            score += 10
        
        return min(100, score)  # Limite à 100%

# ... (adaptation des fonctions existantes pour utiliser LiquidityHunter)

def analyze_pair(pair):
    """Nouvelle version focalisée sur les liquidités"""
    hunter = LiquidityHunter()
    
    # 1. Mise à jour des données
    if not hunter.update_asian_range(pair):
        return
    
    if not hunter.analyze_htf_liquidity(pair):
        return
    
    # 2. Recherche d'opportunités
    opportunity = hunter.find_best_opportunity(pair)
    if not opportunity:
        return
    
    # 3. Vérification finale des conditions
    if opportunity['confidence'] < 70:  # Seuil de confiance minimum
        logger.info(f"Opportunité faible confiance ({opportunity['confidence']}%) pour {pair}")
        return
    
    # 4. Envoi de l'alerte
    reasons = [
        f"Zone: {opportunity['zone_type'].upper()}",
        f"Confiance: {opportunity['confidence']}%",
        f"Direction: {opportunity['direction'].upper()}",
        f"ATR: {calculate_atr_for_pair(pair):.5f}"
    ]
    
    send_trade_alert(
        pair=opportunity['pair'],
        direction=opportunity['direction'],
        entry_price=opportunity['entry'],
        stop_price=opportunity['sl'],
        take_profit=opportunity['tp'],
        reasons=reasons
    )

# ... (le reste du code main reste similaire mais utilise la nouvelle analyse_pair)

if __name__ == "__main__":
    logger.info("🚀 Démarrage du bot de trading OANDA - Mode Sniper Liquidités...")
    
    # Initialisation du chasseur de liquidités
    liquidity_hunter = LiquidityHunter()
    
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
            logger.info("⏱ Session active - Chasse aux liquidités en cours...")
            
            # 1. Mise à jour des données de marché
            try:
                # Pour toutes les paires, priorité à XAU_USD
                for pair in sorted(PAIRS, key=lambda x: 0 if x == "XAU_USD" else 1):
                    try:
                        # Mise à jour des ranges et zones de liquidité
                        liquidity_hunter.update_asian_range(pair)
                        liquidity_hunter.analyze_htf_liquidity(pair)
                        
                        # Analyse spécifique pour l'or
                        if pair == "XAU_USD":
                            analyze_gold()  # Conserve votre analyse spécialisée
                        else:
                            # Recherche d'opportunités de trading
                            opportunity = liquidity_hunter.find_best_opportunity(pair)
                            
                            if opportunity and opportunity['confidence'] >= 70:
                                # Préparation des détails du trade
                                reasons = [
                                    f"Zone: {opportunity['zone_type'].upper()}",
                                    f"Confiance: {opportunity['confidence']}%",
                                    f"ATR: {calculate_atr_for_pair(pair):.5f}",
                                    f"Alignement HTF: {'OUI' if is_trend_aligned(pair, opportunity['direction']) else 'NON'}"
                                ]
                                
                                # Envoi de l'alerte
                                send_trade_alert(
                                    pair=opportunity['pair'],
                                    direction=opportunity['direction'],
                                    entry_price=opportunity['entry'],
                                    stop_price=opportunity['sl'],
                                    take_profit=opportunity['tp'],
                                    reasons=reasons
                                )
                                
                                # Exécution réelle en mode live
                                if not SIMULATION_MODE and opportunity['confidence'] > 80:
                                    place_trade(
                                        pair=opportunity['pair'],
                                        direction=opportunity['direction'],
                                        entry_price=opportunity['entry'],
                                        stop_loss_price=opportunity['sl'],
                                        take_profit_price=opportunity['tp'],
                                        atr=calculate_atr_for_pair(pair),
                                        account_balance=get_account_balance()
                                    )
                    
                    except Exception as e:
                        logger.error(f"Erreur analyse {pair}: {str(e)}")
                        continue
                
                # 2. Gestion des trades existants
                update_closed_trades()
                for pair in list(active_trades):
                    try:
                        manage_open_trade(pair)  # Nouvelle fonction de gestion
                    except Exception as e:
                        logger.error(f"Erreur gestion trade {pair}: {e}")
                
            except Exception as e:
                logger.error(f"Erreur majeure: {str(e)}")
            
            # Pause entre les analyses (15 secondes)
            time.sleep(15)
                
        else:
            logger.info("🛑 Session de trading inactive. Prochaine vérification dans 5 minutes...")
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
