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
    level=logging.DEBUG,
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

def calculate_atr(highs, lows, closes, period=14):
    """Calculate the Average True Range (ATR)"""
    try:
        if len(highs) < period or len(lows) < period or len(closes) < period:
            logger.warning(f"Not enough data to calculate ATR (need {period} periods)")
            return 0.0

        true_ranges = []
        for i in range(1, len(closes)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)

        # Calculate initial ATR as simple average of first 'period' TR values
        atr = sum(true_ranges[:period]) / period
        
        # Calculate subsequent ATR values using Wilder's smoothing method
        for i in range(period, len(true_ranges)):
            atr = (atr * (period - 1) + true_ranges[i]) / period
            
        return atr
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
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
        "XAU_USD": {"min_atr": 0.5, "rsi_overbought": 70, "rsi_oversold": 30},
        "XAG_USD": {"min_atr": 0.3, "rsi_overbought": 65, "rsi_oversold": 35},
        "EUR_USD": {"min_atr": 0.0005, "rsi_overbought": 65, "rsi_oversold": 35},
        "GBP_JPY": {"min_atr": 0.05, "rsi_overbought": 70, "rsi_oversold": 30},
        "USD_JPY": {"min_atr": 0.05, "rsi_overbought": 70, "rsi_oversold": 30},
        "DEFAULT": {"min_atr": 0.5, "rsi_overbought": 65, "rsi_oversold": 35}
    }
    
def detect_pin_bars(candles):
    """Détecte des pin bars dans une série de bougies"""
    pin_bars = []
    for candle in candles:
        try:
            # Extraction des données de la bougie
            open_price = float(candle['mid']['o'])
            high_price = float(candle['mid']['h'])
            low_price = float(candle['mid']['l'])
            close_price = float(candle['mid']['c'])

            # Calcul du corps et des mèches
            body_size = abs(close_price - open_price)
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price

            # Validation pour éviter les divisions par zéro
            if body_size == 0:
                logger.warning("Bougie avec body_size=0 détectée (doji ou données invalides). Ignorée.")
                return

            # Calcul du ratio entre les mèches et le corps
            ratio = round(max(upper_wick, lower_wick) / body_size, 1)

            # Critère pour une pin bar
            if ratio >= PIN_BAR_RATIO_THRESHOLD:
                pin_bar_type = "Bullish" if close_price > open_price else "Bearish"
                pin_bars.append({
                    "type": pin_bar_type,
                    "ratio": ratio,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "size": body_size  # Ajout de la clé 'size'
                })

        except Exception as e:
            logger.error(f"Erreur lors de la détection des pin bars : {e}")
            return

    return pin_bars
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

def detect_ltf_patterns(candles):
    """Détecte des patterns sur des timeframes basses (pin bars, engulfing patterns)"""
    patterns_detected = []

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

    # 2. Détection des signaux
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
    pin_bars = detect_pin_bars(candles)
    engulfing_patterns = detect_engulfing_patterns(candles)
    if pin_bars:
        signals["price_action"] = True
        reasons.append(f"Pin Bar détectée (ratio: {pin_bars[-1]['ratio']}x)")
    if engulfing_patterns:
        signals["price_action"] = True
        reasons.append("Engulfing Pattern détecté")

    # 3. Validation des signaux
    required = CONFIRMATION_REQUIRED.get(pair, 2)
    if sum(signals.values()) < required:
        logger.info(f"Signaux insuffisants ({sum(signals.values())}/{required})")
        return False

    # 4. Détermination de la direction
    bullish_signals = sum([
        signals["rsi"] and rsi < settings["rsi_oversold"],
        signals["macd"] and macd > macd_signal
    ])
    bearish_signals = sum([
        signals["rsi"] and rsi > settings["rsi_overbought"],
        signals["macd"] and macd <= macd_signal
    ])

    if bullish_signals >= bearish_signals:
        direction = "buy"
    else:
        direction = "sell"

    # 5. Validations finales
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


def analyze_pair(pair):
    """Analyse une paire de trading et exécute les trades si conditions remplies"""
    logger.info(f"🔍 Analyse de la paire {pair}...")
    try:
        # 1. Récupérer le range asiatique
        asian_high, asian_low = get_asian_session_range(pair)
        if asian_high is None or asian_low is None:
            logger.warning(f"Impossible de récupérer le range asiatique pour {pair}.")
            return
        
        logger.info(f"Range asiatique pour {pair}: High={asian_high}, Low={asian_low}")
        
        # 2. Analyser les timeframes élevés (HTF)
        fvg_zones, ob_zones = analyze_htf(pair)
        logger.info(f"Zones HTF pour {pair}: FVG={fvg_zones}, OB={ob_zones}")
        
        # 3. Récupérer les données M5
        params = {"granularity": "M5", "count": 50, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        client.request(r)
        candles = r.response['candles']
        
        # 4. Extraire les séries de prix AVANT les calculs
        closes = [float(c['mid']['c']) for c in candles if c['complete']]
        highs = [float(c['mid']['h']) for c in candles if c['complete']]
        lows = [float(c['mid']['l']) for c in candles if c['complete']]
        volumes = [c['volume'] for c in candles if c['complete']]

        # Vérifier que les données sont valides
        if len(closes) < 20 or not highs or not lows:
            logger.warning("Données de prix insuffisantes")
            return

        # 5. Calculer les indicateurs TECHNIQUES ICI
        adx_value = calculate_adx(highs, lows, closes)
        vwap_value = calculate_vwap(closes, volumes)
        
    
        # 5. Calculer les indicateurs techniques
        close_series = pd.Series(closes)
    
        # RSI robuste
        try:
            latest_rsi = calculate_rsi(np.array(closes))
            if np.isnan(latest_rsi):
                latest_rsi = 50.0
                logger.warning(f"RSI invalide pour {pair}, utilisation de 50.0")
        except Exception as e:
            logger.error(f"Erreur calcul RSI pour {pair}: {str(e)}")
            latest_rsi = 50.0
        
        # MACD
        ema12 = close_series.ewm(span=12, adjust=False).mean()
        ema26 = close_series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        latest_macd = macd_line.iloc[-1]
        latest_signal = signal_line.iloc[-1]
        
        # ATR
        atr = np.mean([h - l for h, l in zip(highs[-14:], lows[-14:])])
        logger.debug(f"ATR calculé pour {pair}: {atr:.5f} (14 périodes)")
        
        # 6. Détection de breakout
        breakout_up = any(float(c['mid']['h']) > asian_high for c in candles[-5:] if c['complete'])
        breakout_down = closes[-1] < min(closes[-11:-1])
        breakout_detected = breakout_up or breakout_down
        
        # 7. Détecter les patterns
        ltf_patterns = detect_ltf_patterns(candles)
        logger.info(f"Patterns LTF détectés pour {pair}: {ltf_patterns}")
        
        # 8. Log des indicateurs
        logger.info(f"📊 Indicateurs {pair}: "
                   f"RSI={latest_rsi:.2f}, "
                   f"MACD={latest_macd:.5f}, "
                   f"Signal={latest_signal:.5f}, "
                   f"ADX: {adx_value} (force de tendance), "
                   f"VWAP: {vwap_value:.5f}, "
                   f"ATR={atr:.5f}")
        logger.info(f"Breakout: {'UP' if breakout_up else 'DOWN' if breakout_down else 'NONE'}")

        # Initialisation des variables
        entry_price = stop_price = direction = None
        
        # 9. Vérifier les conditions de trading
        key_zones = fvg_zones + ob_zones + [(asian_low, asian_high)]
        trade_signal = should_open_trade(pair, latest_rsi, latest_macd, latest_signal, 
                                      breakout_detected, closes[-1], key_zones, atr, candles)

        if trade_signal in ("buy", "sell"):
                # Récupérer l'ATR H1
            atr_h1 = get_atr(pair, "H1")
            if atr_h1 <= 0:
                logger.warning(f"Invalid ATR value for {pair}, skipping trade")
                return  # Exit the function instead of using continue
    
            # Calcul dynamique du SL/TP
            sl_pips, tp_pips = dynamic_sl_tp(atr_h1, trade_signal)

            entry_price = closes[-1]
            direction = trade_signal
            stop_price = entry_price - sl_pips if direction == "buy" else entry_price + sl_pips
            take_profit = entry_price + tp_pips if direction == "buy" else entry_price - tp_pips
    
            # Nouveau log détaillé
            logger.info(f"""
            \n📈 SIGNAL DÉTECTÉ 📉
            Paire: {pair}
            Direction: {direction.upper()}
            Entrée: {entry_price:.5f}
            Stop Loss: {stop_price:.5f}
            Take Profit: {take_profit:.5f}
            Ratio R/R: {(take_profit-entry_price)/(entry_price-stop_price):.1f}
            ATR utilisé: {atr:.5f}""")

            # Récupérer les motifs détectés
            raw_patterns = detect_ltf_patterns(candles)
            patterns = []
            for p in raw_patterns:
                if isinstance(p, tuple):
                    patterns.append(p[0].split()[0])
                else:
                    patterns.append(str(p))
            reasons = [
                f"RSI: {latest_rsi:.2f}",
                f"MACD: {latest_macd:.5f}",
                f"ATR: {atr:.5f}",
                f"Patterns: {', '.join(patterns) if patterns else 'Aucun'}"
            ]

            # Envoyer l'email d'alerte
            send_trade_alert(pair, direction, entry_price, stop_price, take_profit, reasons)
    
        else:
            logger.info("📉 Pas de signal de trading valide")
            
    except Exception as e:
        logger.error(f"Erreur analyse {pair}: {str(e)}", exc_info=True)

            

if __name__ == "__main__":
    logger.info("🚀 Démarrage du bot de trading OANDA...")
    
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
            logger.info("⏱ Session active - Analyse des paires...")
            
            # Vérifier les trades ouverts et fermés
            check_active_trades()
            update_closed_trades()
            
            # Analyser chaque paire
            for pair in PAIRS:
                try:
                    analyze_pair(pair)
                except Exception as e:
                    logger.error(f"Erreur critique avec {pair}: {e}")
                    continue  # Continue to next pair if error occurs
            
            # Mettre à jour le SL pour chaque paire active
            for pair in list(active_trades):  # Use list() for a copy
                try:
                    # Récupérer le prix actuel
                    params = {"granularity": "M5", "count": 1, "price": "M"}
                    r = instruments.InstrumentsCandles(instrument=pair, params=params)
                    client.request(r)
                    current_price = float(r.response['candles'][0]['mid']['c'])

                    # Récupérer les détails de la position
                    r = positions.PositionDetails(accountID=OANDA_ACCOUNT_ID, instrument=pair)
                    response = client.request(r)
                    
                    # Déterminer la direction et les prix
                    if float(response['position']['long']['units']) > 0:
                        trade_data = response['position']['long']
                        direction = "buy"
                        entry_price = float(trade_data['averagePrice'])
                        current_sl = float(trade_data['stopLossOrder']['price'])
                        trade_id = trade_data['tradeIDs'][0]
                    elif float(response['position']['short']['units']) < 0:
                        trade_data = response['position']['short']
                        direction = "sell"
                        entry_price = float(trade_data['averagePrice'])
                        current_sl = float(trade_data['stopLossOrder']['price'])
                        trade_id = trade_data['tradeIDs'][0]
                    else:
                        logger.warning(f"Position neutre pour {pair}")
                        continue  # Skip to next pair

                    # Mettre à jour le trailing stop
                    if direction == "buy" and current_price > current_sl + TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001:
                        new_sl = current_price - TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001
                        update_stop_loss(trade_id, new_sl)
                        logger.info(f"Trailing stop mis à jour pour {pair}: {current_sl} -> {new_sl}")
                    elif direction == "sell" and current_price < current_sl - TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001:
                        new_sl = current_price + TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001
                        update_stop_loss(trade_id, new_sl)
                        logger.info(f"Trailing stop mis à jour pour {pair}: {current_sl} -> {new_sl}")
                        
                except Exception as e:
                    logger.error(f"Erreur position pour {pair}: {str(e)}")
                    continue  # Skip to next pair if error occurs

            # Attente avec vérification plus fréquente des trades
            for _ in range(12):  # 12 x 5 secondes = 1 minute
                check_active_trades()
                update_closed_trades()
                time.sleep(5)
        else:
            logger.info("🛑 Session de trading inactive. Prochaine vérification dans 5 minutes...")
            time.sleep(300)
