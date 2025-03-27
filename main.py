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
    """Vérifie les trades actuellement ouverts avec OANDA"""
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        response = client.request(r)
        current_trades = {t['instrument'] for t in response['trades']}
        
        global active_trades
        active_trades = current_trades
        
        logger.info(f"Trades actuellement ouverts: {current_trades}")
        return current_trades
    except Exception as e:
        logger.error(f"Erreur lors de la vérification des trades ouverts: {e}")
        return set()

def get_account_balance():
    """Récupère le solde du compte OANDA"""
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
        client.request(r)
        return float(r.response["account"]["balance"])
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du solde: {e}")
        return 0

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
        asian_start_date = now.date()
        asian_end_date = (now - timedelta(days=-1)).date()
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

def detect_pin_bars(candles):
    """Détecte des pin bars dans une série de bougies"""
    pin_bars = []
    for i in range(len(candles)):
        open_price = float(candles[i]['mid']['o'])
        close_price = float(candles[i]['mid']['c'])
        high_price = float(candles[i]['mid']['h'])
        low_price = float(candles[i]['mid']['l'])

        # Calcul du corps et des ombres
        body = abs(close_price - open_price)
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price

        # Critères pour un pin bar
        if upper_wick > 2 * body or lower_wick > 2 * body:
            pin_type = "Bullish" if lower_wick > upper_wick else "Bearish"
            pin_bars.append((i, pin_type))
    
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
    """Met à jour la liste des trades actifs en supprimant ceux qui sont fermés"""
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        response = client.request(r)
        current_trades = {t['instrument'] for t in response['trades']}
        closed_trades = active_trades - current_trades
        if closed_trades:
            logger.info(f"🔄 Trades fermés détectés: {closed_trades}")
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

def calculate_position_size(account_balance, entry_price, stop_loss_price, pair):
    """Calcule la taille de position selon le risque et le type d'instrument"""
    risk_amount = min(account_balance * (RISK_PERCENTAGE / 100), RISK_AMOUNT_CAP)
    risk_per_unit = abs(entry_price - stop_loss_price)
    
    if risk_per_unit == 0:
        logger.error("Distance SL nulle - trade annulé")
        return 0
    
    # Conversion spéciale pour les paires crypto et XAU/XAG
    if pair in CRYPTO_PAIRS or pair in ["XAU_USD", "XAG_USD"]:
        units = risk_amount / risk_per_unit
    else:
        # Pour les paires forex standard
        units = risk_amount / (risk_per_unit * 10000)  # Conversion en lots standard
    
    # Arrondir selon les conventions OANDA
    if pair in CRYPTO_PAIRS:
        units = round(units, 6)  # 6 décimales pour les cryptos
    elif pair in ["XAU_USD", "XAG_USD"]:
        units = round(units, 2)  # 2 décimales pour l'or et l'argent
    else:
        units = round(units)  # Unités entières pour forex
    
    # Vérification finale : les unités doivent être strictement positives
    if units <= 0:
        logger.error("Unités calculées invalides ou nulles.")
        return 0
    
    return units

def update_stop_loss(order_id, new_stop_loss):
    """Met à jour le Stop Loss d'une position existante"""
    try:
        data = {
            "order": {
                "stopLoss": {
                    "price": "{0:.5f}".format(new_stop_loss)
                }
            }
        }
        r = orders.OrderReplace(accountID=OANDA_ACCOUNT_ID, orderSpecifier=order_id, data=data)
        response = client.request(r)
        if 'orderCancelTransaction' in response:
            logger.info(f"止损更新成功。新止损: {new_stop_loss}")
        else:
            logger.error(f"更新止损失败: {response}")
    except Exception as e:
        logger.error(f"更新止损时发生错误: {e}")

def should_open_trade(pair, rsi, macd, macd_signal, breakout_detected, price, key_zones, atr, candles):
    """Détermine si les conditions pour ouvrir un trade sont remplies"""
    # 1. Vérifications initiales
    if rsi is None or macd is None or macd_signal is None:
        logger.error(f"Indicateurs manquants pour {pair}. Aucun trade ouvert.")
        return False

    # 2. Paramètres ajustés par paire
    PAIR_SETTINGS = {
        "XAU_USD": {"min_atr": 0.5, "rsi_overbought": 70, "rsi_oversold": 30},
        "XAG_USD": {"min_atr": 0.3, "rsi_overbought": 65, "rsi_oversold": 35},
        "EUR_USD": {"min_atr": 0.0005, "rsi_overbought": 65, "rsi_oversold": 35},
        "GBP_JPY": {"min_atr": 0.05, "rsi_overbought": 70, "rsi_oversold": 30},
        "USD_JPY": {"min_atr": 0.05, "rsi_overbought": 70, "rsi_oversold": 30},
        "DEFAULT": {"min_atr": 0.5, "rsi_overbought": 65, "rsi_oversold": 35}
    }
    
    settings = PAIR_SETTINGS.get(pair, PAIR_SETTINGS["DEFAULT"])

    # 3. Vérification de la volatilité
    if atr < settings["min_atr"]:
        logger.info(f"Volatilité trop faible pour {pair} (ATR={atr:.2f}). Aucun trade ouvert.")
        return False

    # 4. Détection des signaux
    signals = {
        "rsi": False,
        "macd": False,
        "price_action": False,
        "breakout": False,
        "zone": False
    }
    reasons = []

    # Vérification des zones clés
    for zone in key_zones:
        if abs(price - zone[0]) <= RETEST_ZONE_RANGE or abs(price - zone[1]) <= RETEST_ZONE_RANGE:
            signals["zone"] = True
            reasons.append("Prix dans zone clé")
            break

    # RSI - Seuil strict
    if rsi > settings["rsi_overbought"]:
        signals["rsi"] = True
        reasons.append(f"RSI {rsi:.1f} > {settings['rsi_overbought']} (survendu)")
    elif rsi < settings["rsi_oversold"]:
        signals["rsi"] = True
        reasons.append(f"RSI {rsi:.1f} < {settings['rsi_oversold']} (suracheté)")

    # MACD -  requise
    macd_crossover = (macd > macd_signal and macd_signal > 0) or (macd < macd_signal and macd_signal < 0)
    if macd_crossover:
        signals["macd"] = True
        reasons.append("Croisement MACD confirmé")

    # Breakout - Plus strict
    if breakout_detected and atr > settings["min_atr"] * 1.5:
        signals["breakout"] = True
        reasons.append("Breakout fort détecté")

    # Price Action - Correction ici
    pin_bars = detect_pin_bars(candles)
    engulfing_patterns = detect_engulfing_patterns(candles)
    
    if pin_bars:
        try:
            # Conversion en float pour le formatage
            pin_bar_size = float(pin_bars[0][1]) if isinstance(pin_bars[0][1], str) else pin_bars[0][1]
            signals["price_action"] = True
            reasons.append(f"Pin Bar confirmé (taille: {pin_bar_size:.1f}×ATR)")
        except (IndexError, ValueError, TypeError) as e:
            logger.warning(f"Erreur formatage Pin Bar: {e}")
            signals["price_action"] = True
            reasons.append("Pin Bar détecté (taille non disponible)")
    
    if engulfing_patterns:
        signals["price_action"] = True
        reasons.append("Engulfing Pattern fort")

    # 5. Logique de décision
    _REQUIRED = {
        "XAU_USD": 3,
        "XAG_USD": 2,
        "GBP_JPY": 2,
        "DEFAULT": 2
    }
    required_confirmations = CONFIRMATION_REQUIRED.get(pair, CONFIRMATION_REQUIRED["DEFAULT"])
    
    valid_signals = sum(signals.values())
    
    if valid_signals < required_confirmations:
        logger.info(f"Signaux insuffisants pour {pair} ({valid_signals}/{required_confirmations} confirmations)")
        return False

    # Vérification cohérence direction
    bullish_signals = 0
    bearish_signals = 0
    
    if signals["rsi"]:
        if rsi < settings["rsi_oversold"]: bullish_signals += 1
        else: bearish_signals += 1
        
    if signals["macd"]:
        if macd > macd_signal: bullish_signals += 1
        else: bearish_signals += 1

    # Décision finale
    if bullish_signals >= bearish_signals and any([signals["breakout"], signals["price_action"], signals["zone"]]):
        logger.info(f"✅ Signal ACHAT confirmé pour {pair} - Raisons: {', '.join(reasons)}")
        return "buy"
    elif bearish_signals > bullish_signals and any([signals["breakout"], signals["price_action"], signals["zone"]]):
        logger.info(f"✅ Signal VENTE confirmé pour {pair} - Raisons: {', '.join(reasons)}")
        return "sell"
    
    logger.info(f"❌ Signaux contradictoires pour {pair} - Raisons: {', '.join(reasons)}")
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

def place_trade(pair, direction, entry_price, stop_price, atr, account_balance):
    """Exécute un trade sur le compte OANDA avec des contrôles renforcés"""
    
    # 1. Contrôles pré-trade
    if pair in active_trades:
        logger.info(f"🚫 Trade déjà actif sur {pair}")
        return None

    # Délai minimum entre trades sur même paire (30 minutes)
    MIN_TRADE_INTERVAL = 1800  
    last_trade = next((t for t in reversed(trade_history) if t['pair'] == pair), None)
    
    if last_trade and (datetime.utcnow() - datetime.fromisoformat(last_trade['timestamp'])).seconds < MIN_TRADE_INTERVAL:
        logger.info(f"⏳ Délai minimum non respecté pour {pair} (attendre {MIN_TRADE_INTERVAL//60} min)")
        return None

    try:
        # 2. Calculs de position avec vérifications
        units = calculate_position_size(account_balance, entry_price, stop_price, pair)
        if units <= 0:
            logger.error("❌ Taille de position invalide")
            return None

        # Paramètres spécifiques aux paires
        PAIR_SETTINGS = {
            "XAU_USD": {"decimal": 2, "min_distance": 0.5},
            "XAG_USD": {"decimal": 2, "min_distance": 0.1},
            "EUR_USD": {"decimal": 5, "min_distance": 0.0005},
            "GBP_JPY": {"decimal": 3, "min_distance": 0.05},
            "USD_JPY": {"decimal": 3, "min_distance": 0.05},
            "DEFAULT": {"decimal": 5, "min_distance": 0.0005}
        }
        settings = PAIR_SETTINGS.get(pair, PAIR_SETTINGS["DEFAULT"])

        # 3. Calcul des niveaux de stop et take profit
        take_profit_price = round(
            entry_price + (ATR_MULTIPLIER_TP * atr if direction == "buy" else -ATR_MULTIPLIER_TP * atr),
            settings["decimal"]
        )
        
        stop_loss_price = round(stop_price, settings["decimal"])
        
        # Validation des distances
        min_distance = settings["min_distance"]
        if abs(entry_price - stop_loss_price) < min_distance:
            logger.warning(f"Distance SL trop faible (<{min_distance}), ajustement automatique")
            stop_loss_price = round(
                entry_price - (min_distance if direction == "buy" else -min_distance),
                settings["decimal"]
            )

        # Validation de la distance du Trailing Stop Loss
        trailing_stop_loss_distance = TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001
        validated_trailing_distance = validate_trailing_stop_loss_distance(pair, trailing_stop_loss_distance)

        # 4. Préparation de l'ordre
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

        # 5. Exécution en mode réel
        if not SIMULATION_MODE:
            try:
                r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
                response = client.request(r)
                
                if 'orderCreateTransaction' in response:
                    trade_id = response['orderCreateTransaction']['id']
                    trade_info = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "pair": pair,
                        "direction": direction,
                        "entry_price": entry_price,
                        "stop_price": stop_loss_price,
                        "take_profit": take_profit_price,
                        "units": units,
                        "atr": atr,
                        "risk_reward_ratio": round(abs(take_profit_price - entry_price) / abs(entry_price - stop_loss_price), 2),
                        "trade_id": trade_id
                    }
                    active_trades.add(pair)
                    trade_history.append(trade_info)
                    
                    # Sauvegarde dans un journal
                    save_trade_to_journal(trade_info)
                    
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
                    response = client.request(orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data))
                    logger.info(f"Trade exécuté après ajustement: {response}")
                    return response['orderCreateTransaction']['id']

        # 6. Mode simulation
        else:
            trade_info = {
                "timestamp": datetime.utcnow().isoformat(),
                "pair": pair,
                "direction": direction,
                "entry_price": entry_price,
                "stop_price": stop_loss_price,
                "take_profit": take_profit_price,
                "units": units,
                "atr": atr,
                "risk_reward_ratio": round(abs(take_profit_price - entry_price) / abs(entry_price - stop_loss_price), 2),
                "trade_id": f"SIM_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            }
            active_trades.add(pair)
            trade_history.append(trade_info)
            logger.info("📊 Trade simulé (non exécuté)")
            return trade_info['trade_id']

    except Exception as e:
        logger.error(f"⛔ Erreur critique: {str(e)}", exc_info=True)
        return None


def save_trade_to_journal(trade_info):
    """Sauvegarde les détails du trade dans un fichier journal"""
    journal_file = "trading_journal.csv"
    file_exists = os.path.exists(journal_file)
    
    with open(journal_file, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=trade_info.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade_info)
def update_trailing_stop(pair, current_price, direction, current_sl):
    """Version plus conservative du trailing stop"""
    try:
        # Augmenter la distance d'activation
        TRAILING_DISTANCE = 50 * 0.0001  # 50 pips pour XAU/USD
        
        # Ne mettre à jour que si le profit est > 2x le risque
        if direction == "buy" and current_price > entry_price + 2*(entry_price - current_sl):
            new_sl = current_price - TRAILING_DISTANCE
            if new_sl > current_sl:
                return new_sl
                
        # Logique similaire pour les shorts
        elif direction == "sell" and current_price < entry_price - 2*(current_sl - entry_price):
            new_sl = current_price + TRAILING_DISTANCE
            if new_sl < current_sl:
                return new_sl
                
        return current_sl
    except Exception as e:
        logger.error(f"Erreur trailing stop: {e}")
        return current_sl

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
        
        # 4. Extraire les séries de prix
        closes = [float(c['mid']['c']) for c in candles if c['complete']]
        highs = [float(c['mid']['h']) for c in candles if c['complete']]
        lows = [float(c['mid']['l']) for c in candles if c['complete']]
        
        if len(closes) < 20:
            logger.warning(f"Pas assez de bougies complètes pour {pair} ({len(closes)}/20)")
            return

        # 5. Calculer les indicateurs techniques
        close_series = pd.Series(closes)
        high_series = pd.Series(highs)
        low_series = pd.Series(lows)
        
        # RSI
        delta = close_series.diff().dropna()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        latest_rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        ema12 = close_series.ewm(span=12, adjust=False).mean()
        ema26 = close_series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        latest_macd = macd_line.iloc[-1]
        latest_signal = signal_line.iloc[-1]
        
        # ATR
        atr = np.mean([h - l for h, l in zip(highs[-14:], lows[-14:])])
        
        # 6. Détection de breakout
        breakout_up = any(float(c['mid']['h']) > asian_high for c in candles[-5:] if c['complete'])
        breakout_down = closes[-1] < min(closes[-11:-1])
        breakout_detected = breakout_up or breakout_down
        
        # 7. Détecter les patterns
        ltf_patterns = detect_ltf_patterns(candles)
        logger.info(f"Patterns LTF détectés pour {pair}: {ltf_patterns}")
        
        # 8. Log des indicateurs
        logger.info(f"📊 Indicateurs {pair}: RSI={latest_rsi:.2f}, MACD={latest_macd:.4f}, Signal MACD={latest_signal:.4f}")
        logger.info(f"Breakout: {'UP' if breakout_up else 'DOWN' if breakout_down else 'NONE'}")
        
        # 9. Vérifier les conditions de trading
        key_zones = fvg_zones + ob_zones + [(asian_low, asian_high)]
        if should_open_trade(pair, latest_rsi, latest_macd, latest_signal, breakout_detected, closes[-1], key_zones, atr, candles):
            logger.info(f"🚀 Trade potentiel détecté sur {pair}")
            entry_price = closes[-1]
            if breakout_up:
                stop_price = entry_price - ATR_MULTIPLIER_SL * atr
                direction = "buy"
            else:
                stop_price = entry_price + ATR_MULTIPLIER_SL * atr
                direction = "sell"
            account_balance = get_account_balance()
            place_trade(pair, direction, entry_price, stop_price, atr, account_balance)
        else:
            logger.info("📉 Pas de conditions suffisantes pour ouvrir un trade.")
            
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de {pair}: {str(e)}", exc_info=True)


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
        
        # Attente avec vérification plus fréquente des trades
        for _ in range(12):  # 12 x 5 secondes = 1 minute
            check_active_trades()
            update_closed_trades()
            time.sleep(5)
    else:
        logger.info("🛑 Session de trading inactive. Prochaine vérification dans 5 minutes...")
        time.sleep(300)  # Attente plus longue hors session

        # Mettre à jour le SL pour chaque paire active
        for pair in active_trades:
            try:
                # Récupérer le prix actuel
                params = {"granularity": "M5", "count": 1, "price": "M"}
                r = instruments.InstrumentsCandles(instrument=pair, params=params)
                client.request(r)
                current_price = float(r.response['candles'][0]['mid']['c'])

                # Récupérer les détails de la position
                r = positions.PositionDetails(accountID=OANDA_ACCOUNT_ID, instrument=pair)
                response = client.request(r)
                trade_id = response['position']['tradeIDs'][0]
                current_sl = float(response['position']['long']['stopLossOrder']['price'])
                direction = "buy" if float(response['position']['long']['units']) > 0 else "sell"

                # Mettre à jour le trailing stop
                new_sl = update_trailing_stop(pair, current_price, direction, current_sl)
                if new_sl != current_sl:
                    update_stop_loss(trade_id, new_sl)
                    logger.info(f"Trailing stop mis à jour pour {pair} : Nouveau SL={new_sl}")
            except Exception as e:
                logger.error(f"Erreur lors de la mise à jour du trailing stop pour {pair}: {e}")

                # Calculer un nouveau SL si nécessaire
                if direction == "buy" and current_price > current_sl + TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001:
                    new_sl = current_price - TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001
                    update_stop_loss(trade_id, new_sl)
                elif direction == "sell" and current_price < current_sl - TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001:
                    new_sl = current_price + TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001
                    update_stop_loss(trade_id, new_sl)
            except Exception as e:
                logger.error(f"Erreur lors de la mise à jour du SL pour {pair}: {e}")

        # Analyser chaque paire
        for pair in PAIRS:
            try:
                analyze_pair(pair)
            except Exception as e:
                logger.error(f"Erreur critique avec {pair}: {e}")
        
        # Attente avec vérification plus fréquente des trades
        for _ in range(12):  # 12 x 5 secondes = 1 minute
            check_active_trades()
            time.sleep(5)
