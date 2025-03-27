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
PAIRS = ["XAU_USD", "EUR_USD", "GBP_JPY","JPY_USD","XAG_USD"]
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
    # Vérifier que les indicateurs sont valides
    if rsi is None or macd is None or macd_signal is None:
        logger.error(f"Indicateurs manquants pour {pair}. Aucun trade ouvert.")
        return False

    signal_detected = False
    reason = []

    # Vérifier la volatilité
    MIN_ATR_THRESHOLD = 0.5
    MAX_ATR_THRESHOLD = 3.0
    if atr < MIN_ATR_THRESHOLD or atr > MAX_ATR_THRESHOLD:
        logger.info(f"Volatilité trop faible ou trop élevée pour {pair} (ATR={atr:.2f}). Aucun trade ouvert.")
        return False

    # Vérifier si le prix est proche d'une zone clé
    for zone in key_zones:
        if abs(price - zone[0]) <= RETEST_ZONE_RANGE or abs(price - zone[1]) <= RETEST_ZONE_RANGE:
            signal_detected = True
            reason.append("Prix proche d'une zone clé")
            break

    # Conditions basées sur RSI
    if rsi > 65:
        signal_detected = True
        reason.append("RSI > 65 : signal de VENTE")
    elif rsi < 35:
        signal_detected = True
        reason.append("RSI < 35 : signal d'ACHAT")

    # Conditions basées sur MACD
    if macd > macd_signal:
        signal_detected = True
        reason.append("MACD croise au-dessus du signal : signal d'ACHAT")
    elif macd < macd_signal:
        signal_detected = True
        reason.append("MACD croise en dessous du signal : signal de VENTE")

    # Détection de breakout
    if breakout_detected:
        signal_detected = True
        reason.append("Breakout détecté sur le range asiatique")

    # Détecter des patterns (pin bars, engulfing)
    pin_bars = detect_pin_bars(candles)
    engulfing_patterns = detect_engulfing_patterns(candles)

    if pin_bars:
        signal_detected = True
        reason.append(f"Pin Bar détecté: {pin_bars}")
    if engulfing_patterns:
        signal_detected = True
        reason.append(f"Engulfing Pattern détecté: {engulfing_patterns}")

    # Priorisation des signaux
    if pin_bars or engulfing_patterns:
        reason.append("Signal prioritaire: Pin Bar ou Engulfing Pattern")
    elif breakout_detected:
        reason.append("Signal secondaire: Breakout détecté")
    elif rsi > 65 or rsi < 35:
        reason.append("Signal basé sur RSI")

    # Résolution des conflits entre signaux
    if "RSI > 65 : signal de VENTE" in reason and "Breakout détecté" in reason:
        logger.warning(f"Conflit de signal pour {pair}: RSI indique VENTE mais Breakout indique ACHAT.")
        return False

    # Logs des raisons du signal
    if signal_detected:
        logger.info(f"💡 Signal détecté pour {pair} → Raisons: {', '.join(reason)}")
    else:
        logger.info(f"🔍 Aucun signal détecté pour {pair}")
    return signal_detected

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

def place_trade(pair, direction, entry_price, stop_price, atr, account_balance):
    """Exécute un trade sur le compte OANDA"""
    if pair in active_trades:
        logger.info(f"🚫 Trade déjà actif sur {pair}, aucun nouveau trade ne sera ouvert.")
        return None
    
    try:
        units = calculate_position_size(account_balance, entry_price, stop_price, pair)
        if units == 0:
            logger.error("❌ Calcul des unités invalide - trade annulé")
            return None
        
        # Calcul du take profit et du stop loss avec arrondi spécifique
        if pair in ["XAU_USD", "XAG_USD"]:
            take_profit_price = round(entry_price + ATR_MULTIPLIER_TP * atr, 2)  # 2 décimales pour XAU_USD et XAG_USD
            stop_loss_price = round(stop_price, 2)  # 2 décimales pour XAU_USD et XAG_USD
        else:
            take_profit_price = round(entry_price + ATR_MULTIPLIER_TP * atr, 5)  # 5 décimales pour forex
            stop_loss_price = round(stop_price, 5)  # 5 décimales pour forex
        
        logger.info(f"\n💖 NOUVEAU TRADE DÉTECTÉ 💖\n"
                    f"Paire: {pair}\n"
                    f"Direction: {direction.upper()}\n"
                    f"Prix d'entrée: {entry_price}\n"
                    f"Stop Loss: {stop_loss_price}\n"
                    f"Take Profit: {take_profit_price}\n"
                    f"Unités: {units}\n"
                    f"Solde compte: {account_balance}")
        
        trade_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "pair": pair,
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_loss_price,
            "take_profit": take_profit_price,
            "units": units
        }
        
        if not SIMULATION_MODE:
            order_data = {
                "order": {
                    "instrument": pair,
                    "units": str(int(units)) if direction == "buy" else str(-int(units)),
                    "type": "MARKET",
                    "positionFill": "DEFAULT",
                    "stopLossOnFill": {
                        "price": "{0:.5f}".format(stop_loss_price) if pair not in ["XAU_USD", "XAG_USD"] else "{0:.2f}".format(stop_loss_price)
                    },
                    "takeProfitOnFill": {
                        "price": "{0:.5f}".format(take_profit_price) if pair not in ["XAU_USD", "XAG_USD"] else "{0:.2f}".format(take_profit_price)
                    },
                    "trailingStopLossOnFill": {
                        "distance": "{0:.5f}".format(TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001)
                    }
                }
            }
            logger.debug(f"Données de l'ordre envoyé à OANDA: {order_data}")
            
            r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
            response = client.request(r)
            
            if 'orderCreateTransaction' in response:
                trade_id = response['orderCreateTransaction']['id']
                logger.info(f"✔️ Trade exécuté avec succès. ID: {trade_id}")
                trade_info['trade_id'] = trade_id
                active_trades.add(pair)
                trade_history.append(trade_info)
                return trade_id
            else:
                logger.error(f"❌ Erreur dans la réponse OANDA: {response}")
                return None
        else:
            trade_info['trade_id'] = "SIMULATED_TRADE_ID"
            trade_history.append(trade_info)
            active_trades.add(pair)
            logger.info("✅ Trade simulé (mode simulation activé)")
            return "SIMULATED_TRADE_ID"
    except Exception as e:
        logger.error(f"❌ Erreur critique lors de la création de l'ordre: {str(e)}", exc_info=True)
        return None

def update_trailing_stop(pair, current_price, direction, current_sl):
    """Met à jour le Stop Loss dynamiquement avec un trailing stop"""
    try:
        # Calcul du nouveau Stop Loss
        if direction == "buy":
            new_sl = current_price - TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001
            if new_sl > current_sl:  # Seulement si le SL peut être amélioré
                return new_sl
        elif direction == "sell":
            new_sl = current_price + TRAILING_ACTIVATION_THRESHOLD_PIPS * 0.0001
            if new_sl < current_sl:  # Seulement si le SL peut être amélioré
                return new_sl
        return current_sl  # Si aucune amélioration n'est possible
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du trailing stop pour {pair}: {e}")
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
