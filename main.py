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
PAIRS = ["XAU_USD", "EUR_USD", "GBP_JPY", "BTC_USD", "ETH_USD"]
RISK_PERCENTAGE = 1
TRAILING_ACTIVATION_THRESHOLD_PIPS = 20
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 3.0
SESSION_START = dtime(7, 0) #7
SESSION_END = dtime(22, 0)
RETEST_TOLERANCE_PIPS = 10
RETEST_ZONE_RANGE = RETEST_TOLERANCE_PIPS * 0.0001
RISK_AMOUNT_CAP = 100
CRYPTO_PAIRS = ["BTC_USD", "ETH_USD"]


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
    asian_start_time = dtime(23, 0)  # 23:00 UTC
    asian_end_time = dtime(7, 0)     # 07:00 UTC

    now = datetime.utcnow()
    if now.time() < asian_end_time:
        # Si nous sommes avant 07:00 UTC, la session asiatique correspond à la veille
        asian_start_date = (now - timedelta(days=1)).date()
        asian_end_date = now.date()
    else:
        # Sinon, la session asiatique correspond à aujourd'hui
        asian_start_date = now.date()
        asian_end_date = (now + timedelta(days=1)).date()

    asian_start = datetime.combine(asian_start_date, asian_start_time).isoformat() + "Z"
    asian_end = datetime.combine(asian_end_date, asian_end_time).isoformat() + "Z"

    # Validation des timestamps
    if datetime.fromisoformat(asian_start[:-1]) > now or datetime.fromisoformat(asian_end[:-1]) > now:
        logger.error(f"Timestamp invalide : 'from' ou 'to' est dans le futur.")
        return None, None
    
    params = {
        "granularity": "M5",
        "from": asian_start,
        "to": asian_end,
        "price": "M"
    }

    try:
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        client.request(r)
        candles = r.response['candles']
        highs = [float(c['mid']['h']) for c in candles if c['complete']]
        lows = [float(c['mid']['l']) for c in candles if c['complete']]

        if not highs or not lows:
            logger.warning(f"Aucune donnée disponible pour le range asiatique de {pair}.")
            return None, None

        asian_high = max(highs)
        asian_low = min(lows)

        logger.info(f"Range asiatique pour {pair}: High={asian_high}, Low={asian_low}")
        return asian_high, asian_low
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du range asiatique pour {pair}: {e}")
        return None, None

def validate_timestamp(timestamp):
    """Vérifie qu'un timestamp est valide et pas dans le futur"""
    now = datetime.utcnow()
    if timestamp > now:
        logger.error(f"Timestamp invalide : {timestamp} est dans le futur.")
        return False
    return True

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

def detect_htf_zones(pair):
    """Détecte des zones clés (FVG, OB) sur des timeframes élevés"""
    htf_params = {"granularity": "H4", "count": 50, "price": "M"}
    try:
        r = instruments.InstrumentsCandles(instrument=pair, params=htf_params)
        client.request(r)
        candles = r.response['candles']
        closes = [float(c['mid']['c']) for c in candles if c['complete']]
        highs = [float(c['mid']['h']) for c in candles if c['complete']]
        lows = [float(c['mid']['l']) for c in candles if c['complete']]

        fvg_zones = []
        ob_zones = []

        for i in range(1, len(candles) - 1):
            if highs[i] < lows[i - 1] and closes[i + 1] > highs[i]:
                fvg_zones.append((highs[i], lows[i - 1]))
            elif lows[i] > highs[i - 1] and closes[i + 1] < lows[i]:
                fvg_zones.append((lows[i], highs[i - 1]))

            if closes[i] > closes[i + 1]:  # Bearish candle
                ob_zones.append((lows[i + 1], highs[i]))
            elif closes[i] < closes[i + 1]:  # Bullish candle
                ob_zones.append((lows[i], highs[i + 1]))

        logger.info(f"Zones HTF pour {pair}: FVG={fvg_zones}, OB={ob_zones}")
        return fvg_zones, ob_zones
    except Exception as e:
        logger.error(f"Erreur lors de la détection des zones HTF pour {pair}: {e}")
        return [], []

def analyze_htf(pair):
    """Analyse les timeframes élevés pour identifier des zones clés (FVG, OB, etc.)"""
    htf_params = {"granularity": "H4", "count": 50, "price": "M"}
    try:
        r = instruments.InstrumentsCandles(instrument=pair, params=htf_params)
        client.request(r)
        candles = r.response['candles']
        closes = [float(c['mid']['c']) for c in candles if c['complete']]
        highs = [float(c['mid']['h']) for c in candles if c['complete']]
        lows = [float(c['mid']['l']) for c in candles if c['complete']]

        # Vérification des données
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
    
    # Conversion spéciale pour les paires crypto et XAU
    if pair in CRYPTO_PAIRS:
        units = risk_amount / risk_per_unit
    elif pair == "XAU_USD":
        # Pour l'or, 1 unité = 1 once, donc ajuster en divisant par 10000
        units = risk_amount / (risk_per_unit * 10000)
    else:
        # Pour les paires forex standard
        risk_per_unit_usd = (risk_per_unit * 10000) / entry_price  # Convertir en USD
        units = risk_amount / risk_per_unit_usd  # Calcul en lots standard
    
    # Arrondir selon les conventions OANDA
    if pair in CRYPTO_PAIRS:
        return round(units, 6)  # 6 décimales pour les cryptos
    elif pair == "XAU_USD":
        return round(units, 2)  # 2 décimales pour l'or
    else:
        return round(units)  # Unités entières pour forex

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

def should_open_trade(pair, rsi, macd, macd_signal, breakout_detected, price, key_zones):
    """Détermine si les conditions pour ouvrir un trade sont remplies"""
    signal_detected = False
    reason = []

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

        # Calcul du take profit
        if direction == "buy":
            take_profit_price = round(entry_price + ATR_MULTIPLIER_TP * atr, 5)
        else:
            take_profit_price = round(entry_price - ATR_MULTIPLIER_TP * atr, 5)
        
        logger.info(f"\n💖 NOUVEAU TRADE DÉTECTÉ 💖\n"
                    f"Paire: {pair}\n"
                    f"Direction: {direction.upper()}\n"
                    f"Prix d'entrée: {entry_price}\n"
                    f"Stop Loss: {stop_price}\n"
                    f"Take Profit: {take_profit_price}\n"
                    f"Unités: {units}\n"
                    f"Solde compte: {account_balance}")
        
        trade_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "pair": pair,
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_price,
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
                        "price": "{0:.5f}".format(stop_price)
                    },
                    "takeProfitOnFill": {
                        "price": "{0:.5f}".format(take_profit_price)
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

def analyze_pair(pair):
    """Analyse une paire de trading et exécute les trades si conditions remplies"""
    logger.info(f"🔍 Analyse de la paire {pair}...")
    try:
        # Récupérer les données M5 pour l'analyse LTF
        params = {"granularity": "M5", "count": 50, "price": "M"}
        try:
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            response = client.request(r)
            candles = response['candles']
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données pour {pair}: {e}")
            return

        # Vérifier si les données sont valides
        if not candles or not all(c['complete'] for c in candles):
            logger.warning(f"Données incomplètes ou invalides pour {pair}.")
            return

        # Extraire les prix
        closes = [float(c['mid']['c']) for c in candles if c['complete']]
        highs = [float(c['mid']['h']) for c in candles if c['complete']]
        lows = [float(c['mid']['l']) for c in candles if c['complete']]

        # Vérifier s'il y a suffisamment de données
        if len(closes) < 26:
            logger.warning("Pas assez de données pour le calcul technique.")
            return

        # Calcul des indicateurs techniques
        close_series = pd.Series(closes)
        high_series = pd.Series(highs)
        low_series = pd.Series(lows)

        # Calcul RSI
        delta = close_series.diff().dropna()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1]

        # Calcul MACD
        ema12 = close_series.ewm(span=12, adjust=False).mean()
        ema26 = close_series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        latest_macd = macd_line.iloc[-1]
        latest_signal = signal_line.iloc[-1]

        # Détection breakout
        breakout_up = closes[-1] > max(closes[-11:-1])
        breakout_down = closes[-1] < min(closes[-11:-1])
        breakout_detected = breakout_up or breakout_down

        # Logs des informations
        logger.info(f"📊 Indicateurs {pair}: RSI={latest_rsi:.2f}, MACD={latest_macd:.4f}, Signal MACD={latest_signal:.4f}")
        logger.info(f"Breakout: {'UP' if breakout_up else 'DOWN' if breakout_down else 'NONE'}")

        # Vérifier les conditions pour ouvrir un trade
        key_zones = [(min(lows), max(highs))]  # Exemple simplifié de zones clés
        if should_open_trade(pair, latest_rsi, latest_macd, latest_signal, breakout_detected, closes[-1], key_zones):
            logger.info(f"🚀 Trade potentiel détecté sur {pair}")
            entry_price = closes[-1]
            atr = np.mean([h - l for h, l in zip(highs[-14:], lows[-14:])])
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
        logger.error(f"Erreur critique lors de l'analyse de {pair}: {e}")

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
                direction = response['position']['long']['units'] > 0 and "buy" or "sell"

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
