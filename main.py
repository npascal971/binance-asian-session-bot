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
SESSION_START = dtime(7, 0)
SESSION_END = dtime(23, 0)
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

SIMULATION_MODE = True  # Mettre à True pour tester sans exécuter de vrais trades

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

def calculate_position_size(account_balance, entry_price, stop_loss_price, pair):
    """Calcule la taille de position selon le risque et le type d'instrument"""
    risk_amount = min(account_balance * (RISK_PERCENTAGE / 100), RISK_AMOUNT_CAP)
    risk_per_unit = abs(entry_price - stop_loss_price)
    
    if risk_per_unit == 0:
        logger.error("Distance SL nulle - trade annulé")
        return 0
    
    # Conversion spéciale pour les paires crypto et XAU
    if pair in CRYPTO_PAIRS or pair == "XAU_USD":
        units = risk_amount / risk_per_unit
    else:
        # Pour les paires forex standard
        units = risk_amount / (risk_per_unit * 10000)  # Conversion en lots standard
    
    # Arrondir selon les conventions OANDA
    if pair in CRYPTO_PAIRS:
        return round(units, 6)  # 6 décimales pour les cryptos
    elif pair == "XAU_USD":
        return round(units, 2)  # 2 décimales pour l'or
    else:
        return round(units)  # Unités entières pour forex

def should_open_trade(pair, rsi, macd, macd_signal, breakout_detected):
    """Détermine si les conditions pour ouvrir un trade sont remplies"""
    signal_detected = False
    reason = []

    if rsi > 70:
        signal_detected = True
        reason.append("RSI > 70 : signal de VENTE")
    elif rsi < 30:
        signal_detected = True
        reason.append("RSI < 30 : signal d'ACHAT")

    if macd > macd_signal:
        signal_detected = True
        reason.append("MACD croise au-dessus du signal : signal d'ACHAT")
    elif macd < macd_signal:
        signal_detected = True
        reason.append("MACD croise en dessous du signal : signal de VENTE")

    if breakout_detected:
        signal_detected = True
        reason.append("Breakout détecté sur le range asiatique")

    if signal_detected:
        logger.info(f"💡 Signal détecté pour {pair} → Raisons: {', '.join(reason)}")
    else:
        logger.info(f"🔍 Aucun signal détecté pour {pair}")

    return signal_detected

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
        params = {"granularity": "M5", "count": 50, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        client.request(r)
        candles = r.response['candles']

        closes = [float(c['mid']['c']) for c in candles if c['complete']]
        highs = [float(c['mid']['h']) for c in candles if c['complete']]
        lows = [float(c['mid']['l']) for c in candles if c['complete']]

        if len(closes) < 26:
            logger.warning("Pas assez de données pour le calcul technique.")
            return

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

        logger.info(f"📊 Indicateurs {pair}:\n"
                   f"RSI: {latest_rsi:.2f}\n"
                   f"MACD: {latest_macd:.4f}\n"
                   f"Signal MACD: {latest_signal:.4f}\n"
                   f"Breakout: {'UP' if breakout_up else 'DOWN' if breakout_down else 'NONE'}")

        if should_open_trade(pair, latest_rsi, latest_macd, latest_signal, breakout_detected):
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
            
            # Vérifier les trades ouverts avant analyse
            check_active_trades()
            
            for pair in PAIRS:
                try:
                    analyze_pair(pair)
                except Exception as e:
                    logger.error(f"Erreur critique avec {pair}: {e}")
            
            # Attente avec vérification plus fréquente des trades
            for _ in range(12):  # 12 x 5 secondes = 1 minute
                check_active_trades()
                time.sleep(5)
        else:
            logger.info("🛑 Session de trading inactive. Prochaine vérification dans 5 minutes...")
            time.sleep(300)  # Attente plus longue hors session
