import os
import time  # Module standard pour sleep()
import logging
from datetime import datetime, timedelta, time as dt_time  # Importation explicite de time
from email.message import EmailMessage
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing
from oandapyV20.endpoints import accounts
import requests
import pytz

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Affiche les logs dans la console
        logging.FileHandler("trading.log")  # Enregistre les logs dans un fichier
    ]
)
logger = logging.getLogger()

# Paramètres globaux
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
SIMULATION_MODE = True  # Passer à False pour le trading réel
ASIAN_SESSION_START = dt_time(0, 0)  # 00h00 UTC
ASIAN_SESSION_END = dt_time(6, 0)    # 06h00 UTC
LONDON_SESSION_START = dt_time(13, 30) # 07h00 UTC
NY_SESSION_END = dt_time(22, 0)     # 16h30 UTC
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
RSI_PERIOD = 14
VOLUME_MA_PERIOD = 20
MAX_RISK_USD = 100
RISK_PERCENTAGE = 1  # Risque en % du solde du compte
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Initialisation des variables globales
asian_ranges = {}
daily_zones = {}
active_trades = set()
end_of_day_processed = False

# Spécifications des instruments
INSTRUMENT_SPECS = {
    "EUR_USD": {"pip": 0.0001, "min_units": 1000, "precision": 0, "margin_rate": 0.02},
    "GBP_USD": {"pip": 0.0001, "min_units": 1000, "precision": 0, "margin_rate": 0.02},
    "USD_JPY": {"pip": 0.01, "min_units": 1000, "precision": 0, "margin_rate": 0.02},
    "XAU_USD": {"pip": 0.01, "min_units": 1, "precision": 2, "margin_rate": 0.02},
}

# Connexion à l'API OANDA
client = oandapyV20.API(access_token=OANDA_API_KEY, environment="practice")

def get_candles(pair, start_time, end_time=None):
    """Récupère les bougies pour une plage horaire spécifique."""
    now = datetime.utcnow()
    
    # Calcul de start_date et end_date
    start_date = datetime.combine(now.date(), start_time)
    end_date = datetime.combine(now.date(), end_time) if end_time else now
    
    # S'assurer que end_date n'est pas dans le futur
    end_date = min(end_date, now)
    
    # Paramètres de la requête
    params = {
        "granularity": "M15",  # Granularité des bougies
        "from": start_date.isoformat() + "Z",
        "to": end_date.isoformat() + "Z",
        "price": "M"  # Mid prices
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

def calculate_macd(pair):
    """
    Calcule le MACD pour une paire.
    Returns:
        tuple: ("BUY" ou "SELL", histogram)
    """
    try:
        # Récupération des bougies pour la session européenne/NY
        candles = get_candles(pair, LONDON_SESSION_START, NY_SESSION_END)
        if not candles or len(candles) < 26:  # Minimum 26 points pour le MACD
            logger.warning(f"⚠️ Données insuffisantes pour MACD ({len(candles)} points)")
            return None, None

        # Extraction des prix de clôture avec validation
        closes = []
        for c in candles:
            try:
                close_price = float(c['mid']['c'])  # Accès sécurisé au prix de clôture
                closes.append(close_price)
            except (KeyError, TypeError, ValueError):
                logger.debug(f"Donnée invalide ignorée: {c}")
                continue

        if len(closes) < 26:  # Vérification après filtrage
            logger.warning(f"⚠️ Trop peu de données valides pour MACD ({len(closes)} points)")
            return None, None

        # Calcul du MACD
        ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # Histogramme
        histogram = macd_line.iloc[-1] - signal_line.iloc[-1]

        # Signal d'achat/vente
        signal = "BUY" if macd_line.iloc[-1] > signal_line.iloc[-1] else "SELL"

        logger.info(f"📊 MACD calculé pour {pair} - Signal: {signal}, Histogramme: {histogram:.5f}")
        return signal, histogram

    except Exception as e:
        logger.error(f"❌ Erreur calcul MACD: {str(e)}")
        return None, None

def initialize_european_ranges():
    """Initialise les plages européennes basées sur les données de la session."""
    global european_ranges
    european_ranges = {}
    for pair in PAIRS:
        candles = get_candles(pair, LONDON_SESSION_START, NY_SESSION_END)
        if not candles:
            logger.warning(f"⚠️ Aucune donnée pour {pair} - Plage européenne ignorée")
            continue
        highs = [float(c['mid']['h']) for c in candles]
        lows = [float(c['mid']['l']) for c in candles]
        european_ranges[pair] = {"high": max(highs), "low": min(lows)}
        logger.info(f"🌍 Range européen {pair}: {min(lows):.5f} - {max(highs):.5f}")

def analyze_pair(pair):
    """Analyse une paire pour détecter des opportunités de trading."""
    try:
        logger.info(f"🔍 Début analyse approfondie pour {pair}")

        # Récupération du range asiatique ou européen
        if ASIAN_SESSION_START <= datetime.utcnow().time() < ASIAN_SESSION_END:
            range_to_use = asian_ranges.get(pair)
        elif LONDON_SESSION_START <= datetime.utcnow().time() <= NY_SESSION_END:
            range_to_use = european_ranges.get(pair)
        else:
            logger.info(f"⚠️ Hors plage horaire définie pour {pair}")
            return

        # Vérification si le prix est dans la plage valide
        current_price = get_current_price(pair)
        if not is_price_in_valid_range(current_price, range_to_use):
            logger.info(f"❌ Prix hors range valide pour {pair}")
            return

        # Calcul des indicateurs techniques
        rsi = calculate_rsi(pair)
        macd_signal = calculate_macd(pair)

        # Logs détaillés
        logger.info(f"📊 Analyse {pair} - Prix: {current_price:.5f}, Range: {range_to_use['low']:.5f} - {range_to_use['high']:.5f}")
        logger.info(f"📈 RSI: {rsi:.2f}, MACD Signal: {macd_signal}")

        # Décision de placement de trade
        if rsi < 40 and macd_signal == "BUY":  # RSI ajusté à 40
            place_trade(pair, "buy", current_price, range_to_use["low"], range_to_use["high"])
        elif rsi > 60 and macd_signal == "SELL":  # RSI ajusté à 60
            place_trade(pair, "sell", current_price, range_to_use["high"], range_to_use["low"])
        else:
            logger.debug(f"❌ Conditions non remplies pour {pair} - RSI: {rsi}, MACD: {macd_signal}")

    except Exception as e:
        logger.error(f"❌ Erreur analyse {pair}: {str(e)}")

def main_loop():
    """Boucle principale du bot."""
    while True:
        try:
            now = datetime.utcnow()
            current_time = now.time()
            logger.info(f"⏳ Heure actuelle: {current_time}")

            # Vérification des trades actifs
            active_trades = check_active_trades()
            logger.info(f"📊 Trades actifs: {len(active_trades)}")

            # Surveillance des TP/SL pour chaque trade actif
            for pair in active_trades:
                check_tp_sl_for_pair(pair)  # Vérifie spécifiquement ce trade

            # Limite globale de 1 trade maximum
            if len(active_trades) >= 1:
                logger.info("⚠️ Limite de 1 trade atteinte - Attente...")
                time.sleep(60)
                continue

            # Détermination de la session active
            if ASIAN_SESSION_START <= current_time < ASIAN_SESSION_END:
                logger.info("🌏 SESSION ASIATIQUE EN COURS")
                analyze_asian_session()
            elif LONDON_SESSION_START <= current_time <= NY_SESSION_END:
                logger.info("🏙️ SESSION LONDRES/NY EN COURS")
                for pair in PAIRS:
                    if pair not in active_trades:  # Éviter les doublons sur la même paire
                        analyze_pair(pair)
            else:
                logger.info("🌆 Hors session - Attente...")

            # Pause avant le prochain cycle
            logger.info("⏰ Pause avant le prochain cycle...")
            time.sleep(60)

        except Exception as e:
            logger.error(f"💥 ERREUR GRAVE: {str(e)}", exc_info=True)

def check_tp_sl_for_pair(pair):
    """Vérifie si le TP ou SL est atteint pour une paire spécifique."""
    try:
        trade = active_trades.get(pair)
        if not trade:
            logger.warning(f"⚠️ Aucun trade actif trouvé pour {pair}")
            return

        current_price = get_current_price(pair)
        direction = trade["direction"]
        stop_loss = trade["stop"]
        take_profit = trade["tp"]

        if direction == "buy":
            if current_price >= take_profit:
                logger.info(f"🎯 Take Profit atteint pour {pair}")
                close_trade(pair)
            elif current_price <= stop_loss:
                logger.info(f"🛑 Stop Loss atteint pour {pair}")
                close_trade(pair)
        elif direction == "sell":
            if current_price <= take_profit:
                logger.info(f"🎯 Take Profit atteint pour {pair}")
                close_trade(pair)
            elif current_price >= stop_loss:
                logger.info(f"🛑 Stop Loss atteint pour {pair}")
                close_trade(pair)

    except Exception as e:
        logger.error(f"❌ Erreur vérification TP/SL pour {pair}: {str(e)}")

def check_active_trades():
    """Retourne une liste des paires avec des trades actifs."""
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        open_trades = client.request(r).get('trades', [])
        active_pairs = [trade['instrument'] for trade in open_trades]
        return active_pairs
    except Exception as e:
        logger.error(f"❌ Erreur récupération trades actifs: {str(e)}")
        return []

def place_trade(pair, direction, entry_price, stop_loss, take_profit):
    """Place un trade avec trailing SL/TP."""
    global SIMULATION_MODE
    account_balance = get_account_balance()
    units = calculate_position_size(pair, account_balance, entry_price, stop_loss)
    
    if units <= 0:
        logger.warning(f"⚠️ Impossible de placer le trade {pair} - Taille de position invalide")
        return
    
    logger.info(f"""
    🚀 NOUVEAU TRADE {'ACHAT' if direction == 'buy' else 'VENTE'} 🚀
    • Paire: {pair}
    • Direction: {direction.upper()}
    • Entrée: {entry_price:.5f}
    • Stop: {stop_loss:.5f}
    • TP: {take_profit:.5f}
    • Unités: {units}
    • Risque: ${min(account_balance * (RISK_PERCENTAGE / 100), MAX_RISK_USD):.2f}
    """)
    
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
                "takeProfitOnFill": {"price": f"{take_profit:.5f}", "timeInForce": "GTC"}
            }
        }
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        response = client.request(r)
        if "orderFillTransaction" in response:
            trade_id = response["orderFillTransaction"]["id"]
            logger.info(f"✅ Trade exécuté! ID: {trade_id}")
            active_trades.add(pair)
            return trade_id
    except Exception as e:
        logger.error(f"❌ Erreur création ordre: {str(e)}")
        return None

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
    avg_loss = np.mean(losses[:period])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def fetch_historical_asian_range(pair):
    """Récupère le range asiatique historique pour une paire."""
    candles = get_candles(pair, ASIAN_SESSION_START, ASIAN_SESSION_END)
    highs = [float(c['mid']['h']) for c in candles if 'mid' in c]
    lows = [float(c['mid']['l']) for c in candles if 'mid' in c]
    if not highs or not lows:
        logger.warning(f"⚠️ Données insuffisantes pour {pair}")
        return None
    return {"high": max(highs), "low": min(lows)}

def analyze_asian_session():
    """Analyse robuste avec gestion des données partielles et réessais intelligents."""
    global asian_ranges
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
    success_count = 0
    max_retries = 3
    
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
        if success_count == len(pairs):
            logger.info("🌍 ANALYSE ASIATIQUE TERMINÉE AVEC SUCCÈS")
            break
        time.sleep(60)
    else:
        logger.warning("⚠️ Échec complet de l'analyse asiatique après plusieurs tentatives")

def get_account_balance():
    """Récupère le solde du compte."""
    try:
        r = accounts.AccountDetails(accountID=OANDA_ACCOUNT_ID)
        response = client.request(r)
        balance = float(response['account']['balance'])
        logger.info(f"💼 Solde du compte récupéré: ${balance:.2f}")
        return balance
    except Exception as e:
        logger.error(f"❌ Erreur récupération solde du compte: {str(e)}")
        return 0

def calculate_position_size(pair, balance, entry_price, stop_loss):
    """Calcule la taille de position en fonction du risque."""
    specs = get_instrument_details(pair)
    distance_pips = abs(entry_price - stop_loss) / specs["pip"]
    risk_per_trade = min(balance * (RISK_PERCENTAGE / 100), MAX_RISK_USD)
    units = risk_per_trade / distance_pips
    units = max(units, specs["min_units"])
    return round(units, specs["precision"])

def get_instrument_details(pair):
    """Retourne les spécifications de l'instrument."""
    specs = INSTRUMENT_SPECS.get(pair, {"pip": 0.0001, "min_units": 1000, "precision": 0})
    return specs

def check_tp_sl():
    """Vérifie les stops et take-profits."""
    active_trades = check_active_trades()
    for pair in active_trades:
        try:
            r = trades.TradeDetails(accountID=OANDA_ACCOUNT_ID, tradeID=pair)
            trade = client.request(r)
            current_price = get_current_price(pair)
            stop_loss = float(trade['stopLossOrder']['price'])
            take_profit = float(trade['takeProfitOrder']['price'])
            
            if current_price <= stop_loss:
                logger.info(f"🛑 Stop Loss atteint pour {pair}")
                close_trade(pair)
            elif current_price >= take_profit:
                logger.info(f"🎯 Take Profit atteint pour {pair}")
                close_trade(pair)
        except Exception as e:
            logger.error(f"❌ Erreur vérification TP/SL {pair}: {str(e)}")

def close_trade(pair):
    """Ferme un trade."""
    try:
        if SIMULATION_MODE:
            logger.info(f"🧪 [SIMULATION] Fermeture du trade {pair}")
            del active_trades[pair]
            return

        r = trades.TradeClose(accountID=OANDA_ACCOUNT_ID, tradeID=pair)
        client.request(r)
        logger.info(f"✅ Trade fermé: {pair}")
        del active_trades[pair]

    except Exception as e:
        logger.error(f"❌ Erreur fermeture trade {pair}: {str(e)}")

def main_loop():
    """Boucle principale du bot."""
    while True:
        try:
            now = datetime.utcnow()
            current_time = now.time()
            
            logger.info(f"⏳ Heure actuelle: {current_time}")
            
            # Vérification des trades actifs
            active_trades = check_active_trades()
            logger.info(f"📊 Trades actifs: {len(active_trades)}")

            # Surveillance des TP/SL pour chaque trade actif
            for pair in active_trades:
                check_tp_sl_for_pair(pair)
            
            # Limite globale de 1 trade maximum
            if len(active_trades) >= 1:
                logger.info("⚠️ Limite de 1 trade atteinte - Attente...")
                time.sleep(60)
                continue
            
             # Détermination de la session active
            for pair in PAIRS:
                if ASIAN_SESSION_START <= current_time < ASIAN_SESSION_END:
                    range_to_use = asian_ranges.get(pair)
                    logger.info(f"🌏 SESSION ASIATIQUE - Range utilisé pour {pair}: {range_to_use}")
                elif LONDON_SESSION_START <= current_time <= NY_SESSION_END:
                    if not european_ranges:  # Initialisation si nécessaire
                        initialize_european_ranges()
                    range_to_use = european_ranges.get(pair)
                    logger.info(f"🌍 SESSION EUROPÉENNE - Range utilisé pour {pair}: {range_to_use}")
                else:
                    logger.info("⚠️ Hors plage horaire définie")
                    continue

                # Si un range est disponible, procéder à l'analyse
                if range_to_use:
                    analyze_pair(pair, range_to_use)
                else:
                    logger.warning(f"⚠️ Aucun range disponible pour {pair} - Analyse ignorée")
            
            # Vérification des stops et take-profits
            check_tp_sl()
            
            logger.info("⏰ Pause avant le prochain cycle...")
            time.sleep(60)
        
        except Exception as e:
            logger.error(f"💥 ERREUR GRAVE: {str(e)}", exc_info=True)
            time.sleep(300)

if __name__ == "__main__":
    try:
        logger.info("✨ DÉMARRAGE DU BOT DE TRADING ✨")
        if SIMULATION_MODE:
            logger.info("🧪 MODE SIMULATION ACTIVÉ")
        else:
            logger.info("🚀 MODE TRADING RÉEL ACTIVÉ")
        
        # Initialisation des données asiatiques
        for pair in PAIRS:
            if pair not in asian_ranges:
                logger.info(f"🌏 Récupération des données asiatiques historiques pour {pair}...")
                historical_range = fetch_historical_asian_range(pair)
                if historical_range:
                    asian_ranges[pair] = historical_range
        
        # Boucle principale
        main_loop()

    except Exception as e:
        logger.error(f"💥 ERREUR GRAVE: {str(e)}", exc_info=True)
        time.sleep(300)
