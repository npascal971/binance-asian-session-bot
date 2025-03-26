import os
import time
import logging
from datetime import datetime, timedelta, time as dt_time
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
import logging


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
ASIAN_SESSION_START = time(0, 0)  # 00h00 UTC
ASIAN_SESSION_END = time(6, 0)    # 06h00 UTC
LONDON_SESSION_START = time(7, 0) # 07h00 UTC
NY_SESSION_END = time(16, 30)     # 16h30 UTC
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
    end_date = datetime.combine(now.date(), end_time) if end_time else now
    start_date = datetime.combine(now.date(), start_time)
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
    
    # Récupération du range asiatique
    asian_range = asian_ranges.get(pair)
    if not asian_range:
        logger.warning(f"⚠️ Aucun range asiatique disponible pour {pair}")
        return
    
    # Récupération du prix actuel
    current_price = get_current_price(pair)
    
    # Vérification si le prix est dans la plage valide
    if not is_price_in_valid_range(current_price, asian_range):
        logger.info(f"❌ Prix hors range valide pour {pair}")
        return
    
    # Calcul des indicateurs techniques
    rsi = calculate_rsi(pair)
    macd_signal = calculate_macd(pair)
    
    # Logs détaillés
    logger.info(f"📊 Analyse {pair} - Prix: {current_price:.5f}, Range: {asian_range['low']:.5f} - {asian_range['high']:.5f}")
    logger.info(f"📈 RSI: {rsi:.2f}, MACD Signal: {macd_signal}")
    
    # Décision de placement de trade
    if rsi < 30 and macd_signal == "BUY":
        place_trade(pair, "buy", current_price, asian_range["low"], asian_range["high"])
    elif rsi > 70 and macd_signal == "SELL":
        place_trade(pair, "sell", current_price, asian_range["high"], asian_range["low"])

def main_loop():
    """Boucle principale du bot."""
    while True:
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
        
        # Session asiatique
        if ASIAN_SESSION_START <= current_time < ASIAN_SESSION_END:
            logger.info("🌏 SESSION ASIATIQUE EN COURS")
            analyze_asian_session()
        else:
            logger.info("🌍 Hors session asiatique")
        
        # Session Londres/NY
        if LONDON_SESSION_START <= current_time <= NY_SESSION_END:
            logger.info("🏙️ SESSION LONDRES/NY EN COURS")
            for pair in PAIRS:
                if pair not in active_trades:
                    analyze_pair(pair)
        else:
            logger.info("🌆 Hors session Londres/NY")
        
        # Vérification des stops et take-profits
        check_tp_sl()
        
        logger.info("⏰ Pause avant le prochain cycle...")
        time.sleep(60)

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
        r = trades.TradeClose(accountID=OANDA_ACCOUNT_ID, tradeID=pair)
        client.request(r)
        logger.info(f"✅ Trade fermé: {pair}")
        active_trades.remove(pair)
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
            
            # Limite globale de 1 trade maximum
            if len(active_trades) >= 1:
                logger.info("⚠️ Limite de 1 trade atteinte - Attente...")
                time.sleep(60)
                continue
            
            # Session asiatique
            if ASIAN_SESSION_START <= current_time < ASIAN_SESSION_END:
                logger.info("🌏 SESSION ASIATIQUE EN COURS")
                analyze_asian_session()
            elif LONDON_SESSION_START <= current_time <= NY_SESSION_END:
                logger.info("🏙️ SESSION LONDRES/NY EN COURS")
                for pair in PAIRS:
                    if pair not in active_trades:
                        analyze_pair(pair)
            else:
                logger.info("🌆 HORS SESSION - Attente...")
            
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
