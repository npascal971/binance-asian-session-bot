import os
import time
import logging
import smtplib
from datetime import datetime, time as dtime
from email.message import EmailMessage
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing

# Chargement des variables d'environnement
load_dotenv()
USE_API_PRICING = True  # True pour utiliser l'Option 2

# Configuration API OANDA
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

client = oandapyV20.API(access_token=OANDA_API_KEY, environment="practice")

# Paramètres de trading
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
RISK_PERCENTAGE = 1  # 1% du capital
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 2.0
SESSION_START = dtime(7, 0)  # 7h00
SESSION_END = dtime(21, 0)   # 21h00
MAX_RISK_USD = 100  # $100 max de risque par trade

# Configuration des logs avec emojis
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("oanda_trading.log")
    ]
)
logger = logging.getLogger()

SIMULATION_MODE = True  # Passer à False pour le trading réel
trade_history = []
active_trades = set()
INSTRUMENT_DETAILS = {}

# ========================
# 🚀 FONCTIONS PRINCIPALES
# ========================


# Configuration des instruments avec valeurs par défaut OANDA
INSTRUMENT_SPECS = {
    "EUR_USD": {
        "pip_location": -4,  # 0.0001
        "min_units": 1000,
        "units_precision": 0,
        "margin_rate": 0.02
    },
    "GBP_USD": {
        "pip_location": -4,  # 0.0001
        "min_units": 1000,
        "units_precision": 0,
        "margin_rate": 0.02
    },
    "USD_JPY": {
        "pip_location": -2,  # 0.01
        "min_units": 1000,
        "units_precision": 0,
        "margin_rate": 0.02
    },
    "XAU_USD": {
        "pip_location": -2,  # 0.01
        "min_units": 1,
        "units_precision": 2,
        "margin_rate": 0.02
    },
    "BTC_USD": {
        "pip_location": 1,  # 10
        "min_units": 0.001,
        "units_precision": 6,
        "margin_rate": 0.05
    }
}

def get_instrument_details(pair):
    """Retourne les spécifications de l'instrument avec fallback intelligent"""
    try:
        # Essayer d'abord avec les valeurs par défaut
        if pair in INSTRUMENT_SPECS:
            logger.info(f"🔧 Using predefined specs for {pair}")
            return INSTRUMENT_SPECS[pair]
        
        # Fallback dynamique pour les paires inconnues
        logger.warning(f"⚠️ Unknown pair {pair}, estimating pip location...")
        
        # Méthode alternative pour estimer la précision
        params = {"instruments": pair}
        r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params=params)
        price = client.request(r)['prices'][0]['closeoutBid']
        
        decimals = len(price.split('.')[1]) if '.' in price else 0
        pip_location = -decimals if decimals > 0 else 0
        
        return {
            "pip_location": pip_location,
            "min_units": 1000,
            "units_precision": max(0, decimals - 1),
            "margin_rate": 0.02
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting details for {pair}: {str(e)}")
        # Fallback ultra-secure
        return {
            "pip_location": -4,
            "min_units": 1000,
            "units_precision": 0,
            "margin_rate": 0.02
        }

def calculate_position_size(pair, account_balance, entry_price, stop_loss):
    """Calcule la taille de position de manière robuste"""
    specs = get_instrument_details(pair)
    
    # Calcul du montant à risquer
    risk_amount = min(account_balance * (RISK_PERCENTAGE / 100), MAX_RISK_USD)
    risk_per_pip = risk_amount / (abs(entry_price - stop_loss) * (10 ** -specs['pip_location']))
    
    units = risk_per_pip * (10 ** specs['pip_location'])
    units = round(units, specs['units_precision'])
    
    # Validation finale
    if units < specs['min_units']:
        logger.error(f"🚨 Calculated units ({units}) below minimum ({specs['min_units']})")
        return 0
    
    logger.info(f"""
    📊 Position Size Calculation:
    Pair: {pair}
    Risk: ${risk_amount:.2f} ({RISK_PERCENTAGE}%)
    Entry: {entry_price:.5f}
    Stop: {stop_loss:.5f}
    Pip Location: 10^{specs['pip_location']}
    Calculated Units: {units}
    """)
    
    return units
def send_notification(trade_info, hit_type):
    """Envoie une notification email pour TP/SL"""
    try:
        emoji = "💰" if hit_type == "TP" else "🛑"
        subject = f"{emoji} {trade_info['pair']} {hit_type} HIT {emoji}"
        
        if hit_type == "TP":
            profit = (trade_info['tp'] - trade_info['entry']) * trade_info['units']
            if trade_info['direction'] == 'sell': profit = -profit
            result = f"PROFIT: ${profit:.2f} ✅"
        else:
            loss = (trade_info['stop'] - trade_info['entry']) * trade_info['units']
            if trade_info['direction'] == 'sell': loss = -loss
            result = f"LOSS: ${abs(loss):.2f} ❌"

        body = f"""
        {emoji * 3} {hit_type} ATTEINT {emoji * 3}
        
        Paire: {trade_info['pair']}
        Direction: {trade_info['direction'].upper()}
        Entrée: {trade_info['entry']:.5f}
        {hit_type}: {trade_info[hit_type.lower()]:.5f}
        Unités: {trade_info['units']}
        
        {result}
        
        Heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_ADDRESS

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            
        logger.info(f"📧 Notification {hit_type} envoyée!")
    except Exception as e:
        logger.error(f"❌ Erreur envoi email: {str(e)}")

def check_tp_sl():
    """Vérifie si TP/SL atteint"""
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        open_trades = client.request(r).get('trades', [])
        
        for trade in open_trades:
            current_price = float(trade['price'])
            tp_price = float(trade['takeProfitOrder']['price'])
            sl_price = float(trade['stopLossOrder']['price'])
            
            if current_price >= tp_price:
                send_notification({
                    'pair': trade['instrument'],
                    'direction': 'buy' if float(trade['currentUnits']) > 0 else 'sell',
                    'entry': float(trade['openPrice']),
                    'stop': sl_price,
                    'tp': tp_price,
                    'units': abs(float(trade['currentUnits']))
                }, "TP")
                
            elif current_price <= sl_price:
                send_notification({
                    'pair': trade['instrument'],
                    'direction': 'buy' if float(trade['currentUnits']) > 0 else 'sell',
                    'entry': float(trade['openPrice']),
                    'stop': sl_price,
                    'tp': tp_price,
                    'units': abs(float(trade['currentUnits']))
                }, "SL")
                
    except Exception as e:
        logger.error(f"❌ Erreur vérification TP/SL: {str(e)}")

def place_trade(pair, direction, entry_price, stop_loss, take_profit):
    """Exécute un trade"""
    if pair in active_trades:
        logger.info(f"⚠️ Trade actif existant sur {pair}")
        return None

    account_balance = get_account_balance()
    units = calculate_position_size(pair, account_balance, entry_price, stop_loss)
    if units <= 0:
        return None

    logger.info(
        f"\n🚀 NOUVEAU TRADE {'ACHAT' if direction == 'buy' else 'VENTE'} 🚀\n"
        f"   📌 Paire: {pair}\n"
        f"   💵 Entrée: {entry_price:.5f}\n"
        f"   🛑 Stop: {stop_loss:.5f}\n"
        f"   🎯 TP: {take_profit:.5f}\n"
        f"   📦 Unités: {units}"
    )

    trade_info = {
        'pair': pair,
        'direction': direction,
        'entry': entry_price,
        'stop': stop_loss,
        'tp': take_profit,
        'units': units,
        'time': datetime.now().isoformat()
    }

    if not SIMULATION_MODE:
        try:
            order_data = {
                "order": {
                    "instrument": pair,
                    "units": str(units) if direction == "buy" else str(-units),
                    "type": "MARKET",
                    "positionFill": "DEFAULT",
                    "stopLossOnFill": {
                        "price": f"{stop_loss:.5f}",
                        "timeInForce": "GTC"
                    },
                    "takeProfitOnFill": {
                        "price": f"{take_profit:.5f}",
                        "timeInForce": "GTC"
                    }
                }
            }

            r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
            response = client.request(r)
            
            if "orderFillTransaction" in response:
                trade_id = response["orderFillTransaction"]["id"]
                logger.info(f"✅ Trade exécuté! ID: {trade_id}")
                trade_info['id'] = trade_id
                active_trades.add(pair)
                trade_history.append(trade_info)
                return trade_id
                
        except Exception as e:
            logger.error(f"❌ Erreur création ordre: {str(e)}")
            return None
    else:
        trade_info['id'] = "SIMULATION"
        active_trades.add(pair)
        trade_history.append(trade_info)
        logger.info("🧪 Mode simulation - Trade non envoyé")
        return "SIMULATION"

# ========================
# 🔄 BOUCLE PRINCIPALE
# ========================

if __name__ == "__main__":
    logger.info("\n"
        "✨✨✨✨✨✨✨✨✨✨✨✨✨\n"
        "   OANDA TRADING BOT v3.1\n"
        "  Boucle 60s pendant session\n"
        "✨✨✨✨✨✨✨✨✨✨✨✨✨"
    )
    
    # Préchargement des spécifications
    for pair in PAIRS:
        get_instrument_details(pair)
        time.sleep(0.5)

    while True:
        now = datetime.now().time()
        
        if SESSION_START <= now <= SESSION_END:
            start_time = time.time()  # 🕒 Mesure du temps d'exécution
            
            logger.info("\n🔎 Analyse des paires...")
            for pair in PAIRS:
                try:
                    # [...] (Votre logique d'analyse et de trading)
                    pass
                except Exception as e:
                    logger.error(f"❌ Erreur sur {pair}: {str(e)}")
            
            # Vérification TP/SL à chaque boucle
            check_tp_sl()
            
            # ⏱ Calcul du temps restant pour 60s total
            elapsed = time.time() - start_time
            sleep_time = max(60 - elapsed, 5)  # Garantit au moins 5s de pause
            logger.info(f"⏳ Prochaine exécution dans {sleep_time:.0f}s")
            time.sleep(sleep_time)
            
        else:
            logger.info("\n😴 Hors session - Prochaine vérification dans 5 minutes")
            time.sleep(300)  # ⏸️ Hors session, pause plus longue
