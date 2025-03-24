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

# Configuration API OANDA
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

client = oandapyV20.API(access_token=OANDA_API_KEY, environment="practice")

# Paramètres de trading (avec BTC et ETH ajoutés)
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD", "BTC_USD", "ETH_USD"]
CRYPTO_PAIRS = ["BTC_USD", "ETH_USD"]
RISK_PERCENTAGE = 1  # 1% du capital
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 2.0
SESSION_START = dtime(0, 5)  # 7h00
SESSION_END = dtime(23, 50)   # 23h50
MAX_RISK_USD = 100  # $100 max de risque par trade
MIN_CRYPTO_UNITS = 0.001  # Unités minimales pour les cryptos

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

# Spécifications des instruments (avec crypto)
INSTRUMENT_SPECS = {
    "EUR_USD": {"pip": 0.0001, "min_units": 1000, "precision": 0, "margin_rate": 0.02},
    "GBP_USD": {"pip": 0.0001, "min_units": 1000, "precision": 0, "margin_rate": 0.02},
    "USD_JPY": {"pip": 0.01, "min_units": 1000, "precision": 0, "margin_rate": 0.02},
    "XAU_USD": {"pip": 0.01, "min_units": 1, "precision": 2, "margin_rate": 0.02},
    "BTC_USD": {"pip": 1, "min_units": MIN_CRYPTO_UNITS, "precision": 6, "margin_rate": 0.05},
    "ETH_USD": {"pip": 0.1, "min_units": MIN_CRYPTO_UNITS, "precision": 6, "margin_rate": 0.05}
}

def get_account_balance():
    """Récupère le solde du compte"""
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID)
        return float(client.request(r)["account"]["balance"])
    except Exception as e:
        logger.error(f"❌ Erreur récupération solde: {str(e)}")
        return 0

def get_instrument_details(pair):
    """Retourne les spécifications de l'instrument"""
    try:
        if pair in INSTRUMENT_SPECS:
            spec = INSTRUMENT_SPECS[pair]
            pip_location = int(np.log10(spec["pip"])) if spec["pip"] > 0 else 0
            return {
                'pip_location': pip_location,
                'min_units': spec["min_units"],
                'units_precision': spec["precision"],
                'margin_rate': spec["margin_rate"]
            }
        return INSTRUMENT_SPECS["EUR_USD"]  # Fallback pour paires inconnues
    except Exception as e:
        logger.error(f"❌ Erreur récupération spécifications: {str(e)}")
        return {
            'pip_location': -4,
            'min_units': MIN_CRYPTO_UNITS if pair in CRYPTO_PAIRS else 1000,
            'units_precision': 6 if pair in CRYPTO_PAIRS else 2,
            'margin_rate': 0.05 if pair in CRYPTO_PAIRS else 0.02
        }

def calculate_position_size(pair, account_balance, entry_price, stop_loss):
    """Calcule la taille de position avec gestion précise du risque"""
    specs = get_instrument_details(pair)
    
    # Calcul du montant à risquer
    risk_amount = min(account_balance * (RISK_PERCENTAGE / 100), MAX_RISK_USD)
    
    # Calcul spécial pour les cryptos
    if pair in CRYPTO_PAIRS:
        units = risk_amount / abs(entry_price - stop_loss)
    else:
        pip_value = 10 ** specs['pip_location']
        distance_pips = abs(entry_price - stop_loss) / (10 ** specs['pip_location'])
        units = risk_amount / (distance_pips * pip_value)
    
    units = round(units, specs['units_precision'])
    
    # Validation finale
    if units < specs['min_units']:
        logger.error(f"🚨 Unités calculées trop faibles ({units}) < minimum ({specs['min_units']})")
        return 0
    
    logger.info(f"""
    📊 Calcul Position {pair}:
    • Risque: ${risk_amount:.2f} ({RISK_PERCENTAGE}%)
    • Entrée: {entry_price:.5f}
    • Stop: {stop_loss:.5f}
    • Pip Location: 10^{specs['pip_location']}
    • Unités: {units}
    """)
    
    return units

def send_trade_notification(trade_info, hit_type):
    """Envoie une notification email pour TP/SL"""
    try:
        emoji = "💰" if hit_type == "TP" else "🛑"
        subject = f"{emoji} {trade_info['pair']} {hit_type} ATTEINT {emoji}"
        
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

def check_active_trades():
    """Vérifie les trades actuellement ouverts"""
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        return {t["instrument"] for t in client.request(r).get('trades', [])}
    except Exception as e:
        logger.error(f"❌ Erreur vérification trades: {str(e)}")
        return set()

def analyze_pair(pair):
    """Analyse une paire et exécute les trades si conditions remplies"""
    try:
        params = {"granularity": "M5", "count": 50, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        candles = client.request(r)['candles']

        closes = [float(c['mid']['c']) for c in candles if c['complete']]
        highs = [float(c['mid']['h']) for c in candles if c['complete']]
        lows = [float(c['mid']['l']) for c in candles if c['complete']]

        if len(closes) < 20:
            logger.warning(f"⚠️ Données insuffisantes pour {pair}")
            return

        # Calcul ATR
        atr = np.mean([h - l for h, l in zip(highs[-14:], lows[-14:])])
        last_close = closes[-1]

        # Stratégie de breakout
        if pair not in active_trades:
            if last_close > max(closes[-10:-1]):  # Breakout haussier
                place_trade(
                    pair=pair,
                    direction="buy",
                    entry_price=last_close,
                    stop_loss=last_close - (ATR_MULTIPLIER_SL * atr),
                    take_profit=last_close + (ATR_MULTIPLIER_TP * atr)
                )
            elif last_close < min(closes[-10:-1]):  # Breakout baissier
                place_trade(
                    pair=pair,
                    direction="sell",
                    entry_price=last_close,
                    stop_loss=last_close + (ATR_MULTIPLIER_SL * atr),
                    take_profit=last_close - (ATR_MULTIPLIER_TP * atr)
                )
    except Exception as e:
        logger.error(f"❌ Erreur analyse {pair}: {str(e)}")

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
        f"   📦 Unités: {units}\n"
        f"   💰 Risque: ${min(account_balance * (RISK_PERCENTAGE / 100), MAX_RISK_USD):.2f}"
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
                send_trade_notification({
                    'pair': trade['instrument'],
                    'direction': 'buy' if float(trade['currentUnits']) > 0 else 'sell',
                    'entry': float(trade['openPrice']),
                    'stop': sl_price,
                    'tp': tp_price,
                    'units': abs(float(trade['currentUnits']))
                }, "TP")
            elif current_price <= sl_price:
                send_trade_notification({
                    'pair': trade['instrument'],
                    'direction': 'buy' if float(trade['currentUnits']) > 0 else 'sell',
                    'entry': float(trade['openPrice']),
                    'stop': sl_price,
                    'tp': tp_price,
                    'units': abs(float(trade['currentUnits']))
                }, "SL")
    except Exception as e:
        logger.error(f"❌ Erreur vérification TP/SL: {str(e)}")

# ========================
# 🔄 BOUCLE PRINCIPALE
# ========================

if __name__ == "__main__":
    logger.info("\n"
        "✨✨✨✨✨✨✨✨✨✨✨✨✨\n"
        "   OANDA TRADING BOT v4.0\n"
        "  Crypto Edition (BTC/ETH)\n"
        "✨✨✨✨✨✨✨✨✨✨✨✨✨"
    )
    
    # Initialisation
    for pair in PAIRS:
        get_instrument_details(pair)
        time.sleep(0.5)

    while True:
        now = datetime.now().time()
        
        if SESSION_START <= now <= SESSION_END:
            start_time = time.time()
            
            # 1. Vérifier les trades actifs
            active_trades = check_active_trades()
            
            # 2. Analyser toutes les paires
            for pair in PAIRS:
                analyze_pair(pair)
            
            # 3. Vérifier les TP/SL
            check_tp_sl()
            
            # 4. Gestion du timing
            elapsed = time.time() - start_time
            sleep_time = max(60 - elapsed, 5)
            logger.info(f"⏳ Prochaine exécution dans {sleep_time:.1f}s")
            time.sleep(sleep_time)
        else:
            logger.info("\n😴 Hors session - Prochaine vérification dans 5 minutes")
            time.sleep(300)
