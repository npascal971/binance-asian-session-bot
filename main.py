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
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
CRYPTO_PAIRS = ["BTC_USD", "ETH_USD"]
RISK_PERCENTAGE = 1  # 1% du capital
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 2.0
ASIAN_SESSION_START = dtime(0, 0)    # 00h00 UTC
ASIAN_SESSION_END = dtime(6, 0)      # 06h00 UTC
LONDON_SESSION_START = dtime(7, 0)   # 07h00 UTC
NY_SESSION_END = dtime(16, 30)       # 16h30 UTC
MAX_RISK_USD = 100  # $100 max de risque par trade
MIN_CRYPTO_UNITS = 0.001  # Unités minimales pour les cryptos

SESSION_START = LONDON_SESSION_START  # On garde pour compatibilité
SESSION_END = NY_SESSION_END
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

SIMULATION_MODE = False  # Passer à False pour le trading réel
trade_history = []
active_trades = set()
end_of_day_processed = False  # Pour éviter les fermetures répétées
# Spécifications des instruments (avec crypto)
INSTRUMENT_SPECS = {
    "EUR_USD": {"pip": 0.0001, "min_units": 1000, "precision": 0, "margin_rate": 0.02},
    "GBP_USD": {"pip": 0.0001, "min_units": 1000, "precision": 0, "margin_rate": 0.02},
    "USD_JPY": {"pip": 0.01, "min_units": 1000, "precision": 0, "margin_rate": 0.02},
    "XAU_USD": {"pip": 0.01, "min_units": 1, "precision": 2, "margin_rate": 0.02},
    "BTC_USD": {"pip": 1, "min_units": 0.001, "precision": 6, "margin_rate": 0.05},
    "ETH_USD": {"pip": 0.1, "min_units": 0.001, "precision": 6, "margin_rate": 0.05}
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
    """Calcule la taille de position avec gestion de risque stricte"""
    specs = get_instrument_details(pair)
    risk_amount = min(account_balance * (RISK_PERCENTAGE / 100), MAX_RISK_USD)
    
    try:
        # Calcul de la distance en pips
        pip_value = 10 ** specs['pip_location']
        distance_pips = abs(entry_price - stop_loss) / pip_value
        
        # Vérification de la distance minimale
        MIN_DISTANCE = 5  # pips minimum
        if distance_pips < MIN_DISTANCE:
            logger.warning(f"⚠️ Distance trop petite ({distance_pips:.1f}p) pour {pair}")
            return 0
            
        # Calcul des unités de base
        units = risk_amount / distance_pips
        
        # Application des contraintes
        if pair not in CRYPTO_PAIRS:
            # Pour Forex: arrondir au multiple de min_units le plus proche
            units = max(round(units / specs['min_units']) * specs['min_units'], specs['min_units'])
            
            # Vérification du risque ajusté
            adjusted_risk = units * distance_pips
            if adjusted_risk > MAX_RISK_USD * 1.1:  # 10% de tolérance
                logger.error(f"🚨 Risque ajusté ${adjusted_risk:.2f} dépasse MAX_RISK_USD")
                return 0
        else:
            # Pour Crypto: précision différente
            units = round(units, specs['units_precision'])
        
        # Validation finale des unités minimales
        if units < specs['min_units']:
            logger.warning(f"⚠️ Forçage des unités au minimum {specs['min_units']}")
            units = specs['min_units']
        
        logger.info(f"""
        📊 Position Validée {pair}:
        • Entrée: {entry_price:.5f}
        • Stop: {stop_loss:.5f}
        • Distance: {distance_pips:.1f} pips
        • Unités: {units}
        • Risque: ${units * distance_pips:.2f}
        """)
        return units
        
    except Exception as e:
        logger.error(f"❌ Erreur calcul position {pair}: {str(e)}")
        return 0
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

def get_candles(pair, start_time, end_time):
    """Récupère les bougies pour une plage horaire spécifique"""
    now = datetime.utcnow()
    start_date = datetime.combine(now.date(), start_time)
    end_date = datetime.combine(now.date(), end_time)
    
    params = {
        "granularity": "M5",
        "from": start_date.isoformat() + "Z",
        "to": end_date.isoformat() + "Z",
        "price": "M"
    }
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    return client.request(r)['candles']

def identify_fvg(candles, lookback=50):
    """
    Identifie les Fair Value Gaps (FVG) sur les données historiques
    Args:
        candles: Liste des bougies OHLC
        lookback: Nombre de bougies à analyser
    Returns:
        Liste des FVG (haussiers et baissiers)
    """
    fvgs = []
    candles = candles[-lookback:]  # On ne garde que les N dernières bougies
    
    for i in range(1, len(candles)):
        prev = candles[i-1]['mid']
        curr = candles[i]['mid']
        
        # FVG haussier (l'actuel Low > précédent High)
        if curr['l'] > prev['h']:
            fvgs.append({
                'type': 'BULLISH',
                'top': float(prev['h']),
                'bottom': float(curr['l']),
                'time': candles[i]['time'],
                'strength': round((float(curr['l']) - float(prev['h']))/float(prev['h'])*10000, 2)  # en pips
            })
        
        # FVG baissier (l'actuel High < précédent Low)
        elif curr['h'] < prev['l']:
            fvgs.append({
                'type': 'BEARISH',
                'top': float(prev['l']),
                'bottom': float(curr['h']),
                'time': candles[i]['time'],
                'strength': round((float(prev['l']) - float(curr['h']))/float(prev['l'])*10000, 2)
            })
    
    # Filtrage des FVG les plus significatifs (force > 5 pips)
    significant_fvgs = [fvg for fvg in fvgs if fvg['strength'] >= 5]
    
    logger.debug(f"🔍 {len(significant_fvgs)} FVG significatifs détectés")
    return significant_fvgs

def identify_order_blocks(candles, lookback=100):
    """
    Identifie les Order Blocks (OB) sur les données historiques
    Args:
        candles: Liste des bougies OHLC
        lookback: Nombre de bougies à analyser
    Returns:
        Dictionnaire des OB (bullish et bearish)
    """
    ob_bullish = []
    ob_bearish = []
    candles = candles[-lookback:]
    
    for i in range(2, len(candles)):
        prev = candles[i-1]['mid']
        curr = candles[i]['mid']
        
        # OB haussier (grosse bougie verte suivie d'une bougie rouge)
        if float(prev['c']) > float(prev['o']) and float(curr['c']) < float(curr['o']):
            ob_bullish.append({
                'high': float(prev['h']),
                'low': float(prev['l']),
                'open': float(prev['o']),
                'close': float(prev['c']),
                'time': prev['time'],
                'volume': candles[i-1]['volume']
            })
        
        # OB baissier (grosse bougie rouge suivie d'une bougie verte)
        elif float(prev['c']) < float(prev['o']) and float(curr['c']) > float(curr['o']):
            ob_bearish.append({
                'high': float(prev['h']),
                'low': float(prev['l']),
                'open': float(prev['o']),
                'close': float(prev['c']),
                'time': prev['time'],
                'volume': candles[i-1]['volume']
            })
    
    # Filtrage des OB par volume et taille
    filtered_ob = {
        'bullish': [ob for ob in ob_bullish if ob['volume'] > 100 and (ob['close'] - ob['open']) > 0.0005],
        'bearish': [ob for ob in ob_bearish if ob['volume'] > 100 and (ob['open'] - ob['close']) > 0.0005]
    }
    
    logger.debug(f"🔍 OB détectés: {len(filtered_ob['bullish'])} haussiers / {len(filtered_ob['bearish'])} baissiers")
    return filtered_ob

def analyze_pair(pair):
    """Analyse SMC avec FVG et Order Blocks optimisés"""
    try:
        # 1. Récupération des données
        params_htf = {"granularity": "H4", "count": 100, "price": "M"}
        htf_candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params_htf))['candles']
        
        # 2. Détection des zones
        fvgs = identify_fvg(htf_candles)
        obs = identify_order_blocks(htf_candles)
        current_price = get_current_price(pair)
        
        # 3. Vérification des conditions
        in_fvg, fvg_zone = is_price_in_fvg(current_price, fvgs)
        near_ob, ob_zone = is_price_near_ob(current_price, obs, 
                                           'BUY' if current_price <= fvg_zone['bottom'] else 'SELL')
        
        if in_fvg and near_ob:
            logger.info(f"🎯 Signal confirmé sur {pair}: FVG + OB")
            direction = 'buy' if fvg_zone['type'] == 'BULLISH' else 'sell'
            
            place_trade(
                pair=pair,
                direction=direction,
                entry_price=current_price,
                stop_loss=fvg_zone['bottom'] if direction == 'buy' else fvg_zone['top'],
                take_profit=calculate_tp_from_structure(fvg_zone, direction)  # À implémenter
            )
            
    except Exception as e:
        logger.error(f"❌ Erreur analyse {pair}: {str(e)}")

def calculate_tp_from_structure(fvg, direction):
    """Calcule le TP basé sur la taille du FVG"""
    fvg_size = abs(fvg['top'] - fvg['bottom'])
    if direction == 'buy':
        return fvg['top'] + fvg_size * 1.5  # TP = 1.5x la taille du FVG
    else:
        return fvg['bottom'] - fvg_size * 1.5

def get_current_price(pair):
    """Récupère le prix actuel"""
    params = {"instruments": pair}
    r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params=params)
    return float(client.request(r)['prices'][0]['bids'][0]['price'])

def is_price_in_fvg(price, fvgs, buffer=0.0002):
    """
    Vérifie si le prix est dans un FVG
    """
    for fvg in fvgs:
        if fvg['type'] == 'BULLISH' and (fvg['bottom'] - buffer) <= price <= (fvg['top'] + buffer):
            return True, fvg
        elif fvg['type'] == 'BEARISH' and (fvg['bottom'] - buffer) <= price <= (fvg['top'] + buffer):
            return True, fvg
    return False, None

def is_price_near_ob(price, obs, direction, buffer=0.0003):
    """
    Vérifie la proximité avec un Order Block
    """
    target_obs = obs['bullish'] if direction == 'BUY' else obs['bearish']
    for ob in target_obs:
        if direction == 'BUY' and (ob['low'] - buffer) <= price <= (ob['high'] + buffer):
            return True, ob
        elif direction == 'SELL' and (ob['low'] - buffer) <= price <= (ob['high'] + buffer):
            return True, ob
    return False, None

def place_trade(pair, direction, entry_price, stop_loss, take_profit):
    """Exécute un trade avec vérifications supplémentaires et gestion du slippage"""
    if pair in active_trades:
        logger.info(f"⚠️ Trade actif existant sur {pair}")
        return None

    # Validation du solde et calcul des unités
    account_balance = get_account_balance()
    if account_balance < 100:
        logger.error("🚨 Solde insuffisant (< $100)")
        return None

    units = calculate_position_size(pair, account_balance, entry_price, stop_loss)
    if units <= 0:
        return None

    # Gestion du buffer pour le slippage
    pip_location = get_instrument_details(pair)['pip_location']
    BUFFER = 0.0002 if "JPY" not in pair else 0.02  # 2 pips standard, 20 pips pour JPY
    adjusted_stop = round(
        stop_loss + (BUFFER if direction == "sell" else -BUFFER),
        abs(pip_location)
    )
    
    logger.info(f"🔧 Stop ajusté: {stop_loss:.5f} → {adjusted_stop:.5f} (Buffer: {BUFFER})")

    # Vérification marge requise
    specs = get_instrument_details(pair)
    margin_required = (units * entry_price) * specs['margin_rate']
    if margin_required > account_balance * 0.5:
        logger.error(f"🚨 Marge insuffisante (requise: ${margin_required:.2f})")
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

def close_all_trades():
    """Ferme tous les trades ouverts"""
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        open_trades = client.request(r).get('trades', [])
        
        if not open_trades:
            logger.info("✅ Aucun trade à fermer")
            return

        for trade in open_trades:
            trade_id = trade['id']
            instrument = trade['instrument']
            units = -float(trade['currentUnits'])  # Inverse la position
            
            if not SIMULATION_MODE:
                data = {
                    "order": {
                        "type": "MARKET",
                        "instrument": instrument,
                        "units": str(units)
                    }
                }
                r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=data)
                client.request(r)
                logger.info(f"🚪 Fermeture {instrument} (ID: {trade_id})")
            else:
                logger.info(f"🧪 [SIMULATION] Fermeture {instrument} (ID: {trade_id})")

        active_trades.clear()
        
    except Exception as e:
        logger.error(f"❌ Erreur fermeture trades: {str(e)}")
        
def check_tp_sl():
    """Vérifie si TP/SL atteint"""
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        open_trades = client.request(r).get('trades', [])
        
        for trade in open_trades:
            try:
                current_price = float(trade['price'])
                tp_order = trade.get('takeProfitOrder', {})
                sl_order = trade.get('stopLossOrder', {})
                
                if not tp_order or not sl_order:
                    continue
                
                tp_price = float(tp_order.get('price', 0))
                sl_price = float(sl_order.get('price', 0))
                
                if not tp_price or not sl_price:
                    continue
                
                # Get the correct open price field
                open_price = float(trade.get('price') or trade.get('openPrice') or trade.get('averagePrice', 0))
                
                if current_price >= tp_price:
                    send_trade_notification({
                        'pair': trade['instrument'],
                        'direction': 'buy' if float(trade['currentUnits']) > 0 else 'sell',
                        'entry': open_price,
                        'stop': sl_price,
                        'tp': tp_price,
                        'units': abs(float(trade['currentUnits']))
                    }, "TP")
                elif current_price <= sl_price:
                    send_trade_notification({
                        'pair': trade['instrument'],
                        'direction': 'buy' if float(trade['currentUnits']) > 0 else 'sell',
                        'entry': open_price,
                        'stop': sl_price,
                        'tp': tp_price,
                        'units': abs(float(trade['currentUnits']))
                    }, "SL")
            except Exception as e:
                logger.error(f"❌ Erreur traitement trade {trade.get('id', 'unknown')}: {str(e)}")
                continue
                
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
    now = datetime.utcnow()
    current_time = now.time()
    weekday = now.weekday()  # 0-4 = Lun-Ven

    # =============================================
    # 1. GESTION FERMETURES (Priorité absolue)
    # =============================================
    
    # A. Week-end - Fermeture totale
    if weekday >= 5:  # 5 = Sam, 6 = Dim
        close_all_trades()
        next_monday = now + timedelta(days=(7 - weekday))
        sleep_hours = (next_monday - now).total_seconds() / 3600
        logger.info(f"⛔ Week-end - Prochain trade lundi {LONDON_SESSION_START.strftime('%H:%M')} UTC")
        time.sleep(min(sleep_hours * 3600, 21600))  # Max 6h de sleep
        continue

    # B. Fin de session NY - Fermeture des trades
    if current_time >= NY_SESSION_END:
        if not end_of_day_processed:
            logger.info("🕒 Fermeture session NY - Liquidation des positions")
            close_all_trades()
            end_of_day_processed = True
        time.sleep(60)
        continue
    else:
        end_of_day_processed = False

    # =============================================
    # 2. ANALYSE ASIE (Sans trading)
    # =============================================
    if ASIAN_SESSION_START <= current_time < ASIAN_SESSION_END:
        if not asian_range_calculated:
            for pair in PAIRS:
                store_asian_range(pair)  # À implémenter
            asian_range_calculated = True
            logger.info("🌏 Range asiatique calculé")
        time.sleep(300)  # Check toutes les 5 min
        continue

    # =============================================
    # 3. PAUSE ENTRE ASIE ET LONDRES
    # =============================================
    if ASIAN_SESSION_END <= current_time < LONDON_SESSION_START:
        time.sleep(60)
        continue

    # =============================================
    # 4. SESSION ACTIVE (Londres + NY)
    # =============================================
    if LONDON_SESSION_START <= current_time <= NY_SESSION_END:
        # Réinitialisation des flags
        asian_range_calculated = False
        
        # Logique de trading
        start_time = time.time()
        
        # A. Vérification trades actifs
        active_trades = check_active_trades()
        
        # B. Analyse des paires
        for pair in PAIRS:
            analyze_pair(pair)  # Votre stratégie SMC
        
        # C. Gestion TP/SL
        check_tp_sl()
        
        # D. Timing
        elapsed = time.time() - start_time
        time.sleep(max(60 - elapsed, 5))
        continue

    # =============================================
    # 5. HORS SESSION (Backup)
    # =============================================
    next_session_start = datetime.combine(
        now.date() + timedelta(days=1) if current_time > NY_SESSION_END else now.date(),
        LONDON_SESSION_START
    )
    sleep_seconds = (next_session_start - now).total_seconds()
    logger.info(f"😴 Prochaine session à {LONDON_SESSION_START.strftime('%H:%M')} UTC")
    time.sleep(min(sleep_seconds, 3600))  # Max 1h
