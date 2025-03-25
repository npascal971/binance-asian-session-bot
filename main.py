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
import requests
from datetime import timedelta
import pytz
UTC = pytz.UTC


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
LONDON_SESSION_STR = LONDON_SESSION_START.strftime('%H:%M')
NY_SESSION_STR = NY_SESSION_END.strftime('%H:%M')
MACRO_UPDATE_HOUR = 8  # 08:00 UTC
MAX_RISK_USD = 100  # $100 max de risque par trade
MIN_CRYPTO_UNITS = 0.001  # Unités minimales pour les cryptos

SESSION_START = LONDON_SESSION_START  # On garde pour compatibilité
SESSION_END = NY_SESSION_END
# Configuration des logs avec emojis
logging.basicConfig(
    level=logging.DEBUG,  # ← Changez de INFO à DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("oanda_trading.log")
    ]
)
logger = logging.getLogger()
# Variables globales
# Ajoutez dans vos variables globales
RISK_REWARD_RATIO = 1.5  # Ratio minimal risque/récompense
MIN_CONFLUENCE_SCORE = 2  # Nombre minimal d'indicateurs favorables
daily_data_updated = False
MACRO_API_KEY = os.getenv("MACRO_API_KEY")  # Clé pour FRED/Quandl
ECONOMIC_CALENDAR_API = "https://api.fxstreet.com/v1/economic-calendar"
asian_ranges = {}  # Dictionnaire pour stocker les ranges
asian_range_calculated = False  # Flag de contrôle

IMPORTANT_EVENTS = {
    "USD": ["CPI", "NFP", "FOMC", "UNEMPLOYMENT"],
    "EUR": ["CPI", "ECB_RATE", "GDP"],
    "JPY": ["BOJ_RATE", "CPI"],
    "XAU": ["REAL_RATES", "INFLATION_EXPECTATIONS"]
}

EVENT_IMPACT = {
    "HIGH": ["NFP", "FOMC", "CPI"],
    "MEDIUM": ["UNEMPLOYMENT", "RETAIL_SALES"],
    "LOW": ["PMI", "CONSUMER_SENTIMENT"]
}
CORRELATION_PAIRS = {
    "EUR_USD": ["USD_INDEX", "XAU_USD", "US10Y"],
    "XAU_USD": ["DXY", "SPX", "US_REAL_RATES"],
    "USD_JPY": ["US10Y", "NIKKEI", "DXY"]
}

CORRELATION_THRESHOLD = 0.7  # Seuil de corrélation significative
SIMULATION_MODE = False  # Passer à False pour le trading réel
trade_history = []
active_trades = set()
end_of_day_processed = False  # Pour éviter les fermetures répétées
daily_zones = {}
RSI_PERIOD = 14
VOLUME_MA_PERIOD = 20

# Spécifications des instruments (avec crypto)
INSTRUMENT_SPECS = {
    "EUR_USD": {"pip": 0.0001, "min_units": 1000, "precision": 0, "margin_rate": 0.02},
    "GBP_USD": {"pip": 0.0001, "min_units": 1000, "precision": 0, "margin_rate": 0.02},
    "USD_JPY": {"pip": 0.01, "min_units": 1000, "precision": 0, "margin_rate": 0.02},
    "XAU_USD": {"pip": 0.01, "min_units": 1, "precision": 2, "margin_rate": 0.02},
    "BTC_USD": {"pip": 1, "min_units": 0.001, "precision": 6, "margin_rate": 0.05},
    "ETH_USD": {"pip": 0.1, "min_units": 0.001, "precision": 6, "margin_rate": 0.05}
}

BUFFER_SETTINGS = {
    'FOREX': 0.0003,    # 3 pips
    'JPY_PAIRS': 0.03,  # 30 pips     
    'CRYPTO': 5.0,
    'EUR_USD': 0.0003,  # 3 pips
    'GBP_USD': 0.0003,
    'USD_JPY': 0.03,    # 30 pips
    'XAU_USD': 0.3      # 30 cents# $5 pour BTC/ETH
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

def calculate_correlation(main_pair, window=30):
    """
    Calcule les corrélations avec les actifs liés
    Retourne un dict {actif: coefficient}
    """
    correlations = {}
    main_prices = get_historical_prices(main_pair, window)
    
    for related_pair in CORRELATION_PAIRS.get(main_pair, []):
        try:
            related_prices = get_historical_prices(related_pair, window)
            if len(related_prices) == len(main_prices):
                corr = np.corrcoef(main_prices, related_prices)[0, 1]
                correlations[related_pair] = corr
        except Exception as e:
            logger.warning(f"⚠️ Erreur corrélation {main_pair}-{related_pair}: {str(e)}")
    
    return correlations

def log_daily_summary():
    """Génère un rapport journalier des performances"""
    try:
        balance = get_account_balance()
        logger.info(f"""
        📊 RÉSUMÉ QUOTIDIEN - {datetime.utcnow().date()}
        • Solde final: ${balance:.2f}
        • Trades ouverts: {len(active_trades)}
        • Sessions analysées:
          - Asiatique: {'OUI' if asian_range_calculated else 'NON'}
          - London/NY: {'OUI' if LONDON_SESSION_START <= datetime.utcnow().time() <= NY_SESSION_END else 'NON'}
        """)
    except Exception as e:
        logger.error(f"❌ Erreur génération rapport: {str(e)}")

def process_asian_session():
    """Gère spécifiquement la session asiatique"""
    global asian_range_calculated
    
    if not asian_range_calculated:
        logger.info("🌏 Début analyse session asiatique")
        
        for pair in PAIRS:
            try:
                # Récupère les données depuis le début de session
                candles = get_candles(pair, ASIAN_SESSION_START, None)
                
                if not candles:
                    logger.warning(f"⚠️ Aucune donnée pour {pair}")
                    continue
                    
                highs = [float(c['mid']['h']) for c in candles if 'mid' in c]
                lows = [float(c['mid']['l']) for c in candles if 'mid' in c]
                
                if highs and lows:
                    # Stocke le range pour usage ultérieur
                    asian_ranges[pair] = {
                        'high': max(highs),
                        'low': min(lows),
                        'time': datetime.utcnow()
                    }
                    logger.debug(f"📊 Range asiatique {pair}: {asian_ranges[pair]['high']:.5f}/{asian_ranges[pair]['low']:.5f}")
            
            except Exception as e:
                logger.error(f"❌ Erreur analyse {pair}: {str(e)}")
        
        asian_range_calculated = True
        logger.info("✅ Analyse session asiatique terminée")
    
    time.sleep(300)  # Attente avant prochaine vérification

def check_correlation(main_pair, direction):
    """
    Vérifie la cohérence inter-marchés
    Retourne True si les corrélations confirment le trade
    """
    correlations = calculate_correlation(main_pair)
    confirmation_score = 0
    
    for pair, corr in correlations.items():
        if abs(corr) >= CORRELATION_THRESHOLD:
            # Vérifie la direction des actifs corrélés
            pair_trend = get_asset_trend(pair)
            
            if (corr > 0 and pair_trend == direction) or (corr < 0 and pair_trend != direction):
                confirmation_score += 1
            else:
                confirmation_score -= 1
    
    return confirmation_score >= 1  # Au moins une confirmation

def get_asset_trend(instrument):
    """
    Détermine la tendance courte d'un actif
    """
    prices = get_historical_prices(instrument, 5)  # 5 dernières heures
    if len(prices) < 2:
        return 'NEUTRAL'
    
    return 'UP' if prices[-1] > prices[0] else 'DOWN'

def get_macro_data(currency, indicator):
    """Récupère les données macro via API"""
    try:
        if indicator == "CPI":
            # Exemple avec FRED (Federal Reserve Economic Data)
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={MACRO_API_KEY}&file_type=json"
            response = requests.get(url).json()
            return float(response["observations"][-1]["value"])
        
        elif indicator == "NFP":
            # Exemple avec Alpha Vantage
            url = f"https://www.alphavantage.co/query?function=NONFARM_PAYROLL&apikey={MACRO_API_KEY}"
            return requests.get(url).json()["data"][0]["value"]
            
    except Exception as e:
        logger.error(f"❌ Erreur macro {indicator}: {str(e)}")
        return None

def macro_filter(pair, direction):
    """
    Vérifie la cohérence macroéconomique
    Retourne True si les conditions sont favorables
    """
    base_currency = pair[:3]
    
    # 1. Vérification des événements imminents
    upcoming = check_economic_calendar(pair)
    if any(e[2] == "HIGH" for e in upcoming):
        logger.warning(f"⚠️ Événement macro majeur imminent - Trade annulé")
        return False
    
    # 2. Analyse des indicateurs clés
    if base_currency == "USD":
        cpi = get_macro_data("USD", "CPI")
        if cpi and cpi > 5.0:  # Inflation élevée
            if direction == "BUY":
                logger.info("✅ Contexte inflationniste favorable aux positions long USD")
                return True
    
    # 3. Contexte des taux d'intérêt
    rates = get_central_bank_rates(base_currency)
    if rates and rates["trend"] != direction:
        logger.warning(f"⚠️ Politique monétaire défavorable")
        return False
        
    return True

def check_economic_calendar(pair):
    """Vérifie les événements à venir pour la paire"""
    base_currency = pair[:3]
    events = IMPORTANT_EVENTS.get(base_currency, [])
    
    try:
        if not ECONOMIC_CALENDAR_API:
            return []
            
        params = {
            "currency": base_currency,
            "importance": "HIGH,MEDIUM",
            "api_key": MACRO_API_KEY
        }
        response = requests.get(ECONOMIC_CALENDAR_API, params=params, timeout=5).json()
        
        return [
            (e["title"], e["date"], e["impact"])
            for e in response.get("events", [])
            if e["title"] in events
        ]
        
    except Exception as e:
        logger.warning(f"⚠️ Erreur calendrier: {str(e)}")
        return []  # Return empty list instead of failing

def get_historical_prices(instrument, periods):
    """
    Récupère les prix historiques selon l'instrument
    """
    if instrument in PAIRS:  # Paires OANDA
        params = {"granularity": "H1", "count": periods, "price": "M"}
        candles = client.request(instruments.InstrumentsCandles(instrument=instrument, params=params))['candles']
        return [float(c['mid']['c']) for c in candles]
    
    # Pour les autres actifs (exemple simplifié)
    elif instrument == "DXY":
        # Implémentez une API DXY (ex: FRED)
        return [...]  # Données fictives
    else:
        logger.warning(f"Instrument non géré: {instrument}")
        return []

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

def check_confluence(pair, direction):
    """
    Vérifie la confluence des indicateurs
    Retourne un score de 0 à 3
    """
    score = 0
    
    # 1. RSI (M30)
    params = {"granularity": "M30", "count": 14}
    candles = client.request(instruments.InstrumentsCandles(instrument=pair, params=params))['candles']
    closes = [float(c['mid']['c']) for c in candles if c['complete']]
    rsi = calculate_rsi(closes)
    
    if (direction == 'BUY' and rsi < 30) or (direction == 'SELL' and rsi > 70):
        score += 1
    
    # 2. Volume relatif (D1)
    current_volume = sum(float(c['volume']) for c in candles[-5:])  # 5 dernières bouches M30
    avg_volume = sum(float(c['volume']) for c in candles[-20:]) / 20
    
    if current_volume > avg_volume * 1.5:
        score += 1
    
    # 3. Confluence avec zones quotidiennes
    current_price = get_current_price(pair)
    if direction == 'BUY' and current_price < daily_zones.get(pair, {}).get('POC', 0):
        score += 1
    elif direction == 'SELL' and current_price > daily_zones.get(pair, {}).get('POC', 0):
        score += 1
    
    return score

def calculate_rsi(prices, period=14):
    """Calcul du RSI"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    return 100 - (100/(1+rs))

def check_active_trades():
    """Vérifie les trades actuellement ouverts"""
    try:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        return {t["instrument"] for t in client.request(r).get('trades', [])}
    except Exception as e:
        logger.error(f"❌ Erreur vérification trades: {str(e)}")
        return set()

def check_htf_trend(pair, timeframe='H4'):
    check_timeframe_validity(timeframe)
    """
    Détermine la tendance sur un timeframe supérieur
    Retourne: 'UP', 'DOWN', 'RANGE' ou 'NEUTRAL'
    
    Args:
        pair: str - Paire de trading (ex: 'EUR_USD')
        timeframe: str - Période valide (S5, M1, M5, M15, H1, H4, D, W, M)
    
    Returns:
        str - Direction de la tendance
    """
    # Validation des paramètres
    valid_timeframes = ["S5", "M1", "M5", "M15", "H1", "H4", "D", "W", "M"]
    if timeframe not in valid_timeframes:
        logger.error(f"❌ Timeframe invalide: {timeframe}. Utilisation de H4 par défaut")
        timeframe = "H4"  # Valeur par défaut sécurisée

    params = {
        "granularity": timeframe,
        "count": 100,
        "price": "M"  # 'M' pour mid, 'B' pour bid, 'A' pour ask
    }

    try:
        # Requête API avec gestion d'erreur
        candles = client.request(
            instruments.InstrumentsCandles(
                instrument=pair, 
                params=params
            )
        )['candles']
        
        # Filtrage des bougies complètes
        closes = []
        for c in candles:
            try:
                if c['complete']:
                    closes.append(float(c['mid']['c']))
            except (KeyError, TypeError):
                continue
        
        if len(closes) < 50:  # Minimum pour les EMA
            logger.warning(f"⚠️ Données insuffisantes pour {pair} (reçu: {len(closes)} bougies)")
            return 'NEUTRAL'

        # Calcul des indicateurs
        series = pd.Series(closes)
        ema20 = series.ewm(span=20, min_periods=20).mean().iloc[-1]
        ema50 = series.ewm(span=50, min_periods=50).mean().iloc[-1]
        last_close = closes[-1]
        
        # Détection des swings
        last_10 = closes[-10:] if len(closes) >= 10 else closes
        last_high = max(last_10)
        last_low = min(last_10)
        range_pct = (last_high - last_low) / last_low if last_low != 0 else 0

        # Décision de tendance
        if ema20 > ema50 and last_close > ema20:
            return 'UP'
        elif ema20 < ema50 and last_close < ema20:
            return 'DOWN'
        elif range_pct < 0.005:  # Range < 0.5%
            return 'RANGE'
        else:
            return 'NEUTRAL'

    except Exception as e:
        logger.error(f"❌ Erreur analyse tendance {pair}: {str(e)}")
        return 'NEUTRAL'  # Retour neutre en cas d'erreur
def update_daily_zones():
    """Met à jour les zones clés quotidiennes"""
    global daily_data_updated
    for pair in PAIRS:
        try:
            # Utilisez None pour end_time pour éviter les problèmes de futur
            candles = get_candles(pair, ASIAN_SESSION_START, None)
            
            if not candles:
                logger.warning(f"⚠️ Aucune donnée pour {pair}")
                continue
                
            highs = [float(c['mid']['h']) for c in candles if 'mid' in c]
            lows = [float(c['mid']['l']) for c in candles if 'mid' in c]
            
            if not highs or not lows:
                continue
                
            daily_zones[pair] = {
                'POC': (max(highs) + min(lows)) / 2,
                'VAH': max(highs),
                'VAL': min(lows),
                'time': datetime.utcnow().date()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur MAJ zones quotidiennes: {str(e)}")
            daily_data_updated = False
            continue
    
    logger.info("📊 Zones quotidiennes mises à jour")
    daily_data_updated = True


def get_candles(pair, start_time, end_time=None):
    """
    Récupère les bougies pour une plage horaire spécifique
    Args:
        pair: Paire de devises (ex: "EUR_USD")
        start_time: Heure de début (datetime.time)
        end_time: Heure de fin (datetime.time) - optionnel
    Returns:
        Liste des bougies
    """
    now = datetime.utcnow()
    
    # Si end_time n'est pas spécifié ou est dans le futur, utiliser maintenant
    if end_time is None or datetime.combine(now.date(), end_time) > now:
        end_date = now
    else:
        end_date = datetime.combine(now.date(), end_time)
    
    start_date = datetime.combine(now.date(), start_time)
    
    # Vérification que start_date est avant end_date
    if start_date >= end_date:
        end_date = start_date + timedelta(hours=1)  # Ajoute 1h si plage invalide
        logger.warning(f"⚠️ Plage temporelle ajustée pour {pair}")

    params = {
        "granularity": "M5",
        "from": start_date.isoformat() + "Z",
        "to": end_date.isoformat() + "Z",
        "price": "M"
    }
    
    try:
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        candles = client.request(r)['candles']
        
        # AJOUTEZ CE LOG ICI ↓
        logger.debug(f"🕯️ Candles reçues pour {pair}: {candles[:1]}")  # Affiche la première bougie
        
        return candles
    except Exception as e:
        logger.error(f"❌ Erreur récupération candles {pair}: {str(e)}")
        return []
        
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
    Version corrigée pour gérer le format OANDA v3
    """
    ob_bullish = []
    ob_bearish = []
    candles = candles[-lookback:]
    
    for i in range(2, len(candles)):
        prev = candles[i-1]
        curr = candles[i]
        
        # Vérification que les données nécessaires existent
        if not all(key in prev['mid'] for key in ['o', 'h', 'l', 'c']):
            continue
            
        prev_open = float(prev['mid']['o'])
        prev_close = float(prev['mid']['c'])
        curr_open = float(curr['mid']['o'])
        curr_close = float(curr['mid']['c'])
        
        # OB haussier (grosse bougie verte suivie d'une bougie rouge)
        if prev_close > prev_open and curr_close < curr_open:
            ob_bullish.append({
                'high': float(prev['mid']['h']),
                'low': float(prev['mid']['l']),
                'open': prev_open,
                'close': prev_close,
                'time': prev.get('time', 'N/A'),  # Gestion du champ time manquant
                'volume': int(prev.get('volume', 0))
            })
        
        # OB baissier (grosse bougie rouge suivie d'une bougie verte)
        elif prev_close < prev_open and curr_close > curr_open:
            ob_bearish.append({
                'high': float(prev['mid']['h']),
                'low': float(prev['mid']['l']),
                'open': prev_open,
                'close': prev_close,
                'time': prev.get('time', 'N/A'),  # Gestion du champ time manquant
                'volume': int(prev.get('volume', 0))
            })
    
    # Filtrage des OB significatifs
    significant_volume = 100  # Volume minimum
    min_size = 0.0005  # Taille minimum en pips
    
    filtered_ob = {
        'bullish': [ob for ob in ob_bullish 
                   if ob['volume'] >= significant_volume 
                   and (ob['close'] - ob['open']) >= min_size],
        'bearish': [ob for ob in ob_bearish 
                   if ob['volume'] >= significant_volume 
                   and (ob['open'] - ob['close']) >= min_size]
    }
    
    logger.debug(f"🔍 OB détectés: {len(filtered_ob['bullish'])} haussiers / {len(filtered_ob['bearish'])} baissiers")
    return filtered_ob
    
def analyze_pair(pair):
    """Version corrigée avec déclaration correcte de 'direction'"""
    try:
        # 1. Vérification initiale des données
        htf_data = get_htf_data(pair)
        if not htf_data or len(htf_data) < 10:
            logger.warning(f"⚠️ Données insuffisantes pour {pair}")
            return

        # 2. Vérification tendance HTF
        trend = check_htf_trend(pair)
        if trend == 'NEUTRAL':
            logger.debug(f"↔️ {pair} en range - Aucun trade")
            return

        # 3. Détermination de la direction ICI (avant utilisation)
        direction = 'BUY' if trend == 'UP' else 'SELL'  # <-- DÉCLARATION AJOUTÉE

        # 4. Récupération du range asiatique
        asian_range = get_asian_range(pair)
        current_price = get_current_price(pair)
        
        # 5. Vérification du contexte de prix
        if not is_price_in_valid_range(current_price, asian_range, trend):
            logger.debug(f"🔍 {pair}: Prix hors range asiatique valide")
            return

        # 6. Détection des zones de trading
        fvgs = identify_fvg(htf_data)
        obs = identify_order_blocks(htf_data)

        # 7. Vérification des zones FVG/OB
        in_fvg, fvg_zone = is_price_in_fvg(current_price, fvgs)
        near_ob, ob_zone = is_price_near_ob(current_price, obs, trend)

        if not (in_fvg and near_ob):
            logger.debug(f"🔍 {pair}: Aucune zone FVG/OB valide")
            return

        # 8. Vérification de la confluence (utilise direction déclarée plus haut)
        confluence_score = check_confluence(pair, direction)
        if confluence_score < MIN_CONFLUENCE_SCORE:
            logger.warning(f"⚠️ {pair}: Confluence insuffisante ({confluence_score}/{MIN_CONFLUENCE_SCORE})")
            return

        # 9. Filtre macroéconomique
        if not macro_filter(pair, direction):
            logger.warning(f"⚠️ {pair}: Contexte macro défavorable")
            return

        # 10. Calcul des niveaux avec référence au range asiatique (nouveau)
        stop_loss = calculate_stop(fvg_zone, ob_zone, asian_range, direction)
        take_profit = calculate_tp(current_price, asian_range, direction, daily_zones.get(pair, {}))
        
        # 11. Journalisation détaillée
        logger.info(f"""
        🎯 Signal confirmé sur {pair}:
        • Direction: {direction}
        • Prix: {current_price:.5f}
        • Range Asiatique: {asian_range['low']:.5f}-{asian_range['high']:.5f}
        • Stop: {stop_loss:.5f} (Risque: {abs(current_price-stop_loss):.1f}pips)
        • TP: {take_profit:.5f} (Gain potentiel: {abs(take_profit-current_price):.1f}pips)
        • Ratio R/R: {abs(take_profit-current_price)/abs(current_price-stop_loss):.1f}
        • Confluence: {confluence_score}/3
        • Tendance HTF: {trend}
        """)

        # 12. Execution du trade
        place_trade(
            pair=pair,
            direction=direction.lower(),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    except Exception as e:
        logger.error(f"❌ Erreur analyse {pair}: {str(e)}", exc_info=True)

def get_buffer_size(pair):
    """Retourne le buffer adapté à chaque instrument"""
    buffers = {
        'EUR_USD': 0.0003,  # 3 pips
        'GBP_USD': 0.0003,
        'USD_JPY': 0.03,    # 30 pips
        'XAU_USD': 0.3,      # 0.3$
        'BTC_USD': 10.0,
        'ETH_USD': 5.0
    }
    return buffers.get(pair, 0.0005)  # Valeur par défaut

def calculate_tp_from_structure(fvg, direction):
    """Calcule le TP basé sur la taille du FVG"""
    fvg_size = abs(fvg['top'] - fvg['bottom'])
    if direction == 'buy':
        return fvg['top'] + fvg_size * 1.5  # TP = 1.5x la taille du FVG
    else:
        return fvg['bottom'] - fvg_size * 1.5
def get_central_bank_rates(currency):
    """Récupère les décisions de taux"""
    try:
        if currency == "USD":
            url = f"https://api.federalreserve.gov/data/DPCREDIT/current?api_key={MACRO_API_KEY}"
            data = requests.get(url).json()
            return {
                "rate": float(data["observations"][0]["value"]),
                "trend": data["trend"]  # 'hawkish'/'dovish'
            }
    except Exception as e:
        logger.error(f"❌ Erreur taux {currency}: {str(e)}")
        return None

def log_macro_context(pair):
    """Affiche le contexte macro"""
    base_currency = pair[:3]
    events = check_economic_calendar(pair)
    
    logger.info(f"""
    📊 Contexte Macro {base_currency}:
    • Prochains événements: {[e[0] for e in events]}
    • Taux directeur: {get_central_bank_rates(base_currency)}
    • Inflation (CPI): {get_macro_data(base_currency, "CPI")}
    """)

def inflation_adjustment(pair):
    """Ajuste le TP/SL en fonction de l'inflation"""
    cpi = get_macro_data(pair[:3], "CPI")
    if not cpi:
        return 1.0
        
    # Exemple: Augmente le TP si inflation haute
    return 1.1 if cpi > 3.0 else 0.9

def get_htf_data(pair, timeframe='H4'):
    """
    Récupère les données HTF avec cache et gestion d'erreur robuste
    Args:
        pair: str - Paire de trading (ex: 'EUR_USD')
        timeframe: str - Période valide (S5, M1, M5, M15, H1, H4, D, W, M)
    Returns:
        list - Liste des bougies ou None en cas d'erreur
    """
    # Initialisation du cache si inexistant
    if not hasattr(get_htf_data, 'cache'):
        get_htf_data.cache = {}
    
    # Validation du timeframe
    valid_timeframes = ["S5", "M1", "M5", "M15", "H1", "H4", "D", "W", "M"]
    if timeframe not in valid_timeframes:
        logger.error(f"❌ Timeframe {timeframe} invalide. Utilisation de H4 par défaut")
        timeframe = "H4"

    # Vérification du cache
    cache_key = f"{pair}_{timeframe}"
    if cache_key in get_htf_data.cache:
        cache_data = get_htf_data.cache[cache_key]
        if time.time() - cache_data['timestamp'] < 900:  # 15 minutes
            return cache_data['data']

    # Configuration des paramètres
    params = {
        "granularity": timeframe,
        "count": 100,
        "price": "M"
    }

    try:
        # Requête API
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        data = client.request(r)['candles']
        
        # Mise en cache
        get_htf_data.cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        return data

    except Exception as e:
        logger.error(f"❌ Erreur récupération HTF {pair} ({timeframe}): {str(e)}")
        return None

def is_price_in_valid_range(price, asian_range, trend):
    """Vérifie si le prix est dans une configuration favorable par rapport au range asiatique"""
    buffer = 0.0003  # 3 pips de marge
    
    # Configuration haussière
    if trend == 'UP' and price > (asian_range['high'] - buffer):
        return True
        
    # Configuration baissière
    if trend == 'DOWN' and price < (asian_range['low'] + buffer):
        return True
        
    return False

def calculate_stop(fvg_zone, ob_zone, asian_range, direction):
    if direction == 'BUY':
        return min(fvg_zone['bottom'], ob_zone['low'], asian_range['low']) - 0.0005
    else:
        return max(fvg_zone['top'], ob_zone['high'], asian_range['high']) + 0.0005

def calculate_tp(entry, asian_range, direction, daily_zone=None):
    """Calcule le take-profit avec référence au range asiatique"""
    range_size = asian_range['high'] - asian_range['low']
    
    if direction == 'BUY':
        base_tp = entry + range_size * 1.5  # 1.5x la taille du range
        if daily_zone:
            return max(base_tp, daily_zone.get('VAH', 0))
        return base_tp
    else:
        base_tp = entry - range_size * 1.5
        if daily_zone:
            return min(base_tp, daily_zone.get('VAL', float('inf')))
        return base_tp

def handle_weekend(now):
    """Gère spécifiquement la fermeture du week-end"""
    global end_of_day_processed
    close_all_trades()
    next_monday = now + timedelta(days=(7 - now.weekday()))
    sleep_seconds = (next_monday - now).total_seconds()
    logger.info(f"⛔ Week-end - Prochaine ouverture à {LONDON_SESSION_START.strftime('%H:%M')} UTC")
    time.sleep(min(sleep_seconds, 21600))  # Max 6h
    end_of_day_processed = True

def log_session_status():
    """Affiche un résumé complet du statut"""
    logger.info(f"""
    🕒 Statut à {datetime.utcnow().strftime('%H:%M UTC')}
    • Trades actifs: {len(active_trades)}
    • Prochain événement macro: {get_next_macro_event()}
    • Liquidité moyenne: {calculate_market_liquidity()}
    """)

def check_high_impact_events():
    """Vérifie les événements macro à haut impact"""
    events = []
    for pair in PAIRS:
        currency = pair[:3]
        events += check_economic_calendar(currency)
    
    # Filtre les événements à haut impact dans les 2h
    critical_events = [
        e for e in events 
        if e["impact"] == "HIGH" 
        and e["time"] - datetime.utcnow() < timedelta(hours=2)
    ]
    
    return len(critical_events) > 0

def process_trading_session():
    """Gère la session de trading active"""
    start_time = time.time()
    
    # Vérification des événements macro imminents
    if check_high_impact_events():
        logger.warning("⚠️ Événement macro majeur - Trading suspendu temporairement")
        time.sleep(300)
        return

    # Analyse normale
    analyze_markets()
    
    # Optimisation du timing
    elapsed = time.time() - start_time
    sleep_time = max(30 - elapsed, 5)  # Cycle plus rapide
    time.sleep(sleep_time)

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

def store_asian_range(pair):
    """Stocke le range asiatique pour une paire"""
    try:
        candles = get_candles(pair, ASIAN_SESSION_START, ASIAN_SESSION_END)
        highs = [float(c['mid']['h']) for c in candles]
        lows = [float(c['mid']['l']) for c in candles]
        asian_ranges[pair] = {
            'high': max(highs),
            'low': min(lows),
            'time': datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"❌ Erreur calcul range {pair}: {str(e)}")

def update_macro_data():
    """Actualise tous les indicateurs macro"""
    global daily_data_updated
    try:
        for currency in set(pair[:3] for pair in PAIRS):
            fetch_macro_indicators(currency)
        daily_data_updated = True
    except Exception as e:
        logger.error(f"❌ Échec MAJ données macro: {str(e)}")

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

def get_asian_range(pair):
    """
    Calcule et retourne le range asiatique sous forme de dictionnaire
    Format:
    {
        'low': float,  # Plus bas de la session
        'high': float, # Plus haut de la session 
        'time': datetime # Heure de calcul
    }
    """
    try:
        candles = get_candles(pair, ASIAN_SESSION_START, ASIAN_SESSION_END)
        if not candles:
            return None
            
        highs = [float(c['mid']['h']) for c in candles if 'mid' in c]
        lows = [float(c['mid']['l']) for c in candles if 'mid' in c]
        
        if not highs or not lows:
            return None
            
        return {
            'low': min(lows),
            'high': max(highs), 
            'time': datetime.utcnow()  # Horodatage du calcul
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur calcul range {pair}: {str(e)}")
        return None

def check_timeframe_validity(tf):
    valid = ["S5","M1","M5","M15","H1","H4","D","W","M"]
    if tf not in valid:
        raise ValueError(f"Timeframe invalide. Utiliser: {valid}")

def analyze_asian_session():
    """
    Analyse complète de la session asiatique pour toutes les paires
    Calcule et stocke les prix high/low pour chaque paire durant la session
    """
    global asian_range_calculated
    
    if asian_range_calculated:
        logger.debug("🔄 Session asiatique déjà analysée")
        return
        
    logger.info("🌏 DÉBUT ANALYSE SESSION ASIATIQUE")
    
    now = datetime.utcnow()
    today = now.date()
    
    # Calcul des dates de début/fin exactes
    start_dt = datetime.combine(today, ASIAN_SESSION_START)
    end_dt = datetime.combine(today, ASIAN_SESSION_END)
    
    # Si on est encore dans la session asiatique
    if now < end_dt:
        end_dt = now  # On utilise l'heure actuelle comme fin
        logger.info("⏳ Session asiatique en cours - Analyse partielle")
    
    logger.info(f"🔍 Plage analysée: {start_dt} à {end_dt} (UTC)")

    for pair in PAIRS:
        try:
            params = {
                "granularity": "H1",  # Bougies horaires
                "from": start_dt.isoformat() + "Z",
                "to": end_dt.isoformat() + "Z",
                "price": "M"  # Prix mid
            }
            
            logger.debug(f"📡 Récupération données pour {pair}...")
            candles = client.request(
                instruments.InstrumentsCandles(
                    instrument=pair,
                    params=params
                )
            )['candles']
            
            # Filtrage des bougies complètes
            valid_candles = [c for c in candles if c['complete']]
            
            if not valid_candles:
                logger.warning(f"⚠️ Aucune donnée valide pour {pair}")
                continue
                
            # Extraction des prix
            highs = [float(c['mid']['h']) for c in valid_candles]
            lows = [float(c['mid']['l']) for c in valid_candles]
            
            if not highs or not lows:
                logger.warning(f"⚠️ Données incomplètes pour {pair}")
                continue
                
            # Stockage des résultats
            asian_ranges[pair] = {
                'high': max(highs),
                'low': min(lows),
                'time': end_dt,
                'candles': len(valid_candles)
            }
            
            logger.info(
                f"📊 {pair}: "
                f"Low={asian_ranges[pair]['low']:.5f} | "
                f"High={asian_ranges[pair]['high']:.5f} | "
                f"Bougies={asian_ranges[pair]['candles']}"
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse {pair}: {str(e)}")
            continue
    
    asian_range_calculated = True
    logger.info("✅ ANALYSE ASIATIQUE TERMINÉE")

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

    # Attente initiale pour la première mise à jour
    if datetime.utcnow().time() < dtime(0, 30):
        logger.info(f"⏳ Attente jusqu'à 00:30 UTC (actuellement {datetime.utcnow().time()})")
    while datetime.utcnow().time() < dtime(0, 30):
        time.sleep(60)
    
    update_daily_zones()  # Premier calcul après 00:30 UTC

# Avant la boucle principale, ajoutez :
logger.info(f"""
🛠️ CONFIGURATION VÉRIFICATION:
- Heure système UTC: {datetime.now(pytz.UTC)}
- Asian Session config: {ASIAN_SESSION_START}-{ASIAN_SESSION_END}
- London Session config: {LONDON_SESSION_START}-{NY_SESSION_END}
""")


# Boucle principale
while True:
    try:
        now = datetime.utcnow()
        current_time = now.time()
        weekday = now.weekday()

        # Détermination précise de la session
        if ASIAN_SESSION_START <= current_time < ASIAN_SESSION_END:
            session_type = "ASIE"
        elif LONDON_SESSION_START <= current_time <= NY_SESSION_END:
            session_type = "LON/NY"
        else:
            session_type = "HORS-SESSION"

        logger.info(f"\n🔄 Cycle début - {now} (UTC) | Session: {session_type}")

        # 1. Gestion Week-End
        if weekday >= 5:
            handle_weekend(now)
            continue

        # 2. Réinitialisation quotidienne (à 00:00 UTC)
        if current_time.hour == 0 and current_time.minute < 30:
            logger.info("🔄 Réinitialisation quotidienne des données")
            daily_data_updated = False
            asian_range_calculated = False
            end_of_day_processed = False

        # 3. Mise à jour des données journalières (après 00:30 UTC)
        if not daily_data_updated and current_time >= dtime(0, 30):
            update_daily_zones()
            
        # 4. Session Asiatique (00:00-06:00 UTC)
        if session_type == "ASIE":
            logger.info(f"🌏 DÉBUT SESSION ASIATIQUE ({current_time})")
            
            if not asian_range_calculated:
                analyze_asian_session()  # Utilise la nouvelle fonction robuste
            else:
                # Vérification périodique du range
                current_price = get_current_price("EUR_USD")  # Paire de référence
                asian_range = asian_ranges.get("EUR_USD", {})
                
                if asian_range:
                    if current_price > asian_range['high']:
                        logger.info(f"🚨 Prix {current_price} > High asiatique {asian_range['high']}")
                    elif current_price < asian_range['low']:
                        logger.info(f"🚨 Prix {current_price} < Low asiatique {asian_range['low']}")
            
            time.sleep(300)  # Vérification toutes les 5 minutes
            continue

        # 5. Pause Entre Sessions (06:00-07:00 UTC)
        if session_type == "HORS-SESSION" and ASIAN_SESSION_END <= current_time < LONDON_SESSION_START:
            logger.debug("⏳ Pause entre sessions - Attente London")
            time.sleep(60)
            continue

        # 6. Session Active (Londres + NY, 07:00-16:30 UTC)
        if session_type == "LON/NY":
            cycle_start = time.time()
            logger.info("🏙️ SESSION LONDRES/NY ACTIVE")
            
            # Vérification des ranges asiatiques (au cas où manqués)
            if not asian_range_calculated:
                logger.warning("⚠️ Aucun range asiatique détecté - Calcul rétroactif")
                analyze_asian_session()
            
            # Vérification des événements macro
            if check_high_impact_events():
                logger.warning("⚠️ Événement macro majeur - Pause de 5min")
                time.sleep(300)
                continue
                
            # Analyse et trading
            active_trades = check_active_trades()
            logger.info(f"📊 Trades actifs: {len(active_trades)}")
            
            for pair in PAIRS:
                if pair not in active_trades:
                    analyze_pair(pair)  # Utilise les ranges asiatiques
                else:
                    logger.debug(f"⏩ Trade actif sur {pair} - Surveillance")
                    check_tp_sl_for_pair(pair)  # Vérifie spécifiquement ce trade
            
            # Timing dynamique
            elapsed = time.time() - cycle_start
            sleep_time = max(30 - elapsed, 5)
            logger.debug(f"⏱ Prochain cycle dans {sleep_time:.1f}s")
            time.sleep(sleep_time)
            continue

        # 7. Après Fermeture (16:30-00:00 UTC)
        if session_type == "HORS-SESSION" and not end_of_day_processed:
            logger.info("🌙 Fermeture de session - Vérification des positions")
            close_all_trades()
            end_of_day_processed = True
            
            # Rapport journalier
            try:
                log_daily_summary()
            except Exception as e:
                logger.error(f"❌ Erreur rapport journalier: {str(e)}")
            
        time.sleep(60)

    except Exception as e:
        logger.error(f"💥 ERREUR GRAVE: {str(e)}", exc_info=True)
        time.sleep(300)  # Long délai après une erreur grave
