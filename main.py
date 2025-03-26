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
from datetime import datetime, timedelta

from functools import lru_cache, wraps
from functools import lru_cache
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
NY_SESSION_END = dtime(16, 25)       # 16h30 UTC
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



def timed_lru_cache(seconds: int, maxsize: int = 128):
    """Décorateur de cache avec expiration temporelle"""
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime
            return func(*args, **kwargs)
        return wrapped_func
    return wrapper_cache

@timed_lru_cache(seconds=3600, maxsize=32)
def get_economic_events(country):
    """
    Récupère les événements économiques avec :
    - Cache LRU intelligent avec expiration
    - Gestion des erreurs robuste
    - Optimisation des requêtes
    """
    params = {
        "function": "ECONOMIC_CALENDAR",
        "apikey": ALPHA_VANTAGE_API_KEY,
        "countries": country,
        "importance": "high,medium",
        "time_from": datetime.utcnow().strftime("%Y%m%dT%H%M"),
        "time_to": (datetime.utcnow() + timedelta(days=1)).strftime("%Y%m%dT%H%M")
    }

    data = safe_api_call(params)
    
    if not data:
        logger.error(f"No data received for {country}")
        return []
    
    try:
        events = data.get('economicCalendar', [])
        
        formatted_events = []
        for event in events:
            try:
                formatted_events.append({
                    'name': event['event'],
                    'time': datetime.strptime(event['time'], '%Y-%m-%dT%H:%M:%S'),
                    'currency': event.get('currency', country),
                    'impact': event.get('impact', 'medium').upper(),
                    'actual': event.get('actual'),
                    'forecast': event.get('forecast'),
                    'previous': event.get('previous')
                })
            except KeyError as e:
                logger.warning(f"Missing key in event data: {str(e)}")
                continue
        
        logger.info(f"Retrieved {len(formatted_events)} valid events for {country}")
        return formatted_events
        
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        return []

def macro_filter(pair, direction):
    """Version optimisée avec :
    - Requêtes groupées
    - Cache intelligent
    - Gestion des timeouts
    """
    base_currency = pair[:3]
    events = get_economic_events(base_currency)
    
    # 1. Détection événements critiques (prochains 4h)
    now = datetime.utcnow()
    critical_events = [
        event for event in events
        if event['impact'] == 'High' and
        datetime.strptime(event['time'], '%Y-%m-%dT%H:%M:%S') - now <= timedelta(hours=4)
    ]
    
    if critical_events:
        next_event = min(critical_events, key=lambda x: x['time'])
        logger.warning(
            f"⛔ Événement critique détecté\n"
            f"• Nom: {next_event['event']}\n"
            f"• Heure: {next_event['time']}\n"
            f"• Impact: {next_event['impact']}\n"
            f"• Devise: {base_currency}"
        )
        return False

    # 2. Analyse contextuelle (inflation/taux)
    if base_currency == "USD":
        cpi = get_macro_data("USD", "CPI")
        if cpi and cpi > 5.0:
            logger.info(f"📈 Contexte inflationniste (CPI: {cpi}%)")
            return direction == "BUY"

    return True

# Configuration Alpha Vantage
ECONOMIC_CALENDAR_API = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # Clé gratuite sur leur site
REQUEST_INTERVAL = 12  # secondes entre les requêtes (5 req/min)
LAST_CALL_TIME = 0

@lru_cache(maxsize=4)  # Cache pour les 4 principales devises
def get_economic_events(country):
    """Filtre les événements par devise"""
    events = get_all_economic_events()
    return [
        {
            'name': e['event'],
            'time': datetime.strptime(e['time'], '%Y-%m-%dT%H:%M:%S'),
            'currency': e.get('currency', country),
            'impact': e.get('impact', 'medium').upper(),
            'actual': e.get('actual'),
            'forecast': e.get('forecast')
        }
        for e in events if e.get('currency', '').upper() == country.upper()
    ]

def check_economic_calendar(pair):
    """Récupère le calendrier économique via Alpha Vantage"""
    base_currency = pair[:3]
    
    params = {
        "function": "ECONOMIC_CALENDAR",
        "apikey": ALPHA_VANTAGE_API_KEY,
        "countries": base_currency,
        "importance": "high,medium",
        "time_from": datetime.utcnow().strftime("%Y%m%dT%H%M"),
        "time_to": (datetime.utcnow() + timedelta(days=1)).strftime("%Y%m%dT%H%M")
    }
    
    try:
        response = requests.get(ECONOMIC_CALENDAR_API, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return [(
            event['event'],
            datetime.strptime(event['time'], '%Y-%m-%dT%H:%M:%S'),
            event['impact'].upper()
        ) for event in data.get('economicCalendar', [])]
        
    except Exception as e:
        logger.warning(f"⚠️ Erreur calendrier Alpha Vantage: {str(e)}")
        return []
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

# Dans votre configuration principale
REQUEST_DELAY = 0.2  # 200ms entre les requêtes
MAX_RETRIES = 3
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def safe_api_call(params):
    """Gestion robuste des appels API avec retry et backoff"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY * (attempt - 1))  # Backoff linéaire
            response = requests.get(
                "https://www.alphavantage.co/query",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {params['countries']}: {str(e)}")
            if attempt == MAX_RETRIES:
                logger.error(f"API call failed after {MAX_RETRIES} attempts")
                return None

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
    """Version corrigée du calcul RSI"""
    try:
        if len(prices) < period + 1:
            return None
            
        deltas = np.diff(prices)
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period]) or 1e-10  # Évite division par 0
        
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))
    except Exception as e:
        logger.error(f"Erreur RSI: {str(e)}")
        return None


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

def get_candles_safely(pair, granularity, count=None, from_time=None, to_time=None):
    """Version corrigée de la récupération des candles"""
    params = {
        "granularity": granularity,
        "price": "M"
    }
    
    if count:
        params["count"] = count
    if from_time and to_time:
        params["from"] = from_time.isoformat() + "Z"
        params["to"] = to_time.isoformat() + "Z"
    
    try:
        # Solution sans timeout dans API.request()
        start_time = time.time()
        candles = client.request(
            instruments.InstrumentsCandles(
                instrument=pair,
                params=params
            )
        )['candles']
        
        # Vérification manuelle du timeout
        if time.time() - start_time > 10:  # 10 secondes max
            logger.warning(f"⚠️ Requête {pair} a pris trop de temps")
            return None
            
        return candles
    except Exception as e:
        logger.error(f"❌ Erreur récupération candles {pair}: {str(e)}")
        return None
        
def analyze_asian_session_v2():
    """Version robuste de l'analyse asiatique"""
    global asian_ranges, asian_range_calculated
    
    logger.info("🌏 NOUVELLE ANALYSE ASIATIQUE EN COURS")
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
    success_count = 0
    
    now = datetime.utcnow()
    start_time = datetime.combine(now.date(), ASIAN_SESSION_START)
    end_time = min(datetime.combine(now.date(), ASIAN_SESSION_END), now)

    for pair in pairs:
        try:
            # Récupération des données sécurisée
            candles = get_candles_safely(
                pair=pair,
                granularity="H1",
                from_time=start_time,
                to_time=end_time
            )
            
            if not candles or len(candles) < 3:
                logger.warning(f"⚠️ Données insuffisantes pour {pair}")
                continue
                
            # Traitement des données
            valid_candles = [c for c in candles if c['complete']]
            highs = [float(c['mid']['h']) for c in valid_candles]
            lows = [float(c['mid']['l']) for c in valid_candles]
            
            if not highs or not lows:
                continue
                
            # Enregistrement des résultats
            asian_ranges[pair] = {
                'high': max(highs),
                'low': min(lows),
                'time': end_time,
                'candles': len(valid_candles)
            }
            success_count += 1
            logger.info(f"✅ {pair} range: {asian_ranges[pair]['low']:.5f}-{asian_ranges[pair]['high']:.5f}")
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse {pair}: {str(e)}")
            continue
    
    # Validation finale
    if success_count >= len(pairs) // 2:  # Au moins 50% de réussite
        asian_range_calculated = True
        logger.info(f"✅ ANALYSE ASIATIQUE COMPLÈTE ({success_count}/{len(pairs)} paires)")
    else:
        logger.error("❌ ANALYSE ASIATIQUE INCOMPLÈTE")


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
        
def identify_fvg(candles, min_strength=5):
    """Version corrigée avec gestion des types"""
    fvgs = []
    
    for i in range(1, len(candles)):
        try:
            prev = candles[i-1]['mid']
            curr = candles[i]['mid']
            
            # Conversion en float
            prev_h = float(prev['h'])
            prev_l = float(prev['l'])
            curr_h = float(curr['h'])
            curr_l = float(curr['l'])
            
            # Détection FVG
            bull_gap = curr_l > prev_h
            bear_gap = curr_h < prev_l
            
            if bull_gap:
                strength = (curr_l - prev_h) * 10000  # en pips
                if strength >= min_strength:
                    fvgs.append({
                        'type': 'BULLISH',
                        'top': prev_h,
                        'bottom': curr_l,
                        'strength': round(strength, 1)
                    })
                    
            elif bear_gap:
                strength = (prev_l - curr_h) * 10000
                if strength >= min_strength:
                    fvgs.append({
                        'type': 'BEARISH',
                        'top': prev_l,
                        'bottom': curr_h,
                        'strength': round(strength, 1)
                    })
                    
        except (KeyError, ValueError) as e:
            logger.warning(f"Erreur traitement FVG bougie {i}: {str(e)}")
            continue
    
    return fvgs

def identify_order_blocks(candles, min_ratio=2):
    """Version corrigée avec conversion explicite des types"""
    blocks = {'bullish': [], 'bearish': []}
    
    for i in range(2, len(candles)):
        try:
            prev = candles[i-1]['mid']
            curr = candles[i]['mid']
            
            # Conversion explicite en float
            prev_o = float(prev['o'])
            prev_c = float(prev['c'])
            prev_h = float(prev['h'])
            prev_l = float(prev['l'])
            curr_c = float(curr['c'])
            curr_o = float(curr['o'])
            
            # Bullish OB
            if (prev_c > prev_o) and (curr_c < curr_o):
                body_size = abs(prev_c - prev_o)
                candle_range = prev_h - prev_l
                if candle_range > 0 and (body_size > (candle_range / min_ratio)):
                    blocks['bullish'].append({
                        'high': prev_h,
                        'low': prev_l,
                        'open': prev_o,
                        'close': prev_c,
                        'time': candles[i-1].get('time')
                    })
                    
            # Bearish OB
            elif (prev_c < prev_o) and (curr_c > curr_o):
                body_size = abs(prev_o - prev_c)
                candle_range = prev_h - prev_l
                if candle_range > 0 and (body_size > (candle_range / min_ratio)):
                    blocks['bearish'].append({
                        'high': prev_h,
                        'low': prev_l,
                        'open': prev_o,
                        'close': prev_c,
                        'time': candles[i-1].get('time')
                    })
                    
        except (KeyError, ValueError) as e:
            logger.warning(f"Erreur traitement bougie {i}: {str(e)}")
            continue
            
    logger.debug(f"OB détectés: {len(blocks['bullish'])} haussiers / {len(blocks['bearish'])} baissiers")
    return blocks
    
def analyze_pair(pair):
    """Version finale corrigée avec gestion robuste des erreurs et optimisations"""
    try:
        logger.info(f"🔍 Début analyse pour {pair}")
        
        # 1. Récupération et validation des données avec timeout
        try:
            candles = get_htf_data(pair, "H4")
            if not candles or len(candles) < 100:
                logger.warning(f"Données insuffisantes pour {pair} (reçu: {len(candles) if candles else 0}/100 bougies)")
                return
        except Exception as e:
            logger.error(f"Erreur récupération données {pair}: {str(e)}")
            return

        # 2. Nettoyage approfondi des données
        closes = []
        volumes = []
        valid_candles = 0
        
        for candle in candles:
            try:
                # Vérification en profondeur de la structure des données
                if not isinstance(candle, dict) or 'mid' not in candle:
                    continue
                    
                mid_data = candle['mid']
                if not isinstance(mid_data, dict) or 'c' not in mid_data:
                    continue
                
                # Conversion sécurisée
                close_price = float(mid_data['c'])
                volume = float(candle.get('volume', 0))
                
                closes.append(close_price)
                volumes.append(volume)
                valid_candles += 1
                
            except (TypeError, ValueError, AttributeError) as e:
                logger.debug(f"Bougie ignorée - erreur traitement: {str(e)}")
                continue

        if valid_candles < 100:
            logger.warning(f"Données valides insuffisantes pour {pair} (reçu: {valid_candles}/100 bougies valides)")
            return

        # 3. Calcul des indicateurs avec vérification renforcée
        current_price = closes[-1]
        
        # EMA200
        ema200 = calculate_ema(closes, 200)
        if ema200 is None:
            logger.warning("Échec calcul EMA200 - données peut-être corrompues")
            return
            
        # RSI
        rsi = calculate_rsi(closes[-15:])  # Only need last 15 periods for RSI14
        if rsi is None:
            logger.warning("Échec calcul RSI - vérifier les données de prix")
            return
            
        # MACD
        macd_line, signal_line = calculate_macd(closes)
        if None in (macd_line, signal_line):
            logger.warning("Échec calcul MACD - série temporelle trop courte?")
            return
            
        macd_hist = macd_line - signal_line

        # 4. Analyse de tendance améliorée
        trend = 'UP' if current_price > ema200 else 'DOWN' if current_price < ema200 else 'NEUTRAL'
        
        # 5. Filtres de confluence stricts
        if trend == 'UP' and not (30 <= rsi < 70 and macd_hist > 0):
            logger.warning(f"Confluence haussière insuffisante (RSI: {rsi:.1f}, MACD: {'↑' if macd_hist > 0 else '↓'})")
            return
        elif trend == 'DOWN' and not (30 < rsi <= 70 and macd_hist < 0):
            logger.warning(f"Confluence baissière insuffisante (RSI: {rsi:.1f}, MACD: {'↑' if macd_hist > 0 else '↓'})")
            return

        # 6. Analyse avancée (range, structures)
        analysis = enhanced_analyze_pair(pair)
        if not analysis:
            logger.warning("Échec analyse avancée")
            return

        # 7. Contexte de prix et range asiatique
        asian_range = asian_ranges.get(pair)
        if not asian_range:
            logger.warning("Aucun range asiatique disponible")
            return

        # 8. Détection des zones techniques (FVG/OB)
        fvgs = identify_fvg(candles)
        obs = identify_order_blocks(candles)
        
        in_fvg, fvg_zone = is_price_in_fvg(current_price, fvgs)
        near_ob, ob_zone = is_price_near_ob(current_price, obs, trend)
        
        if not (in_fvg and near_ob):
            logger.info("Aucune zone technique valide détectée")
            return

        # 9. Calcul des niveaux de trading
        direction = 'BUY' if trend == 'UP' else 'SELL'
        stop_loss = calculate_stop(fvg_zone, ob_zone, asian_range, direction)
        take_profit = calculate_tp(current_price, asian_range, direction, daily_zones.get(pair, {}))
        
        # Ajustement pour les structures de retournement
        if analysis['reversal_structure']:
            logger.info(f"Structure {analysis['reversal_structure']['type']} détectée - ajustement TP")
            take_profit = analysis['reversal_structure'].get('target', take_profit)

        # 10. Validation finale du trade
        rr_ratio = abs(take_profit-current_price)/abs(current_price-stop_loss)
        if rr_ratio < RISK_REWARD_RATIO:
            logger.warning(f"Ratio R/R inacceptable ({rr_ratio:.1f} < {RISK_REWARD_RATIO})")
            return

        if not macro_filter(pair, direction):
            logger.warning("Conditions macroéconomiques défavorables")
            return

        # 11. Journalisation avant exécution
        logger.info(f"""
        ✅ SIGNAL VALIDE {pair} ✅
        Direction: {'ACHAT' if direction == 'BUY' else 'VENTE'}
        Contexte: {'Range' if analysis['range_info'].get('is_range') else 'Tendance'}
        Structure: {analysis['reversal_structure']['type'] if analysis['reversal_structure'] else 'Aucune'}
        Prix: {current_price:.5f} | EMA200: {ema200:.5f}
        RSI(14): {rsi:.1f} | MACD: {'↑' if macd_hist > 0 else '↓'} {macd_hist:.5f}
        Stop: {stop_loss:.5f} | TP: {take_profit:.5f} | R/R: {rr_ratio:.1f}
        Volume récent: {sum(volumes[-5:]):.0f} (moyenne: {sum(volumes[-20:])/20:.0f})
        """)

        # 12. Exécution du trade
        place_trade(
            pair=pair,
            direction=direction.lower(),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    except Exception as e:
        logger.error(f"ERREUR {pair}: {str(e)}", exc_info=True)

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

def get_clean_candle_data(candles):
    """Extrait et nettoie les données des bougies de manière robuste"""
    closes = []
    volumes = []
    
    for candle in candles:
        try:
            # Vérification complète de la structure des données
            if not isinstance(candle, dict):
                continue
                
            mid = candle.get('mid')
            if not mid or not isinstance(mid, dict):
                continue
                
            close = mid.get('c')
            volume = candle.get('volume', 0)
            
            # Conversion sécurisée en float
            closes.append(float(close))
            volumes.append(float(volume)))
            
        except (TypeError, ValueError, AttributeError) as e:
            logger.debug(f"Bougie ignorée - erreur conversion: {str(e)}")
            continue
            
    return closes, volumes

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

# Ajouter ces nouvelles fonctions dans votre code

def detect_range_market(pair, lookback_period=20, threshold_pct=0.005):
    """
    Détecte si le marché est en range basé sur l'ATR et le pourcentage de range
    Args:
        pair: str - Paire de trading
        lookback_period: int - Période d'analyse (en bougies)
        threshold_pct: float - Seuil pour considérer un range (0.5% par défaut)
    Returns:
        dict - {'is_range': bool, 'support': float, 'resistance': float}
    """
    try:
        candles = get_htf_data(pair, "H4")  # Utilise le timeframe H4 pour la détection
        if len(candles) < lookback_period:
            return {'is_range': False}
        
        highs = [float(c['mid']['h']) for c in candles[-lookback_period:]]
        lows = [float(c['mid']['l']) for c in candles[-lookback_period:]]
        closes = [float(c['mid']['c']) for c in candles[-lookback_period:]]
        
        # Calcul de l'ATR pour évaluer la volatilité
        atr = calculate_atr(candles, period=14)
        current_atr = atr[-1] if atr else 0
        
        # Calcul du range en pourcentage
        range_size = (max(highs) - min(lows)) / min(lows)
        is_range = range_size < threshold_pct and current_atr < (threshold_pct * min(lows))
        
        return {
            'is_range': is_range,
            'support': min(lows),
            'resistance': max(highs),
            'pivot': (max(highs) + min(lows)) / 2,
            'range_percent': range_size * 100
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur détection range {pair}: {str(e)}")
        return {'is_range': False}

def calculate_atr(candles, period=14):
    """Calcul de l'Average True Range"""
    try:
        tr_values = []
        for i in range(1, len(candles)):
            prev_close = float(candles[i-1]['mid']['c'])
            high = float(candles[i]['mid']['h'])
            low = float(candles[i]['mid']['l'])
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr_values.append(max(tr1, tr2, tr3))
            
        atr = []
        for i in range(len(tr_values)):
            if i < period:
                atr.append(sum(tr_values[:i+1])/(i+1))
            else:
                atr.append((atr[-1] * (period-1) + tr_values[i])/period)
                
        return atr
    except Exception as e:
        logger.error(f"❌ Erreur calcul ATR: {str(e)}")
        return []

def detect_reversal_structure(pair, timeframe='H1'):
    """
    Détecte les structures de retournement (double top/bottom, tête-épaules, etc.)
    Args:
        pair: str - Paire de trading
        timeframe: str - Timeframe pour l'analyse
    Returns:
        dict - Structure détectée ou None
    """
    try:
        candles = get_htf_data(pair, timeframe)
        if len(candles) < 10:
            return None
            
        closes = [float(c['mid']['c']) for c in candles]
        highs = [float(c['mid']['h']) for c in candles]
        lows = [float(c['mid']['l']) for c in candles]
        
        # Détection double top/bottom
        pattern = detect_double_top_bottom(highs, lows)
        if pattern:
            return pattern
            
        # Détection tête-épaules
        pattern = detect_head_shoulders(highs, lows)
        if pattern:
            return pattern
            
        return None
        
    except Exception as e:
        logger.error(f"❌ Erreur détection structure {pair}: {str(e)}")
        return None

def detect_double_top_bottom(highs, lows):
    """Détection des doubles tops/bottoms"""
    if len(highs) < 5:
        return None
        
    # Double Top
    if (highs[-3] > highs[-4] and 
        highs[-3] > highs[-2] and 
        abs(highs[-1] - highs[-3]) < (highs[-3] * 0.002) and  # Marge de 0.2%
        lows[-2] < lows[-3]):
        return {
            'type': 'DOUBLE_TOP',
            'confirmation_level': highs[-2],
            'target': lows[-3] - (highs[-3] - lows[-3])
        }
        
    # Double Bottom
    elif (lows[-3] < lows[-4] and 
          lows[-3] < lows[-2] and 
          abs(lows[-1] - lows[-3]) < (lows[-3] * 0.002) and 
          highs[-2] > highs[-3]):
        return {
            'type': 'DOUBLE_BOTTOM',
            'confirmation_level': lows[-2],
            'target': highs[-3] + (highs[-3] - lows[-3])
        }
        
    return None

def detect_head_shoulders(highs, lows):
    """Détection des patterns tête-épaules"""
    if len(highs) < 7:
        return None
        
    # Tête-épaules
    if (highs[-4] > highs[-5] and 
        highs[-4] > highs[-3] and 
        highs[-4] > highs[-6] and 
        highs[-6] > highs[-7] and 
        highs[-2] < highs[-4] and 
        abs(highs[-2] - highs[-6]) < (highs[-6] * 0.003)):
        return {
            'type': 'HEAD_SHOULDERS',
            'neckline': min(lows[-5], lows[-3]),
            'target': min(lows[-5], lows[-3]) - (highs[-4] - min(lows[-5], lows[-3]))
        }
        
    # Tête-épaules inversé
    elif (lows[-4] < lows[-5] and 
          lows[-4] < lows[-3] and 
          lows[-4] < lows[-6] and 
          lows[-6] < lows[-7] and 
          lows[-2] > lows[-4] and 
          abs(lows[-2] - lows[-6]) < (lows[-6] * 0.003)):
        return {
            'type': 'INVERSE_HEAD_SHOULDERS',
            'neckline': max(highs[-5], highs[-3]),
            'target': max(highs[-5], highs[-3]) + (max(highs[-5], highs[-3]) - lows[-4])
        }
        
    return None

def get_technical_confirmations(pair):
    """
    Récupère les confirmations techniques (EMA200, RSI, MACD)
    Returns:
        dict - Résultats des indicateurs
    """
    try:
        candles = get_htf_data(pair, "H1")
        if len(candles) < 200:
            return {}
            
        closes = [float(c['mid']['c']) for c in candles]
        
        # EMA200
        ema200 = calculate_ema(closes, 200)[-1]
        
        # RSI
        rsi = calculate_rsi(closes, 14)[-1]
        
        # MACD
        macd_line, signal_line = calculate_macd(closes)
        
        current_price = closes[-1]
        ema_signal = 'BULLISH' if current_price > ema200 else 'BEARISH'
        rsi_signal = 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL'
        macd_signal = 'BULLISH' if macd_line[-1] > signal_line[-1] else 'BEARISH'
        
        return {
            'ema200': {
                'value': ema200,
                'signal': ema_signal,
                'distance_pct': abs(current_price - ema200) / ema200 * 100
            },
            'rsi': {
                'value': rsi,
                'signal': rsi_signal
            },
            'macd': {
                'line': macd_line[-1],
                'signal': signal_line[-1],
                'histogram': macd_line[-1] - signal_line[-1],
                'signal_type': macd_signal
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur confirmations techniques {pair}: {str(e)}")
        return {}

def get_clean_prices(candles):
    """Extrait et nettoie les prix de clôture"""
    prices = []
    for candle in candles:
        try:
            if 'mid' in candle and 'c' in candle['mid']:
                prices.append(float(candle['mid']['c']))
        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Erreur traitement bougie: {str(e)}")
            continue
    return prices

def calculate_ema(prices, period):
    """Version robuste du calcul EMA"""
    try:
        if not prices or len(prices) < period:
            return None
            
        series = pd.Series(prices)
        return float(series.ewm(span=period, adjust=False).mean().iloc[-1])
    except Exception as e:
        logger.error(f"Erreur EMA {period}: {str(e)}")
        return None

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD sécurisé avec gestion d'erreur"""
    try:
        if len(prices) < slow + signal:
            return None, None
            
        series = pd.Series(prices)
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])
    except Exception as e:
        logger.error(f"Erreur MACD: {str(e)}")
        return None, None
        
def enhanced_analyze_pair(pair):
    """
    Version améliorée de l'analyse avec:
    - Détection de range
    - Structures de retournement
    - Confirmations techniques
    """
    try:
        # 1. Vérification du contexte de marché
        range_info = detect_range_market(pair)
        reversal_structure = detect_reversal_structure(pair)
        tech_conf = get_technical_confirmations(pair)
        
        logger.info(f"""
        📊 ANALYSE AMÉLIORÉE {pair}
        • Range Market: {'OUI' if range_info.get('is_range') else 'NON'} 
          (Support: {range_info.get('support', 0):.5f}, Resistance: {range_info.get('resistance', 0):.5f})
        • Structure: {reversal_structure['type'] if reversal_structure else 'Aucune'}
        • EMA200: {tech_conf.get('ema200', {}).get('signal', 'N/A')} ({tech_conf.get('ema200', {}).get('value', 0):.5f})
        • RSI: {tech_conf.get('rsi', {}).get('signal', 'N/A')} ({tech_conf.get('rsi', {}).get('value', 0):.1f})
        • MACD: {tech_conf.get('macd', {}).get('signal_type', 'N/A')} 
          (Hist: {tech_conf.get('macd', {}).get('histogram', 0):.5f})
        """)
        
        # 2. Stratégie pour range market
        if range_info.get('is_range'):
            current_price = get_current_price(pair)
            if current_price > range_info['resistance'] * 0.99:
                logger.info(f"🔔 {pair} proche résistance du range")
                # Stratégie de breakout ou faux breakout
                
            elif current_price < range_info['support'] * 1.01:
                logger.info(f"🔔 {pair} proche support du range")
                # Stratégie de rebond ou cassure
                
        # 3. Stratégie pour structure de retournement
        if reversal_structure:
            logger.info(f"🔔 Structure {reversal_structure['type']} détectée sur {pair}")
            # Implémenter la logique de trading basée sur la structure
            
        # 4. Filtrage par indicateurs
        if (tech_conf.get('ema200', {}).get('signal') == 'BULLISH' and 
            tech_conf.get('rsi', {}).get('signal') == 'OVERSOLD' and 
            tech_conf.get('macd', {}).get('signal_type') == 'BULLISH'):
            logger.info("🎯 Configuration haussière idéale détectée")
            
        elif (tech_conf.get('ema200', {}).get('signal') == 'BEARISH' and 
              tech_conf.get('rsi', {}).get('signal') == 'OVERBOUGHT' and 
              tech_conf.get('macd', {}).get('signal_type') == 'BEARISH'):
            logger.info("🎯 Configuration baissière idéale détectée")
            
        # 5. Analyse des prix par rapport aux indicateurs
        current_price = get_current_price(pair)
        ema200 = tech_conf.get('ema200', {}).get('value', current_price)
        
        if current_price > ema200:
            logger.info(f"📈 Prix au-dessus EMA200 ({ema200:.5f}) - Biais haussier")
        else:
            logger.info(f"📉 Prix sous EMA200 ({ema200:.5f}) - Biais baissier")
            
        return {
            'range_info': range_info,
            'reversal_structure': reversal_structure,
            'technical_confirmations': tech_conf
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur analyse avancée {pair}: {str(e)}")
        return {}

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

def is_price_in_valid_range(current_price, asian_range, trend, buffer_multiplier=1.0):
    """
    Version améliorée avec :
    - Buffer dynamique
    - Seuils adaptatifs
    - Gestion précise des bords
    """
    # Calcul du buffer en fonction de la volatilité
    range_size = asian_range['high'] - asian_range['low']
    buffer = range_size * 0.05 * buffer_multiplier  # 5% de la taille du range
    
    # Seuils avec buffer
    upper_threshold = asian_range['high'] - buffer if trend == 'UP' else asian_range['high'] + buffer
    lower_threshold = asian_range['low'] + buffer if trend == 'DOWN' else asian_range['low'] - buffer
    
    # Validation précise
    if trend == 'UP':
        return lower_threshold <= current_price <= asian_range['high']
    elif trend == 'DOWN':
        return asian_range['low'] <= current_price <= upper_threshold
    else:
        return False

def calculate_dynamic_buffer(pair):
    """Calcule un buffer adapté à la paire"""
    buffers = {
        'EUR_USD': 0.0003,  # 3 pips
        'GBP_USD': 0.0003,
        'USD_JPY': 0.03,    # 3 pips JPY
        'XAU_USD': 0.5,     # 50 cents
        'BTC_USD': 10.0,
        'ETH_USD': 5.0
    }
    return buffers.get(pair, 0.0005)  # Valeur par défaut

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

MAX_RETRIES = 3
RETRY_DELAY = 5
MIN_CANDLES = 3
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
    """Version robuste avec gestion des données partielles et réessais intelligents"""
    global asian_ranges, asian_range_calculated
    
    logger.info("🌏 LANCEMENT ANALYSE ASIATIQUE AVANCÉE")
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
    success_count = 0
    max_retries = 3  # Augmenté à 3 tentatives
    retry_delay = 60  # Augmenté à 60 secondes entre les tentatives

    for attempt in range(1, max_retries + 1):
        logger.info(f"🔁 Tentative {attempt}/{max_retries}")
        
        for pair in pairs:
            if pair in asian_ranges:  # Ne pas réessayer les paires déjà réussies
                continue
                
            try:
                now = datetime.utcnow()
                start_time = datetime.combine(now.date(), ASIAN_SESSION_START)
                end_time = min(datetime.combine(now.date(), ASIAN_SESSION_END), now)
                
                granularity = "M15" if attempt == 3 else "M30" if attempt == 2 else "H1"
                params = {
                    "granularity": granularity,
                    "from": start_time.isoformat() + "Z",
                    "to": end_time.isoformat() + "Z",
                    "price": "M"
                }
                
                logger.debug(f"📡 Récupération {pair} ({granularity})...")
                candles = client.request(
                    instruments.InstrumentsCandles(
                        instrument=pair,
                        params=params
                    )
                )['candles']
                
                if not candles:
                    raise ValueError("Aucune donnée reçue")
                    
                valid_candles = [c for c in candles if c.get('complete', False)]
                if len(valid_candles) < 4:
                    raise ValueError(f"Seulement {len(valid_candles)} bougies valides")
                
                highs = [float(c['mid']['h']) for c in valid_candles]
                lows = [float(c['mid']['l']) for c in valid_candles]
                if not highs or not lows:
                    raise ValueError("Données de prix manquantes")
                
                asian_ranges[pair] = {
                    'high': max(highs),
                    'low': min(lows),
                    'time': end_time,
                    'candles': len(valid_candles),
                    'granularity': granularity
                }
                success_count += 1
                logger.info(f"✅ {pair}: {asian_ranges[pair]['low']:.5f}-{asian_ranges[pair]['high']:.5f} ({len(valid_candles)} bougies {granularity})")
                
            except Exception as e:
                logger.warning(f"⚠️ {pair} tentative {attempt} échouée: {str(e)}")
                if attempt == max_retries:
                    logger.error(f"❌ Échec final pour {pair}")
        
        if success_count >= len(pairs) - 1:
            break
            
        if attempt < max_retries:
            logger.info(f"⏳ Prochaine tentative dans {retry_delay}s...")
            time.sleep(retry_delay)
    
    if success_count >= 2:
        asian_range_calculated = True
        logger.info(f"🏁 ANALYSE TERMINÉE: {success_count}/{len(pairs)} paires valides")
    else:
        logger.warning("💥 ÉCHEC CRITIQUE: Données insuffisantes pour trading, nouvelle tentative en cours...")
        analyze_asian_session()


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
