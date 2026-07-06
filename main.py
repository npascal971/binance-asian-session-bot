import os
import time
import logging
import smtplib
import json
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.transactions as transactions
import oandapyV20.endpoints.positions as positions

load_dotenv()

# =========================================================
# OANDA CONFIG
# =========================================================
OANDA_API_KEY = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_ACCESS_TOKEN")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

client = oandapyV20.API(access_token=OANDA_API_KEY, environment=OANDA_ENVIRONMENT)

# =========================================================
# STRATEGIE V73 - OANDA SCORING SNIPER PRODUCTION + MARKET HOURS + RISK FIX
# Exécution OANDA V67 + détection V63/V65 rééquilibrée
# Setups: FVG_RETEST_PERFECT, NESTED_FVG, WICK_REJECTION
# Scoring qualité: spread, H1 EMA50, rejet M15, RR, anti-doublon. Hard-block seulement sur risque extrême.
# =========================================================
PAIR_LIST = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD",
    "AUD_USD", "AUD_CAD", "AUD_JPY", "XAU_USD", "GBP_JPY"
]

FOREX_OPEN_UTC = dtime(21, 0)   # Dimanche ouverture approximative FX/OANDA
FOREX_CLOSE_UTC = dtime(21, 0)  # Vendredi fermeture approximative FX/OANDA
LOOP_SECONDS = 60

RISK_PERCENTAGE = float(os.getenv("RISK_PERCENTAGE", "1.0"))
MAX_RISK_USD = float(os.getenv("MAX_RISK_USD", "100"))
RISK_REWARD = float(os.getenv("RISK_REWARD", "2.0"))
MAX_TRADES_TOTAL = int(os.getenv("MAX_TRADES_TOTAL", "3"))
ONE_TRADE_PER_PAIR = True

# =========================================================
# V75 - Gestion active des trades ouverts
# =========================================================
ENABLE_SMART_TRAILING = os.getenv("ENABLE_SMART_TRAILING", "true").lower() == "true"
BREAKEVEN_TRIGGER_R = float(os.getenv("BREAKEVEN_TRIGGER_R", "1.0"))
TRAILING_START_R = float(os.getenv("TRAILING_START_R", "1.5"))
BREAKEVEN_OFFSET_PIPS = float(os.getenv("BREAKEVEN_OFFSET_PIPS", "1.0"))
TRAIL_SWING_LOOKBACK = int(os.getenv("TRAIL_SWING_LOOKBACK", "30"))
TRAIL_BUFFER_PIPS = float(os.getenv("TRAIL_BUFFER_PIPS", "2.0"))
MIN_SL_UPDATE_PIPS = float(os.getenv("MIN_SL_UPDATE_PIPS", "1.0"))


# Risk guard production
MAX_DAILY_RISK_PERCENT = float(os.getenv("MAX_DAILY_RISK_PERCENT", "3.0"))
MAX_WEEKLY_LOSS_PERCENT = float(os.getenv("MAX_WEEKLY_LOSS_PERCENT", "6.0"))

# Variables mémoire: réinitialisées au redémarrage.
# Elles servent à couper le bot si la perte réalisée journalière / hebdo dépasse le seuil.
RISK_GUARD_DAY = None
RISK_GUARD_WEEK = None
RISK_GUARD_DAILY_START_BALANCE = None
RISK_GUARD_WEEKLY_START_BALANCE = None

EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14
RSI_PERIOD = 14
ADX_PERIOD = 14

# Seuil de score : score minimum sur 15 pour prendre un trade
MIN_SEQUENCE_SCORE = int(os.getenv("MIN_SEQUENCE_SCORE", "13"))  # Score strict pour NESTED/WICK
V74_CORE_FVG_MIN_SCORE = int(os.getenv("V74_CORE_FVG_MIN_SCORE", "10"))  # Reprise logique V63: FVG perfect propre dès 10/15

# Filtres paramètres (resserrés par rapport à V66)
MIN_FVG_ATR_RATIO = 0.28            # Zone FVG doit faire ≥ 40% ATR (était 0.25)
MAX_ZONE_AGE_H4_CANDLES = 10        # Zone max 6 candles H4 = ~24H (était 8)
RETRACE_TOLERANCE_ATR = 0.18       # Tolérance autour zone (était 0.15)
MIN_REJECTION_BODY_RATIO = 0.45    # Corps de la bougie de rejet (était 0.45)
MIN_IMPULSE_ATR_MULTIPLIER = 1.15  # Impulsion forte requise (était 1.20)
MIN_ADX = 18.0                      # Filtre trend : ADX > 22
RSI_BUY_MIN = 48.0                  # RSI M15 minimum pour BUY (était 48)
RSI_SELL_MAX = 52.0                 # RSI M15 maximum pour SELL (était 52)
MIN_VOLUME_RATIO = 1.20             # Volume impulsion > 1.2x moyenne

# Sessions haute qualité UTC
LONDON_OPEN_START = dtime(7, 0)
LONDON_OPEN_END = dtime(10, 30)
NY_OPEN_START = dtime(13, 0)
NY_OPEN_END = dtime(17, 0)

# TTL des signaux : on ne retrade pas la même zone avant 8H
SIGNAL_KEY_TTL_HOURS = 8

MAX_SPREAD_PIPS = {
    "XAU_USD": 90.0,   # OANDA exprime ici le spread XAU en pips de 0.01: 73p = 0.73$
    "EUR_USD": 2.2,
    "GBP_USD": 2.5,
    "USD_CAD": 2.5,
    "GBP_JPY": 4.0,
    "AUD_JPY": 3.5,
    "AUD_CAD": 3.0,
    "USD_JPY": 2.2,
    "DEFAULT": 2.2,
}

PAIR_DECIMALS = {
    "XAU_USD": 2,
    "USD_JPY": 3,
    "GBP_JPY": 3,
    "AUD_JPY": 3,
    "AUD_CAD": 5,
    "DEFAULT": 5,
}

PIP_VALUE = {
    "XAU_USD": 0.01,
    "USD_JPY": 0.01,
    "GBP_JPY": 0.01,
    "AUD_JPY": 0.01,
    "AUD_CAD": 0.0001,
    "DEFAULT": 0.0001,
}

# Taille minimale / pas d'arrondi des unités.
# Pour respecter strictement le risque, on ARRONDIT TOUJOURS vers le bas.
UNIT_STEP = {
    "XAU_USD": 1,
    "DEFAULT": 1000,
}

MIN_TRADE_UNITS = {
    "XAU_USD": 1,
    "DEFAULT": 1000,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("oanda_v75_scoring_sniper.log")],
)
logger = logging.getLogger("OANDA-V75-Scoring-Sniper")

# Dict signal_key -> timestamp pour expiration TTL (fix: était un set() infini)
last_signal_key: Dict[tuple, datetime] = {}


def is_market_open_utc(now_dt: datetime) -> bool:
    """
    True quand le marché Forex/OANDA est censé être ouvert.
    - Lundi -> jeudi : ouvert toute la journée
    - Vendredi : ouvert jusqu'à 21:00 UTC
    - Samedi : fermé
    - Dimanche : fermé avant 21:00 UTC, ouvert après
    
    Important : cela respecte les week-ends tout en analysant dès l'ouverture du marché,
    sans limiter le bot à Londres/NY.
    """
    weekday = now_dt.weekday()  # lundi=0 ... dimanche=6
    current_time = now_dt.time()

    if weekday == 5:  # samedi
        return False

    if weekday == 6 and current_time < FOREX_OPEN_UTC:  # dimanche avant ouverture
        return False

    if weekday == 4 and current_time >= FOREX_CLOSE_UTC:  # vendredi après fermeture
        return False

    return True


def market_status_text(now_dt: datetime) -> str:
    if is_market_open_utc(now_dt):
        return "Marché Forex : OUVERT"
    weekday = now_dt.weekday()
    if weekday == 5:
        return "Marché Forex : FERMÉ samedi"
    if weekday == 6:
        return f"Marché Forex : FERMÉ dimanche avant {FOREX_OPEN_UTC.strftime('%H:%M')} UTC"
    if weekday == 4:
        return f"Marché Forex : FERMÉ vendredi après {FOREX_CLOSE_UTC.strftime('%H:%M')} UTC"
    return "Marché Forex : FERMÉ"
# V72: cache mémoire uniquement. Il repart vide à chaque vrai redémarrage du process.


@dataclass
class FVGZone:
    direction: str
    low: float
    high: float
    midpoint: float
    age: int
    size: float
    impulse_strength: float
    volume_confirmed: bool = False
    setup_type: str = "FVG_RETEST_PERFECT"


# =========================================================
# UTILITAIRES
# =========================================================
def pip_value(pair: str) -> float:
    return PIP_VALUE.get(pair, PIP_VALUE["DEFAULT"])


def decimals(pair: str) -> int:
    return PAIR_DECIMALS.get(pair, PAIR_DECIMALS["DEFAULT"])


def round_price(pair: str, price: float) -> str:
    return f"{price:.{decimals(pair)}f}"


def price_to_pips(pair: str, distance: float) -> float:
    return abs(distance) / pip_value(pair)


def compact_json(obj, max_len: int = 6000) -> str:
    """Log JSON compact pour diagnostiquer les réponses OANDA sans exploser les logs Render."""
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        text = str(obj)
    return text if len(text) <= max_len else text[:max_len] + " ...[TRONQUÉ]"


def get_account_summary() -> Dict:
    r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
    resp = client.request(r)
    logger.info(
        f"DEBUG OANDA SUMMARY | account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT} "
        f"balance={resp.get('account', {}).get('balance')} "
        f"openTradeCount={resp.get('account', {}).get('openTradeCount')} "
        f"openPositionCount={resp.get('account', {}).get('openPositionCount')}"
    )
    return resp


def get_account_balance() -> float:
    resp = get_account_summary()
    return float(resp["account"]["balance"])



def get_mid_price(pair: str) -> Optional[float]:
    """Récupère un prix mid récent pour convertir les devises de cotation en USD."""
    prices = fetch_pricing_spread(pair)
    if prices:
        return float(prices[2])
    try:
        df = fetch_candles(pair, "M1", 2)
        if not df.empty:
            return float(df["close"].iloc[-1])
    except Exception:
        pass
    return None


def quote_currency(pair: str) -> str:
    """EUR_USD -> USD, USD_JPY -> JPY, XAU_USD -> USD."""
    return pair.split("_")[-1]


def quote_to_usd_factor(pair: str, entry_price: float) -> Optional[float]:
    """
    Convertit 1 unité de devise de cotation en USD.
    Exemple:
    - EUR_USD: quote USD -> 1
    - USD_JPY: 1 JPY -> 1 / USD_JPY USD
    - USD_CAD/AUD_CAD: 1 CAD -> 1 / USD_CAD USD
    - GBP_JPY/AUD_JPY: 1 JPY -> 1 / USD_JPY USD
    """
    quote = quote_currency(pair)

    if quote == "USD":
        return 1.0

    # Si la paire est USD_XXX, le prix d'entrée permet déjà de convertir XXX -> USD.
    # Exemple USD_JPY=161 => 1 JPY = 1/161 USD.
    if pair.startswith("USD_"):
        if entry_price <= 0:
            return None
        return 1.0 / entry_price

    # Cas croisés : AUD_JPY, GBP_JPY => utiliser USD_JPY.
    usd_quote_pair = f"USD_{quote}"
    mid = get_mid_price(usd_quote_pair)
    if mid and mid > 0:
        return 1.0 / mid

    # Cas éventuel EUR_GBP -> GBP_USD si disponible.
    quote_usd_pair = f"{quote}_USD"
    mid = get_mid_price(quote_usd_pair)
    if mid and mid > 0:
        return mid

    logger.error(f"Impossible de convertir la devise de cotation {quote} en USD pour {pair}.")
    return None


def get_risk_guard_state(balance: Optional[float] = None) -> Dict[str, float]:
    """Initialise/réinitialise les seuils daily/weekly et retourne l'état courant."""
    global RISK_GUARD_DAY, RISK_GUARD_WEEK, RISK_GUARD_DAILY_START_BALANCE, RISK_GUARD_WEEKLY_START_BALANCE

    now = datetime.utcnow()
    today = now.date()
    week = now.isocalendar()[:2]  # (year, week)
    if balance is None:
        balance = get_account_balance()

    if RISK_GUARD_DAY != today or RISK_GUARD_DAILY_START_BALANCE is None:
        RISK_GUARD_DAY = today
        RISK_GUARD_DAILY_START_BALANCE = balance
        logger.info(f"RISK GUARD daily reset | start_balance={balance:.2f}")

    if RISK_GUARD_WEEK != week or RISK_GUARD_WEEKLY_START_BALANCE is None:
        RISK_GUARD_WEEK = week
        RISK_GUARD_WEEKLY_START_BALANCE = balance
        logger.info(f"RISK GUARD weekly reset | start_balance={balance:.2f}")

    daily_dd_pct = 0.0
    weekly_dd_pct = 0.0
    if RISK_GUARD_DAILY_START_BALANCE > 0:
        daily_dd_pct = max(0.0, (RISK_GUARD_DAILY_START_BALANCE - balance) / RISK_GUARD_DAILY_START_BALANCE * 100)
    if RISK_GUARD_WEEKLY_START_BALANCE > 0:
        weekly_dd_pct = max(0.0, (RISK_GUARD_WEEKLY_START_BALANCE - balance) / RISK_GUARD_WEEKLY_START_BALANCE * 100)

    return {
        "balance": balance,
        "daily_start": RISK_GUARD_DAILY_START_BALANCE,
        "weekly_start": RISK_GUARD_WEEKLY_START_BALANCE,
        "daily_dd_pct": daily_dd_pct,
        "weekly_dd_pct": weekly_dd_pct,
    }


def risk_guard_allows_new_trade(balance: Optional[float] = None) -> bool:
    """Bloque l'ouverture si la perte réalisée journalière/hebdo dépasse le seuil configuré."""
    state = get_risk_guard_state(balance)
    logger.info(
        f"RISK GUARD | balance={state['balance']:.2f} | daily_dd={state['daily_dd_pct']:.2f}%/{MAX_DAILY_RISK_PERCENT:.2f}% | "
        f"weekly_dd={state['weekly_dd_pct']:.2f}%/{MAX_WEEKLY_LOSS_PERCENT:.2f}%"
    )
    if state["daily_dd_pct"] >= MAX_DAILY_RISK_PERCENT:
        logger.warning("RISK GUARD: limite de perte journalière atteinte, nouveau trade bloqué.")
        return False
    if state["weekly_dd_pct"] >= MAX_WEEKLY_LOSS_PERCENT:
        logger.warning("RISK GUARD: limite de perte hebdomadaire atteinte, nouveau trade bloqué.")
        return False
    return True

def get_open_trades(log_raw: bool = False) -> List[Dict]:
    """Lecture officielle OpenTrades + logs de diagnostic compte/env."""
    r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
    resp = client.request(r)
    open_trades = resp.get("trades", [])
    logger.info(
        f"DEBUG OPEN TRADES | account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT} count={len(open_trades)}"
    )
    if log_raw:
        logger.info(f"DEBUG OPEN TRADES RAW={compact_json(resp)}")
    for t in open_trades:
        logger.info(
            f"DEBUG TRADE | id={t.get('id')} instrument={t.get('instrument')} "
            f"units={t.get('currentUnits')} price={t.get('price')} "
            f"state={t.get('state', 'OPEN')} openTime={t.get('openTime')}"
        )
    return open_trades


def get_open_positions(log_raw: bool = False) -> List[Dict]:
    """Lecture des positions ouvertes OANDA pour vérifier l'état réel du compte."""
    try:
        r = positions.OpenPositions(accountID=OANDA_ACCOUNT_ID)
        resp = client.request(r)
        open_positions = resp.get("positions", [])
        logger.info(
            f"DEBUG OPEN POSITIONS | account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT} count={len(open_positions)}"
        )
        if log_raw:
            logger.info(f"DEBUG OPEN POSITIONS RAW={compact_json(resp)}")
        for p in open_positions:
            long_units = p.get("long", {}).get("units")
            short_units = p.get("short", {}).get("units")
            logger.info(
                f"DEBUG POSITION | instrument={p.get('instrument')} long_units={long_units} short_units={short_units} "
                f"pl={p.get('pl')} resettablePL={p.get('resettablePL')}"
            )
        return open_positions
    except Exception as exc:
        logger.exception(f"DEBUG OPEN POSITIONS impossible: {exc}")
        return []


def log_account_snapshot(label: str) -> None:
    """Snapshot complet compte/trades/positions pour éviter toute zone d'ombre avant exécution."""
    logger.info(f"========== OANDA ACCOUNT SNAPSHOT {label} ==========")
    try:
        summary = get_account_summary()
        acc = summary.get("account", {})
        logger.info(
            f"SNAPSHOT SUMMARY | id={acc.get('id')} balance={acc.get('balance')} NAV={acc.get('NAV')} "
            f"marginUsed={acc.get('marginUsed')} marginAvailable={acc.get('marginAvailable')} "
            f"openTradeCount={acc.get('openTradeCount')} openPositionCount={acc.get('openPositionCount')} "
            f"lastTransactionID={summary.get('lastTransactionID') or acc.get('lastTransactionID')}"
        )
    except Exception as exc:
        logger.exception(f"SNAPSHOT SUMMARY impossible: {exc}")
    get_open_trades(log_raw=True)
    get_open_positions(log_raw=True)
    logger.info(f"========== FIN SNAPSHOT {label} ==========")


def has_open_trade(pair: str) -> bool:
    """V74: vérifie les trades ouverts ET les positions ouvertes, avec preuve dans les logs."""
    open_trades = get_open_trades(log_raw=True)
    trade_exists = any(
        t.get("instrument") == pair and abs(float(t.get("currentUnits", 0))) > 0
        for t in open_trades
    )

    open_positions = get_open_positions(log_raw=True)
    position_exists = False
    for p in open_positions:
        if p.get("instrument") != pair:
            continue
        long_units = float(p.get("long", {}).get("units", 0) or 0)
        short_units = float(p.get("short", {}).get("units", 0) or 0)
        if abs(long_units) > 0 or abs(short_units) > 0:
            position_exists = True
            break

    exists = trade_exists or position_exists
    logger.info(
        f"{pair}: has_open_trade={exists} | trade_exists={trade_exists} | position_exists={position_exists}"
    )
    return exists


def open_trade_count() -> int:
    return len(get_open_trades(log_raw=True))


def extract_oanda_trade_or_order_id(resp: Dict) -> Optional[str]:
    """Retourne l'ID du trade ouvert si possible, sinon l'ID de transaction d'ordre."""
    fill = resp.get("orderFillTransaction", {}) or {}
    trade_opened = fill.get("tradeOpened") or {}
    if trade_opened.get("tradeID"):
        return str(trade_opened["tradeID"])
    trades_opened = fill.get("tradesOpened") or []
    if trades_opened and trades_opened[0].get("tradeID"):
        return str(trades_opened[0]["tradeID"])
    return str(fill.get("id") or resp.get("orderCreateTransaction", {}).get("id") or "") or None


def log_oanda_order_response(pair: str, resp: Dict) -> None:
    """Logs inspirés des samples officiels OANDA: create/fill/cancel/reject."""
    logger.info(f"DEBUG ORDER RESPONSE RAW {pair}={compact_json(resp)}")
    for key in [
        "orderCreateTransaction",
        "orderFillTransaction",
        "orderCancelTransaction",
        "orderRejectTransaction",
    ]:
        tx = resp.get(key)
        if not tx:
            continue
        logger.info(
            f"OANDA {key} | id={tx.get('id')} type={tx.get('type')} "
            f"instrument={tx.get('instrument')} units={tx.get('units')} "
            f"reason={tx.get('reason')} rejectReason={tx.get('rejectReason')}"
        )
    if resp.get("orderRejectTransaction"):
        logger.error(f"ORDRE REJETÉ {pair} | {compact_json(resp.get('orderRejectTransaction'))}")
    if resp.get("orderCancelTransaction"):
        logger.warning(f"ORDRE ANNULÉ {pair} | {compact_json(resp.get('orderCancelTransaction'))}")



def log_recent_oanda_transactions(label: str = "") -> None:
    """Log les dernières transactions OANDA pour diagnostiquer si un ordre est créé/rejeté/rempli."""
    try:
        # AccountChanges depuis le dernier ID connu n'est pas fiable sans état persistant ici.
        # On logge donc le dernierTransactionID du summary et les détails récents si possible.
        summary = get_account_summary()
        last_tx_id = summary.get("lastTransactionID") or summary.get("account", {}).get("lastTransactionID")
        logger.info(f"DEBUG OANDA LAST TRANSACTION {label} | lastTransactionID={last_tx_id}")
        if not last_tx_id:
            return
        start_id = max(1, int(last_tx_id) - 10)
        params = {"from": str(start_id), "to": str(last_tx_id)}
        r = transactions.TransactionIDRange(accountID=OANDA_ACCOUNT_ID, params=params)
        resp = client.request(r)
        logger.info(f"DEBUG OANDA TRANSACTIONS {label} RAW={compact_json(resp, max_len=8000)}")
    except Exception as exc:
        logger.warning(f"DEBUG OANDA TRANSACTIONS impossible {label}: {exc}")

def send_email(subject: str, body: str) -> None:
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = EMAIL_ADDRESS
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, msg.as_string())
        logger.info(f"Email envoyé: {subject}")
    except Exception as exc:
        logger.error(f"Erreur email: {exc}")


def fetch_candles(pair: str, granularity: str, count: int, price: str = "M") -> pd.DataFrame:
    params = {"granularity": granularity, "count": count, "price": price}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    resp = client.request(r)
    rows = []
    for c in resp.get("candles", []):
        if not c.get("complete", False):
            continue
        mid = c.get("mid") or {}
        if not all(k in mid for k in ("o", "h", "l", "c")):
            continue
        rows.append({
            "time": pd.to_datetime(c["time"]),
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": float(c.get("volume", 0)),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.set_index("time", inplace=True)
    return df


def fetch_pricing_spread(pair: str) -> Optional[Tuple[float, float, float]]:
    try:
        params = {"granularity": "M1", "count": 2, "price": "BA"}
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        resp = client.request(r)
        candles = [c for c in resp.get("candles", []) if c.get("complete")]
        if not candles:
            return None
        c = candles[-1]
        bid = float(c["bid"]["c"])
        ask = float(c["ask"]["c"])
        mid = (bid + ask) / 2
        return bid, ask, mid
    except Exception as exc:
        logger.warning(f"Spread indisponible {pair}: {exc}")
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """EMA50/200, ATR, RSI, ADX (+DI, -DI) via lissage de Wilder."""
    df = df.copy()

    df["ema50"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    # ADX avec lissage de Wilder (alpha = 1/période)
    alpha = 1.0 / ADX_PERIOD
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_w = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    df["plus_di"] = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w
    df["minus_di"] = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w
    di_sum = (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
    dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / di_sum
    df["adx"] = dx.ewm(alpha=alpha, adjust=False).mean()

    return df


def session_quality(now: dtime) -> bool:
    """True si on est pendant l'ouverture Londres ou NY."""
    if LONDON_OPEN_START <= now <= LONDON_OPEN_END:
        return True
    if NY_OPEN_START <= now <= NY_OPEN_END:
        return True
    return False


def clean_signal_keys() -> None:
    """Supprime les clés de signal expirées (> TTL heures)."""
    cutoff = datetime.utcnow() - timedelta(hours=SIGNAL_KEY_TTL_HOURS)
    expired = [k for k, ts in last_signal_key.items() if ts < cutoff]
    for k in expired:
        del last_signal_key[k]


# =========================================================
# ANALYSE MULTI-TIMEFRAME
# =========================================================
def daily_bias(df_d1: pd.DataFrame) -> Optional[str]:
    """Biais D1 : EMA50 > EMA200 + prix au-dessus + pente positive."""
    df = add_indicators(df_d1)
    if len(df) < 210:
        return None
    last = df.iloc[-1]
    slope = df["ema50"].iloc[-1] - df["ema50"].iloc[-6]
    if last["close"] > last["ema50"] and last["ema50"] > last["ema200"] and slope > 0:
        return "BUY"
    if last["close"] < last["ema50"] and last["ema50"] < last["ema200"] and slope < 0:
        return "SELL"
    return None


def market_bias(df_h4: pd.DataFrame, df_h1: pd.DataFrame) -> Optional[str]:
    """
    Biais H4+H1 hybride.
    V67 était trop strict avec EMA200, donc on garde un biais sniper mais plus proche V63:
    - prix H4 et H1 du même côté EMA50
    - pente EMA50 H4/H1 cohérente
    - EMA200 donne un bonus ailleurs, mais n'est plus bloquante.
    """
    h4 = add_indicators(df_h4)
    h1 = add_indicators(df_h1)
    if len(h4) < 80 or len(h1) < 80:
        return None

    h4_last = h4.iloc[-1]
    h1_last = h1.iloc[-1]
    h4_slope = h4["ema50"].iloc[-1] - h4["ema50"].iloc[-8]
    h1_slope = h1["ema50"].iloc[-1] - h1["ema50"].iloc[-8]

    buy = (
        h4_last["close"] > h4_last["ema50"] and
        h1_last["close"] > h1_last["ema50"] and
        h4_slope > 0 and h1_slope > 0
    )
    sell = (
        h4_last["close"] < h4_last["ema50"] and
        h1_last["close"] < h1_last["ema50"] and
        h4_slope < 0 and h1_slope < 0
    )
    if buy:
        return "BUY"
    if sell:
        return "SELL"
    return None


def ema200_bonus(df_h4: pd.DataFrame, df_h1: pd.DataFrame, bias: str) -> bool:
    """Bonus si le biais est aussi aligné EMA200, sans en faire un blocage."""
    h4 = add_indicators(df_h4)
    h1 = add_indicators(df_h1)
    if len(h4) < 210 or len(h1) < 210:
        return False
    h4_last = h4.iloc[-1]
    h1_last = h1.iloc[-1]
    if bias == "BUY":
        return bool(h4_last["ema50"] > h4_last["ema200"] and h1_last["close"] > h1_last["ema200"])
    if bias == "SELL":
        return bool(h4_last["ema50"] < h4_last["ema200"] and h1_last["close"] < h1_last["ema200"])
    return False

def h1_candle_confirms(df_h1: pd.DataFrame, bias: str) -> bool:
    """La dernière bougie H1 fermée doit être dans le sens du bias."""
    if len(df_h1) < 3:
        return False
    last = df_h1.iloc[-1]
    if bias == "BUY":
        return bool(last["close"] > last["open"])
    if bias == "SELL":
        return bool(last["close"] < last["open"])
    return False


def detect_impulse_and_fvg(df_h4: pd.DataFrame, bias: str) -> Optional[FVGZone]:
    """
    Détecte la meilleure FVG récente dans le sens du bias.
    Priorité : impulsion forte > taille zone > récence.
    Fix : tri corrigé (était trié par âge croissant en premier).
    """
    df = add_indicators(df_h4)
    if len(df) < 30:
        return None

    vol_avg20 = df["volume"].rolling(20).mean()
    candidates: List[FVGZone] = []

    start = max(2, len(df) - MAX_ZONE_AGE_H4_CANDLES - 2)
    for i in range(start, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        nxt = df.iloc[i + 1]
        atr = float(df["atr"].iloc[i])
        if np.isnan(atr) or atr <= 0:
            continue

        body = abs(curr["close"] - curr["open"])
        impulse_strength = body / atr
        if impulse_strength < MIN_IMPULSE_ATR_MULTIPLIER:
            continue

        vol_avg = float(vol_avg20.iloc[i])
        vol_confirmed = (vol_avg > 0 and curr["volume"] >= vol_avg * MIN_VOLUME_RATIO)

        if bias == "BUY" and nxt["low"] > prev["high"]:
            low, high = prev["high"], nxt["low"]
            size = high - low
            if size / atr < MIN_FVG_ATR_RATIO:
                continue
            age = len(df) - 1 - i
            candidates.append(FVGZone("BUY", low, high, (low + high) / 2, age, size, impulse_strength, vol_confirmed))

        if bias == "SELL" and nxt["high"] < prev["low"]:
            low, high = nxt["high"], prev["low"]
            size = high - low
            if size / atr < MIN_FVG_ATR_RATIO:
                continue
            age = len(df) - 1 - i
            candidates.append(FVGZone("SELL", low, high, (low + high) / 2, age, size, impulse_strength, vol_confirmed))

    if not candidates:
        return None

    # Fix : priorité impulsion forte > taille > récence (était age en premier = INVERSÉ)
    candidates.sort(key=lambda z: (-z.impulse_strength, -z.size, z.age))
    return candidates[0]


def price_retraced_to_zone(pair: str, df_m15: pd.DataFrame, zone: FVGZone) -> bool:
    """Prix touche la zone FVG (avec tolérance pair-specific - fix hardcoding EUR_USD)."""
    df = add_indicators(df_m15)
    atr = float(df["atr"].iloc[-1])
    if np.isnan(atr):
        atr = 0.0
    pv = pip_value(pair)  # Fix : était hardcodé sur EUR_USD
    tolerance = max(atr * RETRACE_TOLERANCE_ATR, pv)
    last = df.iloc[-1]
    touched = last["low"] <= zone.high + tolerance and last["high"] >= zone.low - tolerance
    return bool(touched)


def rejection_confirmed(pair: str, df_m15: pd.DataFrame, zone: FVGZone) -> Tuple[bool, str]:
    """
    Confirmation du rejet M15 :
    - Direction bougie
    - Corps fort (≥ 58% de la mèche totale)
    - Mèche de rejet claire
    - Cassure structure locale
    - RSI calibré (> 52 BUY / < 48 SELL)
    - ADX > 22 (tendance présente)
    - Confirmation multi-bougies (au moins 1 des 2 précédentes dans sens du bias)
    """
    df = add_indicators(df_m15)
    if len(df) < 10:
        return False, "données M15 insuffisantes"

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    total = max(last["high"] - last["low"], 1e-9)
    body = abs(last["close"] - last["open"])
    body_ratio = body / total
    upper_wick = last["high"] - max(last["close"], last["open"])
    lower_wick = min(last["close"], last["open"]) - last["low"]
    rsi = float(df["rsi"].iloc[-1])
    adx_val = float(df["adx"].iloc[-1]) if not np.isnan(df["adx"].iloc[-1]) else 0.0

    if zone.direction == "BUY":
        if last["close"] <= last["open"]:
            return False, "bougie M15 non haussière"
        if body_ratio < MIN_REJECTION_BODY_RATIO:
            return False, f"corps faible {body_ratio:.2f} < {MIN_REJECTION_BODY_RATIO}"
        if lower_wick < upper_wick * 0.8:
            return False, "rejet bas insuffisant (mèche courte)"
        if last["close"] <= prev["high"]:
            return False, "pas de cassure structure M15"
        if rsi < RSI_BUY_MIN:
            return False, f"RSI M15 faible {rsi:.1f} < {RSI_BUY_MIN}"
        if adx_val < MIN_ADX:
            return False, f"ADX M15 faible {adx_val:.1f} < {MIN_ADX}"
        prev_bullish = (prev["close"] > prev["open"]) or (prev2["close"] > prev2["open"])
        if not prev_bullish:
            return False, "pas de momentum haussier M15 récent"
        return True, f"rejet BUY | RSI={rsi:.1f} ADX={adx_val:.1f} corps={body_ratio:.2f}"

    if zone.direction == "SELL":
        if last["close"] >= last["open"]:
            return False, "bougie M15 non baissière"
        if body_ratio < MIN_REJECTION_BODY_RATIO:
            return False, f"corps faible {body_ratio:.2f} < {MIN_REJECTION_BODY_RATIO}"
        if upper_wick < lower_wick * 0.8:
            return False, "rejet haut insuffisant (mèche courte)"
        if last["close"] >= prev["low"]:
            return False, "pas de cassure structure M15"
        if rsi > RSI_SELL_MAX:
            return False, f"RSI M15 élevé {rsi:.1f} > {RSI_SELL_MAX}"
        if adx_val < MIN_ADX:
            return False, f"ADX M15 faible {adx_val:.1f} < {MIN_ADX}"
        prev_bearish = (prev["close"] < prev["open"]) or (prev2["close"] < prev2["open"])
        if not prev_bearish:
            return False, "pas de momentum baissier M15 récent"
        return True, f"rejet SELL | RSI={rsi:.1f} ADX={adx_val:.1f} corps={body_ratio:.2f}"

    return False, "direction inconnue"



def detect_fvg_candidates(df: pd.DataFrame, bias: str, pair: str, timeframe_label: str, max_age: int, min_atr_ratio: float, min_impulse: float) -> List[FVGZone]:
    """Détecte plusieurs FVG récents, moins strict que V67, pour retrouver la richesse V63."""
    df = add_indicators(df)
    if len(df) < 30:
        return []
    candidates: List[FVGZone] = []
    vol_avg20 = df["volume"].rolling(20).mean()
    start = max(2, len(df) - max_age - 2)
    for i in range(start, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        nxt = df.iloc[i + 1]
        atr = float(df["atr"].iloc[i])
        if np.isnan(atr) or atr <= 0:
            continue
        body = abs(curr["close"] - curr["open"])
        impulse_strength = body / atr
        if impulse_strength < min_impulse:
            continue
        vol_avg = float(vol_avg20.iloc[i]) if not np.isnan(vol_avg20.iloc[i]) else 0
        vol_confirmed = vol_avg > 0 and curr["volume"] >= vol_avg * MIN_VOLUME_RATIO
        if bias == "BUY" and nxt["low"] > prev["high"]:
            low, high = prev["high"], nxt["low"]
            size = high - low
            if size / atr < min_atr_ratio:
                continue
            age = len(df) - 1 - i
            z = FVGZone("BUY", low, high, (low + high) / 2, age, size, impulse_strength, vol_confirmed, "FVG_RETEST_PERFECT" if timeframe_label == "H4" else "NESTED_FVG")
            candidates.append(z)
        if bias == "SELL" and nxt["high"] < prev["low"]:
            low, high = nxt["high"], prev["low"]
            size = high - low
            if size / atr < min_atr_ratio:
                continue
            age = len(df) - 1 - i
            z = FVGZone("SELL", low, high, (low + high) / 2, age, size, impulse_strength, vol_confirmed, "FVG_RETEST_PERFECT" if timeframe_label == "H4" else "NESTED_FVG")
            candidates.append(z)
    return candidates


def detect_wick_rejection_candidate(df_m15: pd.DataFrame, bias: str, pair: str) -> Optional[FVGZone]:
    """Réactive le Wick Rejection V63, mais uniquement sur la dernière bougie M15 et dans le sens du biais."""
    df = add_indicators(df_m15)
    if len(df) < 30:
        return None
    last = df.iloc[-1]
    atr = float(df["atr"].iloc[-1]) if not np.isnan(df["atr"].iloc[-1]) else 0.0
    total = max(last["high"] - last["low"], 1e-9)
    body = abs(last["close"] - last["open"])
    upper = last["high"] - max(last["close"], last["open"])
    lower = min(last["close"], last["open"]) - last["low"]
    body_ratio = body / total

    if atr <= 0:
        return None

    # SELL : mèche haute claire + clôture rouge ou cassure du précédent low.
    if bias == "SELL":
        if upper >= max(body * 1.3, lower * 1.2) and body_ratio >= 0.25 and last["close"] < df.iloc[-2]["low"]:
            low = max(last["open"], last["close"])
            high = last["high"]
            z = FVGZone("SELL", low, high, (low + high) / 2, 0, high - low, max(body / atr, 1.0), True, "WICK_REJECTION")
            return z

    # BUY : mèche basse claire + clôture verte ou cassure du précédent high.
    if bias == "BUY":
        if lower >= max(body * 1.3, upper * 1.2) and body_ratio >= 0.25 and last["close"] > df.iloc[-2]["high"]:
            low = last["low"]
            high = min(last["open"], last["close"])
            z = FVGZone("BUY", low, high, (low + high) / 2, 0, high - low, max(body / atr, 1.0), True, "WICK_REJECTION")
            return z
    return None


def cluster_zones(zones: List[FVGZone], pair: str) -> List[FVGZone]:
    """Fusionne les zones proches et garde la plus qualitative pour éviter le bruit V63."""
    if not zones:
        return []
    threshold = 3 * pip_value(pair) if pair != "XAU_USD" else 0.50
    zones = sorted(zones, key=lambda z: (-setup_priority(z.setup_type), -z.impulse_strength, z.age))
    kept: List[FVGZone] = []
    for z in zones:
        if any(z.direction == k.direction and abs(z.midpoint - k.midpoint) <= threshold for k in kept):
            continue
        kept.append(z)
    return kept[:4]


def setup_priority(setup_type: str) -> int:
    return {"FVG_RETEST_PERFECT": 3, "NESTED_FVG": 2, "WICK_REJECTION": 2}.get(setup_type, 1)


def collect_hybrid_zones(pair: str, df_h4: pd.DataFrame, df_h1: pd.DataFrame, df_m15: pd.DataFrame, bias: str) -> List[FVGZone]:
    """Moteur hybride: FVG H4 + Nested H1/M15 + Wick Rejection, puis clustering."""
    zones: List[FVGZone] = []
    zones.extend(detect_fvg_candidates(df_h4, bias, pair, "H4", MAX_ZONE_AGE_H4_CANDLES, MIN_FVG_ATR_RATIO, MIN_IMPULSE_ATR_MULTIPLIER))
    zones.extend(detect_fvg_candidates(df_h1, bias, pair, "H1", 18, 0.18, 0.85))
    zones.extend(detect_fvg_candidates(df_m15, bias, pair, "M15", 48, 0.12, 0.75))
    wick = detect_wick_rejection_candidate(df_m15, bias, pair)
    if wick:
        zones.append(wick)
    zones = [z for z in zones if z.direction == bias]
    clustered = cluster_zones(zones, pair)
    logger.info(f"{pair}: zones hybrides détectées={len(zones)} → après clustering={len(clustered)}")
    for z in clustered:
        logger.info(f"{pair}: zone {z.setup_type} {z.direction} [{z.low:.{decimals(pair)}f}-{z.high:.{decimals(pair)}f}] age={z.age} impulse={z.impulse_strength:.2f}x")
    return clustered


def score_zone(base_score: int, zone: FVGZone, df_h4: pd.DataFrame, df_h1: pd.DataFrame, df_m15: pd.DataFrame, bias: str) -> Tuple[int, List[str]]:
    """Score hybrid inspiré V63 mais en gardant les vetos V67."""
    score = base_score
    details: List[str] = []
    if zone.setup_type == "FVG_RETEST_PERFECT":
        score += 3
        details.append("FVG_PERFECT(+3)")
    elif zone.setup_type == "NESTED_FVG":
        score += 2
        details.append("NESTED_FVG(+2)")
    elif zone.setup_type == "WICK_REJECTION":
        score += 2
        details.append("WICK_REJECTION(+2)")

    if zone.impulse_strength >= 1.7:
        score += 2
        details.append("impulse≥1.7(+2)")
    elif zone.impulse_strength >= 1.15:
        score += 1
        details.append("impulse≥1.15(+1)")
    if zone.age <= 2:
        score += 1
        details.append("zone_fraîche(+1)")
    if zone.volume_confirmed:
        score += 1
        details.append("volume(+1)")
    if h1_candle_confirms(df_h1, bias):
        score += 1
        details.append("H1_candle(+1)")
    if session_quality(datetime.utcnow().time()):
        score += 1
        details.append("session(+1)")
    if ema200_bonus(df_h4, df_h1, bias):
        score += 1
        details.append("EMA200_bonus(+1)")
    return score, details

def spread_ok(pair: str) -> Tuple[bool, str, Optional[float]]:
    """Ancienne compatibilité: True seulement si spread <= seuil."""
    status, msg, spread_pips, _ = spread_quality(pair)
    return status != "BLOCK", msg, spread_pips


def spread_quality(pair: str) -> Tuple[str, str, Optional[float], int]:
    """
    V72: le spread n'est plus un veto sauf extrême.
    - OK: +1
    - HIGH: -1, mais le setup peut survivre si score A+
    - BLOCK: spread indisponible ou > 2x seuil
    """
    prices = fetch_pricing_spread(pair)
    if not prices:
        return "BLOCK", "spread indisponible - trade bloqué par sécurité", None, -99
    bid, ask, _ = prices
    spread_pips = price_to_pips(pair, ask - bid)
    max_spread = MAX_SPREAD_PIPS.get(pair, MAX_SPREAD_PIPS["DEFAULT"])
    if spread_pips > max_spread * 2.0:
        return "BLOCK", f"spread extrême {spread_pips:.1f}p > {max_spread*2.0:.1f}p", spread_pips, -99
    if spread_pips > max_spread:
        return "HIGH", f"spread élevé mais toléré {spread_pips:.1f}p > {max_spread:.1f}p (-1)", spread_pips, -1
    return "OK", f"spread OK {spread_pips:.1f}p (+1)", spread_pips, 1


def m15_rejection_score(pair: str, df_m15: pd.DataFrame, zone: FVGZone) -> Tuple[int, str]:
    """
    V72: transforme le rejet M15 en score au lieu de hard veto.
    Cela évite de tuer un bon setup juste parce que le corps est à 0.30 au lieu de 0.45.
    Score typique: -3 à +5.
    """
    df = add_indicators(df_m15)
    if len(df) < 10:
        return -3, "M15 données insuffisantes (-3)"

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    total = max(last["high"] - last["low"], 1e-9)
    body = abs(last["close"] - last["open"])
    body_ratio = body / total
    upper_wick = last["high"] - max(last["close"], last["open"])
    lower_wick = min(last["close"], last["open"]) - last["low"]
    rsi = float(df["rsi"].iloc[-1])
    adx_val = float(df["adx"].iloc[-1]) if not np.isnan(df["adx"].iloc[-1]) else 0.0

    pts = 0
    reasons = []

    if zone.direction == "BUY":
        if last["close"] > last["open"]:
            pts += 1; reasons.append("bougie BUY(+1)")
        else:
            pts -= 2; reasons.append("bougie non BUY(-2)")

        if body_ratio >= MIN_REJECTION_BODY_RATIO:
            pts += 1; reasons.append(f"corps fort {body_ratio:.2f}(+1)")
        elif body_ratio >= 0.25:
            reasons.append(f"corps moyen {body_ratio:.2f}(0)")
        else:
            pts -= 1; reasons.append(f"corps faible {body_ratio:.2f}(-1)")

        if lower_wick >= upper_wick * 0.8:
            pts += 1; reasons.append("mèche basse rejet(+1)")
        if last["close"] > prev["high"]:
            pts += 1; reasons.append("cassure high M15(+1)")
        elif last["close"] > prev["close"]:
            pts += 0; reasons.append("progression M15(0)")
        else:
            pts -= 1; reasons.append("pas de cassure M15(-1)")
        if rsi >= RSI_BUY_MIN:
            pts += 1; reasons.append(f"RSI={rsi:.1f}(+1)")
        else:
            pts -= 1; reasons.append(f"RSI faible {rsi:.1f}(-1)")
        if (prev["close"] > prev["open"]) or (prev2["close"] > prev2["open"]):
            pts += 1; reasons.append("momentum récent(+1)")

    elif zone.direction == "SELL":
        if last["close"] < last["open"]:
            pts += 1; reasons.append("bougie SELL(+1)")
        else:
            pts -= 2; reasons.append("bougie non SELL(-2)")

        if body_ratio >= MIN_REJECTION_BODY_RATIO:
            pts += 1; reasons.append(f"corps fort {body_ratio:.2f}(+1)")
        elif body_ratio >= 0.25:
            reasons.append(f"corps moyen {body_ratio:.2f}(0)")
        else:
            pts -= 1; reasons.append(f"corps faible {body_ratio:.2f}(-1)")

        if upper_wick >= lower_wick * 0.8:
            pts += 1; reasons.append("mèche haute rejet(+1)")
        if last["close"] < prev["low"]:
            pts += 1; reasons.append("cassure low M15(+1)")
        elif last["close"] < prev["close"]:
            pts += 0; reasons.append("progression M15(0)")
        else:
            pts -= 1; reasons.append("pas de cassure M15(-1)")
        if rsi <= RSI_SELL_MAX:
            pts += 1; reasons.append(f"RSI={rsi:.1f}(+1)")
        else:
            pts -= 1; reasons.append(f"RSI élevé {rsi:.1f}(-1)")
        if (prev["close"] < prev["open"]) or (prev2["close"] < prev2["open"]):
            pts += 1; reasons.append("momentum récent(+1)")

    if adx_val >= MIN_ADX:
        pts += 1; reasons.append(f"ADX={adx_val:.1f}(+1)")
    elif adx_val < 12:
        pts -= 1; reasons.append(f"ADX faible {adx_val:.1f}(-1)")
    else:
        reasons.append(f"ADX neutre {adx_val:.1f}(0)")

    return int(pts), " | ".join(reasons)


def calculate_units(pair: str, entry: float, stop: float, balance: float) -> int:
    """
    Calcule la taille de position pour que le risque ne dépasse pas RISK_PERCENTAGE du compte.

    Correction V72:
    - EUR_USD / GBP_USD / XAU_USD / AUD_USD: distance déjà en USD.
    - USD_JPY / GBP_JPY / AUD_JPY: distance en JPY -> conversion via USD_JPY.
    - USD_CAD / AUD_CAD: distance en CAD -> conversion via USD_CAD.
    - Arrondi toujours vers le bas pour ne jamais dépasser le risque.
    """
    risk_usd = min(balance * (RISK_PERCENTAGE / 100), MAX_RISK_USD)
    distance_quote = abs(entry - stop)
    if distance_quote <= 0:
        return 0

    q_to_usd = quote_to_usd_factor(pair, entry)
    if q_to_usd is None or q_to_usd <= 0:
        logger.error(f"{pair}: conversion devise impossible, trade bloqué.")
        return 0

    risk_per_unit_usd = distance_quote * q_to_usd
    if risk_per_unit_usd <= 0:
        return 0

    raw_units = risk_usd / risk_per_unit_usd
    step = UNIT_STEP.get(pair, UNIT_STEP["DEFAULT"])
    min_trade_units = MIN_TRADE_UNITS.get(pair, MIN_TRADE_UNITS["DEFAULT"])

    # Arrondi vers le bas = respect strict du risque. Ne jamais forcer au minimum si ça dépasse le risque.
    units = int(np.floor(raw_units / step) * step)

    if units < min_trade_units:
        estimated_min_risk = min_trade_units * risk_per_unit_usd
        logger.warning(
            f"{pair}: taille calculée trop faible units={units} < min={min_trade_units}. "
            f"Risque min estimé=${estimated_min_risk:.2f}, risque autorisé=${risk_usd:.2f}. Trade bloqué."
        )
        return 0

    estimated_risk = units * risk_per_unit_usd
    estimated_risk_pct = (estimated_risk / balance * 100) if balance > 0 else 0
    logger.info(
        f"RISK LOT {pair} | balance=${balance:.2f} | risk_cap=${risk_usd:.2f} ({RISK_PERCENTAGE:.2f}%) | "
        f"entry={entry:.{decimals(pair)}f} stop={stop:.{decimals(pair)}f} | "
        f"distance={price_to_pips(pair, distance_quote):.1f}p | quote_to_usd={q_to_usd:.8f} | "
        f"risk_per_unit=${risk_per_unit_usd:.8f} | raw_units={raw_units:.2f} | units={units} | "
        f"estimated_risk=${estimated_risk:.2f} ({estimated_risk_pct:.2f}%)"
    )

    # Protection finale: si l'arrondi ou la conversion dépasse de plus de 1%, on bloque.
    if estimated_risk > risk_usd * 1.01:
        logger.error(f"{pair}: risque estimé ${estimated_risk:.2f} > risque autorisé ${risk_usd:.2f}. Trade bloqué.")
        return 0

    return units

def build_trade_levels(pair: str, zone: FVGZone, entry: float, df_m15: pd.DataFrame) -> Tuple[float, float]:
    df = add_indicators(df_m15)
    atr = float(df["atr"].iloc[-1])
    if np.isnan(atr):
        atr = abs(zone.high - zone.low)
    buffer = max(0.35 * atr, 3 * pip_value(pair))

    if zone.direction == "BUY":
        stop = min(zone.low - buffer, df["low"].tail(6).min() - pip_value(pair))
        risk = abs(entry - stop)
        tp = entry + RISK_REWARD * risk
    else:
        stop = max(zone.high + buffer, df["high"].tail(6).max() + pip_value(pair))
        risk = abs(entry - stop)
        tp = entry - RISK_REWARD * risk
    return stop, tp


def place_trade(pair: str, direction: str, entry: float, stop: float, tp: float, score: int, reason: str) -> Optional[str]:
    """
    Couche OANDA V72 inspirée des samples officiels:
    - logge account/env
    - logge OpenTrades RAW si blocage
    - logge toute la réponse OrderCreate
    - distingue Fill / Cancel / Reject
    - retourne uniquement un trade/order id si l'ordre est bien accepté/rempli.
    """
    logger.info(f"DEBUG PLACE_TRADE START | account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT} execute={EXECUTE_TRADES}")
    logger.info(
        f"DEBUG PLACE_TRADE INPUT | pair={pair} direction={direction} entry={entry:.{decimals(pair)}f} "
        f"stop={stop:.{decimals(pair)}f} tp={tp:.{decimals(pair)}f} score={score} reason={reason}"
    )
    log_recent_oanda_transactions("BEFORE_PLACE_TRADE")
    log_account_snapshot("BEFORE_PLACE_TRADE")

    if ONE_TRADE_PER_PAIR and has_open_trade(pair):
        logger.info(f"{pair}: trade déjà ouvert détecté par OANDA, aucun nouvel ordre.")
        log_recent_oanda_transactions("BLOCKED_OPEN_TRADE_EXISTS")
        return None

    total_open = open_trade_count()
    if total_open >= MAX_TRADES_TOTAL:
        logger.info(f"Limite trades ouverts atteinte ({total_open}/{MAX_TRADES_TOTAL}). Aucun ordre POST /orders ne sera envoyé.")
        log_account_snapshot("BLOCKED_MAX_TRADES_TOTAL")
        return None

    balance = get_account_balance()
    if not risk_guard_allows_new_trade(balance):
        return None

    units = calculate_units(pair, entry, stop, balance)
    if units <= 0:
        logger.warning(f"{pair}: taille position invalide.")
        return None

    signed_units = units if direction == "BUY" else -units
    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": pair,
            "units": str(signed_units),
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": round_price(pair, stop), "timeInForce": "GTC"},
            "takeProfitOnFill": {"price": round_price(pair, tp), "timeInForce": "GTC"},
        }
    }

    rr_actual = abs(tp - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0
    logger.info(
        f"SIGNAL V75 SCORING {pair} {direction} | "
        f"entry≈{entry:.{decimals(pair)}f} SL={stop:.{decimals(pair)}f} TP={tp:.{decimals(pair)}f} "
        f"RR={rr_actual:.2f} score={score}/15 units={units} signed_units={signed_units} | {reason}"
    )
    logger.info(f"ORDER PAYLOAD {pair}={compact_json(order_data)}")

    if not EXECUTE_TRADES:
        logger.info("EXECUTE_TRADES=false : ordre non envoyé à OANDA.")
        logger.info("DEBUG EXECUTION RESULT | status=SIMULATION | aucun POST /orders envoyé")
        return "SIMULATION"

    try:
        logger.info(f"DEBUG POST /orders START | pair={pair} account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT}")
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        resp = client.request(r)
        logger.info(f"DEBUG POST /orders END | pair={pair}")
        log_oanda_order_response(pair, resp)
        log_recent_oanda_transactions("AFTER_ORDER_CREATE")

        if resp.get("orderRejectTransaction"):
            logger.error(f"DEBUG EXECUTION RESULT | status=REJECTED | pair={pair}")
            return None
        if resp.get("orderCancelTransaction") and not resp.get("orderFillTransaction"):
            logger.error(f"DEBUG EXECUTION RESULT | status=CANCELLED_NO_FILL | pair={pair}")
            return None

        trade_id = extract_oanda_trade_or_order_id(resp)
        if not trade_id:
            logger.error(f"ORDRE NON CONFIRMÉ {pair}: aucune transaction Fill/Create exploitable.")
            logger.error(f"DEBUG EXECUTION RESULT | status=NO_TRADE_ID | pair={pair}")
            return None

        logger.info(f"ORDRE RÉEL CONFIRMÉ {pair} | ID={trade_id}")
        logger.info(f"DEBUG EXECUTION RESULT | status=CONFIRMED | pair={pair} trade_or_order_id={trade_id}")

        # Vérification juste après envoi: on relit OpenTrades pour éviter les faux positifs.
        log_account_snapshot("AFTER_ORDER_CREATE")
        open_after = get_open_trades(log_raw=True)
        opened_for_pair = [t for t in open_after if t.get("instrument") == pair]
        if opened_for_pair:
            logger.info(f"CONFIRMATION OPEN TRADE {pair}: {compact_json(opened_for_pair)}")
        else:
            logger.warning(f"ATTENTION {pair}: ordre accepté mais OpenTrades ne montre pas encore la position.")

        send_email(
            f"V73 {pair} {direction} score={score}/15",
            (
                f"Paire: {pair}\nDirection: {direction}\n"
                f"Entrée: {entry:.{decimals(pair)}f}\nSL: {stop:.{decimals(pair)}f}\n"
                f"TP: {tp:.{decimals(pair)}f}\nRR: {rr_actual:.2f}\n"
                f"Score: {score}/15\nTrade/Order ID: {trade_id}\nRaison: {reason}"
            ),
        )
        return trade_id

    except Exception as exc:
        logger.exception(f"Erreur ordre OANDA {pair}: {exc}")
        log_recent_oanda_transactions("EXCEPTION_ORDER_CREATE")
        logger.error(f"DEBUG EXECUTION RESULT | status=EXCEPTION | pair={pair} error={exc}")
        return None



def is_v74_core_fvg_trade(zone: FVGZone, score: int, rr: float, reject_pts: int) -> Tuple[bool, str]:
    """
    Décision V74 inspirée V63.
    On garde l'exécution/risk V73, mais on évite de refuser un FVG PERFECT propre
    juste parce que le score global reste à 10-12/15.

    Conditions obligatoires:
    - setup FVG_RETEST_PERFECT uniquement
    - prix déjà dans la zone (déjà vérifié avant l'appel)
    - RR >= 1.8
    - score >= V74_CORE_FVG_MIN_SCORE
    - M15 pas catastrophique (reject_pts > -3)

    Les NESTED_FVG et WICK_REJECTION restent soumis au score strict MIN_SEQUENCE_SCORE.
    """
    if zone.setup_type != "FVG_RETEST_PERFECT":
        return False, "pas core FVG"
    if rr < 1.8:
        return False, f"RR trop faible {rr:.2f}"
    if score < V74_CORE_FVG_MIN_SCORE:
        return False, f"score core insuffisant {score}/{V74_CORE_FVG_MIN_SCORE}"
    if reject_pts <= -3:
        return False, f"M15 trop contraire reject_pts={reject_pts}"
    return True, f"V74_CORE_FVG_ACCEPT score={score}/15 min_core={V74_CORE_FVG_MIN_SCORE} RR={rr:.2f} reject_pts={reject_pts}"


# =========================================================
# V75 - BREAK-EVEN + TRAILING STRUCTUREL M5
# =========================================================
def trade_direction_from_units(units: float) -> Optional[str]:
    if units > 0:
        return "BUY"
    if units < 0:
        return "SELL"
    return None


def get_current_mid_from_m1(pair: str) -> Optional[float]:
    price = get_mid_price(pair)
    if price is not None:
        return float(price)
    try:
        df = fetch_candles(pair, "M1", 3)
        if not df.empty:
            return float(df["close"].iloc[-1])
    except Exception:
        pass
    return None


def current_trade_sl_tp(trade: Dict) -> Tuple[Optional[float], Optional[float]]:
    sl_order = trade.get("stopLossOrder") or {}
    tp_order = trade.get("takeProfitOrder") or {}
    sl = float(sl_order["price"]) if sl_order.get("price") else None
    tp = float(tp_order["price"]) if tp_order.get("price") else None
    return sl, tp


def infer_initial_risk_from_trade(pair: str, trade: Dict, direction: str, entry: float, sl: Optional[float], tp: Optional[float]) -> Optional[float]:
    """
    On infère le risque initial depuis le TP car V74/V75 ouvre à RR=RISK_REWARD.
    C'est plus fiable qu'utiliser le SL courant, qui peut déjà avoir été déplacé au break-even.
    """
    if tp is not None and RISK_REWARD > 0:
        r = abs(tp - entry) / RISK_REWARD
        if r > 0:
            return r
    if sl is not None:
        r = abs(entry - sl)
        if r > 0:
            return r
    return None


def trade_profit_r(direction: str, entry: float, current_price: float, initial_risk: float) -> float:
    if initial_risk <= 0:
        return 0.0
    if direction == "BUY":
        return (current_price - entry) / initial_risk
    return (entry - current_price) / initial_risk


def find_last_m5_swing(pair: str, direction: str, lookback: int = TRAIL_SWING_LOOKBACK) -> Optional[float]:
    """Retourne le dernier swing M5 utile: swing low pour BUY, swing high pour SELL."""
    try:
        df = fetch_candles(pair, "M5", max(lookback + 10, 40))
        if df.empty or len(df) < 8:
            return None
        df = df.tail(lookback).copy()
        lows = df["low"].values
        highs = df["high"].values
        if direction == "BUY":
            for i in range(len(df) - 3, 1, -1):
                if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                    return float(lows[i])
        else:
            for i in range(len(df) - 3, 1, -1):
                if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                    return float(highs[i])
    except Exception as exc:
        logger.warning(f"{pair}: impossible de calculer swing M5 trailing: {exc}")
    return None


def update_trade_stop_loss(trade_id: str, pair: str, new_sl: float, reason: str) -> bool:
    """Remplace uniquement le Stop Loss du trade via l'endpoint TradeCRCDO."""
    if not EXECUTE_TRADES:
        logger.info(f"V75 TRAILING SIMULATION {pair} trade={trade_id} new_sl={round_price(pair, new_sl)} reason={reason}")
        return True

    data = {
        "stopLoss": {
            "price": round_price(pair, new_sl),
            "timeInForce": "GTC",
        }
    }
    try:
        logger.info(f"V75 SL UPDATE PAYLOAD {pair} trade={trade_id} {compact_json(data)}")
        r = trades.TradeCRCDO(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id, data=data)
        resp = client.request(r)
        logger.info(f"V75 SL UPDATE RESPONSE {pair} trade={trade_id} {compact_json(resp)}")
        if resp.get("stopLossOrderRejectTransaction") or resp.get("orderRejectTransaction"):
            logger.error(f"V75 SL UPDATE REJECTED {pair} trade={trade_id} {compact_json(resp)}")
            return False
        logger.info(f"V75 SL UPDATED {pair} trade={trade_id} SL={round_price(pair, new_sl)} | {reason}")
        return True
    except Exception as exc:
        logger.exception(f"V75 SL UPDATE EXCEPTION {pair} trade={trade_id}: {exc}")
        return False


def maybe_update_sl_for_trade(trade: Dict) -> None:
    pair = trade.get("instrument")
    trade_id = str(trade.get("id"))
    if not pair or not trade_id:
        return

    units = float(trade.get("currentUnits", 0) or 0)
    direction = trade_direction_from_units(units)
    if direction is None:
        return

    entry = float(trade.get("price") or 0)
    if entry <= 0:
        return

    current_price = get_current_mid_from_m1(pair)
    if current_price is None:
        logger.warning(f"{pair}: prix actuel indisponible pour trailing V75")
        return

    current_sl, current_tp = current_trade_sl_tp(trade)
    initial_risk = infer_initial_risk_from_trade(pair, trade, direction, entry, current_sl, current_tp)
    if initial_risk is None or initial_risk <= 0:
        logger.info(f"{pair}: risque initial introuvable, trailing ignoré trade={trade_id}")
        return

    profit_r = trade_profit_r(direction, entry, current_price, initial_risk)
    pv = pip_value(pair)
    min_update = MIN_SL_UPDATE_PIPS * pv
    be_offset = BREAKEVEN_OFFSET_PIPS * pv
    trail_buffer = TRAIL_BUFFER_PIPS * pv

    logger.info(
        f"V75 TRADE MANAGER {pair} trade={trade_id} dir={direction} entry={entry:.{decimals(pair)}f} "
        f"price={current_price:.{decimals(pair)}f} SL={current_sl} TP={current_tp} R={profit_r:.2f}"
    )

    candidates: List[Tuple[float, str]] = []

    # 1) Break-even intelligent à +1R.
    if profit_r >= BREAKEVEN_TRIGGER_R:
        be_sl = entry + be_offset if direction == "BUY" else entry - be_offset
        candidates.append((be_sl, f"break-even +{BREAKEVEN_OFFSET_PIPS:.1f}p à {profit_r:.2f}R"))

    # 2) Trailing structurel seulement après +1.5R.
    if profit_r >= TRAILING_START_R:
        swing = find_last_m5_swing(pair, direction)
        if swing is not None:
            if direction == "BUY":
                swing_sl = swing - trail_buffer
                # Ne jamais repasser sous BE une fois à +1.5R.
                swing_sl = max(swing_sl, entry + be_offset)
            else:
                swing_sl = swing + trail_buffer
                swing_sl = min(swing_sl, entry - be_offset)
            candidates.append((swing_sl, f"trailing swing M5 à {profit_r:.2f}R swing={swing:.{decimals(pair)}f}"))

    if not candidates:
        return

    # Sélection du SL le plus protecteur.
    if direction == "BUY":
        new_sl, reason = max(candidates, key=lambda x: x[0])
        if current_sl is not None and new_sl <= current_sl + min_update:
            logger.info(f"{pair}: SL inchangé, amélioration insuffisante BUY new={new_sl:.{decimals(pair)}f} current={current_sl:.{decimals(pair)}f}")
            return
        # sécurité: SL doit rester sous le prix actuel
        if new_sl >= current_price - pv:
            logger.info(f"{pair}: nouveau SL BUY trop proche/au-dessus du prix, ignoré new={new_sl} price={current_price}")
            return
    else:
        new_sl, reason = min(candidates, key=lambda x: x[0])
        if current_sl is not None and new_sl >= current_sl - min_update:
            logger.info(f"{pair}: SL inchangé, amélioration insuffisante SELL new={new_sl:.{decimals(pair)}f} current={current_sl:.{decimals(pair)}f}")
            return
        if new_sl <= current_price + pv:
            logger.info(f"{pair}: nouveau SL SELL trop proche/sous le prix, ignoré new={new_sl} price={current_price}")
            return

    update_trade_stop_loss(trade_id, pair, new_sl, reason)


def manage_open_trades_v75() -> None:
    """Passe sur tous les trades ouverts et applique BE/trailing structurel."""
    if not ENABLE_SMART_TRAILING:
        return
    try:
        open_trades = get_open_trades(log_raw=False)
        if not open_trades:
            logger.info("V75 TRADE MANAGER: aucun trade ouvert.")
            return
        logger.info(f"V75 TRADE MANAGER: gestion de {len(open_trades)} trade(s) ouvert(s).")
        for trade in open_trades:
            maybe_update_sl_for_trade(trade)
            time.sleep(0.15)
    except Exception as exc:
        logger.exception(f"V75 TRADE MANAGER erreur globale: {exc}")

# =========================================================
# ANALYSE PRINCIPALE AVEC SYSTÈME DE SCORE RÉEL (0-15)
# =========================================================
def analyze_pair(pair: str) -> None:
    logger.info(f"\nV75 SCORING analyse {pair}")
    try:
        spread_status, spread_msg, _, spread_pts = spread_quality(pair)
        logger.info(f"{pair} · {spread_msg}")
        if spread_status == "BLOCK":
            return

        df_d1 = fetch_candles(pair, "D", 300)
        df_h4 = fetch_candles(pair, "H4", 310)
        df_h1 = fetch_candles(pair, "H1", 310)
        df_m15 = fetch_candles(pair, "M15", 220)

        if df_h4.empty or df_h1.empty or df_m15.empty:
            logger.warning(f"{pair}: données insuffisantes.")
            return

        score = 0
        details: List[str] = []

        # Spread devient un composant du score, plus un blocage léger.
        score += spread_pts
        details.append(spread_msg)

        d1_b = daily_bias(df_d1) if not df_d1.empty else None
        bias = market_bias(df_h4, df_h1)
        if not bias:
            logger.info(f"{pair}: pas d'alignement H4/H1 exploitable.")
            return

        # V72: D1 contraire = malus, pas veto. D1 neutre = 0.
        if d1_b == bias:
            score += 2
            details.append(f"D1={d1_b}(+2)")
        elif d1_b and d1_b != bias:
            score -= 2
            details.append(f"D1 conflit {d1_b} vs {bias}(-2)")
            logger.info(f"{pair}: conflit D1={d1_b} vs H4/H1={bias}, converti en malus V73.")
        else:
            details.append("D1 neutre(0)")

        score += 3
        details.append(f"H4/H1={bias}(+3)")
        logger.info(f"{pair}: bias={bias} | D1={d1_b}")

        df_h4_ind = add_indicators(df_h4)
        h4_adx = float(df_h4_ind["adx"].iloc[-1]) if not np.isnan(df_h4_ind["adx"].iloc[-1]) else 0.0
        if h4_adx >= MIN_ADX:
            score += 1
            details.append(f"ADX_H4={h4_adx:.1f}(+1)")
        elif h4_adx < 12:
            score -= 1
            details.append(f"ADX_H4 faible={h4_adx:.1f}(-1)")
        else:
            details.append(f"ADX_H4 neutre={h4_adx:.1f}(0)")

        zones = collect_hybrid_zones(pair, df_h4, df_h1, df_m15, bias)
        if not zones:
            logger.info(f"{pair}: aucune zone V63/V74 exploitable.")
            return

        best_payload = None

        for zone in zones:
            z_score, z_details = score_zone(score, zone, df_h4, df_h1, df_m15, bias)
            all_details = details + z_details

            # La zone/retest reste obligatoire: sinon l'entrée est trop anticipée.
            if not price_retraced_to_zone(pair, df_m15, zone):
                logger.info(f"{pair}: {zone.setup_type} ignoré, prix pas encore en zone.")
                continue
            z_score += 1
            all_details.append("prix_en_zone(+1)")

            # Rejet M15 devient du scoring.
            reject_pts, reject_reason = m15_rejection_score(pair, df_m15, zone)
            z_score += reject_pts
            all_details.append(f"M15_score={reject_pts} [{reject_reason}]")

            # Hard safety: un M15 vraiment contraire empêche seulement les scores faibles.
            if reject_pts <= -3 and z_score < (MIN_SEQUENCE_SCORE + 3):
                logger.info(f"{pair}: {zone.setup_type} M15 trop contraire: {reject_reason} | score provisoire={z_score}")
                continue

            current_price = df_m15["close"].iloc[-1]
            stop, tp = build_trade_levels(pair, zone, current_price, df_m15)
            risk = abs(current_price - stop)
            reward = abs(tp - current_price)
            rr = reward / risk if risk > 0 else 0
            if rr >= 2.5:
                z_score += 2
                all_details.append(f"RR={rr:.2f}(+2)")
            elif rr >= 1.8:
                z_score += 1
                all_details.append(f"RR={rr:.2f}(+1)")
            else:
                logger.info(f"{pair}: {zone.setup_type} RR insuffisant {rr:.2f}")
                continue

            core_ok, core_reason = is_v74_core_fvg_trade(zone, z_score, rr, reject_pts)
            logger.info(f"{pair}: CANDIDAT {zone.setup_type} SCORE={z_score}/15 RR={rr:.2f} | {' | '.join(all_details)}")

            if z_score < MIN_SEQUENCE_SCORE and not core_ok:
                logger.info(
                    f"{pair}: score {z_score}/15 insuffisant (min={MIN_SEQUENCE_SCORE}) "
                    f"et pas accepté core V74 ({core_reason})"
                )
                continue

            if core_ok and z_score < MIN_SEQUENCE_SCORE:
                all_details.append(core_reason)
                logger.info(f"{pair}: accepté par règle V74 core FVG malgré score {z_score}/15 < {MIN_SEQUENCE_SCORE}")

            payload = (zone, current_price, stop, tp, z_score, reject_reason, rr, all_details)
            if best_payload is None or z_score > best_payload[4] or (z_score == best_payload[4] and rr > best_payload[6]):
                best_payload = payload

        if best_payload is None:
            logger.info(f"{pair}: aucun finaliste après scoring V74.")
            return

        zone, current_price, stop, tp, final_score, reject_reason, rr, all_details = best_payload

        clean_signal_keys()
        key = (pair, zone.direction, zone.setup_type, round(zone.midpoint, decimals(pair)))
        if key in last_signal_key:
            logger.info(f"{pair}: signal déjà traité sur cette zone (TTL={SIGNAL_KEY_TTL_HOURS}H).")
            return

        logger.info(f"{pair}: FINALISTE V75 {zone.setup_type} score={final_score}/15 RR={rr:.2f} | {' | '.join(all_details)}")
        logger.info(
            f"DEBUG EXECUTION DECISION | pair={pair} direction={zone.direction} setup={zone.setup_type} "
            f"score={final_score}/15 rr={rr:.2f} key={key} execute={EXECUTE_TRADES} "
            f"ttl_present={key in last_signal_key}"
        )

        trade_id = place_trade(
            pair=pair,
            direction=zone.direction,
            entry=current_price,
            stop=stop,
            tp=tp,
            score=final_score,
            reason=f"{zone.setup_type} | {reject_reason}",
        )

        if trade_id not in (None, False):
            last_signal_key[key] = datetime.utcnow()
            logger.info(f"{pair}: zone enregistrée ({zone.setup_type}) pour {SIGNAL_KEY_TTL_HOURS}H. trade_id={trade_id}")
        else:
            logger.info(f"{pair}: ordre non exécuté, zone NON enregistrée.")

    except Exception as exc:
        logger.exception(f"Erreur analyse {pair}: {exc}")

def main() -> None:
    logger.info("Démarrage OANDA V75 OANDA V63 Decision + Smart BE/Trailing")
    logger.info(f"Compte: {OANDA_ACCOUNT_ID} | env={OANDA_ENVIRONMENT} | EXECUTE_TRADES={EXECUTE_TRADES}")
    logger.info(f"Trading hours: marché FX ouvert dimanche {FOREX_OPEN_UTC.strftime('%H:%M')} UTC -> vendredi {FOREX_CLOSE_UTC.strftime('%H:%M')} UTC")
    balance = get_account_balance()
    logger.info(f"Solde: {balance:.2f} | Paires: {', '.join(PAIR_LIST)}")
    logger.info(f"Score min V75: strict={MIN_SEQUENCE_SCORE}/15 | core_fvg={V74_CORE_FVG_MIN_SCORE}/15 | RR: {RISK_REWARD} | Risk/trade: {RISK_PERCENTAGE}% | Daily max: {MAX_DAILY_RISK_PERCENT}% | Weekly max: {MAX_WEEKLY_LOSS_PERCENT}%")
    get_risk_guard_state(balance)

    while True:
        now_dt = datetime.utcnow()
        logger.info(f"\n===== {now_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC =====")
        logger.info(market_status_text(now_dt))
        try:
            if is_market_open_utc(now_dt):
                # V75: on protège d'abord les positions déjà ouvertes avant de chercher de nouveaux signaux.
                manage_open_trades_v75()
                for pair in PAIR_LIST:
                    analyze_pair(pair)
                    time.sleep(0.3)
                # V75: second passage après les analyses pour réagir si un ordre vient d'être ouvert.
                manage_open_trades_v75()
                time.sleep(LOOP_SECONDS)
            else:
                logger.info("Week-end / marché fermé. Attente 5 min.")
                time.sleep(300)
        except Exception as exc:
            logger.exception(f"Erreur boucle principale: {exc}")
            time.sleep(60)


if __name__ == "__main__":
    main()
