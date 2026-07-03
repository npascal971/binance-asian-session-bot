import os
import time
import logging
import smtplib
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
# STRATEGIE V67.1 - OANDA HYBRID SNIPER
# Exécution OANDA V67 + détection V63/V65 rééquilibrée
# Setups: FVG_RETEST_PERFECT, NESTED_FVG, WICK_REJECTION
# Veto qualité: spread, H1 EMA50, rejet M15, RR, anti-doublon
# =========================================================
PAIR_LIST = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD",
    "AUD_USD", "AUD_CAD", "AUD_JPY", "XAU_USD", "GBP_JPY"
]

SESSION_START = dtime(7, 0)
SESSION_END = dtime(21, 0)
LOOP_SECONDS = 60

RISK_PERCENTAGE = float(os.getenv("RISK_PERCENTAGE", "1.0"))
MAX_RISK_USD = float(os.getenv("MAX_RISK_USD", "100"))
RISK_REWARD = float(os.getenv("RISK_REWARD", "2.0"))
MAX_TRADES_TOTAL = int(os.getenv("MAX_TRADES_TOTAL", "3"))
ONE_TRADE_PER_PAIR = True

EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14
RSI_PERIOD = 14
ADX_PERIOD = 14

# Seuil de score : score minimum sur 15 pour prendre un trade
MIN_SEQUENCE_SCORE = 10  # score mini équilibré pour retrouver des signaux type V63 sans ouvrir les B setups

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
    "XAU_USD": 10.0,   # Était 35.0 - beaucoup trop lâche
    "GBP_JPY": 3.5,
    "AUD_JPY": 3.0,
    "AUD_CAD": 2.2,
    "USD_JPY": 2.0,
    "GBP_USD": 2.0,
    "DEFAULT": 1.8,
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

MIN_UNITS = {
    "XAU_USD": 1,
    "DEFAULT": 1000,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("oanda_v67_1_hybrid_sniper.log")],
)
logger = logging.getLogger("OANDA-V67.1-Hybrid-Sniper")

# Dict signal_key -> timestamp pour expiration TTL (fix: était un set() infini)
last_signal_key: Dict[tuple, datetime] = {}


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


def get_account_balance() -> float:
    r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
    resp = client.request(r)
    return float(resp["account"]["balance"])


def get_open_trades() -> List[Dict]:
    r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
    resp = client.request(r)
    return resp.get("trades", [])


def has_open_trade(pair: str) -> bool:
    return any(t.get("instrument") == pair for t in get_open_trades())


def open_trade_count() -> int:
    return len(get_open_trades())


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
    prices = fetch_pricing_spread(pair)
    if not prices:
        # Fix : on bloque si spread indisponible (était ignoré = risque ouverture en pic de spread)
        return False, "spread indisponible - trade bloqué par sécurité", None
    bid, ask, _ = prices
    spread_pips = price_to_pips(pair, ask - bid)
    max_spread = MAX_SPREAD_PIPS.get(pair, MAX_SPREAD_PIPS["DEFAULT"])
    if spread_pips > max_spread:
        return False, f"spread trop élevé {spread_pips:.1f}p > {max_spread:.1f}p", spread_pips
    return True, f"spread OK {spread_pips:.1f}p", spread_pips


def calculate_units(pair: str, entry: float, stop: float, balance: float) -> int:
    risk_usd = min(balance * (RISK_PERCENTAGE / 100), MAX_RISK_USD)
    distance = abs(entry - stop)
    if distance <= 0:
        return 0

    units_float = risk_usd / distance
    min_units = MIN_UNITS.get(pair, MIN_UNITS["DEFAULT"])

    if pair == "XAU_USD":
        units = int(max(round(units_float, 0), min_units))
    else:
        units = int(max(round(units_float / min_units) * min_units, min_units))

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
    if ONE_TRADE_PER_PAIR and has_open_trade(pair):
        logger.info(f"{pair}: trade déjà ouvert, aucun nouvel ordre.")
        return None
    if open_trade_count() >= MAX_TRADES_TOTAL:
        logger.info(f"Limite trades ouverts atteinte ({MAX_TRADES_TOTAL}).")
        return None

    balance = get_account_balance()
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
        f"SIGNAL V67.1 HYBRID {pair} {direction} | "
        f"entry≈{entry:.{decimals(pair)}f} SL={stop:.{decimals(pair)}f} TP={tp:.{decimals(pair)}f} "
        f"RR={rr_actual:.2f} score={score}/15 units={units} | {reason}"
    )

    if not EXECUTE_TRADES:
        logger.info("EXECUTE_TRADES=false : ordre non envoyé à OANDA.")
        return "SIMULATION"

    try:
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        resp = client.request(r)
        trade_id = (
            resp.get("orderFillTransaction", {}).get("id")
            or resp.get("orderCreateTransaction", {}).get("id")
        )
        logger.info(f"ORDRE RÉEL ENVOYÉ {pair} | ID={trade_id}")
        send_email(
            f"V67.1 {pair} {direction} score={score}/15",
            (
                f"Paire: {pair}\nDirection: {direction}\n"
                f"Entrée: {entry:.{decimals(pair)}f}\nSL: {stop:.{decimals(pair)}f}\n"
                f"TP: {tp:.{decimals(pair)}f}\nRR: {rr_actual:.2f}\n"
                f"Score: {score}/15\nRaison: {reason}"
            ),
        )
        return trade_id
    except Exception as exc:
        logger.error(f"Erreur ordre OANDA {pair}: {exc}")
        return None


# =========================================================
# ANALYSE PRINCIPALE AVEC SYSTÈME DE SCORE RÉEL (0-15)
# =========================================================
def analyze_pair(pair: str) -> None:
    logger.info(f"\nV67.1 HYBRID analyse {pair}")
    try:
        spread_valid, spread_msg, _ = spread_ok(pair)
        logger.info(f"{pair} · {spread_msg}")
        if not spread_valid:
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

        # D1 = bonus seulement, pas blocage permanent sauf conflit fort explicite.
        d1_b = daily_bias(df_d1) if not df_d1.empty else None
        if d1_b:
            score += 2
            details.append(f"D1={d1_b}(+2)")

        bias = market_bias(df_h4, df_h1)
        if not bias:
            logger.info(f"{pair}: pas d'alignement H4/H1 exploitable.")
            return

        if d1_b and d1_b != bias:
            # V63 pouvait trader sans D1, mais on garde ce veto pour éviter les vrais contresens HTF.
            logger.info(f"{pair}: conflit biais D1={d1_b} vs H4/H1={bias}. Trade bloqué.")
            return

        score += 3
        details.append(f"H4/H1={bias}(+3)")
        logger.info(f"{pair}: bias={bias} | D1={d1_b}")

        df_h4_ind = add_indicators(df_h4)
        h4_adx = float(df_h4_ind["adx"].iloc[-1]) if not np.isnan(df_h4_ind["adx"].iloc[-1]) else 0.0
        if h4_adx >= MIN_ADX:
            score += 1
            details.append(f"ADX_H4={h4_adx:.1f}(+1)")
        else:
            logger.info(f"{pair}: ADX H4 faible {h4_adx:.1f}, autorisé mais sans bonus.")

        zones = collect_hybrid_zones(pair, df_h4, df_h1, df_m15, bias)
        if not zones:
            logger.info(f"{pair}: aucune zone V63/V67.1 exploitable.")
            return

        best = None
        best_payload = None

        for zone in zones:
            z_score, z_details = score_zone(score, zone, df_h4, df_h1, df_m15, bias)
            all_details = details + z_details

            if not price_retraced_to_zone(pair, df_m15, zone):
                logger.info(f"{pair}: {zone.setup_type} ignoré, prix pas encore en zone.")
                continue
            all_details.append("prix_en_zone")

            ok_reject, reject_reason = rejection_confirmed(pair, df_m15, zone)
            if not ok_reject:
                logger.info(f"{pair}: {zone.setup_type} rejeté, M15 refusé: {reject_reason}")
                continue
            z_score += 2
            all_details.append("rejet_M15(+2)")

            current_price = df_m15["close"].iloc[-1]
            stop, tp = build_trade_levels(pair, zone, current_price, df_m15)
            risk = abs(current_price - stop)
            reward = abs(tp - current_price)
            rr = reward / risk if risk > 0 else 0
            if rr < 1.8:
                logger.info(f"{pair}: {zone.setup_type} RR insuffisant {rr:.2f}")
                continue

            logger.info(f"{pair}: CANDIDAT {zone.setup_type} SCORE={z_score}/15 RR={rr:.2f} | {' | '.join(all_details)}")

            if z_score < MIN_SEQUENCE_SCORE:
                logger.info(f"{pair}: score {z_score}/15 insuffisant (min={MIN_SEQUENCE_SCORE})")
                continue

            payload = (zone, current_price, stop, tp, z_score, reject_reason, rr, all_details)
            if best is None or z_score > best_payload[4] or (z_score == best_payload[4] and rr > best_payload[6]):
                best = zone
                best_payload = payload

        if best_payload is None:
            logger.info(f"{pair}: aucun finaliste après vetos qualité.")
            return

        zone, current_price, stop, tp, final_score, reject_reason, rr, all_details = best_payload

        clean_signal_keys()
        key = (pair, zone.direction, zone.setup_type, round(zone.midpoint, decimals(pair)))
        if key in last_signal_key:
            logger.info(f"{pair}: signal déjà traité sur cette zone (TTL={SIGNAL_KEY_TTL_HOURS}H).")
            return
        last_signal_key[key] = datetime.utcnow()

        logger.info(f"{pair}: FINALISTE {zone.setup_type} score={final_score}/15 RR={rr:.2f} | {' | '.join(all_details)}")
        place_trade(pair, zone.direction, current_price, stop, tp, final_score, f"{zone.setup_type} | {reject_reason}")

    except Exception as exc:
        logger.exception(f"Erreur analyse {pair}: {exc}")


def main() -> None:
    logger.info("Démarrage OANDA V67.1 Hybrid Sniper")
    logger.info(f"Compte: {OANDA_ACCOUNT_ID} | env={OANDA_ENVIRONMENT} | EXECUTE_TRADES={EXECUTE_TRADES}")
    balance = get_account_balance()
    logger.info(f"Solde: {balance:.2f} | Paires: {', '.join(PAIR_LIST)}")
    logger.info(f"Score min: {MIN_SEQUENCE_SCORE}/15 | RR: {RISK_REWARD} | Risk: {RISK_PERCENTAGE}%")

    while True:
        now = datetime.utcnow().time()
        logger.info(f"\n===== {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC =====")
        try:
            if SESSION_START <= now <= SESSION_END:
                for pair in PAIR_LIST:
                    analyze_pair(pair)
                    time.sleep(0.3)
                time.sleep(LOOP_SECONDS)
            else:
                logger.info("Hors session. Attente 5 min.")
                time.sleep(300)
        except Exception as exc:
            logger.exception(f"Erreur boucle principale: {exc}")
            time.sleep(60)


if __name__ == "__main__":
    main()
