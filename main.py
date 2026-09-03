# ============================================================
# main.py - Version PROD "2R Strict" (v133)
# Stratégie : Biais H4/H1 → Retracement → Confirmation → 2R
# ============================================================

import os
import sys
import time
import logging
import requests
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import oandapyV20
from oandapyV20.endpoints import instruments, pricing, orders, accounts, trades
import talib
import traceback
from ta.momentum import RSIIndicator
from typing import List, Dict, Tuple, Optional

# =========================
# CHARGEMENT .env
# =========================
load_dotenv()
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# =========================
# CONFIGURATION GÉNÉRALE
# =========================
PAIR_LIST = ["GBP_USD", "USD_CAD", "AUD_USD", "XAU_USD", "EUR_USD", "USD_JPY", "AUD_JPY"]
GRANULARITY_D1 = "D"
GRANULARITY_H4 = "H4"
GRANULARITY_H1 = "H1"
GRANULARITY_M15 = "M15"

EMA_MEDIUM = 50
SWING_LOOKBACK = 3
ATR_PERIOD = 14

# Paramètres de gestion
BASE_BREAKEVEN_TRIGGER_R = 0.55
BASE_TRAILING_ACTIVATION_R = 0.80
BASE_TRAILING_STOP_DISTANCE_ATR_MULTIPLIER = 1.5
BASE_TRAILING_STOP_MIN_DISTANCE_PIPS = 8.0

MAX_TRADES_TOTAL = 10
ONE_TRADE_PER_PAIR = True
RISK_PERCENTAGE = 0.75
MAX_RISK_USD = 1250
MAX_MARGIN_USAGE_PER_TRADE_PERCENT = 5.0

OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "101-004-31348578-001")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "true").lower() == "true"

# --- NOUVEAU : marge de sécurité pour le slippage ---
RR_MIN_EXECUTION = 2.0   # exigé avant ordre, pour absorber le slippage normal

PIP_SIZE_V88 = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "USD_CAD": 0.0001, "AUD_CAD": 0.0001,
    "USD_JPY": 0.01, "AUD_JPY": 0.01, "GBP_JPY": 0.01,
    "XAU_USD": 0.01,
}
PRICE_DECIMALS_V88 = {
    "EUR_USD": 5, "GBP_USD": 5, "AUD_USD": 5,
    "USD_CAD": 5, "AUD_CAD": 5,
    "USD_JPY": 3, "AUD_JPY": 3, "GBP_JPY": 3,
    "XAU_USD": 3,
}
UNIT_STEP_BY_PAIR = {
    "XAU_USD": 1, "EUR_USD": 1000, "GBP_USD": 1000,
    "USD_JPY": 1000, "USD_CAD": 1000, "AUD_USD": 1000,
    "AUD_CAD": 1000, "AUD_JPY": 1000, "GBP_JPY": 1000,
    "DEFAULT": 1000,
}
MIN_UNITS_BY_PAIR = {"XAU_USD": 1, "DEFAULT": 1000}
MAX_UNITS_BY_PAIR = {
    "XAU_USD": 100, "EUR_USD": 200000, "GBP_USD": 200000,
    "USD_JPY": 200000, "USD_CAD": 200000, "AUD_USD": 200000,
    "AUD_CAD": 200000, "AUD_JPY": 200000, "GBP_JPY": 200000,
    "DEFAULT": 200000,
}
EXECUTION_COOLDOWN_SECONDS = 60

# ============================================================
# LOGGING
# ============================================================
logger = logging.getLogger("TradingBot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
for noisy in ("urllib3", "requests", "oandapyV20"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# ============================================================
# CACHE OANDA
# ============================================================
_OANDA_CACHE = {}
OANDA_CACHE_TTL = 3.0

def cache_get(key: str, ttl: float = OANDA_CACHE_TTL):
    item = _OANDA_CACHE.get(key)
    if not item:
        return None
    ts, value = item
    if time.time() - ts > ttl:
        _OANDA_CACHE.pop(key, None)
        return None
    return value

def cache_set(key: str, value):
    _OANDA_CACHE[key] = (time.time(), value)

def clear_cache():
    _OANDA_CACHE.clear()

# ============================================================
# MAINTENANCE OANDA
# ============================================================
MAINTENANCE_DETECTED = False
MAINTENANCE_SUSPEND_TIME = 0

def is_oanda_maintenance(error: Exception) -> bool:
    return any(p in str(error).lower() for p in ["maintenance", "temporarily unavailable", "service unavailable"])

def handle_api_error(error: Exception):
    global MAINTENANCE_DETECTED, MAINTENANCE_SUSPEND_TIME
    if is_oanda_maintenance(error):
        MAINTENANCE_DETECTED = True
        MAINTENANCE_SUSPEND_TIME = time.time() + 120
        logger.warning("OANDA maintenance détectée, suspension 120s")
        return True
    return False

def is_maintenance_suspended():
    global MAINTENANCE_DETECTED, MAINTENANCE_SUSPEND_TIME
    if not MAINTENANCE_DETECTED:
        return False
    if time.time() < MAINTENANCE_SUSPEND_TIME:
        return True
    MAINTENANCE_DETECTED = False
    return False

def reset_maintenance():
    global MAINTENANCE_DETECTED, MAINTENANCE_SUSPEND_TIME
    MAINTENANCE_DETECTED = False
    MAINTENANCE_SUSPEND_TIME = 0

# ============================================================
# FONCTIONS OANDA
# ============================================================
def v88_client():
    token = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_ACCESS_TOKEN")
    return oandapyV20.API(access_token=token, environment=os.getenv("OANDA_ENVIRONMENT", "practice"))

def get_candles(api, instrument: str, granularity: str, count: int = 500) -> pd.DataFrame:
    if is_maintenance_suspended():
        return pd.DataFrame()
    try:
        params = {"granularity": granularity, "count": min(count, 500), "price": "M"}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        api.request(r)
        candles = r.response.get("candles", [])
        data = []
        for c in candles:
            mid = c.get("mid")
            if mid:
                data.append({
                    "time": c["time"],
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(c.get("volume", 0))
                })
        df = pd.DataFrame(data)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df.attrs['instrument'] = instrument
        return df
    except Exception as e:
        handle_api_error(e)
        return pd.DataFrame()

def get_price_spread(pair: str) -> dict:
    cached = cache_get(f"pricing:{pair}")
    if cached:
        return cached
    try:
        if is_maintenance_suspended():
            return {"bid": 0, "ask": 0, "mid": 0, "spread": 0}
        api = v88_client()
        r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params={"instruments": pair})
        api.request(r)
        prices = r.response.get("prices", [])
        if prices:
            item = prices[0]
            bid = float(item.get("bids", [{}])[0].get("price", 0))
            ask = float(item.get("asks", [{}])[0].get("price", 0))
            mid = (bid + ask) / 2.0 if bid and ask else 0
            data = {"bid": bid, "ask": ask, "mid": mid, "spread": max(ask - bid, 0)}
            cache_set(f"pricing:{pair}", data)
            return data
    except Exception:
        pass
    return {"bid": 0, "ask": 0, "mid": 0, "spread": 0}

def get_current_price(pair: str) -> float:
    df = get_candles(v88_client(), pair, "M5", 10)
    if not df.empty:
        return float(df["close"].iloc[-1])
    return 0.0

def get_open_trades(force_refresh=False) -> list:
    if is_maintenance_suspended():
        return []
    key = "open_trades_raw"
    if force_refresh:
        _OANDA_CACHE.pop(key, None)
    resp = cache_get(key, ttl=1.0)
    if resp is None:
        try:
            api = v88_client()
            r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
            api.request(r)
            resp = r.response
            cache_set(key, resp)
        except Exception as e:
            handle_api_error(e)
            return []
    return resp.get("trades", [])

def get_account_summary():
    try:
        api = v88_client()
        r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
        api.request(r)
        return r.response
    except:
        return {}

def get_balance():
    return float(get_account_summary().get("account", {}).get("balance", 0))

def get_oanda_margin_rate(pair: str) -> float:
    try:
        api = v88_client()
        r = accounts.AccountInstruments(accountID=OANDA_ACCOUNT_ID, params={"instruments": pair})
        api.request(r)
        instr = r.response.get("instruments", [])
        if instr:
            return float(instr[0].get("marginRate", 0.0333))
    except:
        pass
    return 0.0333

def get_available_margin():
    return float(get_account_summary().get("account", {}).get("marginAvailable", 0))

def get_fx_rate_to_usd(currency: str) -> float:
    if currency == "USD":
        return 1.0
    # simplifié, on utilise un taux fixe pour l'exemple
    return 1.0

def calculate_margin(pair: str, units: int, entry_price: float) -> dict:
    margin_rate = get_oanda_margin_rate(pair)
    notional_usd = units * entry_price
    margin_req = notional_usd * margin_rate
    available = get_available_margin()
    return {"margin_required": margin_req, "margin_available": available, "sufficient": available >= margin_req}

def cap_units_by_margin(pair: str, units: int, entry_price: float, balance: float) -> int:
    margin_info = calculate_margin(pair, units, entry_price)
    if margin_info["sufficient"]:
        return units
    max_margin = balance * (MAX_MARGIN_USAGE_PER_TRADE_PERCENT / 100.0)
    ratio = max_margin / margin_info["margin_required"] if margin_info["margin_required"] > 0 else 0
    capped = int(units * ratio)
    step = UNIT_STEP_BY_PAIR.get(pair, UNIT_STEP_BY_PAIR["DEFAULT"])
    return max(0, int(capped // step * step))

def calculate_units(pair: str, entry: float, stop_loss: float, balance: float, risk_pct: float = None) -> int:
    if risk_pct is None:
        risk_pct = RISK_PERCENTAGE
    risk_usd = min(balance * (risk_pct / 100.0), MAX_RISK_USD)
    distance = abs(entry - stop_loss)
    if distance <= 0:
        return 0
    quote = pair.split("_")[1]
    quote_to_usd = get_fx_rate_to_usd(quote)
    risk_per_unit = distance * quote_to_usd
    if risk_per_unit <= 0:
        return 0
    raw_units = risk_usd / risk_per_unit
    step = UNIT_STEP_BY_PAIR.get(pair, UNIT_STEP_BY_PAIR["DEFAULT"])
    units = int(raw_units // step * step)
    units = cap_units_by_margin(pair, units, entry, balance)
    return max(0, units)

def round_price(pair: str, price: float) -> str:
    decimals = PRICE_DECIMALS_V88.get(pair, 5)
    return f"{float(price):.{decimals}f}"

def is_market_open(now_dt: datetime) -> bool:
    wd = now_dt.weekday()
    t = now_dt.time()
    if wd == 5:
        return False
    if wd == 6 and t < datetime.strptime("21:00", "%H:%M").time():
        return False
    if wd == 4 and t >= datetime.strptime("21:00", "%H:%M").time():
        return False
    return True

def open_trade_count() -> int:
    return len(get_open_trades())

def has_open_trade(pair: str) -> bool:
    for t in get_open_trades():
        if t.get("instrument") == pair:
            return True
    return False

def get_trade_details(trade_id: str) -> dict:
    try:
        api = v88_client()
        r = trades.TradeDetails(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id)
        api.request(r)
        return r.response.get("trade", {})
    except:
        return {}

def get_stop_loss(trade: dict) -> float:
    sl = trade.get("stopLossOrder", {})
    return float(sl.get("price", 0))

def has_trailing_stop(trade: dict) -> bool:
    return bool(trade.get("trailingStopLossOrder", {}).get("id"))

# ============================================================
# INDICATEURS
# ============================================================
def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    try:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        atr = talib.ATR(high, low, close, timeperiod=period)
        return float(atr[-1]) if not np.isnan(atr[-1]) else 0.0001
    except:
        return 0.0001

def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    try:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        return float(talib.ADX(high, low, close, timeperiod=period)[-1])
    except:
        return 0.0

def calculate_momentum(df: pd.DataFrame, period: int = 5) -> float:
    if len(df) < period + 1:
        return 0.0
    return (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period] * 100

def get_last_rsi(prices: pd.Series, period: int = 14) -> float:
    try:
        rsi = talib.RSI(prices.values, timeperiod=period)
        return float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
    except:
        return 50.0

def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> tuple:
    """Détecte les swings sur la base des extrêmes de prix."""
    highs, lows = [], []
    for i in range(lookback, len(df) - lookback):
        if df["high"].iloc[i] == df["high"].iloc[i-lookback:i+lookback+1].max():
            highs.append({"index": i, "time": df.index[i], "price": df["high"].iloc[i]})
        if df["low"].iloc[i] == df["low"].iloc[i-lookback:i+lookback+1].min():
            lows.append({"index": i, "time": df.index[i], "price": df["low"].iloc[i]})
    return highs, lows

def detect_fvg(df: pd.DataFrame, max_lookback_hours: int = 36) -> List[Dict]:
    fvgs = []
    if len(df) < 3:
        return fvgs
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    min_time = now - timedelta(hours=max_lookback_hours)
    idx_times = pd.to_datetime(df.index)
    if idx_times.tz is None:
        idx_times = idx_times.tz_localize('UTC')
    pair = df.attrs.get("instrument", "")
    min_gap = 0.00015 if "JPY" in pair else 0.0002
    for i in range(1, len(df) - 1):
        if idx_times[i] < min_time:
            continue
        prev = df.iloc[i-1]
        nxt = df.iloc[i+1]
        if prev["high"] < nxt["low"] and nxt["low"] - prev["high"] >= min_gap:
            fvgs.append({
                "direction": "BUY",
                "high_level": nxt["low"],
                "low_level": prev["high"],
                "midpoint": (prev["high"] + nxt["low"]) / 2,
                "time": idx_times[i]
            })
        if prev["low"] > nxt["high"] and prev["low"] - nxt["high"] >= min_gap:
            fvgs.append({
                "direction": "SELL",
                "high_level": prev["low"],
                "low_level": nxt["high"],
                "midpoint": (nxt["high"] + prev["low"]) / 2,
                "time": idx_times[i]
            })
    return fvgs

def detect_wick_rejection(df: pd.DataFrame, bias: str) -> list:
    poi = []
    if len(df) < 3:
        return poi
    for i in range(1, len(df) - 1):
        c = df.iloc[i]
        body = abs(c["close"] - c["open"])
        total = c["high"] - c["low"]
        if total == 0:
            continue
        upper = c["high"] - max(c["open"], c["close"])
        lower = min(c["open"], c["close"]) - c["low"]
        if bias in ["BUY", "NEUTRAL"] and lower >= body * 0.7 and lower >= upper * 1.5:
            poi.append({"direction": "BUY", "price_level": c["low"]})
        elif bias in ["SELL", "NEUTRAL"] and upper >= body * 0.7 and upper >= lower * 1.5:
            poi.append({"direction": "SELL", "price_level": c["high"]})
    return poi

def detect_bos(df: pd.DataFrame) -> dict:
    highs, lows = detect_swing_points(df, 5)
    if len(highs) < 1 or len(lows) < 1:
        return {"type": None}
    last_close = df["close"].iloc[-1]
    if last_close > highs[-1]["price"]:
        return {"type": "BOS_BUY", "level": highs[-1]["price"]}
    if last_close < lows[-1]["price"]:
        return {"type": "BOS_SELL", "level": lows[-1]["price"]}
    return {"type": None}

def detect_setups(
    pair: str,
    df_m15: pd.DataFrame,
    df_h1: pd.DataFrame,
    bias: str
) -> List[Dict]:
    """
    Détecte les setups dans le sens du biais.

    Setups autorisés :
        - FVG_RETEST
        - WICK_REJECTION
        - BOS_RETEST

    Le BOS n'est pas pris sur simple cassure :
        BOS -> retest du niveau cassé -> confirmation
    """

    setups = []

    # =========================================================
    # SÉCURITÉ
    # =========================================================

    if df_m15 is None or len(df_m15) < 20:
        return setups

    if bias not in ("BUY", "SELL"):
        return setups

    # =========================================================
    # 1. FVG RETEST
    # =========================================================

    try:
        fvgs = detect_fvg(df_m15)
    except Exception as e:
        logger.warning(
            f"{pair} | FVG detection error: {e}"
        )
        fvgs = []

    for f in fvgs:

        if f.get("direction") != bias:
            continue

        try:
            entry_level = float(
                f["midpoint"]
            )
        except (KeyError, TypeError, ValueError):
            continue

        setups.append({
            "type": "FVG_RETEST",
            "direction": bias,
            "entry_level": entry_level,
            "fvg": f,
        })

        logger.debug(
            f"[SETUP] {pair} | "
            f"FVG_RETEST | "
            f"{bias} | "
            f"entry={entry_level:.5f}"
        )

    # =========================================================
    # 2. WICK REJECTION
    # =========================================================

    try:
        wicks = detect_wick_rejection(
            df_m15,
            bias
        )
    except Exception as e:
        logger.warning(
            f"{pair} | WICK detection error: {e}"
        )
        wicks = []

    for w in wicks:

        if w.get("direction") != bias:
            continue

        try:
            entry_level = float(
                w["price_level"]
            )
        except (KeyError, TypeError, ValueError):
            continue

        setups.append({
            "type": "WICK_REJECTION",
            "direction": bias,
            "entry_level": entry_level,
        })

        logger.debug(
            f"[SETUP] {pair} | "
            f"WICK_REJECTION | "
            f"{bias} | "
            f"entry={entry_level:.5f}"
        )

    # =========================================================
    # 3. BOS RETEST
    # =========================================================
    #
    # IMPORTANT :
    # detect_bos_retest() doit être définie au niveau global
    # du fichier.
    #
    # Elle renvoie None si aucun BOS + retest + confirmation
    # n'est présent.
    # =========================================================

    try:
        bos_setup = detect_bos_retest(
            df_m15,
            bias
        )
    except Exception as e:
        logger.warning(
            f"{pair} | BOS_RETEST detection error: {e}"
        )
        bos_setup = None

    if bos_setup is not None:

        # Sécurité : on force la direction du biais
        bos_setup["direction"] = bias

        try:
            bos_entry = float(
                bos_setup["entry_level"]
            )
        except (
            KeyError,
            TypeError,
            ValueError
        ):
            bos_entry = None

        if bos_entry is not None:

            # =================================================
            # ÉVITER LES DOUBLONS
            # =================================================

            duplicate = any(
                s["type"] == "BOS_RETEST"
                and abs(
                    float(s["entry_level"])
                    - bos_entry
                ) < 1e-10
                for s in setups
            )

            if not duplicate:

                setups.append({
                    **bos_setup,
                    "type": "BOS_RETEST",
                    "direction": bias,
                    "entry_level": bos_entry,
                })

                logger.info(
                    f"[SETUP] {pair} | "
                    f"BOS_RETEST | "
                    f"{bias} | "
                    f"entry={bos_entry:.5f} | "
                    f"confirmation="
                    f"{bos_setup.get('confirmation', 'OK')}"
                )

    # =========================================================
    # 4. NETTOYAGE DES DOUBLONS
    # =========================================================

    unique_setups = []
    seen = set()

    for setup in setups:

        try:
            key = (
                setup["type"],
                setup["direction"],
                round(
                    float(setup["entry_level"]),
                    8
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError
        ):
            continue

        if key in seen:
            continue

        seen.add(key)
        unique_setups.append(setup)

    # =========================================================
    # 5. LOG FINAL
    # =========================================================

    logger.info(
        f"[SETUPS] {pair} | "
        f"BIAS={bias} | "
        f"FVG/WICK/BOS="
        f"{len(unique_setups)}"
    )

    return unique_setups
# ============================================================
# STRATÉGIE SIMPLIFIÉE
# ============================================================
def get_directional_bias(
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame
) -> str:
    """
    Détermine le biais directionnel H4/H1.

    H4 = timeframe directeur.
    H1 = confirmation / retracement.

    Retour :
        BUY
        SELL
        NEUTRAL
    """

    def bias_from_structure(
        df: pd.DataFrame,
        label: str = ""
    ):
        highs, lows = detect_swing_points(df, 5)

        if len(highs) < 2 or len(lows) < 2:
            return "NEUTRAL", 0, 0

        hh = highs[-1]["price"] > highs[-2]["price"]
        hl = lows[-1]["price"] > lows[-2]["price"]

        lh = highs[-1]["price"] < highs[-2]["price"]
        ll = lows[-1]["price"] < lows[-2]["price"]

        buy_signals = int(hh) + int(hl)
        sell_signals = int(lh) + int(ll)

        if buy_signals == 2:
            return "BUY", buy_signals, sell_signals

        if sell_signals == 2:
            return "SELL", buy_signals, sell_signals

        if buy_signals == 1 and sell_signals == 0:
            return "BUY_WEAK", buy_signals, sell_signals

        if sell_signals == 1 and buy_signals == 0:
            return "SELL_WEAK", buy_signals, sell_signals

        return "NEUTRAL", buy_signals, sell_signals

    # =========================================================
    # STRUCTURE H4 / H1
    # =========================================================

    b4, b4_buy, b4_sell = bias_from_structure(
        df_h4,
        "H4"
    )

    b1, b1_buy, b1_sell = bias_from_structure(
        df_h1,
        "H1"
    )

    # =========================================================
    # INDICATEURS H1
    # =========================================================

    try:
        adx_h1 = float(calculate_adx(df_h1))
    except Exception:
        adx_h1 = 0.0

    try:
        momentum_h1 = float(calculate_momentum(df_h1))
    except Exception:
        momentum_h1 = 0.0

    # =========================================================
    # H4 NEUTRAL = PAS DE TRADE
    # =========================================================

    if b4 == "NEUTRAL":
        result = "NEUTRAL"

    # =========================================================
    # H4 BUY
    # =========================================================

    elif b4 == "BUY":

        # H1 confirme la tendance
        if b1 in ("BUY", "BUY_WEAK"):
            result = "BUY"

        # H1 neutre mais momentum haussier
        elif b1 == "NEUTRAL":

            if adx_h1 >= 20 and momentum_h1 > 0.15:
                result = "BUY"
            else:
                result = "NEUTRAL"

        # H1 en retracement baissier faible
        elif b1 == "SELL_WEAK":

            if adx_h1 >= 25 and momentum_h1 > 0.15:
                result = "BUY"
            else:
                result = "NEUTRAL"

        # H1 SELL confirmé = contradiction
        else:
            result = "NEUTRAL"

    # =========================================================
    # H4 SELL
    # =========================================================

    elif b4 == "SELL":

        # H1 confirme la tendance
        if b1 in ("SELL", "SELL_WEAK"):
            result = "SELL"

        # H1 neutre mais momentum baissier
        elif b1 == "NEUTRAL":

            if adx_h1 >= 20 and momentum_h1 < -0.15:
                result = "SELL"
            else:
                result = "NEUTRAL"

        # H1 en retracement haussier faible
        elif b1 == "BUY_WEAK":

            if adx_h1 >= 25 and momentum_h1 < -0.15:
                result = "SELL"
            else:
                result = "NEUTRAL"

        # H1 BUY confirmé = contradiction
        else:
            result = "NEUTRAL"

    else:
        result = "NEUTRAL"

    # =========================================================
    # LOG DIAGNOSTIC
    # =========================================================

    logger.info(
        f"[BIAS_DIAG] "
        f"H4={b4} ({b4_buy}/{b4_sell}) | "
        f"H1={b1} ({b1_buy}/{b1_sell}) | "
        f"ADX_H1={adx_h1:.1f} | "
        f"MOM_H1={momentum_h1:+.2f}% | "
        f"RESULT={result}"
    )

    return result
    
def detect_bos_retest(
    df: pd.DataFrame,
    direction: str
) -> Optional[dict]:
    """
    Détecte un véritable BOS suivi d'un RETEST.

    Structure recherchée :

        BUY :
            1. cassure d'un swing high
            2. retour du prix sur le niveau cassé
            3. maintien au-dessus du niveau
            4. rejet OU micro-break haussier

        SELL :
            1. cassure d'un swing low
            2. retour du prix sur le niveau cassé
            3. maintien sous le niveau
            4. rejet OU micro-break baissier

    IMPORTANT :
    Le BOS et le retest peuvent se produire sur des bougies
    différentes.
    """

    # =========================================================
    # SÉCURITÉ
    # =========================================================

    if df is None or len(df) < 30:
        return None

    if direction not in ("BUY", "SELL"):
        return None

    # =========================================================
    # UNIQUEMENT LES BOUGIES CLÔTURÉES
    # =========================================================

    data = df.iloc[:-1].copy()

    if len(data) < 25:
        return None

    atr = calculate_atr(data)

    if atr is None or atr <= 0:
        return None

    atr = float(atr)

    # =========================================================
    # PARAMÈTRES
    # =========================================================

    # On cherche un BOS relativement récent.
    MAX_BOS_AGE = 8

    # Tolérance autour du niveau de retest.
    # 0.25 ATR permet de ne pas rater les retests légèrement
    # imparfaits.
    RETEST_TOLERANCE_ATR = 0.25

    retest_tolerance = atr * RETEST_TOLERANCE_ATR

    # =========================================================
    # SWINGS
    # =========================================================

    # Les swings doivent être établis AVANT le BOS.
    structure_df = data.iloc[:-MAX_BOS_AGE]

    swing_highs, swing_lows = detect_swing_points(
        structure_df,
        5
    )

    if direction == "BUY" and not swing_highs:
        return None

    if direction == "SELL" and not swing_lows:
        return None

    # =========================================================
    # FONCTION DE CONFIRMATION
    # =========================================================

    def get_confirmation(
        candle,
        previous_candle,
        side: str
    ):
        candle_high = float(candle["high"])
        candle_low = float(candle["low"])
        candle_open = float(candle["open"])
        candle_close = float(candle["close"])

        candle_range = candle_high - candle_low

        if candle_range <= 0:
            return False, 0.0, False

        if side == "BUY":

            lower_wick = (
                min(
                    candle_open,
                    candle_close
                )
                - candle_low
            )

            rejection_ratio = (
                lower_wick / candle_range
            )

            micro_break = (
                candle_close
                > float(previous_candle["high"])
            )

        else:

            upper_wick = (
                candle_high
                - max(
                    candle_open,
                    candle_close
                )
            )

            rejection_ratio = (
                upper_wick / candle_range
            )

            micro_break = (
                candle_close
                < float(previous_candle["low"])
            )

        confirmation_ok = (
            rejection_ratio >= 0.30
            or micro_break
        )

        return (
            confirmation_ok,
            rejection_ratio,
            micro_break
        )

    # =========================================================
    # RECHERCHE DU BOS + RETEST
    # =========================================================

    # On part du BOS le plus récent.
    # Cela évite de prendre un ancien niveau alors qu'un
    # nouveau BOS vient d'apparaître.

    for bos_offset in range(
        1,
        min(MAX_BOS_AGE, len(data) - 2) + 1
    ):

        bos_index = len(data) - 1 - bos_offset

        if bos_index < 2:
            continue

        bos_candle = data.iloc[bos_index]
        before_bos = data.iloc[bos_index - 1]

        # =====================================================
        # BUY
        # =====================================================

        if direction == "BUY":

            # -------------------------------------------------
            # Dernier swing high disponible avant le BOS
            # -------------------------------------------------

            valid_highs = [
                h for h in swing_highs
                if h.get("index", -1) < bos_index
            ]

            if not valid_highs:
                continue

            swing_level = float(
                valid_highs[-1]["price"]
            )

            # -------------------------------------------------
            # BOS HAUSSIER
            # -------------------------------------------------

            bos_confirmed = (
                float(bos_candle["close"])
                > swing_level
                and
                float(before_bos["close"])
                <= swing_level
            )

            if not bos_confirmed:
                continue

            # -------------------------------------------------
            # RETEST APRÈS LE BOS
            # -------------------------------------------------

            retest_found = False

            for retest_index in range(
                bos_index + 1,
                len(data)
            ):

                retest_candle = data.iloc[
                    retest_index
                ]

                retest_low = float(
                    retest_candle["low"]
                )

                retest_close = float(
                    retest_candle["close"]
                )

                # Le prix revient sur le niveau cassé.
                touched_level = (
                    retest_low
                    <= swing_level + retest_tolerance
                )

                # On ne veut pas une cassure profonde
                # qui invaliderait le BOS.
                held_level = (
                    retest_close
                    >= swing_level - retest_tolerance
                )

                if not touched_level or not held_level:
                    continue

                retest_found = True

                # ---------------------------------------------
                # Confirmation du retest
                # ---------------------------------------------

                previous_retest = (
                    data.iloc[retest_index - 1]
                )

                (
                    confirmation_ok,
                    rejection_ratio,
                    micro_break
                ) = get_confirmation(
                    retest_candle,
                    previous_retest,
                    "BUY"
                )

                if not confirmation_ok:
                    continue

                # ---------------------------------------------
                # Le retest doit être récent.
                # ---------------------------------------------

                bars_since_retest = (
                    len(data) - 1 - retest_index
                )

                if bars_since_retest > 2:
                    continue

                current_price = float(
                    data.iloc[-1]["close"]
                )

                return {
                    "type": "BOS_RETEST",
                    "direction": "BUY",
                    "entry_level": swing_level,
                    "bos_level": swing_level,
                    "bos_index": bos_index,
                    "retest_index": retest_index,
                    "confirmation": (
                        "rejection"
                        if rejection_ratio >= 0.30
                        else "micro_break"
                    ),
                    "strength": max(
                        rejection_ratio,
                        1.0 if micro_break else 0.0
                    ),
                    "distance_atr": (
                        abs(
                            current_price
                            - swing_level
                        ) / atr
                    ),
                }

        # =====================================================
        # SELL
        # =====================================================

        else:

            # -------------------------------------------------
            # Dernier swing low disponible avant le BOS
            # -------------------------------------------------

            valid_lows = [
                l for l in swing_lows
                if l.get("index", -1) < bos_index
            ]

            if not valid_lows:
                continue

            swing_level = float(
                valid_lows[-1]["price"]
            )

            # -------------------------------------------------
            # BOS BAISSIER
            # -------------------------------------------------

            bos_confirmed = (
                float(bos_candle["close"])
                < swing_level
                and
                float(before_bos["close"])
                >= swing_level
            )

            if not bos_confirmed:
                continue

            # -------------------------------------------------
            # RETEST APRÈS LE BOS
            # -------------------------------------------------

            retest_found = False

            for retest_index in range(
                bos_index + 1,
                len(data)
            ):

                retest_candle = data.iloc[
                    retest_index
                ]

                retest_high = float(
                    retest_candle["high"]
                )

                retest_close = float(
                    retest_candle["close"]
                )

                # Retour sur le niveau cassé.
                touched_level = (
                    retest_high
                    >= swing_level - retest_tolerance
                )

                # Le prix reste sous le niveau.
                held_level = (
                    retest_close
                    <= swing_level + retest_tolerance
                )

                if not touched_level or not held_level:
                    continue

                retest_found = True

                # ---------------------------------------------
                # Confirmation
                # ---------------------------------------------

                previous_retest = (
                    data.iloc[retest_index - 1]
                )

                (
                    confirmation_ok,
                    rejection_ratio,
                    micro_break
                ) = get_confirmation(
                    retest_candle,
                    previous_retest,
                    "SELL"
                )

                if not confirmation_ok:
                    continue

                # ---------------------------------------------
                # Retest récent
                # ---------------------------------------------

                bars_since_retest = (
                    len(data) - 1 - retest_index
                )

                if bars_since_retest > 2:
                    continue

                current_price = float(
                    data.iloc[-1]["close"]
                )

                return {
                    "type": "BOS_RETEST",
                    "direction": "SELL",
                    "entry_level": swing_level,
                    "bos_level": swing_level,
                    "bos_index": bos_index,
                    "retest_index": retest_index,
                    "confirmation": (
                        "rejection"
                        if rejection_ratio >= 0.30
                        else "micro_break"
                    ),
                    "strength": max(
                        rejection_ratio,
                        1.0 if micro_break else 0.0
                    ),
                    "distance_atr": (
                        abs(
                            current_price
                            - swing_level
                        ) / atr
                    ),
                }

    return None
def get_confirmation_signal(
    df_m15: pd.DataFrame,
    direction: str
) -> Tuple[bool, str]:
    """
    Confirmation M15.

    Une confirmation valide peut être :
    - un rejet significatif de la zone
    OU
    - un micro-break de la bougie précédente.

    Le but est d'éviter de rater un setup simplement parce
    que les deux confirmations ne se produisent pas simultanément.
    """

    if len(df_m15) < 3:
        return False, "données insuffisantes"

    last = df_m15.iloc[-1]
    prev = df_m15.iloc[-2]

    total = last["high"] - last["low"]

    if total <= 0:
        return False, "range nul"

    upper_wick = last["high"] - max(last["open"], last["close"])
    lower_wick = min(last["open"], last["close"]) - last["low"]

    # =============================================================
    # BUY
    # =============================================================

    if direction == "BUY":

        rejection_ratio = lower_wick / total
        rejection = rejection_ratio > 0.40

        micro_break = last["close"] > prev["high"]

        if rejection and micro_break:
            return True, (
                f"rejet + micro-break OK "
                f"(rejet={rejection_ratio:.2f})"
            )

        if rejection:
            return True, (
                f"rejet OK "
                f"(rejet={rejection_ratio:.2f}, sans micro-break)"
            )

        if micro_break:
            return True, (
                f"micro-break OK "
                f"(sans rejet significatif)"
            )

        return False, (
            f"pas de confirmation "
            f"(rejet={rejection_ratio:.2f}, "
            f"micro_break=False)"
        )

    # =============================================================
    # SELL
    # =============================================================

    elif direction == "SELL":

        rejection_ratio = upper_wick / total
        rejection = rejection_ratio > 0.40

        micro_break = last["close"] < prev["low"]

        if rejection and micro_break:
            return True, (
                f"rejet + micro-break OK "
                f"(rejet={rejection_ratio:.2f})"
            )

        if rejection:
            return True, (
                f"rejet OK "
                f"(rejet={rejection_ratio:.2f}, sans micro-break)"
            )

        if micro_break:
            return True, (
                f"micro-break OK "
                f"(sans rejet significatif)"
            )

        return False, (
            f"pas de confirmation "
            f"(rejet={rejection_ratio:.2f}, "
            f"micro_break=False)"
        )

    return False, f"direction inconnue: {direction}"

def calculate_sl_tp_structural(
    df_m15: pd.DataFrame,
    direction: str,
    entry: float,
    pair: str
) -> Tuple[float, float, float]:
    """
    Calcule un SL structurel et un TP à 2R.

    Règles :
    - BUY  : SL sous le dernier swing low M15.
    - SELL : SL au-dessus du dernier swing high M15.
    - Buffer de sécurité de 5 pips.
    - Fallback ATR 1.5x si aucun swing exploitable.
    - SL structurel maximum = 2 ATR.
    - Si le SL structurel dépasse 2 ATR : setup rejeté.
      On ne déplace PAS artificiellement le SL.
    - Distance SL minimum = 10 pips.
    - TP = exactement 2R après arrondi.
    - Garantie finale RR >= 2.0.
    """

    pair = pair.upper()
    direction = direction.upper()
    entry = float(entry)

    if direction not in ("BUY", "SELL"):
        raise ValueError(
            f"Direction inconnue: {direction}"
        )

    if df_m15 is None or len(df_m15) < 20:
        raise ValueError(
            f"Données M15 insuffisantes pour {pair}"
        )

    # ============================================================
    # 1. INDICATEURS
    # ============================================================

    highs, lows = detect_swing_points(
        df_m15,
        5
    )

    pip = float(
        get_pip_value(pair)
    )

    atr = calculate_atr(df_m15)

    if atr is None or atr <= 0:
        atr = pip * 10

    atr = float(atr)

    if pip <= 0:
        raise ValueError(
            f"Valeur pip invalide pour {pair}"
        )

    # ============================================================
    # 2. PARAMÈTRES
    # ============================================================

    SL_BUFFER_PIPS = 5
    MIN_SL_PIPS = 10
    MAX_SL_ATR = 2.0
    FALLBACK_SL_ATR = 1.5
    TARGET_RR = 2.0

    sl_buffer = SL_BUFFER_PIPS * pip
    min_sl_distance = MIN_SL_PIPS * pip
    max_sl_distance = atr * MAX_SL_ATR

    # ============================================================
    # 3. SL STRUCTUREL
    # ============================================================

    if direction == "BUY":

        # Dernier swing low réellement sous l'entrée.
        valid_lows = [
            low for low in lows
            if float(low["price"]) < entry
        ]

        if valid_lows:

            last_swing_low = float(
                valid_lows[-1]["price"]
            )

            # SL sous le swing + buffer.
            sl = (
                last_swing_low
                - sl_buffer
            )

            sl_source = (
                f"SWING_LOW "
                f"{last_swing_low:.5f}"
            )

        else:

            # Aucun swing exploitable.
            sl = (
                entry
                - atr * FALLBACK_SL_ATR
            )

            sl_source = "ATR_FALLBACK"

    else:  # SELL

        # Dernier swing high réellement au-dessus
        # de l'entrée.
        valid_highs = [
            high for high in highs
            if float(high["price"]) > entry
        ]

        if valid_highs:

            last_swing_high = float(
                valid_highs[-1]["price"]
            )

            # SL au-dessus du swing + buffer.
            sl = (
                last_swing_high
                + sl_buffer
            )

            sl_source = (
                f"SWING_HIGH "
                f"{last_swing_high:.5f}"
            )

        else:

            # Aucun swing exploitable.
            sl = (
                entry
                + atr * FALLBACK_SL_ATR
            )

            sl_source = "ATR_FALLBACK"

    # ============================================================
    # 4. PROTECTION : SL DU BON CÔTÉ
    # ============================================================

    if direction == "BUY" and sl >= entry:

        sl = (
            entry
            - max(
                min_sl_distance,
                atr * FALLBACK_SL_ATR
            )
        )

        sl_source = "ATR_FALLBACK_INVALID_STRUCTURE"

    elif direction == "SELL" and sl <= entry:

        sl = (
            entry
            + max(
                min_sl_distance,
                atr * FALLBACK_SL_ATR
            )
        )

        sl_source = "ATR_FALLBACK_INVALID_STRUCTURE"

    # ============================================================
    # 5. RISQUE AVANT ARRONDI
    # ============================================================

    risk_before_rounding = abs(
        entry - sl
    )

    if risk_before_rounding <= 0:
        raise ValueError(
            f"Risque nul {pair}"
        )

    # ============================================================
    # 6. SL MAXIMUM = 2 ATR
    #
    # IMPORTANT :
    # On REJETTE si la structure est trop éloignée.
    # On ne coupe pas artificiellement le SL.
    # ============================================================

    if risk_before_rounding > max_sl_distance:

        logger.debug(
            f"[SL] {pair} | "
            f"{direction} | "
            f"SL structurel trop large | "
            f"risk={risk_before_rounding:.5f} | "
            f"max={max_sl_distance:.5f} | "
            f"atr={atr:.5f} | "
            f"source={sl_source} | "
            f"→ SETUP REJECTED"
        )

        raise ValueError(
            f"SL structurel > {MAX_SL_ATR:.1f} ATR "
            f"(risk={risk_before_rounding:.5f}, "
            f"max={max_sl_distance:.5f})"
        )

    # ============================================================
    # 7. SL MINIMUM
    # ============================================================

    if risk_before_rounding < min_sl_distance:

        logger.debug(
            f"[SL] {pair} | "
            f"{direction} | "
            f"SL structurel trop proche | "
            f"risk={risk_before_rounding:.5f} | "
            f"min={min_sl_distance:.5f} | "
            f"→ ajustement minimum"
        )

        if direction == "BUY":

            sl = (
                entry
                - min_sl_distance
            )

        else:

            sl = (
                entry
                + min_sl_distance
            )

    # ============================================================
    # 8. ARRONDI DU SL
    # ============================================================

    sl = float(
        round_price(
            pair,
            sl
        )
    )

    # ============================================================
    # 9. VÉRIFICATION APRÈS ARRONDI
    # ============================================================

    if direction == "BUY" and sl >= entry:

        sl = float(
            round_price(
                pair,
                entry - min_sl_distance
            )
        )

    elif direction == "SELL" and sl <= entry:

        sl = float(
            round_price(
                pair,
                entry + min_sl_distance
            )
        )

    risk = abs(
        entry - sl
    )

    if risk <= 0:
        raise ValueError(
            f"Risk nul après arrondi {pair}"
        )

    # Vérification finale du plafond 2 ATR
    if risk > max_sl_distance:

        raise ValueError(
            f"SL après arrondi > "
            f"{MAX_SL_ATR:.1f} ATR "
            f"(risk={risk:.5f}, "
            f"max={max_sl_distance:.5f})"
        )

    # ============================================================
    # 10. TP = 2R
    # ============================================================

    if direction == "BUY":

        tp = (
            entry
            + risk * TARGET_RR
        )

    else:

        tp = (
            entry
            - risk * TARGET_RR
        )

    tp = float(
        round_price(
            pair,
            tp
        )
    )

    # ============================================================
    # 11. RR FINAL APRÈS ARRONDI
    # ============================================================

    final_risk = abs(
        entry - sl
    )

    final_reward = abs(
        tp - entry
    )

    if final_risk <= 0:
        raise ValueError(
            f"Risque final nul {pair}"
        )

    rr = (
        final_reward
        / final_risk
    )

    # ============================================================
    # 12. GARANTIE RR >= 2
    # ============================================================

    if rr < TARGET_RR:

        if direction == "BUY":

            tp = float(
                round_price(
                    pair,
                    entry
                    + final_risk * 2.01
                )
            )

        else:

            tp = float(
                round_price(
                    pair,
                    entry
                    - final_risk * 2.01
                )
            )

        final_reward = abs(
            tp - entry
        )

        rr = (
            final_reward
            / final_risk
        )

    # ============================================================
    # 13. GARANTIE FINALE
    # ============================================================

    if rr < TARGET_RR:

        raise ValueError(
            f"RR final insuffisant après arrondi "
            f"(RR={rr:.3f})"
        )

    # ============================================================
    # 14. LOG
    # ============================================================

    logger.debug(
        f"[SLTP] {pair} | "
        f"{direction} | "
        f"ENTRY={entry:.5f} | "
        f"SL={sl:.5f} | "
        f"TP={tp:.5f} | "
        f"RISK={final_risk:.5f} | "
        f"ATR={atr:.5f} | "
        f"SL_ATR={final_risk / atr:.2f} | "
        f"RR={rr:.3f} | "
        f"SOURCE={sl_source}"
    )

    return (
        sl,
        tp,
        final_risk
    )
def has_enough_room_to_tp(
    df_h1: pd.DataFrame,
    direction: str,
    entry: float,
    tp: float
) -> bool:
    """
    Vérifie que le TP à 2R dispose d'un espace structurel suffisant.

    On ne bloque pas un trade simplement parce qu'un swing H1
    historique se trouve sur le chemin.

    BUY :
        on recherche uniquement les résistances H1 significatives
        proches du TP.

    SELL :
        on recherche uniquement les supports H1 significatifs
        proches du TP.

    Le TP est considéré bloqué uniquement si un swing se trouve
    dans les 15 derniers pourcents du trajet vers le TP.
    """

    if df_h1 is None or len(df_h1) < 20:
        return True

    try:
        highs, lows = detect_swing_points(df_h1, 5)
    except Exception as e:
        logger.warning(
            f"[TP_SPACE] erreur swings H1: {e}"
        )
        return True

    total_distance = abs(tp - entry)

    if total_distance <= 0:
        return False

    # Zone réellement critique autour du TP.
    # On laisse le prix traverser les petits swings intermédiaires.
    critical_zone = total_distance * 0.15

    # =========================================================
    # BUY
    # =========================================================

    if direction == "BUY":

        for h in highs:

            level = float(h["price"])

            if not (entry < level < tp):
                continue

            distance_to_tp = tp - level

            # Seulement un swing très proche du TP bloque.
            if distance_to_tp <= critical_zone:

                logger.debug(
                    f"[TP_SPACE] BUY | "
                    f"résistance H1 proche du TP | "
                    f"level={level:.5f} | "
                    f"TP={tp:.5f} | "
                    f"distance={distance_to_tp:.5f}"
                )

                return False

    # =========================================================
    # SELL
    # =========================================================

    elif direction == "SELL":

        for l in lows:

            level = float(l["price"])

            if not (tp < level < entry):
                continue

            distance_to_tp = level - tp

            # Seulement un swing très proche du TP bloque.
            if distance_to_tp <= critical_zone:

                logger.debug(
                    f"[TP_SPACE] SELL | "
                    f"support H1 proche du TP | "
                    f"level={level:.5f} | "
                    f"TP={tp:.5f} | "
                    f"distance={distance_to_tp:.5f}"
                )

                return False

    return True
def evaluate_setup(
    pair: str,
    direction: str,
    entry: dict,
    df_m15: pd.DataFrame,
    df_h1: pd.DataFrame,
    current_price: float
) -> dict:

    # =========================================================
    # TYPES DE SETUPS AUTORISÉS
    # =========================================================

    setup_type = entry.get("type")

    if setup_type not in (
        "FVG_RETEST",
        "WICK_REJECTION",
        "BOS_RETEST",
    ):
        return {
            "passed": False,
            "reason": (
                f"type non autorisé: "
                f"{setup_type}"
            ),
        }

    # =========================================================
    # ENTRY LEVEL
    # =========================================================

    try:
        entry_level = float(
            entry["entry_level"]
        )
    except Exception:
        return {
            "passed": False,
            "reason": "entry_level invalide",
        }

    # =========================================================
    # ATR
    # =========================================================

    atr_price = calculate_atr(df_m15)

    if atr_price is None or atr_price <= 0:
        return {
            "passed": False,
            "reason": "ATR invalide",
        }

    # =========================================================
    # DISTANCE MAXIMALE
    # =========================================================

    distance_ratio = (
        abs(current_price - entry_level)
        / atr_price
    )

    if distance_ratio > 2.0:
        return {
            "passed": False,
            "reason": (
                f"prix hors zone "
                f"(target={entry_level:.5f}, "
                f"price={current_price:.5f}, "
                f"dist={distance_ratio:.2f}ATR, "
                f"max=2.0ATR)"
            ),
        }

    # =========================================================
    # CONFIRMATION
    #
    # BOS_RETEST possède déjà sa confirmation.
    # Pour FVG/WICK on utilise la confirmation classique.
    # =========================================================

    if setup_type == "BOS_RETEST":

        confirmation_ok = True

        confirmation_msg = (
            f"BOS_RETEST "
            f"{entry.get('confirmation', 'OK')}"
        )

    else:

        confirmation_ok, confirmation_msg = (
            get_confirmation_signal(
                df_m15,
                direction
            )
        )

        if not confirmation_ok:

            last = df_m15.iloc[-1]

            total = (
                float(last["high"])
                - float(last["low"])
            )

            if total <= 0:
                rejection_ratio = 0.0

            elif direction == "BUY":

                rejection_ratio = (
                    min(
                        float(last["open"]),
                        float(last["close"])
                    )
                    - float(last["low"])
                ) / total

            else:

                rejection_ratio = (
                    float(last["high"])
                    - max(
                        float(last["open"]),
                        float(last["close"])
                    )
                ) / total

            prev = df_m15.iloc[-2]

            micro_break = (
                float(last["close"])
                > float(prev["high"])
                if direction == "BUY"
                else
                float(last["close"])
                < float(prev["low"])
            )

            return {
                "passed": False,
                "reason": (
                    f"confirmation: "
                    f"{confirmation_msg} "
                    f"(rejet={rejection_ratio:.2f}, "
                    f"micro_break={micro_break})"
                ),
            }

    # =========================================================
    # SL / TP STRUCTURELS
    # =========================================================

    sl, tp, risk = calculate_sl_tp_structural(
        df_m15,
        direction,
        entry_level,
        pair
    )

    if sl is None or tp is None:
        return {
            "passed": False,
            "reason": "SL/TP impossible à calculer",
        }

    sl = float(sl)
    tp = float(tp)
    risk = float(risk)

    # =========================================================
    # DISTANCE MINIMALE SL
    # =========================================================

    pip = (
        0.01
        if "JPY" in pair
        else 0.0001
    )

    min_sl_distance = pip * 10

    if abs(entry_level - sl) < min_sl_distance:

        return {
            "passed": False,
            "reason": (
                f"SL trop proche "
                f"({abs(entry_level - sl):.5f} "
                f"< {min_sl_distance:.5f})"
            ),
        }

    # =========================================================
    # RR THÉORIQUE
    # =========================================================

    risk_distance = abs(
        entry_level - sl
    )

    reward_distance = abs(
        tp - entry_level
    )

    if risk_distance <= 0:
        return {
            "passed": False,
            "reason": "risque nul",
        }

    rr = (
        reward_distance
        / risk_distance
    )

    # =========================================================
    # VÉRIFICATION ESPACE H1
    # =========================================================

    if not has_enough_room_to_tp(
        df_h1,
        direction,
        entry_level,
        tp
    ):
        return {
            "passed": False,
            "reason": (
                f"RR réel impossible "
                f"(swing H1 bloque le TP "
                f"à {tp:.5f})"
            ),
        }

    # =========================================================
    # RR MINIMUM = 2R
    # =========================================================

    if rr < 2.0:

        return {
            "passed": False,
            "reason": (
                f"RR={rr:.3f} < 2.0 "
                f"(SL={sl:.5f}, "
                f"TP={tp:.5f}, "
                f"entry={entry_level:.5f})"
            ),
        }

    # =========================================================
    # SUCCÈS
    # =========================================================

    try:
        adx = calculate_adx(df_h1)
    except Exception:
        adx = 0.0

    try:
        momentum = calculate_momentum(df_m15)
    except Exception:
        momentum = 0.0

    try:
        rsi = get_last_rsi(
            df_m15["close"]
        )
    except Exception:
        rsi = 0.0

    return {
        "passed": True,

        "type": setup_type,

        "direction": direction,

        "entry_level": entry_level,

        "sl": sl,

        "tp": tp,

        "risk": risk,

        "rr": rr,

        "confirmation": confirmation_msg,

        "metrics": {
            "atr": price_to_pips(
                atr_price,
                pair
            ),

            "adx": adx,

            "momentum": momentum,

            "rsi": rsi,

            "session": get_session_label(),
        },
    }
    
def get_session_label() -> str:
    h = datetime.utcnow().hour
    if 7 <= h < 16:
        return "LONDON"
    if 12 <= h < 21:
        return "NY"
    if 21 <= h or h < 7:
        return "ASIA"
    return "OTHER"

def price_to_pips(price_diff: float, pair: str) -> float:
    pip = 0.01 if "JPY" in pair else 0.0001
    return abs(price_diff) / pip

def get_pip_value(pair: str) -> float:
    """
    Taille de pip / unité de prix propre à chaque instrument.

    IMPORTANT :
    - Forex classique : 0.0001
    - JPY : 0.01
    - XAU/USD : 0.01

    Utilise PIP_SIZE_V88 comme source unique.
    """
    pair = pair.upper()

    return float(
        PIP_SIZE_V88.get(
            pair,
            0.01 if "JPY" in pair else 0.0001
        )
    )

# ============================================================
# CLASSE TRADE TRACKER (MFE/MAE)
# ============================================================
class TradeTracker:
    def __init__(self):
        self.trades = {}

    def add_trade(self, trade_id, pair, direction, entry, sl, tp, setup_type, eqs=0):
        self.trades[trade_id] = {
            "pair": pair, "direction": direction, "entry": entry,
            "sl": sl, "tp": tp, "setup_type": setup_type, "eqs": eqs,
            "highest": entry, "lowest": entry, "mfe": 0, "mae": 0,
            "closed": False, "exit_price": None, "exit_r": None
        }

    def update_price(self, trade_id, price):
        if trade_id not in self.trades or self.trades[trade_id]["closed"]:
            return
        t = self.trades[trade_id]
        if t["direction"] == "BUY":
            t["highest"] = max(t["highest"], price)
            t["lowest"] = min(t["lowest"], price)
            mfe = (price - t["entry"]) / get_pip_value(t["pair"])
            mae = (t["entry"] - price) / get_pip_value(t["pair"])
        else:
            t["highest"] = max(t["highest"], price)
            t["lowest"] = min(t["lowest"], price)
            mfe = (t["entry"] - price) / get_pip_value(t["pair"])
            mae = (price - t["entry"]) / get_pip_value(t["pair"])
        t["mfe"] = max(t["mfe"], mfe)
        t["mae"] = min(t["mae"], mae)

    def close_trade(self, trade_id, exit_price, r_multiple):
        if trade_id not in self.trades:
            return
        t = self.trades[trade_id]
        t["closed"] = True
        t["exit_price"] = exit_price
        t["exit_r"] = r_multiple
        logger.info(f"[MFE/MAE] {t['pair']} | MFE={t['mfe']:.1f} | MAE={t['mae']:.1f} | R={r_multiple:.2f}")

    def get_trade(self, trade_id):
        return self.trades.get(trade_id)

# ============================================================
# STATISTIQUES SIMPLIFIÉES
# ============================================================
class TradingStats:
    def __init__(self):
        self.stats = defaultdict(lambda: {"total":0, "accepted":0, "rejected":0, "wins":0, "losses":0, "profit":0, "loss":0})

    def record_signal(self, pair, accepted, reason="", entry=0, sl=0, tp=0, score=0, direction="", metrics=None):
        self.stats[pair]["total"] += 1
        if accepted:
            self.stats[pair]["accepted"] += 1
        else:
            self.stats[pair]["rejected"] += 1

    def record_close(self, trade_id, pair, setup_type, eqs, r, pl, close_price=None, is_estimate=False, trade_info=None):
        if pl > 0:
            self.stats[pair]["wins"] += 1
            self.stats[pair]["profit"] += pl
        elif pl < 0:
            self.stats[pair]["losses"] += 1
            self.stats[pair]["loss"] += abs(pl)

    def log_summary(self):
        logger.info("="*80)
        logger.info("📊 STATISTIQUES GLOBALES")
        for pair, s in self.stats.items():
            total = s["total"]
            accepted = s["accepted"]
            rejected = s["rejected"]
            wins = s["wins"]
            losses = s["losses"]
            wr = f"{wins/(wins+losses)*100:.1f}%" if wins+losses > 0 else "0%"
            pf = f"{s['profit']/s['loss']:.2f}" if s['loss'] > 0 else "∞"
            logger.info(f"{pair:10} | Signaux:{total:3} | Acceptés:{accepted:3} | Rejetés:{rejected:3} | Wins:{wins:3} | Losses:{losses:3} | WR:{wr:>6} | PF:{pf:>6}")
        logger.info("="*80)

stats = TradingStats()
trade_tracker = TradeTracker()
open_trade_details = {}
stagnant_trade_tracker = {}
last_execution_attempt = {}

# ============================================================
# EXÉCUTION ORDRE (VERSION 2R STRICT)
# ============================================================
def execute_trade(pair: str, direction: str, entry_price: float, stop_loss: float, take_profit: float,
                  score: int, entry_type: str, eqs: int, setup_type: str, metrics: dict) -> str | None:
    global last_execution_attempt

    pair = pair.upper()
    direction = direction.upper()

    # ========================================================
    # 1. COOLDOWN
    # ========================================================
    now = time.time()

    if (
        pair in last_execution_attempt
        and now - last_execution_attempt[pair] < EXECUTION_COOLDOWN_SECONDS
    ):
        logger.warning(f"[ORDER] Cooldown actif pour {pair}")
        return None

    last_execution_attempt[pair] = now

    # ========================================================
    # 2. PARAMÈTRES
    # ========================================================
    expected_entry = float(entry_price)
    sl = float(stop_loss)
    tp = float(take_profit)

    risk = abs(expected_entry - sl)
    reward = abs(tp - expected_entry)

    if risk <= 0:
        logger.error(f"[ORDER] {pair} | Risque nul")
        return None

    rr = reward / risk

    # ========================================================
    # 3. RR INITIAL STRICT
    # ========================================================
    if rr < RR_MIN_EXECUTION:
        logger.warning(
            f"[ORDER] {pair} | RR={rr:.2f} < {RR_MIN_EXECUTION} → rejet"
        )
        return None

    # ========================================================
    # 4. VÉRIFICATIONS
    # ========================================================
    if ONE_TRADE_PER_PAIR and has_open_trade(pair):
        logger.info(f"[ORDER] {pair}: trade déjà ouvert")
        return None

    if open_trade_count() >= MAX_TRADES_TOTAL:
        logger.info(f"[ORDER] Limite trades atteinte ({MAX_TRADES_TOTAL})")
        return None

    if is_maintenance_suspended():
        logger.warning(f"[ORDER] {pair} | OANDA maintenance")
        return None

    # ========================================================
    # 5. VÉRIFICATION DU PRIX ACTUEL AVANT MARKET ORDER
    #
    # IMPORTANT :
    # On ne déclenche PAS une MARKET ORDER si le marché
    # est déjà trop éloigné du niveau d'entrée calculé.
    # ========================================================
    pricing = get_price_spread(pair)

    bid = float(pricing.get("bid", 0) or 0)
    ask = float(pricing.get("ask", 0) or 0)

    if direction == "BUY":
        market_entry = ask if ask > 0 else float(pricing.get("mid", 0) or 0)
    else:
        market_entry = bid if bid > 0 else float(pricing.get("mid", 0) or 0)

    if market_entry <= 0:
        logger.warning(
            f"[ORDER] {pair} | Prix marché indisponible → rejet"
        )
        return None

    pip = get_pip_value(pair)

    # Tolérance d'entrée.
    #
    # Forex : 5 pips minimum
    # JPY    : 5 pips
    # XAU    : 20 pips = 0.20
    # Indices : 5 pips selon leur taille configurée
    #
    # On ajoute également une petite marge proportionnelle pour
    # éviter qu'un instrument à prix élevé soit bloqué inutilement.
    if pair == "XAU_USD":
        max_entry_deviation = max(pip * 20, expected_entry * 0.00005)
    elif "JPY" in pair:
        max_entry_deviation = max(pip * 5, expected_entry * 0.00003)
    else:
        max_entry_deviation = max(pip * 5, expected_entry * 0.00003)

    entry_deviation = abs(market_entry - expected_entry)

    logger.info(
        f"[ENTRY_CHECK] {pair} | {direction} | "
        f"TARGET={expected_entry:.5f} | MARKET={market_entry:.5f} | "
        f"DEV={entry_deviation:.5f} | MAX={max_entry_deviation:.5f}"
    )

    if entry_deviation > max_entry_deviation:
        logger.warning(
            f"[ENTRY_REJECT] {pair} | marché trop éloigné du niveau d'entrée | "
            f"target={expected_entry:.5f} | market={market_entry:.5f} | "
            f"écart={entry_deviation:.5f} > max={max_entry_deviation:.5f}"
        )
        return None

    # ========================================================
    # 6. BALANCE / RISK
    # ========================================================
    balance = get_balance()

    if balance <= 0:
        logger.error(f"[ORDER] {pair} | Balance invalide")
        return None

    min_sl_distance = pip * 10

    if risk < min_sl_distance:
        logger.warning(
            f"[ORDER] {pair} | SL trop proche "
            f"({risk:.5f} < {min_sl_distance:.5f}) → rejet"
        )
        return None

    # ========================================================
    # 7. RISK PERCENTAGE
    # ========================================================
    hour = datetime.utcnow().hour
    is_asia = (21 <= hour or hour < 7)
    risk_pct = 0.5 if is_asia else RISK_PERCENTAGE

    units = calculate_units(
        pair,
        expected_entry,
        sl,
        balance,
        risk_pct
    )

    if units <= 0:
        logger.error(f"[ORDER] {pair} | Units invalides: {units}")
        return None

    # ========================================================
    # 8. MARGE
    # ========================================================
    margin_info = calculate_margin(
        pair,
        units,
        expected_entry
    )

    if not margin_info["sufficient"]:
        units = cap_units_by_margin(
            pair,
            units,
            expected_entry,
            balance
        )

        if units <= 0:
            logger.error(
                f"[RISK] {pair} | Marge insuffisante"
            )
            return None

    # ========================================================
    # 9. MARKET ORDER
    # ========================================================
    signed_units = units if direction == "BUY" else -units

    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": pair,
            "units": str(int(signed_units)),
            "positionFill": "DEFAULT",

            "stopLossOnFill": {
                "price": round_price(pair, sl),
                "timeInForce": "GTC"
            },

            "takeProfitOnFill": {
                "price": round_price(pair, tp),
                "timeInForce": "GTC"
            }
        }
    }

    logger.info(
        f"[ORDER_EXPECTED] {pair} | {direction} | "
        f"ENTRY={expected_entry:.5f} | "
        f"SL={sl:.5f} | "
        f"TP={tp:.5f} | "
        f"RR={rr:.2f} | "
        f"UNITS={units}"
    )

    if not EXECUTE_TRADES:
        logger.info("[ORDER] EXECUTE_TRADES=false")
        return "SIMULATION"

    # ========================================================
    # 10. ENVOI OANDA
    # ========================================================
    try:
        api = v88_client()

        r = orders.OrderCreate(
            accountID=OANDA_ACCOUNT_ID,
            data=order_data
        )

        api.request(r)
        resp = r.response

        # ----------------------------------------------------
        # REJET OANDA
        # ----------------------------------------------------
        if resp.get("orderRejectTransaction"):
            reject = resp["orderRejectTransaction"]

            logger.error(
                f"[ORDER] REJECT {pair}: "
                f"{reject.get('rejectReason')}"
            )

            return None

        # ----------------------------------------------------
        # RÉCUPÉRATION DU FILL
        # ----------------------------------------------------
        fill = resp.get("orderFillTransaction", {})

        trade_id = None
        actual_entry = None

        if fill.get("tradeOpened"):
            trade_id = fill["tradeOpened"].get("tradeID")

        if fill.get("price") is not None:
            try:
                actual_entry = float(fill["price"])
            except Exception:
                actual_entry = None

        # ----------------------------------------------------
        # FALLBACK : RECHERCHE DU TRADE OUVERT
        # ----------------------------------------------------
        if not trade_id:

            time.sleep(1)

            open_trades = get_open_trades(
                force_refresh=True
            )

            candidates = []

            for t in open_trades:

                if t.get("instrument") != pair:
                    continue

                current_units = float(
                    t.get("currentUnits", 0)
                )

                t_direction = (
                    "BUY"
                    if current_units > 0
                    else "SELL"
                )

                if t_direction != direction:
                    continue

                t_entry = float(
                    t.get("price", 0)
                )

                if t_entry <= 0:
                    continue

                candidates.append(
                    (
                        abs(t_entry - expected_entry),
                        t
                    )
                )

            if candidates:

                candidates.sort(
                    key=lambda x: x[0]
                )

                _, best_trade = candidates[0]

                trade_id = best_trade.get("id")

                actual_entry = float(
                    best_trade.get("price")
                )

        # ----------------------------------------------------
        # TRADE NON CONFIRMÉ
        # ----------------------------------------------------
        if not trade_id:

            logger.error(
                f"[ORDER] {pair} | "
                f"Trade non confirmé"
            )

            return None

        # ----------------------------------------------------
        # FALLBACK PRIX
        # ----------------------------------------------------
        if actual_entry is None:

            try:

                trade_details = get_trade_details(
                    trade_id
                )

                if trade_details:
                    actual_entry = float(
                        trade_details.get(
                            "price",
                            expected_entry
                        )
                    )

            except Exception as e:

                logger.warning(
                    f"[ORDER] {pair} | "
                    f"Impossible récupérer fill réel: {e}"
                )

        if actual_entry is None:
            actual_entry = expected_entry

        # ====================================================
        # 11. VÉRIFICATION DU FILL RÉEL
        # ====================================================
        fill_deviation = abs(
            actual_entry - expected_entry
        )

        # IMPORTANT :
        # Le prix de fill ne doit pas pouvoir détruire
        # complètement le RR.
        risk_real = abs(
            actual_entry - sl
        )

        reward_real = abs(
            tp - actual_entry
        )

        rr_real = (
            reward_real / risk_real
            if risk_real > 0
            else 0
        )

        slippage_pips = (
            (actual_entry - expected_entry) / pip
        )

        logger.info(
            f"[FILL_REAL] {pair} | "
            f"ID={trade_id} | "
            f"ENTRY_EXPECTED={expected_entry:.5f} | "
            f"ENTRY_FILLED={actual_entry:.5f} | "
            f"SLIPPAGE={slippage_pips:+.2f} pips | "
            f"RR_REAL={rr_real:.2f}"
        )

        # ====================================================
        # 12. PROTECTION ABSOLUE DU RR
        # ====================================================
        if rr_real < RR_MIN_EXECUTION:

            logger.error(
                f"[ORDER_ABORT] {pair} | "
                f"RR réel {rr_real:.2f} < "
                f"{RR_MIN_EXECUTION} après exécution"
            )

            # Le trade existe déjà.
            # On ne laisse surtout pas courir une position
            # avec un RR dégradé.
            try:

                close_data = {
                    "units": "ALL"
                }

                close_request = trades.TradeClose(
                    accountID=OANDA_ACCOUNT_ID,
                    tradeID=str(trade_id),
                    data=close_data
                )

                api.request(close_request)

                logger.error(
                    f"[ORDER_ABORT] {pair} | "
                    f"Trade {trade_id} fermé immédiatement "
                    f"car RR réel insuffisant"
                )

            except Exception as close_error:

                logger.critical(
                    f"[ORDER_ABORT] {pair} | "
                    f"IMPOSSIBLE DE FERMER LE TRADE "
                    f"{trade_id}: {close_error}"
                )

            return None

        # ====================================================
        # 13. LOG SLIPPAGE
        # ====================================================
        if fill_deviation > max_entry_deviation:

            logger.warning(
                f"[SLIPPAGE] {pair} | "
                f"Fill hors tolérance malgré contrôle pré-ordre | "
                f"écart={fill_deviation:.5f} | "
                f"max={max_entry_deviation:.5f}"
            )

        # ====================================================
        # 14. ENREGISTREMENT
        # ====================================================
        trade_tracker.add_trade(
            trade_id,
            pair,
            direction,
            actual_entry,
            sl,
            tp,
            setup_type,
            eqs
        )

        open_trade_details[str(trade_id)] = {
            "entry": actual_entry,
            "sl": sl,
            "tp": tp,
            "direction": direction,
            "setup_type": setup_type,
            "eqs": eqs,
            "pair": pair,
            "units": units,
            **metrics
        }

        logger.info(
            f"[ORDER_CONFIRMED] {pair} | "
            f"{direction} | "
            f"ID={trade_id} | "
            f"ENTRY={actual_entry:.5f} | "
            f"SL={sl:.5f} | "
            f"TP={tp:.5f} | "
            f"RR={rr_real:.2f}"
        )

        return str(trade_id)

    except Exception as e:

        logger.error(
            f"[ORDER] Erreur {pair}: {e}"
        )

        if is_oanda_maintenance(e):
            handle_api_error(e)

        return None

# ============================================================
# GESTION DES POSITIONS (BE / TRAILING)
# ============================================================
def modify_sl(trade_id: str, pair: str, new_sl: float, adjust_tp: bool = False) -> bool:
    try:
        if is_maintenance_suspended():
            return False
        api = v88_client()
        data = {"stopLoss": {"price": round_price(pair, new_sl), "timeInForce": "GTC"}}
        # TP ne doit pas être modifié
        r = trades.TradeCRCDO(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id, data=data)
        api.request(r)
        logger.info(f"[BE] SL modifié pour {trade_id} -> {new_sl:.5f}")
        clear_cache()
        return True
    except Exception as e:
        logger.error(f"[BE] Erreur modif SL {trade_id}: {e}")
        return False

def create_trailing_stop(trade_id: str, pair: str, distance: float) -> bool:
    try:
        if is_maintenance_suspended():
            return False
        api = v88_client()
        data = {"order": {"type": "TRAILING_STOP_LOSS", "tradeID": trade_id, "distance": str(distance), "timeInForce": "GTC"}}
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=data)
        api.request(r)
        logger.info(f"[TSL] Trailing stop créé pour {trade_id}, distance={distance:.5f}")
        clear_cache()
        return True
    except Exception as e:
        logger.error(f"[TSL] Erreur création trailing {trade_id}: {e}")
        return False

def check_breakeven():
    try:
        if is_maintenance_suspended():
            return
        open_trades = get_open_trades()
        logger.info(f"[BE] Scan de {len(open_trades)} trades ouverts")
        for t in open_trades:
            trade_id = str(t.get("id"))
            pair = t.get("instrument")
            direction = "BUY" if float(t.get("currentUnits", 0)) > 0 else "SELL"
            entry = float(t.get("price"))
            current_sl = get_stop_loss(t)
            if current_sl <= 0:
                continue

            current_price = get_current_price(pair)
            if current_price <= 0:
                continue

            trade_tracker.update_price(trade_id, current_price)

            # Récupérer le SL initial
            trade_info = open_trade_details.get(trade_id, {})
            initial_sl = trade_info.get("sl", current_sl)
            if initial_sl <= 0:
                initial_sl = current_sl

            if direction == "BUY":
                profit = current_price - entry
                initial_risk = entry - initial_sl
            else:
                profit = entry - current_price
                initial_risk = current_sl - entry
            if initial_risk <= 0:
                initial_risk = abs(entry - current_sl)
            r = profit / initial_risk if initial_risk > 0 else 0.0

            # BE à 0.55R (ne modifie pas le TP)
            is_already_be = (direction == "BUY" and current_sl >= entry) or (direction == "SELL" and current_sl <= entry)
            if not is_already_be and r >= BASE_BREAKEVEN_TRIGGER_R:
                pip = get_pip_value(pair)
                offset = max(0, pip * 1.0)
                if direction == "BUY":
                    be_sl = entry + offset
                else:
                    be_sl = entry - offset
                if (direction == "BUY" and be_sl > current_sl) or (direction == "SELL" and be_sl < current_sl):
                    if modify_sl(trade_id, pair, be_sl, adjust_tp=False):
                        logger.info(f"[BE] SL déplacé à {be_sl:.5f} pour {trade_id}")
                        current_sl = be_sl

            # Trailing stop (ne modifie pas le TP)
            trade_details = get_trade_details(trade_id)
            if has_trailing_stop(trade_details):
                continue

            if r >= BASE_TRAILING_ACTIVATION_R:
                atr = calculate_atr(get_candles(v88_client(), pair, "M15", 40))
                pip = get_pip_value(pair)
                distance = max(atr * BASE_TRAILING_STOP_DISTANCE_ATR_MULTIPLIER, pip * BASE_TRAILING_STOP_MIN_DISTANCE_PIPS)
                distance = round(distance, PRICE_DECIMALS_V88.get(pair, 5))
                if distance > 0:
                    if create_trailing_stop(trade_id, pair, distance):
                        logger.info(f"[TSL] Trailing activé pour {trade_id}")
    except Exception as e:
        logger.error(f"[BE] Erreur: {e}")

# ============================================================
# SUIVI DES TRADES FERMÉS
# ============================================================
def check_closed_trades():
    try:
        if is_maintenance_suspended():
            return
        current_open = get_open_trades(force_refresh=True)
        open_ids = {str(t.get("id")) for t in current_open}
        for trade_id in list(open_trade_details.keys()):
            trade_id = str(trade_id)
            if trade_id in open_ids:
                continue
            trade_info = open_trade_details.pop(trade_id, None)
            if not trade_info:
                continue
            pair = trade_info.get("pair", "UNKNOWN")
            setup_type = trade_info.get("setup_type", "UNKNOWN")
            eqs = trade_info.get("eqs", 0)
            direction = trade_info.get("direction", "").upper()
            entry = float(trade_info.get("entry", 0))
            sl = float(trade_info.get("sl", 0))
            units = float(trade_info.get("units", 0))
            pip = get_pip_value(pair)
            risk_pips = abs(entry - sl) / pip if sl > 0 else 0

            # Récupérer les détails de fermeture
            close_price = None
            pl = 0.0
            is_estimate = True
            trade_data = get_trade_details(trade_id)
            if trade_data:
                avg_close = trade_data.get("averageClosePrice")
                realized = trade_data.get("realizedPL")
                if avg_close:
                    close_price = float(avg_close)
                    is_estimate = False
                if realized:
                    pl = float(realized)
            if close_price is None or close_price <= 0:
                close_price = get_current_price(pair)
                is_estimate = True
                if direction == "BUY":
                    pl = (close_price - entry) * units
                else:
                    pl = (entry - close_price) * units
            if risk_pips > 0:
                if direction == "BUY":
                    r_multiple = (close_price - entry) / pip / risk_pips
                else:
                    r_multiple = (entry - close_price) / pip / risk_pips
            else:
                r_multiple = 0.0

            logger.info(f"[CLOSE] {pair} | {direction} | R={r_multiple:.2f} | PL={pl:.2f} | {'EST' if is_estimate else 'CONF'}")
            stats.record_close(trade_id, pair, setup_type, eqs, r_multiple, pl, close_price, is_estimate, trade_info)
            trade_tracker.close_trade(trade_id, close_price, r_multiple)
    except Exception as e:
        logger.error(f"[CLOSE] Erreur: {e}")

# ============================================================
# FONCTIONS DE DÉDUPLICATION ET FILTRES
# ============================================================
def strict_keep_best_per_direction(scored_entries, min_score_gap=5):
    # Simplifié : garde le meilleur score par direction
    best = {}
    for item in scored_entries:
        direction = item["entry"].get("direction", "").upper()
        score = item["confidence"].get("score", 0)  # On utilisera le RR comme score
        if direction not in best or score > best[direction]["score"]:
            best[direction] = {"entry": item["entry"], "score": score}
    return [{"entry": v["entry"], "confidence": {"score": v["score"]}} for v in best.values()]

def is_signal_sent_recently(pair, direction, price, zone_start, zone_end):
    return False  # Simplifié

def mark_signal_sent(pair, direction, entry_level, zone_start, zone_end):
    pass

# ============================================================
# FONCTION PRINCIPALE DE SCAN
# ============================================================
def advanced_main():
    try:
        api = v88_client()
        logger.info("✅ API OANDA initialisée")
        logger.info("🎯 MODE 2R STRICT : Biais → Retracement → Confirmation → 2R")
    except Exception as e:
        logger.error(f"❌ Échec API: {e}")
        return

    # Petite fonction interne pour diagnostiquer la structure HTF
    def _struct_brief(df, label):
        highs, lows = detect_swing_points(df, 5)
        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1]["price"] > highs[-2]["price"]
            hl = lows[-1]["price"] > lows[-2]["price"]
            lh = highs[-1]["price"] < highs[-2]["price"]
            ll = lows[-1]["price"] < lows[-2]["price"]
            if hh and hl: return f"{label} BULLISH"
            if lh and ll: return f"{label} BEARISH"
            return f"{label} MIXED (HH={hh},HL={hl},LH={lh},LL={ll})"
        return f"{label} INDETERMINE"

    for pair in PAIR_LIST:
        if has_open_trade(pair):
            logger.info(f"[INFO] {pair}: trade déjà ouvert")
            continue

        try:
            df_h4 = get_candles(api, pair, GRANULARITY_H4, 300)
            df_h1 = get_candles(api, pair, GRANULARITY_H1, 200)
            df_m15 = get_candles(api, pair, GRANULARITY_M15, 250)
            if any(df.empty for df in [df_h4, df_h1, df_m15]):
                continue

            current_price = float(df_m15["close"].iloc[-1])
            bias = get_directional_bias(df_h4, df_h1)
            
            # --- LOG DIAGNOSTIC STRUCTURE NEUTRAL (enrichi) ---
            if bias == "NEUTRAL":
                logger.info(f"{pair} | BIAS_DIAG: H4={_struct_brief(df_h4,'H4')} | H1={_struct_brief(df_h1,'H1')} | -> NEUTRAL")
                continue

            setups = detect_setups(pair, df_m15, df_h1, bias)
            
            # --- LOG DES SETUPS DÉTECTÉS (enrichi) ---
            if not setups:
                logger.info(f"{pair} | Aucun setup {bias} détecté")
                continue
            else:
                setup_summary = ", ".join([f"{s.get('type')}@{round(s.get('entry_level',0),5)}" for s in setups[:3]])
                logger.info(f"{pair} | SETUPS_FOUND: {len(setups)} | Types: {setup_summary}{' ...' if len(setups)>3 else ''}")

            valid_trades = []
            for entry in setups:
                result = evaluate_setup(pair, bias, entry, df_m15, df_h1, current_price)
                if result["passed"]:
                    valid_trades.append({
                        "entry": entry,
                        "sl": result["sl"],
                        "tp": result["tp"],
                        "risk": result["risk"],
                        "rr": result["rr"],
                        "metrics": result["metrics"]
                    })
                else:
                    logger.info(f"[REJECT] {pair} {bias} {entry.get('type')} : {result['reason']}")

            if not valid_trades:
                continue

            # Sélection du meilleur RR
            valid_trades.sort(key=lambda x: x["rr"], reverse=True)
            best = valid_trades[0]

            entry_level = float(best["entry"]["entry_level"])
            stop_loss = best["sl"]
            take_profit = best["tp"]
            rr = best["rr"]
            metrics = best["metrics"]

            logger.info(f"[TRADE] {pair} {bias} @{entry_level:.5f} | SL={stop_loss:.5f} | TP={take_profit:.5f} | RR={rr:.2f}")

            trade_id = execute_trade(
                pair=pair, direction=bias, entry_price=entry_level,
                stop_loss=stop_loss, take_profit=take_profit,
                score=0, entry_type=best["entry"]["type"], eqs=0,
                setup_type=best["entry"]["type"], metrics=metrics
            )

            if trade_id:
                logger.info(f"✅ {pair} trade exécuté (ID {trade_id})")
                send_telegram(pair, bias, entry_level, stop_loss, take_profit, rr, best["entry"]["type"])
            else:
                logger.error(f"❌ {pair} échec exécution")

        except Exception as e:
            logger.error(f"💥 Erreur sur {pair}: {e}")

    stats.log_summary()
# ============================================================
# TELEGRAM (optionnel)
# ============================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(pair, direction, entry, sl, tp, rr, setup_type):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        msg = f"{'🟢' if direction=='BUY' else '🔴'} TRADE\nPair: {pair}\nDirection: {direction}\nEntry: {entry:.5f}\nSL: {sl:.5f}\nTP: {tp:.5f}\nRR: {rr:.2f}\nSetup: {setup_type}"
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except:
        pass

# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
if __name__ == "__main__":
    logger.info("🚀 Démarrage du Bot 2R Strict - Version Optimisée")
    logger.info("✅ SL structurel | TP = 2R (immuable) | RR ≥ 2.0 avant ordre")
    logger.info("✅ Structure H1 assouplie (2/3) + tolérance retracement H4 fort")
    logger.info("✅ Distance max 2.0 ATR | Confirmation rejet OU micro-break")
    logger.info("✅ SL limité à 2×ATR | has_enough_room_to_tp() assoupli")
    logger.info(f"✅ MAX TRADES: {MAX_TRADES_TOTAL}")
    if DEMO_MODE:
        logger.info("🔬 MODE DEMO ACTIVÉ")
    if DEBUG_MODE:
        logger.info("🔍 MODE DEBUG ACTIVÉ")

    last_signal_scan = time.time()
    SIGNAL_SCAN_INTERVAL = 900  # 15 min
    FAST_LOOP_INTERVAL = 30

    while True:
        try:
            if is_maintenance_suspended():
                logger.warning("⏳ Maintenance OANDA, pause 10s")
                time.sleep(10)
                continue

            clear_cache()
            current_open = open_trade_count()
            logger.info(f"[SCAN] Trades ouverts: {current_open}/{MAX_TRADES_TOTAL}")

            check_closed_trades()
            check_breakeven()

            if time.time() - last_signal_scan >= SIGNAL_SCAN_INTERVAL:
                logger.info("⏰ Scan des signaux")
                last_signal_scan = time.time()
                if current_open < MAX_TRADES_TOTAL:
                    # --- AJOUT : vérification du marché ouvert ---
                    now_utc = datetime.utcnow()
                    if not is_market_open(now_utc):
                        logger.info(f"Marché fermé ({now_utc.strftime('%A %H:%M')} UTC) → pas de scan")
                    else:
                        advanced_main()
                else:
                    logger.info("Limite trades atteinte")

            time.sleep(FAST_LOOP_INTERVAL)

        except KeyboardInterrupt:
            logger.info("🛑 Arrêt demandé")
            break
        except Exception as e:
            logger.error(f"💥 Erreur critique: {e}")
            traceback.print_exc()
            time.sleep(30)
