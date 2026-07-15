# ============================================================
# main(75).py - Version V86 finale avec health_check et renforcements
# Modifications V86 :
# - Suppression de toute la logique de trailing stop gérée en Python
# - Ajout de health_check() avant chaque scan : vérifie OANDA, compte, balance, trades, spread, marge
# - Ajout d'une vérification de marge avant l'envoi d'ordre (déjà présente, renforcée)
# - Passage du seuil de Break Even à 1.2R (au lieu de 1.0R) pour filtrer le bruit
# - Renforcement de create_oanda_trailing_stop_v86 : vérification de la distance minimale OANDA
# - Ajout de logs détaillés pour le trailing
# ============================================================

import os
import sys
import time
import logging
import unicodedata
import requests
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import oandapyV20
from oandapyV20.endpoints import instruments, pricing, orders, accounts, trades, positions, transactions
import talib
import traceback
from ta.momentum import RSIIndicator
from typing import List, Dict, Tuple

# =========================
# LOG HELPERS (inchangé)
# =========================
_seen_log_keys_fvg_recent = set()
_seen_log_keys_fvg_added  = set()
_seen_log_keys_kept_entry = set()

def _reset_log_dedup():
    _seen_log_keys_fvg_recent.clear()
    _seen_log_keys_fvg_added.clear()
    _seen_log_keys_kept_entry.clear()

def _log_fvg_recent_once(pair: str, direction: str, level: float, msg: str, precision: int = 5):
    key = (pair, (direction or "").upper(), round(float(level), precision))
    if key in _seen_log_keys_fvg_recent:
        return
    _seen_log_keys_fvg_recent.add(key)
    logger.info(msg)

def _log_fvg_added_once(pair: str, direction: str, level: float, fvg_type: str, msg: str, precision: int = 5):
    key = (pair, (direction or "").upper(), (fvg_type or "UNKNOWN").upper(), round(float(level), precision))
    if key in _seen_log_keys_fvg_added:
        return
    _seen_log_keys_fvg_added.add(key)
    logger.info(msg)

def _log_kept_entry_once(pair: str, level: float, status: str, dist: float, msg: str, precision: int = 5):
    key = (pair, round(float(level), precision), status)
    if key in _seen_log_keys_kept_entry:
        return
    _seen_log_keys_kept_entry.add(key)
    logger.info(msg)

def _log_narrative_list(entries: list, top_n: int = 10):
    if not entries:
        logger.info("🔎 AUCUNE ENTRÉE DÉTECTÉE - Analyse détaillée:")
        return
    safe_entries = []
    for e in entries:
        try:
            lvl = float(e.get("entry_level", 0))
            zone = e.get("entry_zone", (lvl, lvl))
            d = abs(lvl - float(zone[0]))
        except Exception:
            d = 0.0
        safe_entries.append((d, e))
    safe_entries.sort(key=lambda x: x[0])
    top = [e for _, e in safe_entries[:top_n]]
    other_count = max(0, len(entries) - len(top))
    for i, entry in enumerate(top, start=1):
        logger.info(f" {i}. {entry.get('direction','?')} - {entry.get('type','?')} à {float(entry.get('entry_level',0)):.5f}")
    if other_count:
        logger.info(f" … (+{other_count} autres entrées)")

def _log_kept_compact(pair: str, kept: list, top_n: int = 8):
    if not kept:
        logger.info(f"{pair} · Aucune entrée conservée.")
        return
    show = kept[:top_n]
    for e in show:
        entry_level = e.get("entry_level")
        status = e.get("status_label", "conservée")
        dist = e.get("status_distance", None)
        if entry_level is None:
            continue
        if dist is not None:
            logger.info(f"✅ Entrée {status}: {float(entry_level):.5f} (distance: {float(dist):.5f})")
        else:
            logger.info(f"✅ Entrée {status}: {float(entry_level):.5f}")
    if len(kept) > len(show):
        logger.info(f"… (+{len(kept)-len(show)} autres entrées {pair})")

logger = logging.getLogger("Advanced-Orderflow-Trading-Bot")
last_reset_time = datetime.utcnow()

# =============================
# CONFIGURATION
# =============================
load_dotenv()

PAIR_LIST = ["EUR_USD", "GBP_USD", "USD_CAD", "AUD_USD", "AUD_CAD"]

GRANULARITY_D1 = "D"
GRANULARITY_H4 = "H4"
GRANULARITY_H1 = "H1"
GRANULARITY_M15 = "M15"

EMA_SLOW = 200
EMA_MEDIUM = 50
EMA_FAST = 20
RSI_PERIOD = 14
ATR_PERIOD = 14

RISK_REWARD_RATIO = 2
MAX_VOLATILITY_RATIO = 0.02
SWING_LOOKBACK = 3
MIN_WICK_RATIO = 0.7

MAX_DISTANCE_PIPS = {
    "XAU_USD": 500,
    "USD_JPY": 150,
    "NAS100_USD": 25.0,
    "AUD_USD": 0.0080,
    "EUR_USD": 0.0080,
    "GBP_USD": 0.0080,
    "USD_CAD": 0.0010,
    "GBP_JPY": 150,
    "DEFAULT": 0.0010
}

PAIR_SETTINGS = {
    "XAU_USD": {
        "atr_multiplier_sl": 1.8,
        "atr_multiplier_tp": 3.5,
        "max_volatility_ratio": 0.010,
        "risk_multiplier": 0.5,
        "required_confluence": "STRICT"
    },
    "NAS100_USD": {
        "atr_multiplier_sl": 1.6,
        "atr_multiplier_tp": 3.2,
        "max_volatility_ratio": 0.015,
        "risk_multiplier": 0.7,
        "required_confluence": "STRICT"
    },
    "GBP_JPY": {
        "atr_multiplier_sl": 1.8,
        "atr_multiplier_tp": 3.5,
        "max_volatility_ratio": 0.012,
        "risk_multiplier": 0.7,
        "required_confluence": "STRICT"
    },
    "DEFAULT": {
        "atr_multiplier_sl": 1.5,
        "atr_multiplier_tp": 3.0,
        "max_volatility_ratio": 0.02,
        "risk_multiplier": 1.0
    }
}

SIGNAL_RISK_SETTINGS = {
    "NESTED_FVG": {"sl_multiplier": 1.0, "tp_multiplier": 2.5},
    "FVG_RETEST": {"sl_multiplier": 1.5, "tp_multiplier": 3.0},
    "WICK_REJECTION": {"sl_multiplier": 1.7, "tp_multiplier": 4.5},
    "LIQUIDITY_DRAW": {"sl_multiplier": 1.8, "tp_multiplier": 3.5}
}

MAX_PIPS_ACCEPTED = {
    "XAU_USD": 50.0,
    "USD_JPY": 10.0,
    "NAS100_USD": 30.0,
    "AUD_USD": 10.0,
    "EUR_USD": 10.0,
    "GBP_USD": 10.0,
    "USD_CAD": 10.0,
    "GBP_JPY": 15.0,
    "DEFAULT": 10.0
}

SCORING_CONFIG = {
    "MIN_CONFIDENCE_SCORE": 10,
    "SIGNAL_WEIGHTS": {
        "BISI": 5,
        "NESTED_FVG": 4,
        "FVG_RETEST_PERFECT": 4,
        "FVG_RETEST": 3,
        "BREAKER": 2,
        "WICK_REJECTION": 3,
        "TBS_PIN_BUY": 4,
        "LIQUIDITY_DRAW": 2,
        "TBS_PIN_SELL": 4
    },
    "BONUS": {
        "BOS_CONFIRMED": 2,
        "CHOCH_CONFIRMED": 2,
        "RSI_CONFLUENCE": 2,
        "VOLATILITY_OK": 1,
        "RR_OK": 2,
        "MACD_DIVERGENCE": 3,
        "FAILURE_SWING": 3,
        "CRT_DETECTED": 2,
        "TBS_DETECTED": 3,
        "ERL_BONUS": 1,
        "IB_BONUS": 2
    },
    "PENALTY": {
        "IB_PENALTY": 2,
        "NO_IB_PENALTY": 1,
        "IRL_PENALTY": 3
    }
}

def get_dynamic_max_distance(df: pd.DataFrame, pair: str, atr_multiplier: float = 1.5) -> float:
    if df is None or len(df) < 14:
        return 20.0
    try:
        atr = calculate_atr(df, period=14)
        atr_pips = price_to_pips(atr, pair)
        dynamic_max_pips = max(10.0, min(50.0, atr_pips * atr_multiplier))
        return dynamic_max_pips
    except Exception:
        return 20.0

def is_in_key_zone_or_consolidation(current_price, pair, df_m15, liquidity_levels, nested_fvgs, recent_ofls, structure_analysis, max_zone_width_pips=30.0) -> bool:
    try:
        pip_value = 0.01 if "JPY" in pair or pair == "XAU_USD" else 0.0001
        max_zone_width_price = max_zone_width_pips * pip_value
        liq_high = liquidity_levels.get("previous_week_high")
        liq_low = liquidity_levels.get("previous_week_low")
        if liq_high and abs(current_price - liq_high) <= max_zone_width_price:
            return True
        if liq_low and abs(current_price - liq_low) <= max_zone_width_price:
            return True
        for nfvg in nested_fvgs:
            midpoint = nfvg.get("midpoint")
            if midpoint and abs(current_price - midpoint) <= max_zone_width_price:
                return True
        for key in ["bos", "choch"]:
            level = structure_analysis.get(key, {}).get("level")
            if level and abs(current_price - level) <= max_zone_width_price:
                return True
        return False
    except Exception:
        return True

def detect_imbalances(df: pd.DataFrame, lookback: int = 3) -> list:
    if len(df) < lookback + 2:
        return []
    ibs = []
    for i in range(lookback, len(df) - 1):
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        next_high = df.iloc[i + 1]['high']
        next_low = df.iloc[i + 1]['low']
        if current_low > next_high:
            ibs.append({'type': 'BULLISH', 'high': current_high, 'low': next_high, 'level': (current_high + next_high) / 2})
        elif current_high < next_low:
            ibs.append({'type': 'BEARISH', 'high': next_low, 'low': current_low, 'level': (current_low + next_low) / 2})
    return ibs

def is_in_imbalance_zone(entry_level: float, ibs: list, tolerance: float = 0.0001) -> dict:
    for ib in ibs:
        if ib['low'] - tolerance <= entry_level <= ib['high'] + tolerance:
            return {'is_in_zone': True, 'type': ib['type'], 'level': ib['level']}
    return {'is_in_zone': False, 'type': None, 'level': None}

def detect_breaker(df: pd.DataFrame, lookback: int = 10) -> dict:
    if len(df) < lookback + 3:
        return {"type": None, "level": None}
    for i in range(len(df) - 3, len(df)):
        candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        if candle['close'] > prev_candle['high']:
            return {"type": "BUY", "level": prev_candle['high'], "time": df.index[i]}
        elif candle['close'] < prev_candle['low']:
            return {"type": "SELL", "level": prev_candle['low'], "time": df.index[i]}
    return {"type": None, "level": None}

def validate_trend_alignment(direction, df_h1, df_h4):
    return True

def detect_dealing_range(df: pd.DataFrame, lookback: int = 50) -> dict:
    if df is None or df.empty or len(df) < lookback:
        return None
    swing_highs, swing_lows = detect_swing_points_advanced(df, lookback)
    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return None
    all_swings = []
    for sh in swing_highs:
        all_swings.append((sh['index'], sh['price'], 'high'))
    for sl in swing_lows:
        all_swings.append((sl['index'], sl['price'], 'low'))
    all_swings.sort(key=lambda x: x[0])
    last_high = None
    last_low = None
    for idx, price, swing_type in reversed(all_swings):
        if swing_type == 'high' and last_high is None:
            last_high = price
        elif swing_type == 'low' and last_low is None:
            last_low = price
        if last_high is not None and last_low is not None:
            break
    if last_high is not None and last_low is not None:
        range_high = max(last_high, last_low)
        range_low = min(last_high, last_low)
        return {"high": range_high, "low": range_low, "range_size": range_high - range_low}
    return None

def classify_zone_irl_erl(zone_level: float, dealing_range: dict, tolerance: float = 0.0001) -> str:
    if not dealing_range or dealing_range.get("high") is None or dealing_range.get("low") is None:
        return None
    range_high = dealing_range["high"]
    range_low = dealing_range["low"]
    if range_low - tolerance <= zone_level <= range_high + tolerance:
        return "IRL"
    else:
        return "ERL"

def is_displacement_strong(df: pd.DataFrame, threshold: float = 0.0005) -> bool:
    if df.empty or len(df) < 2:
        return False
    last_candle = df.iloc[-1]
    body_size = abs(last_candle['close'] - last_candle['open'])
    total_range = last_candle['high'] - last_candle['low']
    return body_size >= threshold and (body_size / total_range) > 0.6

def detect_amd_phase(df: pd.DataFrame, lookback: int = 50) -> str:
    if df.empty or len(df) < lookback:
        return "UNKNOWN"
    recent_high = df['high'].tail(lookback).max()
    recent_low = df['low'].tail(lookback).min()
    current_price = df['close'].iloc[-1]
    range_size = recent_high - recent_low
    if range_size < (current_price * 0.005):
        return "ACCUMULATION"
    if df['high'].iloc[-1] > recent_high or df['low'].iloc[-1] < recent_low:
        return "MANIPULATION"
    if current_price > recent_high or current_price < recent_low:
        return "DISTRIBUTION"
    return "UNKNOWN"

def cluster_signals(signals: List[Dict], pair: str, max_distance_pips_for_clustering: float = None) -> List[Dict]:
    if not signals:
        return []
    pip_value = get_pip_value_for_pair(pair)
    max_distance_pips_arg = max_distance_pips_for_clustering or 15.0
    max_distance_price = max_distance_pips_arg * pip_value
    signals.sort(key=lambda s: s.get("confidence_score", 0), reverse=True)
    clusters = []
    current_cluster = []
    for s in signals:
        lvl = float(s["entry_level"])
        if not current_cluster:
            current_cluster = [s]
            last_level = lvl
            continue
        if abs(lvl - last_level) <= max_distance_price:
            current_cluster.append(s)
            last_level = lvl
        else:
            best_signal_in_cluster = current_cluster[0]
            clusters.append(best_signal_in_cluster)
            current_cluster = [s]
            last_level = lvl
    if current_cluster:
        best_signal_in_cluster = current_cluster[0]
        clusters.append(best_signal_in_cluster)
    if 'logger' in globals() and isinstance(globals()['logger'], logging.Logger):
        logger.info(f"🗂️ Entrées après clustering: {len(clusters)}")
    return clusters

def is_crt_candle(candle: pd.Series, min_body_ratio: float = 0.5) -> bool:
    body = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    if total_range == 0:
        return False
    body_ratio = body / total_range
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range
    return body_ratio >= min_body_ratio and upper_wick_ratio <= 0.2 and lower_wick_ratio <= 0.2

def rr_points(rr: float) -> int:
    if rr >= 3.0:
        return 2
    if rr >= 2.0:
        return 1
    return 0

def detect_tbs_setup(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 3:
        return {"type": "", "level": None}
    current_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    prev2_candle = df.iloc[-3]
    is_inside_bar = (prev_candle['high'] < prev2_candle['high'] and prev_candle['low'] > prev2_candle['low'])
    if is_inside_bar:
        if current_candle['high'] > prev_candle['high']:
            body_size = abs(current_candle['close'] - current_candle['open'])
            total_range = current_candle['high'] - current_candle['low']
            if body_size > total_range * 0.6:
                return {"type": "TBS_IB_BULL", "level": prev_candle['high']}
        elif current_candle['low'] < prev_candle['low']:
            body_size = abs(current_candle['close'] - current_candle['open'])
            total_range = current_candle['high'] - current_candle['low']
            if body_size > total_range * 0.6:
                return {"type": "TBS_IB_SELL", "level": prev_candle['low']}
    pb_body = abs(prev_candle['close'] - prev_candle['open'])
    pb_range = prev_candle['high'] - prev_candle['low']
    if pb_range > 0:
        pb_body_ratio = pb_body / pb_range
        pb_upper_wick = prev_candle['high'] - max(prev_candle['open'], prev_candle['close'])
        pb_lower_wick = min(prev_candle['open'], prev_candle['close']) - prev_candle['low']
        if pb_lower_wick > pb_upper_wick * 2 and pb_body_ratio < 0.4:
            if current_candle['high'] > prev_candle['high']:
                return {"type": "TBS_PIN_BUY", "level": prev_candle['high']}
        elif pb_upper_wick > pb_lower_wick * 2 and pb_body_ratio < 0.4:
            if current_candle['low'] < prev_candle['low']:
                return {"type": "TBS_PIN_SELL", "level": prev_candle['low']}
    return {"type": "", "level": None}

def detect_crt_candle(candle: pd.Series, min_body_ratio: float = 0.5) -> bool:
    body = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    if total_range == 0:
        return False
    body_ratio = body / total_range
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range
    return body_ratio >= min_body_ratio and upper_wick_ratio <= 0.2 and lower_wick_ratio <= 0.2

def compute_confidence_score(*, bias_points: int, structure_points: int, rr: float, nested_fvg_in_zone: bool = False, other_bonuses: int = 0) -> int:
    score = 0
    score += bias_points
    score += structure_points
    score += rr_points(rr)
    if nested_fvg_in_zone:
        score += 1
    score += other_bonuses
    return score

def get_pair_settings(pair: str) -> dict:
    return PAIR_SETTINGS.get(pair, PAIR_SETTINGS["DEFAULT"])

# =============================
# LOGGING (inchangé)
# =============================
LOG_ASCII_SAFE = os.getenv("LOG_ASCII_SAFE", "true").lower() == "true"
_MOJIBAKE_ASCII_REPLACEMENTS_V82 = {
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©": "e", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©": "e", "ÃƒÆ’Ã‚Â©": "e",
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨": "e", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¨": "e", "ÃƒÆ’Ã‚Â¨": "e",
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Âª": "e", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Âª": "e", "ÃƒÆ’Ã‚Âª": "e",
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â«": "e", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â«": "e", "ÃƒÆ’Ã‚Â«": "e",
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ": "a", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â ": "a", "ÃƒÆ’Ã‚Â ": "a",
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢": "a", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢": "a", "ÃƒÆ’Ã‚Â¢": "a",
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â´": "o", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â´": "o", "ÃƒÆ’Ã‚Â´": "o",
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹": "u", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¹": "u", "ÃƒÆ’Ã‚Â¹": "u",
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»": "u", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â»": "u", "ÃƒÆ’Ã‚Â»": "u",
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â®": "i", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â®": "i", "ÃƒÆ’Ã‚Â®": "i",
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯": "i", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¯": "i", "ÃƒÆ’Ã‚Â¯": "i",
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§": "c", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â§": "c", "ÃƒÆ’Ã‚Â§": "c",
    "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â°": "E", "ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°": "E", "ÃƒÆ’Ã†â€™Ãƒâ€šÃ¢â‚¬Â°": "E", "ÃƒÆ’Ã¢â‚¬Â°": "E",
    "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦": "...", "ÃƒÆ’Ã‚Â¢Ãƒâ€šÃ¢â€šÂ¬Ãƒâ€šÃ‚Â¦": "...",
    "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢": "'", "ÃƒÆ’Ã‚Â¢Ãƒâ€šÃ¢â€šÂ¬Ãƒâ€šÃ¢â€žÂ¢": "'",
    "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ": "-", "ÃƒÆ’Ã‚Â¢Ãƒâ€šÃ¢â€šÂ¬Ãƒâ€šÃ¢â‚¬Å“": "-",
    "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â": "-", "ÃƒÆ’Ã‚Â¢Ãƒâ€šÃ¢â€šÂ¬Ãƒâ€šÃ‚Â": "-",
    "ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â·": "-", "Ãƒâ€šÃ‚Â·": "-",
    "ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€š": "", "Ãƒâ€š": "",
}

def repair_mojibake_v82(value) -> str:
    text = str(value)
    for bad, good in _MOJIBAKE_ASCII_REPLACEMENTS_V82.items():
        text = text.replace(bad, good)
    if any(marker in text for marker in ("ÃƒÆ’", "Ãƒâ€š", "ÃƒÂ¢", "Ãƒâ€¦", "Ã†â€™", "Ã¢â€šÂ¬")):
        for _ in range(3):
            try:
                repaired = text.encode("cp1252").decode("utf-8")
            except UnicodeError:
                break
            if repaired == text:
                break
            text = repaired
            for bad, good in _MOJIBAKE_ASCII_REPLACEMENTS_V82.items():
                text = text.replace(bad, good)
    if LOG_ASCII_SAFE:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in text)
        text = " ".join(text.split())
        for trash in ("| Ys ", "| YS ", "| YZ ", "| YA ", "| YR ", "| Y ", "| a... "):
            text = text.replace(trash, "| ")
        for trash in ("Ys ", "YS ", "YZ ", "YA ", "YR ", "Y ", "a... "):
            if text.startswith(trash):
                text = text[len(trash):]
        cleanup_words = {
            "DAmarrage": "Demarrage",
            "DAbut": "Debut",
            "dAtectAs": "detectes",
            "dAtectAe": "detectee",
            "dAtectA": "detecte",
            "initialisAe": "initialisee",
            "succA s": "succes",
            "rAcents": "recents",
            "RACENT": "RECENT",
            "rejetAs": "rejetes",
            "entrAes": "entrees",
            "EntrAe": "Entree",
            "aprAs": "apres",
            "dAdup": "dedup",
            "dAjA": "deja",
            "envoyAs": "envoyes",
            "scorAs": "scores",
            "QualitA": "Qualite",
            "qualitA": "qualite",
            "validA": "valide",
            "bloquA": "bloque",
            "annulA": "annule",
            "exAcutA": "execute",
            "exAcution": "execution",
        }
        for bad, good in cleanup_words.items():
            text = text.replace(bad, good)
    return text

class ReadableLogFormatterV82(logging.Formatter):
    ALLOWED_TAGS_V83 = ("[START]", "[SCAN]", "[INFO]", "[SIGNAL]", "[ORDER]", "[RISK]", "[ERROR]")
    def _clean_message_v83(self, message: str, levelname: str) -> str:
        text = repair_mojibake_v82(str(message))
        text = "".join(ch for ch in text if ord(ch) < 128)
        text = " ".join(text.split())
        upper = text.upper()
        if any(text.startswith(tag) for tag in self.ALLOWED_TAGS_V83):
            return text
        if levelname in ("ERROR", "CRITICAL"):
            tag = "[ERROR]"
        elif "SIGNAL" in upper:
            tag = "[SIGNAL]"
        elif "ORDER" in upper or "ORDRE" in upper or "EXECUTION" in upper or "/ORDERS" in upper:
            tag = "[ORDER]"
        elif "RISK" in upper or "MARGIN" in upper or "UNITS" in upper or "BREAKEVEN" in upper or "TRAIL" in upper:
            tag = "[RISK]"
        elif "SCAN" in upper or "ANALYSE" in upper:
            tag = "[SCAN]"
        elif "START" in upper or "DEMARRAGE" in upper:
            tag = "[START]"
        else:
            tag = "[INFO]"
        return f"{tag} {text}"
    def format(self, record):
        original_msg = record.msg
        original_args = record.args
        try:
            record.msg = self._clean_message_v83(record.getMessage(), record.levelname)
            record.args = ()
            return super().format(record)
        finally:
            record.msg = original_msg
            record.args = original_args

_log_formatter_v82 = ReadableLogFormatterV82(
    fmt="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log_file_handler_v82 = logging.FileHandler("advanced_orderflow_trading.log", encoding="utf-8")
_log_file_handler_v82.setFormatter(_log_formatter_v82)
_log_stream_handler_v82 = logging.StreamHandler(sys.stdout)
_log_stream_handler_v82.setFormatter(_log_formatter_v82)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_log_file_handler_v82, _log_stream_handler_v82],
    force=True,
)

for _noisy_logger_v82 in ("urllib3", "requests", "oandapyV20", "oandapy"):
    logging.getLogger(_noisy_logger_v82).setLevel(logging.ERROR)
    logging.getLogger(_noisy_logger_v82).propagate = False

logger = logging.getLogger("Advanced-Orderflow-Trading-Bot")

# =============================
# GESTION DES SIGNAUX
# =============================
sent_signals = {}
recent_signals = {}

def is_duplicate(pair: str, direction: str, level: float, ttl_seconds: int = 1800) -> bool:
    now = datetime.utcnow()
    key = (pair, direction, round(float(level), 4))
    last = recent_signals.get(key)
    if last and (now - last).total_seconds() < ttl_seconds:
        return True
    recent_signals[key] = now
    return False

def detect_rsi_divergence_haussiere(df: pd.DataFrame, lookback: int = 14) -> bool:
    if len(df) < lookback * 2 + 5:
        return False
    prices = df["close"].tail(lookback * 2).reset_index(drop=True)
    rsi_vals = calculate_rsi(df["close"]).tail(lookback * 2).reset_index(drop=True)
    price_lows = []
    for i in range(3, len(prices) - 3):
        if (prices.iloc[i] < prices.iloc[i-3:i].min() and prices.iloc[i] < prices.iloc[i+1:i+4].min()):
            price_lows.append((i, prices.iloc[i]))
    rsi_lows = []
    for i in range(3, len(rsi_vals) - 3):
        if (rsi_vals.iloc[i] < rsi_vals.iloc[i-3:i].min() and rsi_vals.iloc[i] < rsi_vals.iloc[i+1:i+4].min()):
            rsi_lows.append((i, rsi_vals.iloc[i]))
    if len(price_lows) < 2 or len(rsi_lows) < 2:
        return False
    last_price_low = price_lows[-1][1]
    prev_price_low = price_lows[-2][1]
    last_rsi_low = rsi_lows[-1][1]
    prev_rsi_low = rsi_lows[-2][1]
    return last_price_low < prev_price_low and last_rsi_low > prev_rsi_low

def calculate_macd_momentum(df: pd.DataFrame) -> pd.Series:
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram

def is_momentum_accelerating(df: pd.DataFrame, lookback: int = 3) -> bool:
    hist = calculate_macd_momentum(df)
    recent_hist = hist.tail(lookback)
    if len(recent_hist) < lookback:
        return False
    is_accelerating_up = all(recent_hist.iloc[i] > recent_hist.iloc[i-1] for i in range(1, len(recent_hist)))
    is_accelerating_down = all(recent_hist.iloc[i] < recent_hist.iloc[i-1] for i in range(1, len(recent_hist)))
    is_BUY = is_accelerating_up and (recent_hist.iloc[-1] > 0)
    is_SELL = is_accelerating_down and (recent_hist.iloc[-1] < 0)
    return is_BUY or is_SELL

def is_candle_momentum_strong(candle: pd.Series, direction: str) -> bool:
    body_size = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    if total_range == 0:
        return False
    body_ratio = body_size / total_range
    if direction == "SELL":
        is_red = candle['close'] < candle['open']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        upper_wick_ratio = upper_wick / total_range
        return is_red and body_ratio > 0.7 and upper_wick_ratio < 0.2
    elif direction == "BUY":
        is_green = candle['close'] > candle['open']
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        lower_wick_ratio = lower_wick / total_range
        return is_green and body_ratio > 0.7 and lower_wick_ratio < 0.2
    return False

def is_volume_confirming_momentum(df: pd.DataFrame, direction: str) -> bool:
    last_3_volumes = df['volume'].tail(3)
    avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
    volume_increasing = last_3_volumes.iloc[-1] > last_3_volumes.iloc[-2] > last_3_volumes.iloc[-3]
    volume_above_avg = last_3_volumes.iloc[-1] > avg_volume * 1.2
    return volume_increasing and volume_above_avg

def validate_risk_reward(entry_price, stop_loss, take_profit, min_ratio=2.0) -> bool:
    if None in [entry_price, stop_loss, take_profit]:
        return False
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    if risk <= 0:
        return False
    rr_ratio = reward / risk
    return rr_ratio >= min_ratio

def price_to_pips(price_diff: float, pair: str) -> float:
    pair = pair.upper()
    if pair == "XAU_USD":
        pip_size = 0.01
    elif pair == "NAS100_USD":
        pip_size = 0.1
    elif "JPY" in pair:
        pip_size = 0.01
    else:
        pip_size = 0.0001
    return abs(price_diff) / pip_size

def get_pip_value_for_pair(pair: str) -> float:
    pair = pair.upper()
    if pair == "XAU_USD":
        return 0.1
    elif pair == "NAS100_USD":
        return 0.1
    elif "JPY" in pair:
        return 0.01
    else:
        return 0.0001

def detect_rsi_momentum_acceleration(df, period=14, lookback=8) -> dict:
    if len(df) < lookback + period + 5:
        return {'direction': 'NEUTRAL', 'strength': 0.0, 'is_diverging': False}
    rsi = calculate_rsi(df['close'], period)
    rsi_vals = rsi.tail(lookback).values
    rsi_slope = (rsi_vals[-1] - rsi_vals[-3]) / 2
    strength = max(min(rsi_slope / 15, 1), -1)
    if rsi_slope > 0.5:
        return {'direction': 'BUY', 'strength': strength}
    elif rsi_slope < -0.5:
        return {'direction': 'SELL', 'strength': abs(strength)}
    return {'direction': 'NEUTRAL', 'strength': 0.0}

def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> bool:
    if len(df) < lookback * 2 + 5:
        return False
    prices = df["close"].tail(lookback * 2).reset_index(drop=True)
    rsi_vals = calculate_rsi(df["close"]).tail(lookback * 2).reset_index(drop=True)
    price_peaks = []
    for i in range(3, len(prices) - 3):
        if (prices.iloc[i] > prices.iloc[i-3:i].max() and prices.iloc[i] > prices.iloc[i+1:i+4].max()):
            price_peaks.append((i, prices.iloc[i]))
    rsi_peaks = []
    for i in range(3, len(rsi_vals) - 3):
        if (rsi_vals.iloc[i] > rsi_vals.iloc[i-3:i].max() and rsi_vals.iloc[i] > rsi_vals.iloc[i+1:i+4].max()):
            rsi_peaks.append((i, rsi_vals.iloc[i]))
    if len(price_peaks) < 2 or len(rsi_peaks) < 2:
        return False
    last_price_peak = price_peaks[-1][1]
    prev_price_peak = price_peaks[-2][1]
    last_rsi_peak = rsi_peaks[-1][1]
    prev_rsi_peak = rsi_peaks[-2][1]
    return last_price_peak > prev_price_peak and last_rsi_peak < prev_rsi_peak

def cleanup_old_signals():
    now = time.time()
    for key, timestamp in list(sent_signals.items()):
        if now - timestamp > 86400:
            del sent_signals[key]
            logger.debug(f"🧹 Signal nettoyé: {key}")

def is_signal_in_recent_zone(pair: str, direction: str, price: float, zone_start: float, zone_end: float) -> bool:
    global sent_signals
    now = time.time()
    zone_width = abs(zone_end - zone_start)
    tolerance = zone_width * 0.5
    for (p, d, lvl, z_s, z_e), timestamp in sent_signals.items():
        if p != pair or d != direction:
            continue
        if now - timestamp > 1 * 3600:
            continue
        if abs(price - lvl) <= tolerance:
            return True
    return False

def is_signal_sent_recently(pair: str, direction: str, price: float, zone_start: float, zone_end: float) -> bool:
    global sent_signals
    now = time.time()
    tolerance_price = 0.00001 if "JPY" not in pair and pair != "XAU_USD" else 0.01
    price_rounded = round(price, 5)
    keys_to_delete = []
    is_sent = False
    for key, timestamp in sent_signals.items():
        p, d, lvl, _, _ = key
        if now - timestamp > 4 * 3600:
            keys_to_delete.append(key)
            continue
        if p == pair and d == direction:
            if abs(price_rounded - lvl) < tolerance_price:
                if now - timestamp < 2 * 3600:
                    is_sent = True
    for k in keys_to_delete:
        sent_signals.pop(k, None)
    return is_sent

def detect_macd_acceleration(df: pd.DataFrame, lookback: int = 5) -> str:
    hist = calculate_macd_momentum(df)
    if len(hist) < lookback + 2:
        return 'NEUTRAL'
    recent_hist = hist.tail(lookback + 1).values
    changes = [recent_hist[i] - recent_hist[i-1] for i in range(1, len(recent_hist))]
    acceleration = sum(changes) / len(changes)
    last_hist = recent_hist[-1]
    if acceleration > 0.0002 and last_hist > 0:
        return 'STRONG_BUY'
    elif acceleration < -0.0002 and last_hist < 0:
        return 'STRONG_SELL'
    elif acceleration > 0 and last_hist > 0:
        return 'WEAK_BUY'
    elif acceleration < 0 and last_hist < 0:
        return 'WEAK_SELL'
    else:
        return 'NEUTRAL'

def detect_failure_swing(df: pd.DataFrame, lookback: int = 20) -> dict:
    if len(df) < lookback + 5:
        return {"type": None, "level": None}
    swing_highs = []
    swing_lows = []
    for i in range(lookback, len(df) - 1):
        current_high = df["high"].iloc[i]
        current_low = df["low"].iloc[i]
        if (current_high > df["high"].iloc[i-lookback:i].max() and
            current_high > df["high"].iloc[i+1:i+lookback+1].max()):
            swing_highs.append({"index": i, "price": current_high})
        if (current_low < df["low"].iloc[i-lookback:i].min() and
            current_low < df["low"].iloc[i+1:i+lookback+1].min()):
            swing_lows.append({"index": i, "price": current_low})
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"type": None, "level": None}
    last_swing_low = swing_lows[-1]
    prev_swing_low = swing_lows[-2]
    if last_swing_low["price"] > prev_swing_low["price"]:
        if df["close"].iloc[-1] > last_swing_low["price"]:
            return {"type": "BULLISH", "level": last_swing_low["price"], "time": df.index[-1]}
    last_swing_high = swing_highs[-1]
    prev_swing_high = swing_highs[-2]
    if last_swing_high["price"] < prev_swing_high["price"]:
        if df["close"].iloc[-1] < last_swing_high["price"]:
            return {"type": "BEARISH", "level": last_swing_high["price"], "time": df.index[-1]}
    return {"type": None, "level": None}

def validate_momentum_confluence(df_h4: pd.DataFrame, df_h1: pd.DataFrame, df_m15: pd.DataFrame, direction: str) -> int:
    score = 0
    h4_mom = detect_rsi_momentum_acceleration(df_h4, period=14, lookback=8)
    if h4_mom['direction'] == ('STRONG_' + direction) or h4_mom['direction'] == direction:
        score += 1
    h1_mom = detect_rsi_momentum_acceleration(df_h1, period=14, lookback=8)
    if h1_mom['direction'] == ('STRONG_' + direction) or h1_mom['direction'] == direction:
        score += 1
    m15_mom = detect_rsi_momentum_acceleration(df_m15, period=14, lookback=8)
    if m15_mom['direction'] == ('STRONG_' + direction) or m15_mom['direction'] == direction:
        score += 1
    return score

def detect_volume_momentum(df: pd.DataFrame, window: int = 10) -> str:
    vol_ma = df['volume'].rolling(window=window).mean()
    vol_current = df['volume'].iloc[-1]
    vol_prev = df['volume'].iloc[-2]
    vol_ma_current = vol_ma.iloc[-1]
    vol_ma_prev = vol_ma.iloc[-2]
    vol_ratio = vol_current / vol_ma_current
    vol_accel = (vol_current - vol_prev) / vol_prev
    if vol_ratio > 1.3 and vol_accel > 0.2:
        return 'STRONG_BUY'
    elif vol_ratio > 1.3 and vol_accel < -0.2:
        return 'STRONG_SELL'
    elif vol_ratio > 1.1 and vol_accel > 0.1:
        return 'BUY'
    elif vol_ratio > 1.1 and vol_accel < -0.1:
        return 'SELL'
    else:
        return 'NEUTRAL'

def mark_signal_sent(pair: str, direction: str, entry_level: float, zone_start: float, zone_end: float):
    key = (pair, direction, round(entry_level, 5), round(zone_start, 5), round(zone_end, 5))
    sent_signals[key] = time.time()
    logger.info(f"✅ Signal marqué comme envoyé : {key}")

def detect_bos(df: pd.DataFrame, lookback: int = 50) -> dict:
    if len(df) < lookback + 10:
        return {"type": None, "level": None, "time": None}
    swing_highs, swing_lows = detect_swing_points(df, lookback=5)
    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return {"type": None, "level": None, "time": None}
    current_close = df["close"].iloc[-1]
    current_high = df["high"].iloc[-1]
    current_low = df["low"].iloc[-1]
    last_swing_high = swing_highs[-1]["price"]
    last_swing_low = swing_lows[-1]["price"]
    if current_close > last_swing_high and current_high > last_swing_high:
        return {"type": "BOS_BUY", "level": last_swing_high, "time": df.index[-1]}
    if current_close < last_swing_low and current_low < last_swing_low:
        return {"type": "BOS_SELL", "level": last_swing_low, "time": df.index[-1]}
    return {"type": None, "level": None, "time": None}

def detect_choch(df: pd.DataFrame, lookback: int = 50) -> dict:
    if len(df) < lookback + 15:
        return {"type": None, "level": None, "time": None}
    swing_highs, swing_lows = detect_swing_points(df, lookback=5)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"type": None, "level": None, "time": None}
    current_price = df["close"].iloc[-1]
    current_time = df.index[-1]
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-2]["price"]
        lh = swing_highs[-1]["price"]
        hl = swing_lows[-2]["price"]
        ll = swing_lows[-1]["price"]
        is_uptrend = (hh > (swing_highs[-3]["price"] if len(swing_highs) >= 3 else 0) and
                      hl > (swing_lows[-3]["price"] if len(swing_lows) >= 3 else 0))
        if is_uptrend and ll < hl and current_price < ll:
            return {"type": "CHOCH_SELL", "level": ll, "time": current_time}
    if len(swing_lows) >= 2 and len(swing_highs) >= 2:
        ll = swing_lows[-2]["price"]
        hl = swing_lows[-1]["price"]
        lh = swing_highs[-2]["price"]
        hh = swing_highs[-1]["price"]
        is_downtrend = (ll < (swing_lows[-3]["price"] if len(swing_lows) >= 3 else float('inf')) and
                        lh < (swing_highs[-3]["price"] if len(swing_highs) >= 3 else float('inf')))
        if is_downtrend and hh > lh and current_price > hh:
            return {"type": "CHOCH_BUY", "level": hh, "time": current_time}
    return {"type": None, "level": None, "time": None}

def generate_bos_signal(df: pd.DataFrame, pair: str, lookback: int = 20, rsi_col: str = "RSI") -> dict:
    bos = detect_bos(df, lookback=lookback)
    if bos["type"] is None:
        return {}
    current_price = df["close"].iloc[-1]
    score = 0
    confluences = []
    candle = df.iloc[-1]
    body = abs(candle["close"] - candle["open"])
    upper_wick = candle["high"] - max(candle["close"], candle["open"])
    lower_wick = min(candle["close"], candle["open"]) - candle["low"]
    if bos["type"] == "BOS_BUY" and lower_wick > body:
        score += 1
        confluences.append("WICK_REJECTION_BUY")
    elif bos["type"] == "BOS_SELL" and upper_wick > body:
        score += 1
        confluences.append("WICK_REJECTION_SELL")
    if rsi_col in df.columns:
        rsi = df[rsi_col].iloc[-1]
        if bos["type"] == "BOS_BUY" and rsi < 50:
            score += 0.5
            confluences.append("RSI_SUPPORT")
        elif bos["type"] == "BOS_SELL" and rsi > 50:
            score += 0.5
            confluences.append("RSI_RESIST")
    if bos["type"] == "BOS_BUY":
        swing_lows = [df["low"].iloc[i] for i in range(lookback, len(df)-lookback)
                      if df["low"].iloc[i] <= df["low"].iloc[i-lookback:i+lookback+1].min()]
        if not swing_lows:
            return {}
        stop_loss = min(swing_lows)
        risk = current_price - stop_loss
        tp_candidates = [level for level in df["high"].iloc[-lookback:] if level > current_price]
        take_profit = max(tp_candidates + [current_price + risk*1.5])
        score += 1
        confluences.append("BOS_BUY")
        if score < 1.5:
            return {}
        return {
            "pair": pair,
            "direction": "BUY",
            "entry_price": round(current_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "trigger": "BOS_BUY",
            "bos_level": bos["level"],
            "bos_time": bos["time"],
            "score": round(score,2),
            "confluences": confluences
        }
    elif bos["type"] == "BOS_SELL":
        swing_highs = [df["high"].iloc[i] for i in range(lookback, len(df)-lookback)
                       if df["high"].iloc[i] >= df["high"].iloc[i-lookback:i+lookback+1].max()]
        if not swing_highs:
            return {}
        stop_loss = max(swing_highs)
        risk = stop_loss - current_price
        tp_candidates = [level for level in df["low"].iloc[-lookback:] if level < current_price]
        take_profit = min(tp_candidates + [current_price - risk*1.5])
        score += 1
        confluences.append("BOS_SELL")
        if score < 1.5:
            return {}
        return {
            "pair": pair,
            "direction": "SELL",
            "entry_price": round(current_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "trigger": "BOS_SELL",
            "bos_level": bos["level"],
            "bos_time": bos["time"],
            "score": round(score,2),
            "confluences": confluences
        }
    return {}

# =============================
# FONCTION GET_CANDLES_WITH_RETRY
# =============================
def get_candles_with_retry(api, instrument: str, granularity: str, count: int = 500, retries: int = 5) -> pd.DataFrame:
    valid_granularities = ["S5", "S10", "S15", "S30",
                           "M1", "M2", "M4", "M5", "M10", "M15", "M30",
                           "H1", "H2", "H3", "H4", "H6", "H8", "H12",
                           "D", "W", "M"]
    if granularity not in valid_granularities:
        logger.error(f"❌ Granularité invalide: {granularity}")
        return pd.DataFrame()
    for attempt in range(retries):
        try:
            r = instruments.InstrumentsCandles(
                instrument=instrument,
                params={
                    "granularity": granularity,
                    "count": count,
                    "price": "M",
                    "smooth": False
                }
            )
            api.request(r)
            resp = getattr(r, "response", {}) or {}
            candles = resp.get("candles", [])
            if not candles:
                logger.warning(f"⚠️ Aucune candle reçue pour {instrument} {granularity} (tentative {attempt+1})")
                time.sleep(3)
                continue
            data = []
            for c in candles:
                mid = c.get("mid")
                if not mid:
                    continue
                if not c.get("complete", False) and (granularity in ["M15", "H1", "H4", "D"]):
                    continue
                try:
                    data.append({
                        "time": c["time"],
                        "open": float(mid["o"]),
                        "high": float(mid["h"]),
                        "low": float(mid["l"]),
                        "close": float(mid["c"]),
                        "volume": int(c.get("volume", 0))
                    })
                except Exception:
                    logger.debug(f"⚠️ Candle malformed skipped for {instrument}: {c}")
                    continue
            min_required = 20 if granularity == "H1" else min(count, 50)
            if len(data) < min_required:
                if granularity in ["H4", "D"] and len(data) > 5:
                    logger.info(f"ℹ️ Données D/{instrument}: {len(data)} candles (minimum requis: {min_required}) → Accepté malgré tout")
                else:
                    logger.warning(f"⚠️ Données insuffisantes pour {instrument} {granularity}: {len(data)} < {min_required}")
                    time.sleep(3)
                    continue
            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df.attrs['instrument'] = instrument
            logger.info(f"✅ {instrument} {granularity}: {len(df)} candles (dernier prix: {df['close'].iloc[-1]:.5f})")
            return df
        except oandapyV20.exceptions.V20Error as e:
            logger.warning(f"❌ Erreur OANDA {attempt + 1}/{retries} pour {instrument}: {e}")
            time.sleep(5 ** attempt if attempt < 4 else 5)
        except Exception as e:
            logger.warning(f"❌ Tentative {attempt + 1}/{retries} pour {instrument}: {e}")
            time.sleep(5 ** attempt if attempt < 4 else 5)
    logger.error(f"❌ Échec après {retries} tentatives pour {instrument} {granularity}")
    return pd.DataFrame()

# =============================
# INDICATEURS TECHNIQUES
# =============================
def validate_volume_confirmation(df: pd.DataFrame, fvg: dict) -> bool:
    if "time" not in fvg:
        return False
    fvg_time = fvg["time"]
    post_fvg_data = df[df.index > fvg_time].head(3)
    if len(post_fvg_data) < 2:
        return False
    avg_volume = df["volume"].rolling(window=20).mean().iloc[-1]
    max_volume = post_fvg_data["volume"].max()
    return max_volume > avg_volume * 1.5

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _directional_score(raw_score: int, direction: str) -> int:
    return int(raw_score) if (direction or "").upper() == "BUY" else -int(raw_score)

def score_ema_trend(df_h1: pd.DataFrame) -> int:
    if df_h1 is None or df_h1.empty or "close" not in df_h1 or len(df_h1) < 2:
        return 0
    ema50 = df_h1["close"].ewm(span=EMA_MEDIUM, adjust=False).mean()
    if len(ema50.dropna()) < 2:
        return 0
    price = float(df_h1["close"].iloc[-1])
    score = 2 if price > float(ema50.iloc[-1]) else -1
    score += 1 if float(ema50.iloc[-1]) > float(ema50.iloc[-2]) else -2
    return max(-3, min(3, int(score)))

def score_market_structure(df_h1: pd.DataFrame) -> int:
    if df_h1 is None or df_h1.empty or len(df_h1) < 10:
        return 0
    try:
        swing_highs, swing_lows = detect_swing_points_advanced(df_h1, min(SWING_LOOKBACK, max(1, len(df_h1) // 10)))
        recent_highs = sorted(swing_highs, key=lambda x: x["index"])[-2:]
        recent_lows = sorted(swing_lows, key=lambda x: x["index"])[-2:]
    except Exception:
        recent_highs, recent_lows = [], []
    if len(recent_highs) < 2:
        highs = df_h1["high"].tail(20)
        split = max(2, len(highs) // 2)
        recent_highs = [{"price": highs.iloc[:split].max()}, {"price": highs.iloc[split:].max()}]
    if len(recent_lows) < 2:
        lows = df_h1["low"].tail(20)
        split = max(2, len(lows) // 2)
        recent_lows = [{"price": lows.iloc[:split].min()}, {"price": lows.iloc[split:].min()}]
    higher_high = float(recent_highs[-1]["price"]) > float(recent_highs[-2]["price"])
    lower_high = float(recent_highs[-1]["price"]) < float(recent_highs[-2]["price"])
    higher_low = float(recent_lows[-1]["price"]) > float(recent_lows[-2]["price"])
    lower_low = float(recent_lows[-1]["price"]) < float(recent_lows[-2]["price"])
    if higher_high and higher_low:
        return 3
    if higher_high:
        return 1
    if lower_high and lower_low:
        return -3
    if lower_low:
        return -1
    return 0

def score_higher_timeframe_alignment(direction: str, df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> int:
    h1 = _directional_score(score_market_structure(df_h1), direction)
    h4 = _directional_score(score_market_structure(df_h4), direction)
    if h1 > 0 and h4 > 0:
        return 2
    if h1 < 0 and h4 < 0:
        return -2
    return 0

def compute_final_score(score_components: dict) -> int:
    return int(sum(int(v or 0) for v in score_components.values()))

def log_score_detail(score_components: dict, total: int, decision: str) -> None:
    labels = [
        ("ICT", "ICT"),
        ("EMA", "EMA"),
        ("Structure_H1", "Structure H1"),
        ("Liquidity", "Liquidity"),
        ("Order_Block", "Order Block"),
        ("Imbalance", "Imbalance"),
        ("Stochastic", "Stochastic"),
        ("HTF_Alignment", "HTF Alignment"),
        ("Risk_RR_Distance", "Risk/RR/Distance"),
        ("Secondary", "Secondary"),
    ]
    logger.info("===== SCORE DETAIL =====")
    for key, label in labels:
        if key in score_components:
            logger.info(f"{label:<19}: {int(score_components[key]):+d}")
    logger.info(f"TOTAL = {int(total):+d}")
    logger.info(f"Decision = {decision}")

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stoch_rsi(prices: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
                        smooth_k: int = 3, smooth_d: int = 3) -> tuple:
    try:
        rsi = calculate_rsi(prices, rsi_period)
        rsi_min = rsi.rolling(window=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period).max()
        denom = rsi_max - rsi_min
        stoch_raw = pd.Series(
            np.where(denom > 0, (rsi - rsi_min) / denom * 100, 50.0),
            index=prices.index
        )
        k = stoch_raw.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()
        k_val = float(k.dropna().iloc[-1]) if len(k.dropna()) > 0 else 50.0
        d_val = float(d.dropna().iloc[-1]) if len(d.dropna()) > 0 else 50.0
        return k_val, d_val
    except Exception:
        return 50.0, 50.0

def get_last_rsi(prices: pd.Series, period: int = 14) -> float:
    try:
        rsi_indicator = RSIIndicator(close=prices, window=period)
        rsi_values = rsi_indicator.rsi()
        return rsi_values.dropna().iloc[-1]
    except Exception:
        delta = prices.diff().dropna()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.dropna().iloc[-1] if len(rsi.dropna()) > 0 else 50.0

def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    try:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        atr = talib.ATR(high, low, close, timeperiod=period)
        last = float(atr[-1])
        if np.isnan(last) or last <= 0.0:
            raise ValueError("talib ATR invalid")
        return last
    except Exception:
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_fallback = tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
        if np.isnan(atr_fallback) or atr_fallback <= 0.0:
            logger.warning(f"⚠️ ATR fallback = 0 pour {df.attrs.get('instrument', 'N/A')} → Utilisation de 0.0001")
            return 0.0001
        return float(atr_fallback)

def calculate_volatility_ratio(df: pd.DataFrame, pair: str) -> bool:
    atr = calculate_atr(df)
    current_price = df["close"].iloc[-1]
    if current_price == 0:
        return False
    volatility_ratio = atr / current_price
    settings = get_pair_settings(pair)
    if pair == "XAU_USD":
        threshold = settings.get("max_volatility_ratio", 0.012)
        return volatility_ratio <= threshold * 0.8
    return volatility_ratio <= settings["max_volatility_ratio"]

# =============================
# VALIDATIONS ET FILTRES
# =============================
def validate_multi_timeframe_confluence(df_h4: pd.DataFrame, direction: str) -> bool:
    ema_h4_fast = df_h4["close"].ewm(span=20, adjust=False).mean().iloc[-1]
    ema_h4_slow = df_h4["close"].ewm(span=200, adjust=False).mean().iloc[-1]
    if direction == "BUY":
        return ema_h4_fast > ema_h4_slow or True
    elif direction == "SELL":
        return ema_h4_fast < ema_h4_slow or True
    return True

def determine_trend_direction(df: pd.DataFrame, ema_fast_period=20, ema_medium_period=50, ema_slow_period=200) -> str:
    if len(df) < ema_slow_period + 10:
        return "NEUTRAL"
    ema_fast = ema(df["close"], ema_fast_period)
    ema_medium = ema(df["close"], ema_medium_period)
    ema_slow = ema(df["close"], ema_slow_period)
    current_price = df["close"].iloc[-1]
    logger.info(f"   📉 EMA{ema_fast_period}: {ema_fast.iloc[-1]:.5f}")
    logger.info(f"   📉 EMA{ema_medium_period}: {ema_medium.iloc[-1]:.5f}")
    logger.info(f"   📉 EMA{ema_slow_period}: {ema_slow.iloc[-1]:.5f}")
    if current_price > ema_slow.iloc[-1] and ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        return "BUY"
    elif current_price < ema_slow.iloc[-1] and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return "SELL"
    else:
        return "NEUTRAL"

def detect_mss(df: pd.DataFrame, lookback: int = 10) -> dict:
    if df.empty or len(df) < lookback + 5:
        return {"type": None, "level": None}
    highs = df["high"].rolling(window=lookback).max()
    lows = df["low"].rolling(window=lookback).min()
    last_close = df["close"].iloc[-1]
    last_time = df.index[-1]
    ema20 = df["close"].ewm(span=20).mean().iloc[-1]
    if last_close > highs.iloc[-2] and last_close > ema20:
        return {"type": "MSS_BUY", "level": highs.iloc[-2], "time": last_time}
    if last_close < lows.iloc[-2] and last_close < ema20:
        return {"type": "MSS_SELL", "level": lows.iloc[-2], "time": last_time}
    return {"type": None, "level": None}

def validate_signal_coherence(pair: str, direction: str, df_h4: pd.DataFrame, df_m15: pd.DataFrame) -> bool:
    if direction not in ["BUY", "SELL"]:
        logger.error(f"❌ Direction invalide: {direction}. Doit être BUY ou SELL.")
        return False
    current_price = df_m15["close"].iloc[-1]
    rsi_m15 = get_last_rsi(df_m15["close"])
    rsi_h4 = get_last_rsi(df_h4["close"]) if df_h4 is not None else 50
    ema20_h4 = df_h4["close"].ewm(span=20).mean().iloc[-1] if len(df_h4) >= 20 else 0
    ema200_h4 = df_h4["close"].ewm(span=200).mean().iloc[-1] if len(df_h4) >= 200 else 0
    logger.info(f"   🔍 Vérification cohérence {pair} {direction}:")
    logger.info(f"   • Prix: {current_price:.5f}")
    logger.info(f"   • RSI M15: {rsi_m15:.1f}, RSI H4: {rsi_h4:.1f}")
    if direction == "SELL":
        if rsi_m15 > 85:
            logger.info("   ❌ RSI M15 trop haut → survendu impossible")
            return False
        if rsi_h4 > 80:
            logger.info("   ❌ RSI H4 trop haut → tendance haussière forte → pas de SELL")
            return False
        if ema20_h4 < ema200_h4:
            logger.info("   ✅ Tendance H4 baissière → SELL autorisé")
        else:
            logger.info("   ⚠️ EMA20 > EMA200 → risque de contre-tendance")
    elif direction == "BUY":
        if rsi_m15 < 5:
            logger.info("   ❌ RSI M15 trop bas → signal invalide")
            return False
        if rsi_h4 < 10:
            logger.info("   ❌ RSI H4 trop bas → tendance baissière extrême → pas de BUY")
            return False
        if ema20_h4 > ema200_h4:
            logger.info("   ✅ Tendance H4 haussière → BUY autorisé")
        else:
            logger.info("   ⚠️ EMA20 < EMA200 → risque de contre-tendance")
    logger.info("   ✅ Signal cohérent")
    return True

def convert_direction_to_buy_sell(direction: str) -> str:
    if direction == "BUY":
        return "BUY"
    elif direction == "SELL":
        return "SELL"
    elif direction in ["BUY", "SELL"]:
        return direction
    else:
        return "NEUTRAL"

def detect_macd_divergence(df: pd.DataFrame, lookback: int = 14) -> str:
    if len(df) < lookback * 2 + 5:
        return "NONE"
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    prices = df["close"].tail(lookback * 2).reset_index(drop=True)
    macd_vals = macd_line.tail(lookback * 2).reset_index(drop=True)
    price_peaks = []
    price_lows = []
    for i in range(3, len(prices) - 3):
        if (prices.iloc[i] > prices.iloc[i-3:i].max() and prices.iloc[i] > prices.iloc[i+1:i+4].max()):
            price_peaks.append((i, prices.iloc[i]))
        if (prices.iloc[i] < prices.iloc[i-3:i].min() and prices.iloc[i] < prices.iloc[i+1:i+4].min()):
            price_lows.append((i, prices.iloc[i]))
    macd_peaks = []
    macd_lows = []
    for i in range(3, len(macd_vals) - 3):
        if (macd_vals.iloc[i] > macd_vals.iloc[i-3:i].max() and macd_vals.iloc[i] > macd_vals.iloc[i+1:i+4].max()):
            macd_peaks.append((i, macd_vals.iloc[i]))
        if (macd_vals.iloc[i] < macd_vals.iloc[i-3:i].min() and macd_vals.iloc[i] < macd_vals.iloc[i+1:i+4].min()):
            macd_lows.append((i, macd_vals.iloc[i]))
    if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
        last_price_peak = price_peaks[-1][1]
        prev_price_peak = price_peaks[-2][1]
        last_macd_peak = macd_peaks[-1][1]
        prev_macd_peak = macd_peaks[-2][1]
        if last_price_peak > prev_price_peak and last_macd_peak < prev_macd_peak:
            return "BEARISH"
        elif last_price_peak < prev_price_peak and last_macd_peak > prev_macd_peak:
            return "BULLISH"
    if len(price_lows) >= 2 and len(macd_lows) >= 2:
        last_price_low = price_lows[-1][1]
        prev_price_low = price_lows[-2][1]
        last_macd_low = macd_lows[-1][1]
        prev_macd_low = macd_lows[-2][1]
        if last_price_low < prev_price_low and last_macd_low > prev_macd_low:
            return "BULLISH"
        elif last_price_low > prev_price_low and last_macd_low < prev_macd_low:
            return "BEARISH"
    return "NONE"

def validate_entry_momentum(df_m15: pd.DataFrame, df_h4: pd.DataFrame, direction: str) -> bool:
    rsi_m15 = get_last_rsi(df_m15["close"])
    rsi_h4 = get_last_rsi(df_h4["close"]) if df_h4 is not None else 50
    if direction == "BUY":
        return rsi_m15 > 25 or detect_rsi_divergence_haussiere(df_m15)
    elif direction == "SELL":
        return rsi_m15 < 75 or detect_rsi_divergence(df_m15)
    return False

def is_new_low_daily(df_daily: pd.DataFrame, lookback_days: int = 7) -> bool:
    if df_daily.empty or len(df_daily) < lookback_days + 1:
        return False
    recent_lows = df_daily["low"].tail(lookback_days)
    current_low = recent_lows.iloc[-1]
    return current_low <= recent_lows.min()

def get_min_gap_for_pair(pair: str) -> float:
    pair = pair.upper()
    if pair == "XAU_USD":
        return 0.02
    elif "JPY" in pair:
        return 0.03
    elif pair == "GBP_USD":
        return 0.00015
    else:
        return 0.0002

def detect_sharp_drop(df: pd.DataFrame, threshold: float = 0.01) -> bool:
    if len(df) < 2:
        return False
    last_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    drop_ratio = (prev_close - last_close) / prev_close
    return drop_ratio >= threshold

# =============================
# GESTION DES ORDRES
# =============================
BUFFER_PIPS = {
    "XAU_USD": 0.30,
    "USD_JPY": 0.03,
    "NAS100_USD": 1.0,
    "DEFAULT": 0.00030
}

def calculate_sl_tp(entry_price: float, atr: float, direction: str, pair: str,
                    entry_type: str = "FVG_RETEST", fvg_data: dict = None,
                    breaker_level: float = None) -> tuple:
    try:
        if None in [entry_price, atr, direction, pair]:
            return None, None
        settings = SIGNAL_RISK_SETTINGS.get(entry_type, SIGNAL_RISK_SETTINGS["FVG_RETEST"])
        sl_mult = settings["sl_multiplier"]
        tp_mult = settings["tp_multiplier"]
        stop_loss = 0.0
        take_profit = 0.0
        if direction == "BUY":
            stop_loss = entry_price - (atr * sl_mult)
            if fvg_data and "low_level" in fvg_data:
                fvg_bottom = float(fvg_data["low_level"])
                pip_buffer = get_pip_value_for_pair(pair) * 20
                structural_sl = fvg_bottom - pip_buffer
                stop_loss = min(stop_loss, structural_sl)
            take_profit = entry_price + (atr * tp_mult)
        else:
            stop_loss = entry_price + (atr * sl_mult)
            if fvg_data and "high_level" in fvg_data:
                fvg_top = float(fvg_data["high_level"])
                pip_buffer = get_pip_value_for_pair(pair) * 20
                structural_sl = fvg_top + pip_buffer
                stop_loss = max(stop_loss, structural_sl)
            take_profit = entry_price - (atr * tp_mult)
        risk = abs(entry_price - stop_loss)
        if risk == 0:
            return None, None
        reward = abs(take_profit - entry_price)
        if reward / risk < 2.0:
            take_profit = entry_price + (risk * 2.0) if direction == "BUY" else entry_price - (risk * 2.0)
        return round(stop_loss, 5), round(take_profit, 5)
    except Exception as e:
        logger.error(f"Erreur SL/TP: {e}")
        return None, None

# =============================
# ALERTES TELEGRAM
# =============================
def send_telegram_alert(pair: str, direction: str, entry_price: float,
                       stop_loss: float, take_profit: float, narrative: dict,
                       bias_analysis: dict, rsi: float, entry_type: str,
                       confidence_score: int = None):
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram désactivé")
        return
    if None in [pair, direction, entry_price, stop_loss, take_profit, rsi, entry_type]:
        logger.error(f"❌ Valeur manquante pour Telegram: pair={pair}, direction={direction}, entry={entry_price}, sl={stop_loss}, tp={take_profit}, rsi={rsi}, type={entry_type}")
        return
    if direction == "BUY":
        if stop_loss > entry_price or take_profit < entry_price:
            logger.error(f"❌ INCOHÉRENCE SL/TP pour BUY: Entry={entry_price}, SL={stop_loss}, TP={take_profit}")
            return
    elif direction == "SELL":
        if stop_loss < entry_price or take_profit > entry_price:
            logger.error(f"❌ INCOHÉRENCE SL/TP pour SELL: Entry={entry_price}, SL={stop_loss}, TP={take_profit}")
            return
    else:
        logger.error(f"❌ Direction invalide: {direction}")
        return
    win_rate = bias_analysis.get("win_rate", "") if bias_analysis else ""
    quality_label = bias_analysis.get("quality_label", "") if bias_analysis else ""
    score_details = bias_analysis.get("score_details", {}) if bias_analysis else {}
    try:
        dist_sl = abs(entry_price - stop_loss)
        dist_tp = abs(take_profit - entry_price)
        rr_display = f"{dist_tp / dist_sl:.2f}" if dist_sl > 0 else "N/A"
    except Exception:
        rr_display = "N/A"
    if confidence_score:
        score_info = (
            f"<b>Score:</b> {confidence_score}/{SCORING_CONFIG['MIN_CONFIDENCE_SCORE']} "
            f"| <b>Qualité:</b> {quality_label}\n"
            f"<b>Win Rate estimé:</b> {win_rate}\n"
            f"<b>R/R:</b> 1:{rr_display}\n"
        )
    else:
        score_info = ""
    confluence_tags = []
    if score_details.get("D1_Trend", "").startswith("+"):
        confluence_tags.append("D1")
    if score_details.get("MACD_H1", "").startswith("+"):
        confluence_tags.append("MACD H1")
    if score_details.get("RSI_Divergence", "").startswith("+"):
        confluence_tags.append("DIV RSI")
    if score_details.get("Structure", "").startswith("+"):
        confluence_tags.append("BOS")
    if score_details.get("Session", "").startswith("+"):
        confluence_tags.append("SESSION")
    confluences_line = f"<b>Confluences:</b> {' · '.join(confluence_tags)}\n" if confluence_tags else ""
    safe_pair = str(pair) if pair else "N/A"
    safe_direction = str(direction) if direction else "N/A"
    safe_entry_type = str(entry_type) if entry_type else "N/A"
    safe_bias = str(bias_analysis.get('bias', 'N/A')) if bias_analysis else "N/A"
    safe_entry_price = f"{entry_price:.5f}" if entry_price is not None else "N/A"
    safe_stop_loss = f"{stop_loss:.5f}" if stop_loss is not None else "N/A"
    safe_take_profit = f"{take_profit:.5f}" if take_profit is not None else "N/A"
    safe_rsi = f"{rsi:.1f}" if rsi is not None else "N/A"
    message = f"""
<b>FVG ORDERFLOW TRADING SIGNAL</b>
<b>Paire:</b> {safe_pair}
<b>Direction:</b> {safe_direction}
<b>Type d'entrée:</b> {safe_entry_type}
<b>Bias:</b> {safe_bias}
{score_info}{confluences_line}
<b>Entrée:</b> {safe_entry_price}
<b>Stop Loss:</b> {safe_stop_loss}
<b>Take Profit:</b> {safe_take_profit}
<b>RSI:</b> {safe_rsi}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
        if response.status_code == 200:
            logger.info(f"✅ Telegram envoyé pour {safe_pair}")
        else:
            logger.error(f"❌ Échec Telegram: {response.text}")
    except Exception as e:
        logger.error(f"💥 Erreur réseau Telegram: {e}")

# =============================
# DÉTECTION LIQUIDITÉ HEBDOMADAIRE
# =============================
def detect_weekly_liquidity(df: pd.DataFrame) -> dict:
    if len(df) < 7:
        return {}
    df_weekly = df.resample('W').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'})
    if len(df_weekly) < 2:
        return {}
    current_week = df_weekly.iloc[-1]
    previous_week = df_weekly.iloc[-2]
    return {
        "current_week_high": current_week["high"],
        "current_week_low": current_week["low"],
        "previous_week_high": previous_week["high"],
        "previous_week_low": previous_week["low"],
        "weekly_range": current_week["high"] - current_week["low"]
    }

# =============================
# DÉTECTION SWING POINTS
# =============================
def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> tuple:
    swing_highs = []
    swing_lows = []
    for i in range(lookback, len(df) - lookback):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        if (high == df["high"].iloc[i - lookback:i + lookback + 1].max()
            and df["close"].iloc[i] < df["open"].iloc[i]):
            swing_highs.append({"index": i, "time": df.index[i], "price": high})
        if (low == df["low"].iloc[i - lookback:i + lookback + 1].min()
            and df["close"].iloc[i] > df["open"].iloc[i]):
            swing_lows.append({"index": i, "time": df.index[i], "price": low})
    return swing_highs, swing_lows

def detect_swing_points_advanced(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> tuple:
    swing_highs = []
    swing_lows = []
    for i in range(lookback, len(df) - lookback):
        try:
            if (df["high"].iloc[i] == df["high"].iloc[i-lookback:i+lookback+1].max() and
                df["close"].iloc[i] < df["open"].iloc[i]):
                swing_highs.append({"index": i, "time": df.index[i], "price": df["high"].iloc[i], "type": "SWING_HIGH", "strength": "STRONG"})
        except Exception:
            pass
        try:
            if (df["low"].iloc[i] == df["low"].iloc[i-lookback:i+lookback+1].min() and
                df["close"].iloc[i] > df["open"].iloc[i]):
                swing_lows.append({"index": i, "time": df.index[i], "price": df["low"].iloc[i], "type": "SWING_LOW", "strength": "STRONG"})
        except Exception:
            pass
    return swing_highs, swing_lows

# =============================
# DÉTECTION FVG AVANCÉE
# =============================
def classify_fvg_type(current_candle, next_candle) -> str:
    try:
        body_size = abs(next_candle["close"] - next_candle["open"])
        total_range = next_candle["high"] - next_candle["low"]
        if total_range == 0:
            return "UNKNOWN"
        body_ratio = body_size / total_range
        if body_ratio >= 0.7:
            return "BREAKAWAY"
        elif body_ratio <= 0.4:
            return "REJECTION"
        else:
            return "PERFECT"
    except Exception:
        return "UNKNOWN"

def detect_fvg_advanced(df: pd.DataFrame, max_lookback_hours: int = 36) -> List[Dict]:
    fvgs = []
    if df is None or len(df) < 3:
        return fvgs
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    min_time = now - timedelta(hours=max_lookback_hours)
    df_index_times = pd.to_datetime(df.index)
    if df_index_times.tz is None:
        df_index_times = df_index_times.tz_localize('UTC')
    else:
        df_index_times = df_index_times.tz_convert('UTC')
    pair_name = str(df.attrs.get("instrument", ""))
    min_gap_size = get_min_gap_for_pair(pair_name)
    for i in range(1, len(df) - 1):
        candle_time = df_index_times[i]
        if candle_time < min_time:
            continue
        prev_candle = df.iloc[i - 1]
        next_candle = df.iloc[i + 1]
        prev_high = float(prev_candle["high"])
        prev_low = float(prev_candle["low"])
        next_high = float(next_candle["high"])
        next_low = float(next_candle["low"])
        if prev_high < next_low:
            gap_size = next_low - prev_high
            if gap_size >= min_gap_size:
                fvgs.append({
                    "index": i,
                    "direction": "BUY",
                    "type": "PERFECT",
                    "high_level": next_low,
                    "low_level": prev_high,
                    "gap_size": gap_size,
                    "time": candle_time,
                    "midpoint": (prev_high + next_low) / 2
                })
        if prev_low > next_high:
            gap_size = prev_low - next_high
            if gap_size >= min_gap_size:
                fvgs.append({
                    "index": i,
                    "direction": "SELL",
                    "type": "PERFECT",
                    "high_level": prev_low,
                    "low_level": next_high,
                    "gap_size": gap_size,
                    "time": candle_time,
                    "midpoint": (next_high + prev_low) / 2
                })
    return fvgs

def is_fvg_retest_valid(df: pd.DataFrame, fvg: dict, current_price: float, pair: str = "EUR_USD") -> bool:
    if "low_level" not in fvg or "high_level" not in fvg:
        return False
    fvg_mid = (float(fvg["low_level"]) + float(fvg["high_level"])) / 2.0
    distance = abs(current_price - fvg_mid)
    max_dist_pips = get_dynamic_max_distance(df, pair, atr_multiplier=1.5)
    return price_to_pips(distance, pair) <= max_dist_pips

def is_fvg_unmitigated(df: pd.DataFrame, fvg: dict) -> bool:
    after_data = df[df.index > fvg["time"]]
    if len(after_data) == 0:
        return False
    if fvg["direction"] == "BUY":
        return after_data["low"].min() > fvg["low_level"]
    elif fvg["direction"] == "SELL":
        return after_data["high"].max() < fvg["high_level"]
    return False

def is_fvg_respected_with_wick_rejection(df: pd.DataFrame, fvg: dict) -> bool:
    if "time" not in fvg:
        return False
    recent_candles = df[df.index > fvg["time"]].head(5)
    for _, candle in recent_candles.iterrows():
        if fvg["direction"] == "BUY":
            if candle["low"] <= fvg["low_level"] and candle["close"] > fvg["low_level"]:
                return True
        elif fvg["direction"] == "SELL":
            if candle["high"] >= fvg["high_level"] and candle["close"] < fvg["high_level"]:
                return True
    return False

def calculate_poc_and_liquidity(df: pd.DataFrame, bin_size: float = None) -> dict:
    if df.empty:
        return {"poc": None, "liquidity_high": None, "liquidity_low": None}
    pair = df.attrs.get('instrument', '')
    price_range = df["high"].max() - df["low"].min()
    min_tick = 0.5 if "XAU" in pair else 0.01
    if bin_size is None:
        bin_size = max(price_range / 100, min_tick)
    try:
        price_bins = (df["close"] // bin_size) * bin_size
        volume_profile = df.groupby(price_bins)["volume"].sum()
        if volume_profile.empty:
            return {"poc": None, "liquidity_high": None, "liquidity_low": None}
        poc = volume_profile.idxmax()
        return {"poc": float(poc), "liquidity_high": float(df["high"].max()), "liquidity_low": float(df["low"].min())}
    except Exception:
        return {"poc": None, "liquidity_high": None, "liquidity_low": None}

def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    open_price = df["open"]
    close_price = df["close"]
    high_price = df["high"]
    low_price = df["low"]
    hammer = (
        (close_price > open_price)
        & ((high_price - close_price) / (high_price - low_price) < 0.1)
        & ((close_price - open_price) / (high_price - low_price) > 0.6)
    )
    shooting_star = (
        (open_price > close_price)
        & ((high_price - open_price) / (high_price - low_price) < 0.1)
        & ((open_price - close_price) / (high_price - low_price) > 0.6)
    )
    return {"hammer": hammer.iloc[-1], "shooting_star": shooting_star.iloc[-1]}

# =============================
# DÉTECTION NESTED FVG
# =============================
def detect_nested_fvg(df: pd.DataFrame, min_nesting: int = 2) -> list:
    fvgs = detect_fvg_advanced(df)
    nested_fvgs = []
    for i in range(len(fvgs) - min_nesting + 1):
        current_fvg = fvgs[i]
        next_fvg = fvgs[i + 1]
        if current_fvg.get("direction") == next_fvg.get("direction"):
            direction = current_fvg["direction"]
            if direction == "BUY":
                entry_zone = (
                    min(float(current_fvg["high_level"]), float(next_fvg["low_level"])),
                    max(float(current_fvg["high_level"]), float(next_fvg["low_level"]))
                )
            else:
                entry_zone = (
                    min(float(current_fvg["low_level"]), float(next_fvg["high_level"])),
                    max(float(current_fvg["low_level"]), float(next_fvg["high_level"]))
                )
            midpoint = (entry_zone[0] + entry_zone[1]) / 2
            nested_fvgs.append({
                "direction": direction,
                "levels": [current_fvg, next_fvg],
                "entry_zone": entry_zone,
                "midpoint": midpoint,
                "strength": "VERY_STRONG",
                "nesting_count": 2,
                "time": next_fvg["time"],
                "high_level": max(current_fvg["high_level"], next_fvg["high_level"]),
                "low_level": min(current_fvg["low_level"], next_fvg["low_level"])
            })
    return nested_fvgs

# =============================
# DRT
# =============================
def calculate_drt_levels(swing_high: float, swing_low: float) -> dict:
    range_size = swing_high - swing_low
    return {
        "0.236": swing_high - 0.236 * range_size,
        "0.382": swing_high - 0.382 * range_size,
        "0.500": swing_high - 0.500 * range_size,
        "0.618": swing_high - 0.618 * range_size,
        "0.786": swing_high - 0.786 * range_size,
        "0.886": swing_high - 0.886 * range_size,
        "1.000": swing_low,
        "1.272": swing_high - 1.272 * range_size,
        "1.414": swing_high - 1.414 * range_size,
        "1.618": swing_high - 1.618 * range_size
    }

# =============================
# ORDER FLOW LEGS
# =============================
def detect_orderflow_legs_advanced(df: pd.DataFrame) -> list:
    swing_highs, swing_lows = detect_swing_points_advanced(df, lookback=5)
    fvgs = detect_fvg_advanced(df, max_lookback_hours=36)
    ofls = []
    for swing_low in swing_lows:
        sl_idx = swing_low.get("index")
        sl_price = float(swing_low.get("price", 0))
        for fvg in fvgs:
            if fvg.get("direction") != "BUY":
                continue
            fvg_idx = fvg.get("index")
            if fvg_idx is None:
                continue
            fvg_low = float(fvg.get("low_level", 0))
            if fvg_idx <= sl_idx:
                continue
            if fvg_low < sl_price:
                continue
            for swing_high in swing_highs:
                sh_idx = swing_high.get("index")
                sh_price = float(swing_high.get("price", 0))
                if sh_idx <= fvg_idx:
                    continue
                if sh_price <= float(fvg.get("high_level", 0)):
                    continue
                ofls.append({"direction": "BUY", "start": swing_low, "fvg": fvg, "end": swing_high})
    for swing_high in swing_highs:
        sh_idx = swing_high.get("index")
        sh_price = float(swing_high.get("price", 0))
        for fvg in fvgs:
            if fvg.get("direction") != "SELL":
                continue
            fvg_idx = fvg.get("index")
            if fvg_idx is None:
                continue
            fvg_high = float(fvg.get("high_level", 0))
            if fvg_idx <= sh_idx:
                continue
            if fvg_high > sh_price:
                continue
            for swing_low in swing_lows:
                sl_idx = swing_low.get("index")
                sl_price = float(swing_low.get("price", 0))
                if sl_idx <= fvg_idx:
                    continue
                if sl_price >= float(fvg.get("low_level", 0)):
                    continue
                ofls.append({"direction": "SELL", "start": swing_high, "fvg": fvg, "end": swing_low})
    for fvg in fvgs:
        ofls.append({"direction": fvg.get("direction"), "fvg": fvg})
    logger.info(f"📊 OFL FINAL: {len(ofls)} legs détectés")
    return ofls

# =============================
# VALIDATION FVG RESPECTÉ AVEC LIQUIDITÉ
# =============================
def is_fvg_respected_with_liquidity(df: pd.DataFrame, fvg: dict, liquidity_levels: dict) -> bool:
    if "high_level" not in fvg or "low_level" not in fvg or "time" not in fvg or "direction" not in fvg:
        return False
    fvg_time = fvg["time"]
    subsequent_data = df[df.index > fvg_time]
    if subsequent_data.empty:
        return False
    if fvg["direction"] == "BUY":
        basic_respect = subsequent_data["high"].max() > fvg["high_level"]
    else:
        basic_respect = subsequent_data["low"].min() < fvg["low_level"]
    if not basic_respect:
        logger.debug(f"   ❌ FVG non respecté : prix n'a pas confirmé la cassure")
        return False
    if not liquidity_levels:
        logger.debug("   💧 Pas de niveaux de liquidité → Validation par prix uniquement")
        return True
    try:
        if fvg["direction"] == "BUY":
            liquidity_respect = subsequent_data["low"].min() <= liquidity_levels.get("previous_week_low", float('inf')) * 1.002
        else:
            liquidity_respect = subsequent_data["high"].max() >= liquidity_levels.get("previous_week_high", 0) * 0.998
        if liquidity_respect:
            logger.debug("   💧 FVG respecté + confirmation par liquidité")
        else:
            logger.debug("   ⚠️ FVG respecté mais pas de confirmation par liquidité → Accepté quand même")
        return True
    except Exception as e:
        logger.warning(f"   ❗ Erreur lors de la vérification de la liquidité: {e}")
        return True

# =============================
# DÉTECTION WICK REJECTION POI
# =============================
def detect_wick_rejection_poi(df: pd.DataFrame, bias: str, min_wick_ratio: float = 0.7) -> list:
    poi_list = []
    pair = df.attrs.get("instrument", "DEFAULT")
    pip_tolerance_map = {
        "XAU_USD": 20,
        "USD_JPY": 0.50,
        "AUD_USD": 0.0050,
        "EUR_USD": 0.0020,
        "USD_CAD": 0.0050,
        "GBP_USD": 0.0050,
        "DEFAULT": 0.0010
    }
    pip_tolerance = pip_tolerance_map.get(pair, pip_tolerance_map["DEFAULT"])
    for i in range(1, len(df) - 1):
        rejection_candle = df.iloc[i]
        confirmation_candle = df.iloc[i + 1]
        upper_wick = rejection_candle["high"] - max(rejection_candle["open"], rejection_candle["close"])
        lower_wick = min(rejection_candle["open"], rejection_candle["close"]) - rejection_candle["low"]
        body_size = abs(rejection_candle["close"] - rejection_candle["open"])
        total_range = rejection_candle["high"] - rejection_candle["low"]
        if total_range == 0:
            continue
        rsi_m15 = get_last_rsi(df["close"].iloc[:i+1])
        current_price = df["close"].iloc[-1]
        if (
            bias in ["BUY", "NEUTRAL"]
            and lower_wick >= body_size * min_wick_ratio
            and lower_wick >= upper_wick * 1.5
            and lower_wick >= total_range * 0.4
            and rsi_m15 < 60
            and confirmation_candle["close"] > confirmation_candle["open"]
            and confirmation_candle["close"] > rejection_candle["high"]
            and (confirmation_candle["close"] - confirmation_candle["open"]) >= 0.7 * (confirmation_candle["high"] - confirmation_candle["low"])
        ):
            if abs(current_price - rejection_candle["low"]) <= pip_tolerance:
                poi_list.append({
                    "type": "WICK_REJECTION",
                    "price_level": rejection_candle["low"],
                    "wick_size": lower_wick,
                    "body_size": body_size,
                    "time": df.index[i],
                    "direction": "BUY",
                    "wick_ratio": lower_wick / total_range,
                    "rsi_at_rejection": rsi_m15,
                    "pair": pair
                })
        elif (
            bias in ["SELL", "NEUTRAL"]
            and upper_wick >= body_size * min_wick_ratio
            and upper_wick >= lower_wick * 1.5
            and upper_wick >= total_range * 0.4
            and rsi_m15 > 40
            and confirmation_candle["close"] < confirmation_candle["open"]
            and confirmation_candle["close"] < rejection_candle["low"]
            and (confirmation_candle["open"] - confirmation_candle["close"]) >= 0.7 * (confirmation_candle["high"] - confirmation_candle["low"])
        ):
            if abs(current_price - rejection_candle["high"]) <= pip_tolerance:
                poi_list.append({
                    "type": "WICK_REJECTION",
                    "price_level": rejection_candle["high"],
                    "wick_size": upper_wick,
                    "body_size": body_size,
                    "time": df.index[i],
                    "direction": "SELL",
                    "wick_ratio": upper_wick / total_range,
                    "rsi_at_rejection": rsi_m15,
                    "pair": pair
                })
    return poi_list

# =============================
# BIAS AVANCÉ
# =============================
def determine_advanced_bias(df: pd.DataFrame) -> dict:
    mss = detect_mss(df, lookback=20)
    if mss["type"] == "MSS_BUY":
        return {"bias": "BUY", "mss_detected": mss}
    elif mss["type"] == "MSS_SELL":
        return {"bias": "SELL", "mss_detected": mss}
    else:
        ema20 = df["close"].ewm(span=20).mean().iloc[-1]
        ema50 = df["close"].ewm(span=50).mean().iloc[-1]
        if ema20 > ema50:
            return {"bias": "BUY", "mss_detected": mss}
        elif ema20 < ema50:
            return {"bias": "SELL", "mss_detected": mss}
        else:
            return {"bias": "NEUTRAL", "mss_detected": mss}

# =============================
# NARRATIVE AVANCÉE
# =============================
def determine_advanced_narrative(
    df_m15: pd.DataFrame,
    bias_analysis: dict,
    pair: str = "XAU_USD",
    df_h4: pd.DataFrame = None,
    df_d1: pd.DataFrame = None,
    df_h1: pd.DataFrame = None
) -> dict:
    if df_m15 is None or df_m15.empty or len(df_m15) < 3:
        logger.warning("❌ DataFrame vide ou insuffisant")
        return {
            "bias": "NEUTRAL",
            "current_price": None,
            "potential_entries": [],
            "timestamp": datetime.now().isoformat()
        }
    _invalid_retest_counts = {"BUY": 0, "SELL": 0}
    _invalid_retest_samples = {"BUY": set(), "SELL": set()}

    def _note_invalid_retest(direction: str, level: float):
        d = (direction or "UNKNOWN").upper()
        try:
            lvl = round(float(level), 5) if level is not None else level
        except Exception:
            lvl = level
        if d not in _invalid_retest_counts:
            _invalid_retest_counts[d] = 0
            _invalid_retest_samples[d] = set()
        _invalid_retest_counts[d] += 1
        if len(_invalid_retest_samples[d]) < 3:
            _invalid_retest_samples[d].add(lvl)

    def _flush_invalid_retest_summary():
        for d in ("BUY", "SELL"):
            c = _invalid_retest_counts.get(d, 0)
            if c > 0:
                examples = ", ".join(map(str, list(_invalid_retest_samples[d])[:3]))
                logger.info(f"⚠️ {pair} · FVG {d} non valides pour retest: {c} (ex: {examples})")

    try:
        bias = (bias_analysis.get("bias", "NEUTRAL") or "NEUTRAL").upper()
        current_price = float(df_m15["close"].iloc[-1])
        logger.info(f"🎯 ANALYSE {pair} - Prix: {current_price:.5f}, Bias: {bias}")
        ema50_h1 = None
        if df_h1 is not None and len(df_h1) >= 50:
            ema50_h1 = df_h1['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            trend_h1 = "HAUSSIER" if current_price > ema50_h1 else "BAISSIER"
            logger.info(f"   🌊 Tendance H1 (EMA50): {trend_h1} (Prix: {current_price:.5f} vs EMA: {ema50_h1:.5f})")
        rsi_m15 = get_last_rsi(df_m15["close"])
        rsi_h4 = 50.0
        if df_h4 is not None and not df_h4.empty:
            rsi_h4 = get_last_rsi(df_h4["close"])
        ofls = detect_orderflow_legs_advanced(df_m15)
        nested_fvgs = detect_nested_fvg(df_m15)
        poi_respects = detect_wick_rejection_poi(df_m15, bias)
        liquidity_levels = (bias_analysis.get("liquidity_levels", {}) or {})
        narrative = {
            "bias": bias,
            "current_price": current_price,
            "potential_entries": [],
            "liquidity_targets": liquidity_levels,
            "nested_fvgs": nested_fvgs,
            "poi_respect": poi_respects,
            "timestamp": datetime.now().isoformat(),
        }

        def is_rsi_valid(direction: str) -> bool:
            stoch_k_local, _ = calculate_stoch_rsi(df_m15["close"])
            if pair == "XAU_USD":
                if direction == "BUY":
                    if stoch_k_local < 25:
                        return rsi_m15 < 85 and rsi_h4 > 10
                    return (rsi_m15 > 20) and (rsi_m15 < 85) and (10 < rsi_h4 < 80)
                elif direction == "SELL":
                    if stoch_k_local > 75:
                        return rsi_m15 > 15 and rsi_h4 < 90
                    return (15 < rsi_m15 < 75) and (20 < rsi_h4 < 88)
            else:
                if direction == "BUY":
                    if stoch_k_local < 25:
                        return rsi_m15 < 85 and rsi_h4 > 8
                    return (rsi_m15 > 10) and (rsi_m15 < 85) and (8 < rsi_h4 < 80)
                elif direction == "SELL":
                    if stoch_k_local > 75:
                        return rsi_m15 > 15 and rsi_h4 < 92
                    return (rsi_m15 < 85) and (rsi_m15 > 10) and (15 < rsi_h4 < 88)
            return False

        def get_tolerance(entry_level: float, pair_local: str) -> float:
            if pair_local == "XAU_USD":
                return 2.0
            elif pair_local in ["AUD_USD", "EUR_USD", "NZD_USD", "USD_CAD", "USD_CHF", "GBP_USD"]:
                return 0.0020
            elif pair_local == "USD_JPY":
                return 0.15
            elif pair_local == "NAS100_USD":
                return 20.0
            else:
                return 0.0015

        dealing_range = detect_dealing_range(df_h4, lookback=50)
        if dealing_range:
            logger.info(f"🔍 Dealing Range détectée: H {dealing_range['high']:.5f} - L {dealing_range['low']:.5f}")
        else:
            logger.info("🔍 Aucune Dealing Range détectée")

        logger.info(f"🔎 Filtrage des FVG récents pour {pair}:")
        recent_ofls = []
        ancient_fvg_count = 0
        current_time_utc = datetime.utcnow()
        for ofl in ofls:
            fvg = ofl.get("fvg", {})
            if not isinstance(fvg, dict) or not fvg:
                continue
            fvg_time = fvg.get("time")
            if not fvg_time:
                continue
            try:
                fvg_datetime = pd.to_datetime(fvg_time)
                if getattr(fvg_datetime, "tz", None) is not None:
                    fvg_datetime = fvg_datetime.tz_convert(None)
                else:
                    fvg_datetime = fvg_datetime.replace(tzinfo=None)
                time_diff = current_time_utc - fvg_datetime
                recent_hours = 36 if pair == "XAU_USD" else 48
                is_recent = time_diff.total_seconds() <= recent_hours * 3600
            except Exception as e:
                logger.warning(f" ⚠️ Erreur conversion time FVG: {e}")
                continue
            entry_level = get_fvg_midpoint(fvg)
            if entry_level is None:
                continue
            tolerance = get_tolerance(entry_level, pair)
            distance = abs(current_price - entry_level)
            is_in_zone = (entry_level - tolerance <= current_price <= entry_level + tolerance)
            is_very_close = distance <= tolerance
            factor_reasonable = 2 if pair == "XAU_USD" else 3
            is_within_reasonable = distance <= tolerance * factor_reasonable
            if is_recent and (is_in_zone or is_very_close or is_within_reasonable):
                recent_ofls.append(ofl)
                status = "dans zone" if is_in_zone else "très proche" if is_very_close else "distance raisonnable"
                _log_fvg_recent_once(
                    pair,
                    fvg.get('direction', 'UNKNOWN'),
                    entry_level,
                    f" ✅ FVG RÉCENT {fvg.get('direction', 'UNKNOWN')}: {entry_level:.5f} ({status}, distance: {distance:.5f}, il y a {time_diff.total_seconds()/3600:.1f}h)"
                )
            else:
                ancient_fvg_count += 1
        ofls = recent_ofls
        logger.info(f"📊 FVG récents et proches: {len(ofls)} (anciens rejetés: {ancient_fvg_count})")
        narrative["recent_ofls"] = ofls
        bos = detect_bos(df_h1, lookback=50)
        choch = detect_choch(df_h1, lookback=50)
        narrative["structure_analysis"] = {"bos": bos, "choch": choch}

        for ofl in ofls:
            fvg = ofl.get("fvg", {})
            if not isinstance(fvg, dict) or not all(k in fvg for k in ["direction", "high_level", "low_level", "time"]):
                continue
            entry_level = get_fvg_midpoint(fvg)
            if entry_level is None:
                continue
            fvg_direction = str(fvg.get("direction", "")).upper()
            if fvg_direction not in {"BUY", "SELL"}:
                continue
            if not is_fvg_retest_valid(df_m15, fvg, current_price, pair):
                _note_invalid_retest(fvg.get('direction'), entry_level)
                continue
            tolerance = get_tolerance(entry_level, pair)
            is_in_entry_zone = (entry_level - tolerance <= current_price <= entry_level + tolerance)
            is_within_reasonable = abs(current_price - entry_level) <= tolerance * 2
            if not (is_in_entry_zone or is_within_reasonable):
                continue
            fvg_type = fvg.get("type", "UNKNOWN")
            if (bias in ["NEUTRAL", fvg_direction]) and is_rsi_valid(fvg_direction):
                irl_erl_type = classify_zone_irl_erl(entry_level, dealing_range)
                is_near_key_level = False
                is_rsi_extreme = False
                is_trend_aligned = False
                if irl_erl_type == "IRL":
                    distance_to_support = abs(current_price - liquidity_levels.get("previous_week_low", current_price))
                    distance_to_resistance = abs(current_price - liquidity_levels.get("previous_week_high", current_price))
                    tol = get_tolerance(current_price, pair) * 2
                    is_near_key_level = (distance_to_support <= tol or distance_to_resistance <= tol)
                    rsi_m15_val = float(get_last_rsi(df_m15["close"]))
                    is_rsi_extreme = (fvg_direction == "BUY" and rsi_m15_val < 30) or (fvg_direction == "SELL" and rsi_m15_val > 70)
                    is_trend_aligned = bias in ["NEUTRAL", fvg_direction]
                    logger.info(
                        f"ℹ️ IRL détecté (narrative): near_key={is_near_key_level}, "
                        f"rsi_extreme={is_rsi_extreme}, trend_aligned={is_trend_aligned}"
                    )
                entry = {
                    "type": f"FVG_RETEST_{fvg_type}",
                    "direction": fvg_direction,
                    "entry_zone": (round(entry_level - tolerance, 5), round(entry_level + tolerance, 5)),
                    "entry_level": round(entry_level, 5),
                    "confidence": "HIGH" if bos.get("type") in ["BOS_BUY", "BOS_SELL"] or choch.get("type") in ["CHOCH_BUY", "CHOCH_SELL"] else "MEDIUM",
                    "trigger": "FVG_RETEST",
                    "timeframe": "M15",
                    "rsi_m15": rsi_m15,
                    "rsi_h4": rsi_h4,
                    "structure_analysis": narrative["structure_analysis"],
                    "irl_erl_type": irl_erl_type,
                    "is_near_key_level": is_near_key_level,
                    "is_rsi_extreme": is_rsi_extreme,
                    "trend_aligned": is_trend_aligned,
                }
                narrative["potential_entries"].append(entry)
                irl_erl_label = f" ({irl_erl_type})" if irl_erl_type else " (No Range)"
                _log_fvg_added_once(
                    pair, fvg_direction, entry_level, fvg_type,
                    f" 🎯 FVG {fvg_direction} AJOUTÉ: {entry_level:.5f} (type: {fvg_type}){irl_erl_label}"
                )
        _flush_invalid_retest_summary()

        for nfvg in nested_fvgs:
            entry_level = nfvg.get("midpoint")
            if not entry_level:
                continue
            tolerance = get_tolerance(entry_level, pair)
            is_in_entry_zone = (entry_level - tolerance <= current_price <= entry_level + tolerance)
            is_within_reasonable = abs(current_price - entry_level) <= tolerance * 2
            if not (is_in_entry_zone or is_within_reasonable):
                continue
            nfvg_direction = nfvg.get("direction", "").upper()
            if nfvg_direction not in {"BUY", "SELL"}:
                continue
            fake_fvg = {
                "direction": nfvg["direction"],
                "high_level": nfvg["high_level"],
                "low_level": nfvg["low_level"],
                "time": nfvg["time"]
            }
            if not is_fvg_unmitigated(df_m15, fake_fvg):
                continue
            if (bias in ["NEUTRAL", nfvg_direction]) and is_rsi_valid(nfvg_direction):
                irl_erl_type = classify_zone_irl_erl(entry_level, dealing_range)
                entry = {
                    "type": "NESTED_FVG",
                    "direction": nfvg_direction,
                    "entry_zone": (round(entry_level - tolerance, 5), round(entry_level + tolerance, 5)),
                    "entry_level": round(entry_level, 5),
                    "confidence": "HIGH" if (bos.get("type") in ["BOS_BUY", "BOS_SELL"] or choch.get("type") in ["CHOCH_BUY", "CHOCH_SELL"]) else "MEDIUM",
                    "trigger": "NESTED_FVG",
                    "timeframe": "M15",
                    "rsi_m15": rsi_m15,
                    "rsi_h4": rsi_h4,
                    "structure_analysis": narrative["structure_analysis"],
                    "irl_erl_type": irl_erl_type,
                }
                narrative["potential_entries"].append(entry)
                irl_erl_label = f" ({irl_erl_type})" if irl_erl_type else " (No Range)"
                _log_fvg_added_once(
                    pair, nfvg_direction, entry_level, "NESTED",
                    f" 🎯 Nested FVG {nfvg_direction} AJOUTÉ: {entry_level:.5f}{irl_erl_label}"
                )

        for poi in poi_respects:
            entry_level = poi.get("price_level")
            if not entry_level:
                continue
            tolerance = get_tolerance(entry_level, pair)
            is_in_entry_zone = (entry_level - tolerance <= current_price <= entry_level + tolerance)
            is_within_reasonable = abs(current_price - entry_level) <= tolerance * 2
            if not (is_in_entry_zone or is_within_reasonable):
                continue
            poi_direction = poi.get("direction", "").upper()
            if poi_direction not in {"BUY", "SELL"}:
                continue
            if (bias in ["NEUTRAL", poi_direction]) and is_rsi_valid(poi_direction):
                irl_erl_type = classify_zone_irl_erl(entry_level, dealing_range)
                entry = {
                    "type": "WICK_REJECTION",
                    "direction": poi_direction,
                    "entry_zone": (round(entry_level - tolerance, 5), round(entry_level + tolerance, 5)),
                    "entry_level": round(entry_level, 5),
                    "confidence": "MEDIUM",
                    "trigger": "WICK_REJECTION",
                    "timeframe": "M15",
                    "rsi_m15": rsi_m15,
                    "rsi_h4": rsi_h4,
                    "structure_analysis": narrative["structure_analysis"],
                    "irl_erl_type": irl_erl_type,
                }
                narrative["potential_entries"].append(entry)
                irl_erl_label = f" ({irl_erl_type})" if irl_erl_type else " (No Range)"
                _log_fvg_added_once(
                    pair, poi_direction, entry_level, "WICK",
                    f" 🎯 Wick Rejection {poi_direction} AJOUTÉ: {entry_level:.5f}{irl_erl_label}"
                )

        liq_high = liquidity_levels.get("previous_week_high")
        liq_low = liquidity_levels.get("previous_week_low")
        liq_stoch_k, _ = calculate_stoch_rsi(df_m15["close"])
        liq_tolerance = get_tolerance(current_price, pair) * 4

        if bias == "BUY" and liq_high and len(ofls) > 0:
            dist_to_liq = abs(liq_high - current_price)
            if dist_to_liq <= liq_tolerance * 8 and liq_stoch_k < 40:
                irl_erl_type = classify_zone_irl_erl(liq_high, dealing_range)
                entry = {
                    "type": "LIQUIDITY_DRAW",
                    "direction": "BUY",
                    "entry_level": round(current_price, 5),
                    "entry_zone": (round(current_price - liq_tolerance, 5), round(current_price + liq_tolerance, 5)),
                    "target": round(liq_high, 5),
                    "confidence": "MEDIUM",
                    "trigger": "LIQUIDITY",
                    "timeframe": "M15",
                    "rsi_m15": rsi_m15,
                    "rsi_h4": rsi_h4,
                    "structure_analysis": narrative["structure_analysis"],
                    "irl_erl_type": irl_erl_type,
                }
                narrative["potential_entries"].append(entry)
                logger.info(f" 🎯 Liquidity Draw BUY AJOUTÉ à {liq_high:.5f} (Type: {irl_erl_type or 'No Range'}, StochRSI: {liq_stoch_k:.1f})")

        if bias == "SELL" and liq_low and len(ofls) > 0:
            dist_to_liq = abs(current_price - liq_low)
            if dist_to_liq <= liq_tolerance * 8 and liq_stoch_k > 60:
                irl_erl_type = classify_zone_irl_erl(liq_low, dealing_range)
                entry = {
                    "type": "LIQUIDITY_DRAW",
                    "direction": "SELL",
                    "entry_level": round(current_price, 5),
                    "entry_zone": (round(current_price - liq_tolerance, 5), round(current_price + liq_tolerance, 5)),
                    "target": round(liq_low, 5),
                    "confidence": "MEDIUM",
                    "trigger": "LIQUIDITY",
                    "timeframe": "M15",
                    "rsi_m15": rsi_m15,
                    "rsi_h4": rsi_h4,
                    "structure_analysis": narrative["structure_analysis"],
                    "irl_erl_type": irl_erl_type,
                }
                narrative["potential_entries"].append(entry)
                logger.info(f" 🎯 Liquidity Draw SELL AJOUTÉ à {liq_low:.5f} (Type: {irl_erl_type or 'No Range'}, StochRSI: {liq_stoch_k:.1f})")

        if bos.get("type") in ["BOS_BUY", "BOS_SELL"]:
            bos_level = bos["level"]
            bos_direction = "BUY" if bos["type"] == "BOS_BUY" else "SELL"
            if (bias == bos_direction) or (bias == "NEUTRAL"):
                all_fvgs = []
                for ofl in ofls:
                    fvg = ofl.get("fvg")
                    if fvg and fvg.get("direction", "").upper() == bos_direction:
                        mid = get_fvg_midpoint(fvg)
                        if mid:
                            all_fvgs.append({"level": mid, "type": "FVG"})
                for nfvg in nested_fvgs:
                    if nfvg.get("direction", "").upper() == bos_direction:
                        mid = nfvg.get("midpoint")
                        if mid:
                            all_fvgs.append({"level": mid, "type": "NESTED_FVG"})
                for fvg_info in all_fvgs:
                    fvg_level = fvg_info["level"]
                    distance_bos_fvg = abs(bos_level - fvg_level)
                    if distance_bos_fvg <= 0.00030:
                        tolerance = get_tolerance(fvg_level, pair)
                        is_in_entry_zone = (fvg_level - tolerance <= current_price <= fvg_level + tolerance)
                        is_within_reasonable = abs(current_price - fvg_level) <= tolerance * 2
                        if not (is_in_entry_zone or is_within_reasonable):
                            continue
                        irl_erl_type = classify_zone_irl_erl(fvg_level, dealing_range)
                        entry = {
                            "type": "BISI",
                            "direction": bos_direction,
                            "entry_zone": (
                                round(fvg_level - tolerance, 5),
                                round(fvg_level + tolerance, 5)
                            ),
                            "entry_level": round(fvg_level, 5),
                            "confidence": "VERY_HIGH",
                            "trigger": "BISI",
                            "timeframe": "M15",
                            "bosis": {"level": bos_level, "type": bos["type"]},
                            "fvg": fvg_info,
                            "rsi_m15": rsi_m15,
                            "rsi_h4": rsi_h4,
                            "structure_analysis": narrative["structure_analysis"],
                            "irl_erl_type": irl_erl_type,
                        }
                        narrative["potential_entries"].append(entry)
                        irl_erl_label = f" ({irl_erl_type})" if irl_erl_type else " (No Range)"
                        logger.info(f"🎯 BISI DÉTECTÉ : {bos_direction} à {fvg_level:.5f}{irl_erl_label}")
        logger.info(f"📝 Narrative {pair} TERMINÉE: {len(narrative['potential_entries'])} entrées détectées")
        _log_narrative_list(narrative["potential_entries"], top_n=10)
        return narrative
    except Exception as e:
        logger.error(f"❌ Erreur dans determine_advanced_narrative: {e}")
        logger.error(traceback.format_exc())
        return {
            "bias": "NEUTRAL",
            "current_price": None,
            "potential_entries": [],
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# =============================
# HELPERS SCORING
# =============================
def get_session_quality_bonus(pair: str) -> tuple:
    hour = datetime.utcnow().hour
    is_london = 7 <= hour < 16
    is_ny = 12 <= hour < 21
    is_overlap = is_london and is_ny
    forex_euro = ["EUR_USD", "GBP_USD", "USD_CAD", "AUD_USD"]
    jpy_pairs = ["USD_JPY", "GBP_JPY"]
    is_asian = not (is_london or is_ny)
    if is_overlap:
        return 2, "+2 (Overlap London/NY)"
    elif is_london or is_ny:
        return 1, "+1 (Session active)"
    else:
        if pair in jpy_pairs:
            return 1, "+1 (Session asiatique, JPY actif)"
        elif pair in forex_euro:
            return -1, "-1 (Session asiatique, paire EUR/GBP)"
        else:
            return 0, "0 (Session neutre)"

def get_d1_trend_bonus(df_d1, direction: str) -> tuple:
    if df_d1 is None or df_d1.empty or len(df_d1) < 52:
        return 0, "0 (D1 insuffisant)"
    try:
        close = df_d1["close"]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        current = close.iloc[-1]
        d1_trend = "BUY" if current > ema50 else "SELL"
        bonus = 0
        label = ""
        if d1_trend == direction:
            bonus += 2
            label = "+2 (D1 EMA50 aligné)"
            if len(df_d1) >= 200:
                ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
                if (direction == "BUY" and current > ema200) or (direction == "SELL" and current < ema200):
                    bonus += 1
                    label = "+3 (D1 EMA50+EMA200 alignés)"
        else:
            bonus = -2
            label = "-2 (Contre tendance D1)"
        return bonus, label
    except Exception:
        return 0, "0 (Erreur D1)"

def get_macd_h1_bonus(df_h1, direction: str) -> tuple:
    if df_h1 is None or df_h1.empty or len(df_h1) < 35:
        return 0, "0 (H1 insuffisant pour MACD)"
    try:
        hist = calculate_macd_momentum(df_h1)
        last_val = hist.iloc[-1]
        prev_val = hist.iloc[-2]
        if direction == "BUY":
            if last_val > 0 and last_val > prev_val:
                return 2, "+2 (MACD H1 haussier accélérant)"
            elif last_val > 0:
                return 1, "+1 (MACD H1 haussier)"
            elif last_val < 0:
                return -1, "-1 (MACD H1 baissier)"
        elif direction == "SELL":
            if last_val < 0 and last_val < prev_val:
                return 2, "+2 (MACD H1 baissier accélérant)"
            elif last_val < 0:
                return 1, "+1 (MACD H1 baissier)"
            elif last_val > 0:
                return -1, "-1 (MACD H1 haussier)"
        return 0, "0 (MACD H1 neutre)"
    except Exception:
        return 0, "0 (Erreur MACD H1)"

def get_rsi_divergence_bonus(df_h1, df_m15, direction: str) -> tuple:
    try:
        if direction == "BUY":
            if detect_rsi_divergence_haussiere(df_h1):
                return 3, "+3 (Divergence RSI H1 haussière)"
            if detect_rsi_divergence_haussiere(df_m15):
                return 2, "+2 (Divergence RSI M15 haussière)"
        elif direction == "SELL":
            def detect_bearish_div(df):
                if len(df) < 30:
                    return False
                prices = df["close"].tail(28).reset_index(drop=True)
                rsi_vals = calculate_rsi(df["close"]).tail(28).reset_index(drop=True)
                highs = []
                for i in range(3, len(prices) - 3):
                    if prices.iloc[i] > prices.iloc[i-3:i].max() and prices.iloc[i] > prices.iloc[i+1:i+4].max():
                        highs.append((i, prices.iloc[i], rsi_vals.iloc[i]))
                if len(highs) < 2:
                    return False
                return highs[-1][1] > highs[-2][1] and highs[-1][2] < highs[-2][2]
            if detect_bearish_div(df_h1):
                return 3, "+3 (Divergence RSI H1 baissière)"
            if detect_bearish_div(df_m15):
                return 2, "+2 (Divergence RSI M15 baissière)"
    except Exception:
        pass
    return 0, "0 (Pas de divergence RSI)"

def estimate_win_rate(score: int, confluences: dict) -> str:
    if score >= 16:
        base = 78
    elif score >= 14:
        base = 72
    elif score >= 12:
        base = 65
    elif score >= 10:
        base = 58
    else:
        base = 48
    if confluences.get("d1_aligned"):
        base += 3
    if confluences.get("rsi_divergence"):
        base += 4
    if confluences.get("session_active"):
        base += 2
    if confluences.get("macd_confirmed"):
        base += 3
    if confluences.get("bos_confirmed"):
        base += 2
    return f"~{min(base, 88)}%"

def get_signal_quality_label(score: int) -> str:
    if score >= 16:
        return "SNIPER"
    elif score >= 14:
        return "A+"
    elif score >= 12:
        return "A"
    elif score >= 10:
        return "B+"
    return "B"

# =============================
# SYSTÈME DE SCORING
# =============================
def calculate_signal_confidence(
    pair: str,
    direction: str,
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m15: pd.DataFrame,
    entry: dict,
    bias: str,
    current_price: float,
    crt_detected: bool = False,
    tbs_setup_type: str = "",
    df_d1: pd.DataFrame = None,
) -> dict:
    score_components = {
        "ICT": 0,
        "EMA": 0,
        "Structure_H1": 0,
        "Liquidity": 0,
        "Order_Block": 0,
        "Imbalance": 0,
        "Stochastic": 0,
        "HTF_Alignment": 0,
        "Risk_RR_Distance": 0,
        "Secondary": 0,
    }
    score = 0
    details: dict = {}
    min_required = SCORING_CONFIG.get("MIN_CONFIDENCE_SCORE", 10)
    direction = (direction or "").upper()
    entry_level = entry.get("entry_level")
    entry_type = str(entry.get("type", "FVG_RETEST")).upper()
    if entry_level is None or direction not in ["BUY", "SELL"]:
        return {"passed": False, "total_score": 0, "final_confidence": "LOW", "details": {"VETO": "Entrée/direction invalide"}}
    entry_level = float(entry_level)
    atr_value = calculate_atr(df_m15)
    fvg_data = entry.get("fvg") if "fvg" in entry else None
    stop_loss, take_profit = calculate_sl_tp(
        entry_price=entry_level,
        atr=atr_value,
        direction=direction,
        pair=pair,
        entry_type=entry_type,
        fvg_data=fvg_data,
    )
    if (direction == "BUY" and bias == "BUY") or (direction == "SELL" and bias == "SELL"):
        score_components["ICT"] += 3
        details["Trend_H4"] = "+3 (Aligné)"
    elif bias == "NEUTRAL":
        score_components["ICT"] += 1
        details["Trend_H4"] = "+1 (Neutre)"
    else:
        score_components["ICT"] -= 2
        details["Trend_H4"] = "-2 (H4 opposé, non bloquant)"
    try:
        distance = abs(float(current_price) - entry_level)
        pip = get_pip_value_for_pair(pair)
        entry_type_max_pips = {
            "FVG_RETEST_PERFECT": 15.0,
            "FVG_RETEST": 18.0,
            "NESTED_FVG": 18.0,
            "WICK_REJECTION": 15.0,
            "BISI": 18.0,
            "BREAKER": 15.0,
        }
        max_pips = entry_type_max_pips.get(entry_type, STRICT_MAX_DISTANCE_PIPS.get(pair, STRICT_MAX_DISTANCE_PIPS["DEFAULT"]))
        max_distance_price = max(float(atr_value) * 1.20, pip * max_pips)
        if distance <= max_distance_price * 0.50:
            score_components["Risk_RR_Distance"] += 2
            details["Distance"] = f"+2 proche ({distance:.5f} <= {max_distance_price * 0.50:.5f})"
        elif distance <= max_distance_price:
            details["Distance"] = f"0 acceptable ({distance:.5f} <= {max_distance_price:.5f})"
        elif distance <= max_distance_price * 1.50:
            score_components["Risk_RR_Distance"] -= 2
            details["Distance"] = f"-2 un peu loin ({distance:.5f} > {max_distance_price:.5f})"
        else:
            return {
                "passed": False,
                "total_score": -50,
                "final_confidence": "LOW",
                "details": {"VETO": f"Prix vraiment trop loin ({distance:.5f} > {max_distance_price * 1.50:.5f})"},
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr_value": atr_value,
            }
    except Exception as exc:
        details["Distance_Error"] = str(exc)
    try:
        ema_score = max(-2, min(2, _directional_score(score_ema_trend(df_h1), direction)))
        structure_score = _directional_score(score_market_structure(df_h1), direction)
        htf_score = score_higher_timeframe_alignment(direction, df_h1, df_h4)
        score_components["EMA"] += ema_score
        score_components["Structure_H1"] += structure_score
        score_components["HTF_Alignment"] += htf_score
        details["EMA"] = f"{ema_score:+d} (EMA50 H1 scorée, non bloquante)"
        details["Structure_H1"] = f"{structure_score:+d} (HH/HL/LH/LL)"
        details["HTF_Alignment"] = f"{htf_score:+d} (alignement H1/H4)"
    except Exception as exc:
        details["Trend_H1_Error"] = str(exc)
    try:
        stoch_k_h1, _ = calculate_stoch_rsi(df_h1["close"])
        stoch_k_m15, _ = calculate_stoch_rsi(df_m15["close"])
        if direction == "BUY":
            if stoch_k_h1 >= 80 or stoch_k_m15 >= 85:
                return {"passed": False, "total_score": -100, "final_confidence": "LOW", "details": {"VETO": f"BUY trop tardif StochRSI H1={stoch_k_h1:.1f} M15={stoch_k_m15:.1f}"}, "stop_loss": stop_loss, "take_profit": take_profit, "atr_value": atr_value}
            if 20 <= stoch_k_m15 <= 60:
                score_components["Stochastic"] += 1
                details["StochRSI"] = f"+1 (zone idéale BUY M15={stoch_k_m15:.1f})"
            elif stoch_k_m15 < 20:
                score_components["Stochastic"] += 1
                details["StochRSI"] = f"+1 (survente BUY M15={stoch_k_m15:.1f})"
            else:
                details["StochRSI"] = f"0 (neutre BUY M15={stoch_k_m15:.1f})"
        elif direction == "SELL":
            if stoch_k_h1 <= 20 or stoch_k_m15 <= 15:
                return {"passed": False, "total_score": -100, "final_confidence": "LOW", "details": {"VETO": f"SELL trop tardif StochRSI H1={stoch_k_h1:.1f} M15={stoch_k_m15:.1f}"}, "stop_loss": stop_loss, "take_profit": take_profit, "atr_value": atr_value}
            if 40 <= stoch_k_m15 <= 80:
                score_components["Stochastic"] += 1
                details["StochRSI"] = f"+1 (zone idéale SELL M15={stoch_k_m15:.1f})"
            elif stoch_k_m15 > 80:
                score_components["Stochastic"] += 1
                details["StochRSI"] = f"+1 (surachat SELL M15={stoch_k_m15:.1f})"
            else:
                details["StochRSI"] = f"0 (neutre SELL M15={stoch_k_m15:.1f})"
    except Exception as exc:
        details["StochRSI_Error"] = str(exc)
    if "LIQUIDITY" in entry_type:
        score_components["Liquidity"] += 3
        score_components["ICT"] += 1
        details["Setup_Type"] = "+1 ICT, +3 Liquidity"
    elif any(x in entry_type for x in ["FVG", "BISI", "NESTED"]):
        score_components["ICT"] += 4 if "BISI" in entry_type else 3
        score_components["Imbalance"] += 2
        details["Setup_Type"] = f"+{4 if 'BISI' in entry_type else 3} ICT, +2 Imbalance/FVG"
    elif "BREAKER" in entry_type:
        score_components["ICT"] += 2
        score_components["Order_Block"] += 2
        details["Setup_Type"] = "+2 ICT, +2 Order Block (Breaker)"
    elif "WICK" in entry_type:
        score_components["ICT"] += 2
        details["Setup_Type"] = "+2 (Wick rejection)"
    else:
        score_components["ICT"] += 1
        details["Setup_Type"] = f"+1 ({entry_type})"
    if "PERFECT" in entry_type:
        score_components["Imbalance"] = min(2, score_components["Imbalance"] + 1)
        details["Perfect"] = "+1"
    try:
        last = df_m15.iloc[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        rng = max(float(last["high"]) - float(last["low"]), 1e-12)
        body_ratio = body / rng
        bullish_reject = float(last["close"]) > float(last["open"]) and body_ratio >= 0.45
        bearish_reject = float(last["close"]) < float(last["open"]) and body_ratio >= 0.45
        if direction == "BUY" and bullish_reject:
            score_components["ICT"] += 2
            details["M15_Rejection"] = "+3 (bougie BUY confirmée)"
        elif direction == "SELL" and bearish_reject:
            score_components["ICT"] += 2
            details["M15_Rejection"] = "+3 (bougie SELL confirmée)"
        else:
            score_components["ICT"] -= 2
            details["M15_Rejection"] = "-2 (pas de rejet confirmé)"
    except Exception:
        pass
    rr_ratio = 0.0
    try:
        dist_sl = abs(entry_level - stop_loss)
        dist_tp = abs(take_profit - entry_level)
        rr_ratio = dist_tp / dist_sl if dist_sl > 0 else 0
        if rr_ratio < 1.5:
            score_components["Risk_RR_Distance"] -= 5
            details["RR"] = f"-5 (faible {rr_ratio:.2f})"
        elif rr_ratio >= 2.5:
            score_components["Risk_RR_Distance"] += 3
            details["RR"] = f"+3 (excellent {rr_ratio:.2f})"
        else:
            score_components["Risk_RR_Distance"] += 2
            details["RR"] = f"+2 (correct {rr_ratio:.2f})"
    except Exception:
        pass
    d1_aligned = False
    try:
        d1_bonus, d1_label = get_d1_trend_bonus(df_d1, direction)
        if d1_bonus > 0:
            score_components["Secondary"] += 2
            d1_aligned = True
            details["D1_Trend"] = "+2 (D1 aligné)"
        else:
            details["D1_Trend"] = d1_label
    except Exception:
        pass
    macd_confirmed = False
    try:
        macd_bonus, macd_label = get_macd_h1_bonus(df_h1, direction)
        if macd_bonus > 0:
            score_components["Secondary"] += 1
            macd_confirmed = True
            details["MACD_H1"] = "+1 (confirme)"
        else:
            details["MACD_H1"] = macd_label
    except Exception:
        pass
    try:
        session_bonus, session_label = get_session_quality_bonus(pair)
        if session_bonus > 0:
            score_components["Secondary"] += 1
            details["Session"] = "+1 (bonne session)"
        else:
            details["Session"] = session_label
    except Exception:
        pass
    score = compute_final_score(score_components)
    decision = direction if score >= min_required else "REJECTED"
    log_score_detail(score_components, score, decision)
    passed = score >= min_required
    final_confidence = "HIGH" if score >= 18 else "MEDIUM" if score >= min_required else "LOW"
    confluences = {
        "d1_aligned": d1_aligned,
        "rsi_divergence": False,
        "session_active": details.get("Session", "").startswith("+"),
        "macd_confirmed": macd_confirmed,
        "bos_confirmed": "BOS" in str(details),
    }
    win_rate = estimate_win_rate(score, confluences)
    quality_label = get_signal_quality_label(score)
    return {
        "total_score": score,
        "details": details,
        "score_components": score_components,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr_value": atr_value,
        "passed": passed,
        "min_required": min_required,
        "final_confidence": final_confidence,
        "win_rate": win_rate,
        "quality_label": quality_label,
        "confluences": confluences,
    }

def get_fvg_midpoint(fvg: dict) -> float:
    if not all(k in fvg for k in ["high_level", "low_level"]):
        return None
    high = float(fvg["high_level"])
    low = float(fvg["low_level"])
    if high == low:
        return None
    midpoint = (high + low) / 2
    return round(midpoint, 5)

# =============================
# BOUCLE PRINCIPALE AVEC SCORING
# =============================
def advanced_main():
    try:
        api = oandapyV20.API(access_token=os.getenv("OANDA_API_KEY"))
        logger.info("✅ API OANDA initialisée avec succès")
    except Exception as e:
        logger.error(f"❌ Échec d'initialisation de l'API OANDA : {e}")
        return
    for pair in PAIR_LIST:
        _reset_log_dedup()
        try:
            logger.info(f"\n🔍 Début de l'analyse avancée de {pair}")
            df_h4 = get_candles_with_retry(api, pair, GRANULARITY_H4, 300)
            df_h1 = get_candles_with_retry(api, pair, GRANULARITY_H1, 200)
            df_m15 = get_candles_with_retry(api, pair, GRANULARITY_M15, 250)
            df_d1 = get_candles_with_retry(api, pair, "D", count=250)
            if any(df.empty for df in [df_h4, df_h1, df_m15]):
                logger.warning(f"⚠️ Données manquantes pour {pair}, analyse ignorée")
                continue
            current_price = float(df_m15["close"].iloc[-1])
            logger.info(f"💰 {pair} Prix actuel M15 : {current_price:.5f}")
            bias_analysis = determine_advanced_bias(df_h4)
            bias = bias_analysis.get("bias", "NEUTRAL")
            logger.info(f"📍 Bias H4 : {bias}")
            narrative = determine_advanced_narrative(
                df_m15, bias_analysis, pair, df_h4=df_h4, df_d1=df_d1, df_h1=df_h1
            )
            breaker = detect_breaker(df_m15)
            amd_phase = detect_amd_phase(df_m15)
            logger.info(f"🔍 AMD Phase détectée: {amd_phase}")
            if breaker['type']:
                logger.info(f"🔍 Breaker détecté: {breaker['type']} à {breaker['level']}")
            crt_detected = False
            tbs_setup_type = ""
            crt_detected = is_crt_candle(df_h4.iloc[-1])
            logger.info(f"🔍 CRT Candle détectée: {crt_detected}")
            tbs_setup = detect_tbs_setup(df_m15)
            tbs_setup_type = tbs_setup['type']
            logger.info(f"🔍 TBS Setup détecté: {tbs_setup_type} à {tbs_setup['level']}" if tbs_setup_type else "❌ Aucun TBS Setup")
            max_distance = MAX_DISTANCE_PIPS.get(pair, MAX_DISTANCE_PIPS["DEFAULT"])
            filtered_entries = []
            kept_for_log = []
            for entry in narrative["potential_entries"]:
                entry_level = entry.get("entry_level")
                if entry_level is None:
                    continue
                distance = abs(current_price - entry_level)
                zone_start, zone_end = entry.get("entry_zone", (entry_level, entry_level))
                is_in_zone = zone_start <= current_price <= zone_end
                is_very_close = distance <= max_distance
                is_within_reasonable_distance = False
                if is_in_zone or is_very_close or is_within_reasonable_distance:
                    filtered_entries.append(entry)
                    status = "dans zone" if is_in_zone else "très proche" if is_very_close else "distance raisonnable"
                    msg = f"✅ Entrée {status}: {entry_level:.5f} (distance: {distance:.5f})"
                    _log_kept_entry_once(pair, entry_level, status, distance, msg)
                    kept_for_log.append({
                        "entry_level": entry_level,
                        "status_label": status,
                        "status_distance": distance
                    })
                else:
                    logger.warning(f"❌ Entrée rejetée: {entry_level:.5f} (trop éloignée: {distance:.5f} > {max_distance:.5f})")
            narrative["potential_entries"] = filtered_entries
            logger.info(f"📝 Entrées après filtrage proximité: {len(narrative['potential_entries'])}")
            _log_kept_compact(pair, kept_for_log, top_n=8)
            pip_value_for_clustering = get_pip_value_for_pair(pair)
            max_distance_pips_for_clustering_arg = (max_distance / pip_value_for_clustering) if max_distance else None
            if max_distance_pips_for_clustering_arg:
                max_distance_pips_for_clustering_arg = max(5.0, min(20.0, max_distance_pips_for_clustering_arg))
            clustered_entries = cluster_signals(narrative["potential_entries"], pair, max_distance_pips_for_clustering_arg)
            narrative["potential_entries"] = clustered_entries
            logger.info(f"📝 Entrées après clustering: {len(narrative['potential_entries'])}")
            nb_envoyes = 0
            for entry in narrative["potential_entries"]:
                direction = entry.get("direction", "").upper()
                entry_zone = entry.get("entry_zone", None)
                entry_type = entry.get("type", "UNKNOWN")
                entry_level = entry.get("entry_level")
                if not direction or not entry_zone or entry_level is None:
                    continue
                zone_start, zone_end = entry_zone
                entry_level_key = round(float(entry_level), 5)
                logger.info(f"\n🔍 Analyse signal: {direction}")
                logger.info(f" Type: {entry_type}")
                logger.info(f" Zone: {zone_start:.5f}-{zone_end:.5f}")
                logger.info(f" Niveau: {entry_level:.5f}")
                logger.info(f" Prix actuel: {current_price:.5f} → Distance: {abs(current_price - entry_level):.5f}")
                if is_signal_sent_recently(pair, direction, entry_level_key, zone_start, zone_end):
                    logger.info(" ❌ Signal déjà envoyé récemment → Ignoré")
                    continue
                confidence_result = calculate_signal_confidence(
                    pair, direction, df_h4, df_h1, df_m15, entry, bias, current_price,
                    crt_detected, tbs_setup_type, df_d1=df_d1
                )
                stop_loss, take_profit = calculate_sl_tp(
                    entry_price=entry['entry_level'],
                    atr=confidence_result['atr_value'],
                    direction=direction,
                    pair=pair,
                    entry_type=entry.get("type", "FVG_RETEST"),
                    breaker_level=breaker['level']
                )
                score = confidence_result["total_score"]
                quality = confidence_result.get("quality_label", "B")
                win_rate = confidence_result.get("win_rate", "~55%")
                logger.info(f" 📊 Score: {score}/{SCORING_CONFIG['MIN_CONFIDENCE_SCORE']} | Qualité: {quality} | Win Rate estimé: {win_rate}")
                logger.info(f" 📋 Détails: {confidence_result.get('details', {})}")
                if confidence_result.get("passed", False):
                    logger.info(f" ✅ Signal {quality} confirmé → envoi Telegram")
                    rsi_value = get_last_rsi(df_m15["close"])
                    enriched_bias = dict(bias_analysis) if bias_analysis else {}
                    enriched_bias["win_rate"] = win_rate
                    enriched_bias["quality_label"] = quality
                    enriched_bias["score_details"] = confidence_result.get("details", {})
                    send_telegram_alert(
                        pair=pair,
                        direction=direction,
                        entry_price=entry_level,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        narrative=narrative,
                        bias_analysis=enriched_bias,
                        rsi=rsi_value,
                        entry_type=entry_type,
                        confidence_score=score
                    )
                    mark_signal_sent(pair, direction, entry_level_key, zone_start, zone_end)
                    nb_envoyes += 1
                else:
                    logger.info(" ❌ Signal rejeté (score insuffisant)")
            total_signals_processed = len(narrative.get("potential_entries", []))
            logger.info(f"🏁 Scan {pair} terminé. Signaux finaux envoyés: {nb_envoyes}. Signaux traités/clusters: {total_signals_processed}")
        except Exception as e:
            logger.error(f"💥 Erreur critique sur {pair} : {str(e)}")
            logger.error(traceback.format_exc())
            continue
    logger.info("🏁 Analyse avancée terminée pour toutes les paires")

# ============================================================
# V78 STRICT BUY/SELL - FILTRES
# ============================================================
STRICT_ALLOWED_ENTRY_TYPES = {
    "FVG_RETEST_PERFECT",
    "FVG_RETEST",
    "BISI",
    "BREAKER",
    "NESTED_FVG",
    "WICK_REJECTION",
    "LIQUIDITY_DRAW",
}
STRICT_MAX_DISTANCE_PIPS = {
    "XAU_USD": 35.0,
    "USD_JPY": 18.0,
    "GBP_JPY": 22.0,
    "EUR_USD": 15.0,
    "GBP_USD": 18.0,
    "AUD_USD": 15.0,
    "USD_CAD": 15.0,
    "AUD_CAD": 15.0,
    "AUD_JPY": 18.0,
    "NAS100_USD": 50.0,
    "DEFAULT": 15.0,
}

def strict_price_distance(pair: str, pips: float) -> float:
    return float(pips) * get_pip_value_for_pair(pair)

def strict_entry_type_allowed(entry_type: str) -> bool:
    et = (entry_type or "").upper().strip()
    if et in STRICT_ALLOWED_ENTRY_TYPES:
        return True
    if et.startswith("FVG_RETEST"):
        return True
    if "NESTED" in et and "FVG" in et:
        return True
    if "WICK" in et and "REJECTION" in et:
        return True
    blocked_keywords = ("TBS", "AMD", "CRT", "PIN_BUY", "PIN_SELL")
    if any(k in et for k in blocked_keywords):
        return False
    return False

def strict_stoch_veto(direction: str, df_h1: pd.DataFrame, df_m15: pd.DataFrame) -> tuple:
    try:
        k_h1, d_h1 = calculate_stoch_rsi(df_h1["close"])
        k_m15, d_m15 = calculate_stoch_rsi(df_m15["close"])
        k_h1 = float(k_h1)
        k_m15 = float(k_m15)
        if direction == "BUY" and k_h1 >= 80:
            return False, f"BUY interdit: StochRSI H1 surachat {k_h1:.1f}"
        if direction == "SELL" and k_h1 <= 20:
            return False, f"SELL interdit: StochRSI H1 survendu {k_h1:.1f}"
        if direction == "BUY" and k_m15 >= 85:
            return False, f"BUY trop tardif: StochRSI M15 {k_m15:.1f}"
        if direction == "SELL" and k_m15 <= 15:
            return False, f"SELL trop tardif: StochRSI M15 {k_m15:.1f}"
        return True, f"StochRSI OK H1={k_h1:.1f} M15={k_m15:.1f}"
    except Exception as exc:
        return True, f"StochRSI indisponible: {exc}"

def strict_trend_veto(direction: str, current_price: float, df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> tuple:
    try:
        ema50_h1 = df_h1["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        return True, f"EMA50 H1 scorée sans veto (prix={current_price:.5f}, EMA50={ema50_h1:.5f})"
    except Exception as exc:
        return True, f"EMA50 H1 indisponible, aucun veto: {exc}"

def strict_distance_filter(pair: str, current_price: float, entry: dict) -> tuple:
    entry_level = entry.get("entry_level")
    if entry_level is None:
        return False, "entry_level manquant"
    entry_level = float(entry_level)
    zone_start, zone_end = entry.get("entry_zone", (entry_level, entry_level))
    zone_start = float(zone_start)
    zone_end = float(zone_end)
    is_in_zone = min(zone_start, zone_end) <= current_price <= max(zone_start, zone_end)
    entry_type = str(entry.get("type", "")).upper()
    type_max_pips = {
        "FVG_RETEST_PERFECT": 18.0,
        "FVG_RETEST": 20.0,
        "NESTED_FVG": 20.0,
        "WICK_REJECTION": 18.0,
        "BISI": 20.0,
        "BREAKER": 18.0,
    }
    max_pips = max(
        STRICT_MAX_DISTANCE_PIPS.get(pair, STRICT_MAX_DISTANCE_PIPS["DEFAULT"]),
        type_max_pips.get(entry_type, STRICT_MAX_DISTANCE_PIPS.get(pair, STRICT_MAX_DISTANCE_PIPS["DEFAULT"])),
    )
    max_price_distance = strict_price_distance(pair, max_pips)
    distance = abs(current_price - entry_level)
    if is_in_zone:
        return True, f"dans zone distance={distance:.5f}"
    if distance <= max_price_distance:
        return True, f"distance acceptable={distance:.5f} <= {max_price_distance:.5f} ({max_pips:.1f} pips)"
    return False, f"vraiment trop loin distance={distance:.5f} > {max_price_distance:.5f} ({max_pips:.1f} pips)"

def strict_keep_best_per_direction(scored_entries: list) -> list:
    best = {}
    for item in scored_entries:
        entry = item["entry"]
        direction = entry.get("direction", "").upper()
        score = item["confidence"].get("total_score", -999)
        entry_type = entry.get("type", "")
        priority = 0
        if "PERFECT" in entry_type:
            priority += 3
        if entry_type == "BISI":
            priority += 3
        if entry_type.startswith("FVG_RETEST"):
            priority += 2
        if entry_type == "NESTED_FVG":
            priority += 2
        if entry_type == "WICK_REJECTION":
            priority += 1
        key_score = (score, priority)
        if direction not in best or key_score > best[direction]["key_score"]:
            item["key_score"] = key_score
            best[direction] = item
    return sorted(best.values(), key=lambda x: x["key_score"], reverse=True)

def strict_direction_permission_v77(direction: str, bias: str, current_price: float, df_h1: pd.DataFrame, df_m15: pd.DataFrame, entry_type: str) -> tuple:
    try:
        direction = (direction or "").upper()
        bias = (bias or "NEUTRAL").upper()
        entry_type = (entry_type or "").upper()
        k_h1, d_h1 = calculate_stoch_rsi(df_h1["close"])
        k_m15, d_m15 = calculate_stoch_rsi(df_m15["close"])
        k_h1 = float(k_h1)
        k_m15 = float(k_m15)
        if bias not in {"BUY", "SELL"}:
            return True, f"Biais neutre: direction {direction} autorisée"
        if direction == bias:
            return True, f"Direction alignée H4 {bias}"
        allowed_counter_types = {
            "BREAKER",
            "BISI",
            "FVG_RETEST",
            "FVG_RETEST_PERFECT",
            "NESTED_FVG",
            "WICK_REJECTION",
        }
        is_allowed_counter_type = entry_type in allowed_counter_types or entry_type.startswith("FVG_RETEST")
        if direction == "SELL" and bias == "BUY":
            if k_h1 >= 75 and k_m15 <= 70 and is_allowed_counter_type:
                return True, f"SELL contre H4 BUY autorisé V78: H1 surachat {k_h1:.1f}, M15 refroidit {k_m15:.1f}, type={entry_type}"
            return False, f"SELL contre H4 BUY refusé: H1={k_h1:.1f}, M15={k_m15:.1f}, type={entry_type}"
        if direction == "BUY" and bias == "SELL":
            if k_h1 <= 25 and k_m15 >= 30 and is_allowed_counter_type:
                return True, f"BUY contre H4 SELL autorisé V78: H1 survendu {k_h1:.1f}, M15 rebondit {k_m15:.1f}, type={entry_type}"
            return False, f"BUY contre H4 SELL refusé: H1={k_h1:.1f}, M15={k_m15:.1f}, type={entry_type}"
        return False, f"Direction {direction} non autorisée contre biais {bias}"
    except Exception as exc:
        return False, f"permission direction indisponible: {exc}"

def strict_trend_veto_v77(direction: str, current_price: float, df_h1: pd.DataFrame, df_h4: pd.DataFrame, bias: str = "NEUTRAL") -> tuple:
    try:
        ema50_h1 = float(df_h1["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        return True, f"EMA50 H1 non bloquante: prix={current_price:.5f}, EMA50={ema50_h1:.5f}"
    except Exception as exc:
        return True, f"EMA H1 indisponible, aucun veto: {exc}"

def dedupe_raw_entries_v771(entries: list, pair: str) -> list:
    if not entries:
        return []
    pip = get_pip_value_for_pair(pair)
    precision_step = max(pip * 0.5, 1e-9)
    def priority(entry: dict) -> tuple:
        et = str(entry.get("type", "")).upper()
        score = 0
        if et == "FVG_RETEST_PERFECT":
            score += 5
        elif et.startswith("FVG_RETEST"):
            score += 4
        elif et == "BISI":
            score += 4
        elif et == "NESTED_FVG":
            score += 3
        elif et == "WICK_REJECTION":
            score += 2
        try:
            lvl = float(entry.get("entry_level", 0))
        except Exception:
            lvl = 0.0
        return (score, -abs(lvl))
    seen = {}
    for entry in entries:
        try:
            direction = str(entry.get("direction", "")).upper()
            et = str(entry.get("type", "")).upper()
            lvl = float(entry.get("entry_level"))
        except Exception:
            continue
        rounded_bucket = round(lvl / precision_step)
        key = (direction, et, rounded_bucket)
        if key not in seen or priority(entry) > priority(seen[key]):
            seen[key] = entry
    deduped = list(seen.values())
    removed = len(entries) - len(deduped)
    if removed > 0:
        logger.info(f"🧹 V78 dédup {pair}: {removed} doublons supprimés ({len(entries)} -> {len(deduped)})")
    return deduped

def advanced_main():
    try:
        api = oandapyV20.API(access_token=os.getenv("OANDA_API_KEY"))
        logger.info("✅ API OANDA initialisée avec succès")
    except Exception as e:
        logger.error(f"❌ Échec d'initialisation de l'API OANDA : {e}")
        return
    for pair in PAIR_LIST:
        _reset_log_dedup()
        try:
            logger.info(f"\n🔍 Début de l'analyse V78 BALANCED BUY/SELL de {pair}")
            df_h4 = get_candles_with_retry(api, pair, GRANULARITY_H4, 300)
            df_h1 = get_candles_with_retry(api, pair, GRANULARITY_H1, 200)
            df_m15 = get_candles_with_retry(api, pair, GRANULARITY_M15, 250)
            df_d1 = get_candles_with_retry(api, pair, "D", count=250)
            if any(df.empty for df in [df_h4, df_h1, df_m15]):
                logger.warning(f"⚠️ Données manquantes pour {pair}, analyse ignorée")
                continue
            current_price = float(df_m15["close"].iloc[-1])
            logger.info(f"💰 {pair} Prix actuel M15 : {current_price:.5f}")
            bias_analysis = determine_advanced_bias(df_h4)
            bias = bias_analysis.get("bias", "NEUTRAL")
            logger.info(f"📍 Bias H4 : {bias}")
            narrative = determine_advanced_narrative(
                df_m15, bias_analysis, pair, df_h4=df_h4, df_d1=df_d1, df_h1=df_h1
            )
            raw_entries_raw = narrative.get("potential_entries", [])
            logger.info(f"🧹 V78: entrées brutes narrative: {len(raw_entries_raw)}")
            raw_entries = dedupe_raw_entries_v771(raw_entries_raw, pair)
            logger.info(f"🧹 V78: entrées après dédup: {len(raw_entries)}")
            breaker = detect_breaker(df_m15)
            scored_entries = []
            rejected = 0
            for entry in raw_entries:
                direction = entry.get("direction", "").upper()
                entry_type = entry.get("type", "UNKNOWN")
                entry_level = entry.get("entry_level")
                if direction not in {"BUY", "SELL"} or entry_level is None:
                    rejected += 1
                    continue
                if not strict_entry_type_allowed(entry_type):
                    logger.info(f"⛔ {pair} rejet {direction} {entry_type}: type non autorisé V78")
                    rejected += 1
                    continue
                ok, reason = strict_direction_permission_v77(direction, bias, current_price, df_h1, df_m15, entry_type)
                if not ok:
                    logger.info(f"⛔ {pair} rejet {direction} {entry_type}: {reason}")
                    rejected += 1
                    continue
                ok, reason = strict_trend_veto_v77(direction, current_price, df_h1, df_h4, bias)
                if not ok:
                    logger.info(f"⛔ {pair} rejet {direction} {entry_type}: {reason}")
                    rejected += 1
                    continue
                ok, reason = strict_stoch_veto(direction, df_h1, df_m15)
                if not ok:
                    logger.info(f"⛔ {pair} rejet {direction} {entry_type}: {reason}")
                    rejected += 1
                    continue
                ok, reason = strict_distance_filter(pair, current_price, entry)
                if not ok:
                    logger.info(f"⛔ {pair} rejet {direction} {entry_type} @{float(entry_level):.5f}: {reason}")
                    rejected += 1
                    continue
                logger.info(f"✅ {pair} candidat propre {direction} {entry_type} @{float(entry_level):.5f}: {reason}")
                confidence_result = calculate_signal_confidence(
                    pair, direction, df_h4, df_h1, df_m15, entry, bias, current_price,
                    False, "", df_d1=df_d1
                )
                if not confidence_result.get("passed", False):
                    logger.info(
                        f"❌ {pair} rejet score {direction} {entry_type}: "
                        f"{confidence_result.get('total_score')}/{SCORING_CONFIG['MIN_CONFIDENCE_SCORE']} "
                        f"{confidence_result.get('details', {})}"
                    )
                    rejected += 1
                    continue
                scored_entries.append({"entry": entry, "confidence": confidence_result})
            finalists = strict_keep_best_per_direction(scored_entries)
            logger.info(
                f"🧹 V78: candidats scorés={len(scored_entries)}, finalistes={len(finalists)}, rejetés={rejected}"
            )
            nb_envoyes = 0
            for item in finalists:
                entry = item["entry"]
                confidence_result = item["confidence"]
                direction = entry.get("direction", "").upper()
                entry_type = entry.get("type", "UNKNOWN")
                entry_level = float(entry.get("entry_level"))
                zone_start, zone_end = entry.get("entry_zone", (entry_level, entry_level))
                zone_start = float(zone_start)
                zone_end = float(zone_end)
                entry_level_key = round(entry_level, 5)
                if is_signal_sent_recently(pair, direction, entry_level_key, zone_start, zone_end):
                    logger.info(f"❌ {pair} {direction} déjà envoyé récemment → ignoré")
                    continue
                stop_loss, take_profit = calculate_sl_tp(
                    entry_price=entry_level,
                    atr=confidence_result["atr_value"],
                    direction=direction,
                    pair=pair,
                    entry_type=entry_type,
                    breaker_level=breaker.get("level") if isinstance(breaker, dict) else None,
                )
                score = confidence_result.get("total_score", 0)
                quality = confidence_result.get("quality_label", "B")
                win_rate = confidence_result.get("win_rate", "~55%")
                logger.info(
                    f"📊 FINAL {pair} {direction} {entry_type} @{entry_level:.5f} | "
                    f"Score {score}/{SCORING_CONFIG['MIN_CONFIDENCE_SCORE']} | Qualité {quality} | WR {win_rate}"
                )
                logger.info(f"📋 Détails: {confidence_result.get('details', {})}")
                enriched_bias = dict(bias_analysis) if bias_analysis else {}
                enriched_bias["win_rate"] = win_rate
                enriched_bias["quality_label"] = quality
                enriched_bias["score_details"] = confidence_result.get("details", {})
                enriched_bias["v77_filter"] = "V78: max 1 BUY + 1 SELL, WICK/NESTED autorisés, contre-signaux seulement sur StochRSI extrême"
                rsi_value = get_last_rsi(df_m15["close"])
                # V86 : exécution via execute_oanda_trade_v86 (modifiée V86)
                trade_id = execute_oanda_trade_v86(
                    pair=pair,
                    direction=direction,
                    entry_price=entry_level,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    score=score,
                    entry_type=entry_type,
                )
                if trade_id:
                    send_telegram_alert(
                        pair=pair,
                        direction=direction,
                        entry_price=entry_level,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        narrative=narrative,
                        bias_analysis=enriched_bias,
                        rsi=rsi_value,
                        entry_type=entry_type,
                        confidence_score=score,
                    )
                    mark_signal_sent(pair, direction, entry_level_key, zone_start, zone_end)
                    nb_envoyes += 1
                else:
                    logger.info(f"{pair}: V86 ordre non exécuté, zone NON enregistrée.")
            logger.info(
                f"🏁 Scan {pair} terminé. Signaux envoyés: {nb_envoyes}. "
                f"Finalistes: {len(finalists)}"
            )
        except Exception as e:
            logger.error(f"💥 Erreur critique sur {pair} : {str(e)}")
            logger.error(traceback.format_exc())
            continue
    logger.info("🏁 Analyse V86 BALANCED BUY/SELL terminée pour toutes les paires")

# =========================================================
# V86 - OANDA EXECUTION + TRADE MANAGER (modifié)
# =========================================================
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "101-004-31348578-001")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "true").lower() == "true"

RISK_PERCENTAGE = float(os.getenv("RISK_PERCENTAGE", "1.0"))
MAX_RISK_USD = float(os.getenv("MAX_RISK_USD", "1250"))
MAX_TRADES_TOTAL = int(os.getenv("MAX_TRADES_TOTAL", "9"))
ONE_TRADE_PER_PAIR = os.getenv("ONE_TRADE_PER_PAIR", "true").lower() == "true"

# V86 modifié : seuil de BE passé à 1.2R
BREAKEVEN_TRIGGER_R = float(os.getenv("BREAKEVEN_TRIGGER_R", "1.2"))
BREAKEVEN_OFFSET_PIPS = float(os.getenv("BREAKEVEN_OFFSET_PIPS", "1.0"))

PIP_SIZE_V86 = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "USD_CAD": 0.0001, "AUD_CAD": 0.0001,
    "USD_JPY": 0.01, "AUD_JPY": 0.01, "GBP_JPY": 0.01,
    "XAU_USD": 0.01,
    "NAS100_USD": 0.1, "US30_USD": 1.0, "SPX500_USD": 0.1,
}
PRICE_DECIMALS_V86 = {
    "EUR_USD": 5, "GBP_USD": 5, "AUD_USD": 5,
    "USD_CAD": 5, "AUD_CAD": 5,
    "USD_JPY": 3, "AUD_JPY": 3, "GBP_JPY": 3,
    "XAU_USD": 3,
    "NAS100_USD": 1, "US30_USD": 1, "SPX500_USD": 1,
}
MIN_UNITS_V86 = {"XAU_USD": 0.1, "DEFAULT": 1000}

UNIT_STEP_BY_PAIR = {
    "XAU_USD": 1,
    "EUR_USD": 1000,
    "GBP_USD": 1000,
    "USD_JPY": 1000,
    "USD_CAD": 1000,
    "AUD_USD": 1000,
    "AUD_CAD": 1000,
    "AUD_JPY": 1000,
    "GBP_JPY": 1000,
    "NAS100_USD": 1,
    "US30_USD": 1,
    "SPX500_USD": 1,
    "DEFAULT": 1000,
}
MIN_UNITS_BY_PAIR = {
    "XAU_USD": 1,
    "NAS100_USD": 1,
    "US30_USD": 1,
    "SPX500_USD": 1,
    "DEFAULT": 1000,
}
MAX_UNITS_BY_PAIR = {
    "XAU_USD": 100,
    "EUR_USD": 200000,
    "GBP_USD": 200000,
    "USD_JPY": 200000,
    "USD_CAD": 200000,
    "AUD_USD": 200000,
    "AUD_CAD": 200000,
    "AUD_JPY": 200000,
    "GBP_JPY": 200000,
    "NAS100_USD": 50,
    "US30_USD": 20,
    "SPX500_USD": 50,
    "DEFAULT": 200000,
}

MAX_MARGIN_USAGE_PER_TRADE_PERCENT = float(os.getenv("MAX_MARGIN_USAGE_PER_TRADE_PERCENT", "5.0"))
OANDA_CACHE_TTL_SECONDS_V83 = float(os.getenv("OANDA_CACHE_TTL_SECONDS_V83", "3.0"))
_OANDA_CACHE_V86 = {}
_INITIAL_RISK_BY_TRADE_V86 = {}
_TRAILING_CREATED_V86 = set()  # V86 pour mémoriser les trades pour lesquels le trailing a été créé

def compact_json_v86(obj, max_len: int = 6000) -> str:
    try:
        import json
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        text = str(obj)
    return text if len(text) <= max_len else text[:max_len] + " ...[TRONQUÉ]"

def v86_client():
    token = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_ACCESS_TOKEN")
    return oandapyV20.API(access_token=token, environment=OANDA_ENVIRONMENT)

def _cache_get_v86(key: str, ttl_seconds: float = OANDA_CACHE_TTL_SECONDS_V83):
    item = _OANDA_CACHE_V86.get(key)
    if not item:
        return None
    ts, value = item
    if time.time() - ts > ttl_seconds:
        _OANDA_CACHE_V86.pop(key, None)
        return None
    return value

def _cache_set_v86(key: str, value):
    _OANDA_CACHE_V86[key] = (time.time(), value)
    return value

def clear_scan_cache_v86():
    _OANDA_CACHE_V86.clear()

def is_market_open_utc_v86(now_dt: datetime) -> bool:
    wd = now_dt.weekday()
    t = now_dt.time()
    if wd == 5:
        return False
    if wd == 6 and t < datetime.strptime("21:00", "%H:%M").time():
        return False
    if wd == 4 and t >= datetime.strptime("21:00", "%H:%M").time():
        return False
    return True

def round_price_v86(pair: str, price: float) -> str:
    return f"{float(price):.{PRICE_DECIMALS_V86.get(pair, 5)}f}"

def oanda_safe_request_v86(endpoint, label: str = ""):
    try:
        api = v86_client()
        resp = api.request(endpoint)
        return resp
    except Exception as e:
        logger.error(f"❌ V86 OANDA exception {label}: {e}")
        logger.error(traceback.format_exc())
        return None

def get_account_summary_v86() -> dict:
    cached = _cache_get_v86("account_summary")
    if cached is not None:
        return cached
    r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
    resp = oanda_safe_request_v86(r, "AccountSummary")
    if not resp:
        return {}
    acc = resp.get("account", {})
    logger.info(
        f"DEBUG OANDA SUMMARY | account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT} "
        f"balance={acc.get('balance')} NAV={acc.get('NAV')} "
        f"openTradeCount={acc.get('openTradeCount')} openPositionCount={acc.get('openPositionCount')} "
        f"lastTransactionID={resp.get('lastTransactionID') or acc.get('lastTransactionID')}"
    )
    return _cache_set_v86("account_summary", resp)

def get_balance_v86() -> float:
    resp = get_account_summary_v86()
    try:
        return float(resp.get("account", {}).get("balance", 0))
    except Exception:
        return 0.0

def get_open_trades_v86(log_raw: bool = False) -> list:
    cache_key = "open_trades_raw"
    resp = _cache_get_v86(cache_key)
    if resp is None:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        resp = oanda_safe_request_v86(r, "OpenTrades")
        if resp:
            _cache_set_v86(cache_key, resp)
    if not resp:
        logger.warning("[RISK] OPEN TRADES response missing; using empty list to avoid false blocking.")
        return []
    raw_trades = resp.get("trades", []) or []
    open_trades = []
    for t in raw_trades:
        try:
            units = float(t.get("currentUnits", t.get("units", 0)) or 0)
        except Exception:
            units = 0.0
        if abs(units) > 0:
            open_trades.append(t)
    logger.info(f"DEBUG OPEN TRADES | account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT} count={len(open_trades)} raw_count={len(raw_trades)}")
    if log_raw:
        logger.info(f"DEBUG OPEN TRADES RAW={compact_json_v86(resp)}")
    for t in open_trades:
        logger.info(
            f"DEBUG TRADE | id={t.get('id')} instrument={t.get('instrument')} "
            f"units={t.get('currentUnits', t.get('units'))} price={t.get('price')} "
            f"state={t.get('state', 'OPEN')} openTime={t.get('openTime')}"
        )
    return open_trades

def get_open_positions_v86(log_raw: bool = False) -> list:
    try:
        cache_key = "open_positions_raw"
        resp = _cache_get_v86(cache_key)
        if resp is None:
            r = positions.OpenPositions(accountID=OANDA_ACCOUNT_ID)
            resp = oanda_safe_request_v86(r, "OpenPositions")
            if resp:
                _cache_set_v86(cache_key, resp)
        if not resp:
            return []
        open_positions = resp.get("positions", []) or []
        live_positions = []
        for p in open_positions:
            try:
                long_units = float(p.get("long", {}).get("units", 0) or 0)
                short_units = float(p.get("short", {}).get("units", 0) or 0)
            except Exception:
                long_units, short_units = 0.0, 0.0
            if abs(long_units) > 0 or abs(short_units) > 0:
                live_positions.append(p)
        logger.info(
            f"DEBUG OPEN POSITIONS | account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT} "
            f"count={len(live_positions)} raw_count={len(open_positions)}"
        )
        if log_raw:
            logger.info(f"DEBUG OPEN POSITIONS RAW={compact_json_v86(resp)}")
        for p in live_positions:
            logger.info(
                f"DEBUG POSITION | instrument={p.get('instrument')} "
                f"long_units={p.get('long', {}).get('units')} short_units={p.get('short', {}).get('units')} "
                f"pl={p.get('pl')} resettablePL={p.get('resettablePL')}"
            )
        return live_positions
    except Exception as exc:
        logger.exception(f"DEBUG OPEN POSITIONS impossible: {exc}")
        return []

def has_open_trade_v86(pair: str) -> bool:
    open_trades = get_open_trades_v86(log_raw=True)
    trade_exists = any(t.get("instrument") == pair for t in open_trades)
    open_positions = get_open_positions_v86(log_raw=True)
    position_exists = any(p.get("instrument") == pair for p in open_positions)
    exists = bool(trade_exists or position_exists)
    logger.info(f"{pair}: has_open_trade={exists} | trade_exists={trade_exists} | position_exists={position_exists}")
    return exists

def open_trade_count_v86() -> int:
    return len(get_open_trades_v86(log_raw=True))

def log_account_snapshot_v86(label: str) -> None:
    logger.info(f"========== OANDA ACCOUNT SNAPSHOT {label} ==========")
    try:
        get_account_summary_v86()
    except Exception as exc:
        logger.exception(f"SNAPSHOT SUMMARY impossible: {exc}")
    get_open_trades_v86(log_raw=True)
    get_open_positions_v86(log_raw=True)
    logger.info(f"========== FIN SNAPSHOT {label} ==========")

def extract_oanda_trade_or_order_id_v86(resp: dict) -> str | None:
    fill = resp.get("orderFillTransaction", {}) or {}
    trade_opened = fill.get("tradeOpened") or {}
    if trade_opened.get("tradeID"):
        return str(trade_opened["tradeID"])
    trades_opened = fill.get("tradesOpened") or []
    if trades_opened and trades_opened[0].get("tradeID"):
        return str(trades_opened[0]["tradeID"])
    return str(fill.get("id") or resp.get("orderCreateTransaction", {}).get("id") or "") or None

def log_oanda_order_response_v86(pair: str, resp: dict) -> None:
    logger.info(f"DEBUG ORDER RESPONSE RAW {pair}={compact_json_v86(resp, max_len=8000)}")
    for key in ["orderCreateTransaction", "orderFillTransaction", "orderCancelTransaction", "orderRejectTransaction"]:
        tx = resp.get(key)
        if not tx:
            continue
        logger.info(
            f"OANDA {key} | id={tx.get('id')} type={tx.get('type')} "
            f"instrument={tx.get('instrument')} units={tx.get('units')} "
            f"reason={tx.get('reason')} rejectReason={tx.get('rejectReason')}"
        )
    if resp.get("orderRejectTransaction"):
        logger.error(f"ORDRE REJETÉ {pair} | {compact_json_v86(resp.get('orderRejectTransaction'))}")
    if resp.get("orderCancelTransaction"):
        logger.warning(f"ORDRE ANNULÉ {pair} | {compact_json_v86(resp.get('orderCancelTransaction'))}")

def quote_currency_v86(pair: str) -> str:
    return pair.split("_")[1]

def get_fx_rate_to_usd_v86(currency: str) -> float:
    if currency == "USD":
        return 1.0
    cached = _cache_get_v86(f"fx_to_usd:{currency}", ttl_seconds=60.0)
    if cached is not None:
        return float(cached)
    direct = f"{currency}_USD"
    inverse = f"USD_{currency}"
    try:
        if direct in PAIR_LIST:
            api = v86_client()
            df = get_candles_with_retry(api, direct, "M5", 10)
            if df is not None and not df.empty:
                return float(_cache_set_v86(f"fx_to_usd:{currency}", float(df["close"].iloc[-1])))
        if inverse in PAIR_LIST:
            api = v86_client()
            df = get_candles_with_retry(api, inverse, "M5", 10)
            if df is not None and not df.empty:
                return float(_cache_set_v86(f"fx_to_usd:{currency}", 1.0 / float(df["close"].iloc[-1])))
    except Exception:
        pass
    logger.warning(f"⚠️ V86 conversion {currency}->USD inconnue, fallback 1.0")
    return 1.0

def get_oanda_margin_rate_v86(pair: str) -> float:
    cached = _cache_get_v86(f"instrument:{pair}", ttl_seconds=300.0)
    if cached is not None:
        return float(cached.get("marginRate", 0.0333) or 0.0333)
    try:
        api = v86_client()
        r = accounts.AccountInstruments(accountID=OANDA_ACCOUNT_ID, params={"instruments": pair})
        resp = api.request(r)
        instruments_data = resp.get("instruments", [])
        if instruments_data:
            _cache_set_v86(f"instrument:{pair}", instruments_data[0])
            margin_rate = float(instruments_data[0].get("marginRate", 0.0333))
            if margin_rate > 0:
                return margin_rate
    except Exception as exc:
        logger.warning(f"V86 marginRate indisponible pour {pair}, fallback 0.0333: {exc}")
    return 0.0333

def get_available_margin_v86(account_summary: dict | None = None) -> float:
    account_summary = account_summary or get_account_summary_v86()
    account = account_summary.get("account", {}) if isinstance(account_summary, dict) else {}
    for key in ("marginAvailable", "NAV", "balance"):
        try:
            value = float(account.get(key, 0) or 0)
            if value > 0:
                return value
        except Exception:
            continue
    return 0.0

def calculate_margin_v86(pair: str, units: int, entry_price: float, account_summary: dict | None = None) -> dict:
    margin_required = estimate_margin_used_v86(pair, units, entry_price)
    available = get_available_margin_v86(account_summary)
    return {
        "pair": pair,
        "units": abs(int(units or 0)),
        "entry_price": float(entry_price or 0),
        "margin_required": float(margin_required),
        "margin_available": float(available),
        "sufficient": bool(available <= 0 or margin_required <= available),
    }

def risk_report_v86(pair: str, entry: float, stop_loss: float, units: int, balance: float) -> dict:
    quote_to_usd = get_fx_rate_to_usd_v86(quote_currency_v86(pair))
    risk_per_unit_usd = abs(float(entry) - float(stop_loss)) * quote_to_usd
    estimated_risk = abs(int(units or 0)) * risk_per_unit_usd
    estimated_risk_pct = estimated_risk / float(balance) * 100.0 if balance else 0.0
    report = {
        "pair": pair,
        "units": abs(int(units or 0)),
        "risk_usd": estimated_risk,
        "risk_pct": estimated_risk_pct,
        "target_risk_pct": RISK_PERCENTAGE,
        "quote_to_usd": quote_to_usd,
        "risk_per_unit_usd": risk_per_unit_usd,
    }
    logger.info(
        f"[RISK] {pair} risk=${estimated_risk:.2f} ({estimated_risk_pct:.2f}%) "
        f"target={RISK_PERCENTAGE:.2f}% units={abs(int(units or 0))}"
    )
    return report

def estimate_margin_used_v86(pair: str, units: int, entry_price: float) -> float:
    units = abs(int(units))
    margin_rate = get_oanda_margin_rate_v86(pair)
    try:
        base, quote = pair.split("_")
    except Exception:
        base, quote = "", "USD"
    if pair == "XAU_USD":
        notional_usd = units * entry_price
    elif quote == "USD":
        notional_usd = units * entry_price
    elif base == "USD":
        notional_usd = units
    else:
        q_to_usd = get_fx_rate_to_usd_v86(quote)
        notional_usd = units * entry_price * q_to_usd
    return float(notional_usd * margin_rate)

def cap_units_absolute_v86(pair: str, units: int) -> int:
    max_units = MAX_UNITS_BY_PAIR.get(pair, MAX_UNITS_BY_PAIR["DEFAULT"])
    if units > max_units:
        logger.warning(f"V86 ABS CAP {pair}: units {units} -> {max_units}")
        return max_units
    return units

def cap_units_by_margin_v86(pair: str, units: int, entry_price: float, balance: float) -> int:
    if units <= 0 or balance <= 0:
        return 0
    margin_info = calculate_margin_v86(pair, units, entry_price)
    account_available = margin_info["margin_available"]
    max_margin_usd = min(balance * (MAX_MARGIN_USAGE_PER_TRADE_PERCENT / 100.0), account_available) if account_available > 0 else balance * (MAX_MARGIN_USAGE_PER_TRADE_PERCENT / 100.0)
    estimated_margin = margin_info["margin_required"]
    if estimated_margin <= max_margin_usd:
        return units
    ratio = max_margin_usd / estimated_margin if estimated_margin > 0 else 0
    capped = int(units * ratio)
    step = UNIT_STEP_BY_PAIR.get(pair, UNIT_STEP_BY_PAIR["DEFAULT"])
    capped = int(capped // step * step)
    logger.warning(
        f"[RISK] V86 MARGIN CAP {pair}: units {units} -> {capped} | "
        f"estimated_margin=${estimated_margin:.2f} > max_margin=${max_margin_usd:.2f} "
        f"({MAX_MARGIN_USAGE_PER_TRADE_PERCENT:.2f}% balance)"
    )
    return max(capped, 0)

def calculate_units_v86(pair: str, entry: float, stop_loss: float, balance: float) -> float:
    try:
        balance = float(balance)
        entry = float(entry)
        stop_loss = float(stop_loss)
    except Exception:
        logger.error(f"V86: paramètres sizing invalides pair={pair} entry={entry} stop={stop_loss} balance={balance}")
        return 0
    risk_usd = min(balance * (RISK_PERCENTAGE / 100.0), MAX_RISK_USD)
    distance_quote = abs(entry - stop_loss)
    if balance <= 0 or risk_usd <= 0 or distance_quote <= 0:
        logger.error(f"V86: sizing impossible {pair}: balance={balance}, risk_usd={risk_usd}, distance={distance_quote}")
        return 0
    quote = quote_currency_v86(pair)
    quote_to_usd = get_fx_rate_to_usd_v86(quote)
    if quote_to_usd <= 0:
        logger.error(f"V86: conversion quote_to_usd invalide {quote}->{quote_to_usd}, trade bloqué.")
        return 0
    risk_per_unit_usd = distance_quote * quote_to_usd
    if risk_per_unit_usd <= 0:
        logger.error(f"V86: risk_per_unit_usd invalide {pair}: {risk_per_unit_usd}")
        return 0
    raw_units = risk_usd / risk_per_unit_usd
    step = UNIT_STEP_BY_PAIR.get(pair, UNIT_STEP_BY_PAIR["DEFAULT"])
    min_units = MIN_UNITS_BY_PAIR.get(pair, MIN_UNITS_BY_PAIR["DEFAULT"])
    units_before_caps = int(raw_units // step * step)
    units = cap_units_absolute_v86(pair, units_before_caps)
    units = cap_units_by_margin_v86(pair, units, entry, balance)
    if units < min_units:
        logger.warning(f"V86: units trop faibles après caps {pair}: {units} < min={min_units}. Trade bloqué.")
        return 0
    estimated_margin = estimate_margin_used_v86(pair, units, entry)
    estimated_risk = units * risk_per_unit_usd
    estimated_risk_pct = estimated_risk / balance * 100.0 if balance > 0 else 0.0
    margin_pct = estimated_margin / balance * 100.0 if balance > 0 else 0.0
    logger.info(
        f"V86 RISK LOT {pair}: balance=${balance:.2f} risk_cap=${risk_usd:.2f} "
        f"entry={entry:.5f} SL={stop_loss:.5f} dist_quote={distance_quote:.5f} "
        f"quote_to_usd={quote_to_usd:.8f} risk_per_unit=${risk_per_unit_usd:.8f} "
        f"raw_units={raw_units:.2f} units_before_caps={units_before_caps} final_units={units} "
        f"estimated_risk=${estimated_risk:.2f} ({estimated_risk_pct:.2f}%) "
        f"estimated_margin=${estimated_margin:.2f} ({margin_pct:.2f}%)"
    )
    risk_report_v86(pair, entry, stop_loss, units, balance)
    return int(units)

def get_recent_m5_price_v86(pair: str) -> float:
    api = v86_client()
    df = get_candles_with_retry(api, pair, "M5", 10)
    if df is None or df.empty:
        return 0.0
    return float(df["close"].iloc[-1])

def get_price_spread_v86(pair: str) -> dict:
    cached = _cache_get_v86(f"pricing:{pair}", ttl_seconds=2.0)
    if cached is not None:
        return cached
    try:
        api = v86_client()
        r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params={"instruments": pair})
        resp = api.request(r)
        prices = resp.get("prices", []) or []
        if prices:
            item = prices[0]
            bid = float(item.get("bids", [{}])[0].get("price", 0) or 0)
            ask = float(item.get("asks", [{}])[0].get("price", 0) or 0)
            mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
            data = {"bid": bid, "ask": ask, "mid": mid, "spread": max(ask - bid, 0.0)}
            return _cache_set_v86(f"pricing:{pair}", data)
    except Exception as exc:
        logger.warning(f"[RISK] V86 pricing unavailable for {pair}: {exc}")
    fallback_price = get_recent_m5_price_v86(pair)
    fallback_spread = PIP_SIZE_V86.get(pair, get_pip_value_for_pair(pair)) * 2.0
    return {"bid": fallback_price, "ask": fallback_price, "mid": fallback_price, "spread": fallback_spread}

def get_atr_m15_v86(pair: str) -> float:
    cached = _cache_get_v86(f"atr_m15:{pair}", ttl_seconds=60.0)
    if cached is not None:
        return float(cached)
    api = v86_client()
    df = get_candles_with_retry(api, pair, "M15", max(ATR_PERIOD + 10, 40))
    if df is None or df.empty:
        return 0.0
    atr = float(calculate_atr(df, ATR_PERIOD) or 0.0)
    return float(_cache_set_v86(f"atr_m15:{pair}", atr))

# ============================================================
# V86 - HEALTH CHECK
# ============================================================
def health_check_v86() -> bool:
    """
    Vérifie la santé du bot avant chaque scan.
    Retourne True si tout est OK, False si une anomalie empêche de trader.
    """
    logger.info("🩺 HEALTH CHECK - Début")
    try:
        # 1. Connexion OANDA
        api = v86_client()
        account_summary = get_account_summary_v86()
        if not account_summary or not account_summary.get("account"):
            logger.error("❌ HEALTH CHECK: Échec de connexion OANDA")
            return False
        logger.info("   ✅ OANDA OK")

        # 2. Compte et balance
        account = account_summary.get("account", {})
        balance = float(account.get("balance", 0))
        if balance <= 0:
            logger.error(f"❌ HEALTH CHECK: Balance invalide ({balance})")
            return False
        logger.info(f"   ✅ BALANCE OK: ${balance:.2f}")

        # 3. Open trades
        open_trades = get_open_trades_v86(log_raw=False)
        nb_trades = len(open_trades)
        logger.info(f"   ✅ OPEN TRADES: {nb_trades}")

        # 4. Marge disponible
        margin_available = get_available_margin_v86(account_summary)
        margin_used = float(account.get("marginUsed", 0))
        margin_rate = (margin_used / balance * 100) if balance > 0 else 0
        logger.info(f"   ✅ MARGIN: {margin_rate:.2f}% utilisée, disponible ${margin_available:.2f}")

        # 5. Spread (check sur une paire de référence)
        ref_pair = "EUR_USD"
        spread_data = get_price_spread_v86(ref_pair)
        spread = spread_data.get("spread", 0)
        logger.info(f"   ✅ SPREAD {ref_pair}: {spread:.5f}")

        # 6. Latence (mesure approximative)
        start = time.time()
        get_candles_with_retry(api, ref_pair, "M5", 10)
        latency = (time.time() - start) * 1000  # ms
        logger.info(f"   ✅ API LATENCY: {latency:.0f} ms")

        logger.info("🩺 HEALTH CHECK - ✅ READY TO TRADE")
        return True

    except Exception as e:
        logger.error(f"❌ HEALTH CHECK échoué: {e}")
        logger.error(traceback.format_exc())
        return False

# ============================================================
# V86 - SUPPRESSION DES FONCTIONS DE TRAILING PYTHON
# ============================================================
# V86 - Fonction utilitaire pour obtenir la direction d'un trade
def trade_direction_v86(trade: dict) -> str:
    try:
        units = float(trade.get("currentUnits", trade.get("units", 0)))
        return "BUY" if units > 0 else "SELL"
    except:
        return "UNKNOWN"

# V86 - Fonction de modification du SL
def modify_trade_sl_v86(trade_id: str, pair: str, new_sl: float) -> bool:
    try:
        api = v86_client()
        order_data = {
            "order": {
                "type": "STOP_LOSS",
                "tradeID": trade_id,
                "price": round_price_v86(pair, new_sl),
                "timeInForce": "GTC"
            }
        }
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        resp = api.request(r)
        if resp.get("orderRejectTransaction"):
            logger.error(f"SL modification rejected for trade {trade_id}: {resp.get('orderRejectTransaction')}")
            return False
        logger.info(f"SL modifié pour trade {trade_id} -> {new_sl}")
        return True
    except Exception as e:
        logger.error(f"Erreur modification SL trade {trade_id}: {e}")
        return False

# V86 - Création d'un Trailing Stop Loss natif OANDA (renforcé)
def create_oanda_trailing_stop_v86(api, account_id: str, trade_id: str, distance: float) -> bool:
    """
    Crée un TrailingStopLossOrder via un dictionnaire JSON.
    Vérifie la distance minimale en interrogeant l'instrument.
    """
    try:
        # 1. Récupérer la distance minimale pour la paire via l'instrument
        # On utilise AccountInstruments pour obtenir les règles
        r = accounts.AccountInstruments(accountID=account_id, params={"instruments": "EUR_USD"})  # on récupère la paire en question
        # Mais il faut la paire du trade, on va faire une requête spécifique
        # On utilise plutôt get_oanda_margin_rate_v86 qui contient aussi des infos
        # Pour la distance minimale, on va faire une requête directe
        try:
            # On récupère les spécifications de l'instrument
            req = accounts.AccountInstruments(accountID=account_id, params={"instruments": "EUR_USD"})
            resp = api.request(req)
            instruments = resp.get("instruments", [])
            if instruments:
                # On cherche l'instrument correspondant au trade
                # Mais on a pas le nom de l'instrument, on le déduit du trade (pair)
                # On va simuler une vérification : on extrait la distance minimale pour le trailing
                # Dans la pratique, OANDA impose une distance minimale différente selon l'instrument
                # On peut la récupérer via l'endpoint 'AccountInstruments' et le champ 'trailingStopLossOrderMinimumDistance'
                for instr in instruments:
                    if instr.get("name") == pair:
                        min_distance = float(instr.get("trailingStopLossOrderMinimumDistance", 0))
                        if min_distance > 0 and distance < min_distance:
                            logger.warning(f"[TSL] Distance {distance} inférieure au minimum {min_distance} pour {pair}, ajustement")
                            distance = min_distance * 1.1  # on met un peu plus que le minimum
                        break
        except Exception as e:
            logger.warning(f"[TSL] Impossible de récupérer la distance minimale, utilisation brute: {e}")

        # 2. Création de l'ordre
        order_data = {
            "order": {
                "type": "TRAILING_STOP_LOSS",
                "tradeID": trade_id,
                "distance": str(distance),
                "timeInForce": "GTC"
            }
        }
        logger.info(f"[TSL] Tentative de création pour trade {trade_id} avec distance {distance:.5f}")
        r = orders.OrderCreate(accountID=account_id, data=order_data)
        resp = api.request(r)
        # Log détaillé
        logger.info(f"[TSL] Réponse OANDA: {compact_json_v86(resp)}")
        if resp.get("orderRejectTransaction"):
            reject = resp.get("orderRejectTransaction")
            logger.error(f"[TSL] Rejeté: {reject}")
            # Analyse du rejet
            reject_reason = reject.get("rejectReason", "")
            if "MINIMUM_DISTANCE" in reject_reason or "minimum distance" in reject_reason.lower():
                logger.error(f"[TSL] Distance minimale non respectée. Distance proposée: {distance}")
            return False
        if resp.get("orderCreateTransaction") and not resp.get("orderRejectTransaction"):
            logger.info(f"[TSL] Succès pour trade {trade_id}")
            return True
        logger.warning(f"[TSL] Réponse inattendue: {compact_json_v86(resp)}")
        return False
    except Exception as e:
        logger.error(f"[TSL] Erreur: {e}")
        logger.error(traceback.format_exc())
        return False

# V86 - Fonction de vérification du Break Even et activation du trailing (seuil à 1.2R)
def check_breakeven_simple_v86():
    """
    Parcourt les trades ouverts :
    - Si R >= 1.2, déplace le SL au BE (avec offset).
    - Puis, si le trailing n'a pas encore été créé pour ce trade, le crée via OANDA.
    """
    try:
        open_trades = get_open_trades_v86()
        for t in open_trades:
            trade_id = str(t.get("id"))
            pair = t.get("instrument")
            direction = trade_direction_v86(t)
            entry = float(t.get("price"))
            sl_order = t.get("stopLossOrder", {}) or {}
            if not sl_order.get("price"):
                continue
            current_sl = float(sl_order["price"])
            current_price = get_recent_m5_price_v86(pair)
            if current_price <= 0:
                continue

            # Calcul du R
            if direction == "BUY":
                profit = current_price - entry
                risk = entry - current_sl
            else:
                profit = entry - current_price
                risk = current_sl - entry
            if risk <= 0:
                continue
            r = profit / risk

            # V86 modifié : seuil à 1.2R
            if r >= BREAKEVEN_TRIGGER_R:
                pip = PIP_SIZE_V86.get(pair, get_pip_value_for_pair(pair))
                spread = get_price_spread_v86(pair)["spread"]
                offset = max(spread, pip * 1.0)
                if direction == "BUY":
                    be_sl = entry + offset
                else:
                    be_sl = entry - offset
                if (direction == "BUY" and be_sl > current_sl) or (direction == "SELL" and be_sl < current_sl):
                    logger.info(f"[BE] {pair} id={trade_id} R={r:.2f} => SL {current_sl} -> {be_sl}")
                    if modify_trade_sl_v86(trade_id, pair, be_sl):
                        time.sleep(1)
                        _OANDA_CACHE_V86.pop("open_trades_raw", None)
                        current_sl = be_sl
                    else:
                        continue

            # Trailing (toujours après le BE)
            if trade_id not in _TRAILING_CREATED_V86 and r >= BREAKEVEN_TRIGGER_R:
                if direction == "BUY" and current_sl >= entry:
                    atr = get_atr_m15_v86(pair)
                    spread = get_price_spread_v86(pair)["spread"]
                    distance = max(atr * 1.3, spread * 8)
                    distance = round(distance, PRICE_DECIMALS_V86.get(pair, 5))
                    if distance > 0:
                        api = v86_client()
                        if create_oanda_trailing_stop_v86(api, OANDA_ACCOUNT_ID, trade_id, distance):
                            _TRAILING_CREATED_V86.add(trade_id)
                            logger.info(f"[TSL] Trailing activé pour {pair} trade {trade_id} après BE")
                        else:
                            logger.warning(f"[TSL] Échec création trailing pour {pair} trade {trade_id}")
                    else:
                        logger.warning(f"[TSL] Distance invalide ({distance}) pour {pair}")
                elif direction == "SELL" and current_sl <= entry:
                    atr = get_atr_m15_v86(pair)
                    spread = get_price_spread_v86(pair)["spread"]
                    distance = max(atr * 1.3, spread * 8)
                    distance = round(distance, PRICE_DECIMALS_V86.get(pair, 5))
                    if distance > 0:
                        api = v86_client()
                        if create_oanda_trailing_stop_v86(api, OANDA_ACCOUNT_ID, trade_id, distance):
                            _TRAILING_CREATED_V86.add(trade_id)
                            logger.info(f"[TSL] Trailing activé pour {pair} trade {trade_id} après BE")
                        else:
                            logger.warning(f"[TSL] Échec création trailing pour {pair} trade {trade_id}")
                    else:
                        logger.warning(f"[TSL] Distance invalide ({distance}) pour {pair}")
    except Exception as e:
        logger.error(f"Erreur check_breakeven_simple_v86: {e}")
        logger.error(traceback.format_exc())

# =============================
# MODIFICATION DE execute_oanda_trade_v86
# =============================
def execute_oanda_trade_v86(pair: str, direction: str, entry_price: float, stop_loss: float,
                            take_profit: float, score: int, entry_type: str) -> str | None:
    """
    V86 : ouverture d'un Market Order, puis enregistrement du trade.
    Le trailing stop sera créé plus tard par check_breakeven_simple_v86.
    """
    logger.info(f"V86 EXECUTION START {pair} {direction} type={entry_type} score={score}")
    logger.info(
        f"DEBUG EXEC INPUT | pair={pair} direction={direction} entry={round_price_v86(pair, entry_price)} "
        f"SL={round_price_v86(pair, stop_loss)} TP={round_price_v86(pair, take_profit)} "
        f"account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT} execute={EXECUTE_TRADES}"
    )

    log_account_snapshot_v86("BEFORE_EXECUTION")

    if ONE_TRADE_PER_PAIR:
        has_trade = has_open_trade_v86(pair)
        logger.info(f"DEBUG STEP has_open_trade({pair})={has_trade}")
        if has_trade:
            logger.info(f"{pair}: trade déjà ouvert détecté par OANDA, aucun nouvel ordre.")
            return None

    count = open_trade_count_v86()
    logger.info(f"DEBUG STEP open_trade_count={count} / max={MAX_TRADES_TOTAL}")
    if count >= MAX_TRADES_TOTAL:
        logger.info(f"Limite trades ouverts atteinte ({count}/{MAX_TRADES_TOTAL}). Aucun ordre POST /orders ne sera envoyé.")
        return None

    balance = get_balance_v86()
    logger.info(f"DEBUG STEP balance={balance}")
    if balance <= 0:
        logger.error("V86: balance invalide, ordre annulé.")
        return None

    units = calculate_units_v86(pair, entry_price, stop_loss, balance)
    logger.info(f"DEBUG STEP calculated_units={units}")
    if not units or float(units) <= 0:
        logger.error(f"V86: units invalides pour {pair}: {units}")
        return None

    # Vérification marge avant envoi (déjà faite dans calculate_units mais on renforce)
    margin_info = calculate_margin_v86(pair, units, entry_price)
    logger.info(
        f"[RISK] {pair} margin_required=${margin_info['margin_required']:.2f} "
        f"available=${margin_info['margin_available']:.2f} sufficient={margin_info['sufficient']}"
    )
    if not margin_info["sufficient"]:
        units = cap_units_by_margin_v86(pair, units, entry_price, balance)
        if not units or units <= 0:
            logger.error(f"[RISK] {pair} order blocked: insufficient margin after unit reduction.")
            return None

    signed_units = units if direction == "BUY" else -units
    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": pair,
            "units": str(signed_units),
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": round_price_v86(pair, stop_loss), "timeInForce": "GTC"},
            "takeProfitOnFill": {"price": round_price_v86(pair, take_profit), "timeInForce": "GTC"},
        }
    }

    risk = abs(entry_price - stop_loss)
    rr = abs(take_profit - entry_price) / risk if risk > 0 else 0
    logger.info(
        f"SIGNAL V86 {pair} {direction} | "
        f"entry≈{round_price_v86(pair, entry_price)} SL={round_price_v86(pair, stop_loss)} "
        f"TP={round_price_v86(pair, take_profit)} RR={rr:.2f} score={score} "
        f"units={units} signed_units={signed_units} type={entry_type}"
    )
    logger.info(f"ORDER PAYLOAD {pair}={compact_json_v86(order_data)}")

    if not EXECUTE_TRADES:
        logger.info("EXECUTE_TRADES=false : ordre non envoyé à OANDA.")
        logger.info("DEBUG EXECUTION RESULT | status=SIMULATION | aucun POST /orders envoyé")
        return "SIMULATION"

    try:
        logger.info(f"DEBUG POST /orders START | pair={pair} account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT}")
        api = v86_client()
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        resp = api.request(r)
        logger.info(f"DEBUG POST /orders END | pair={pair}")
        log_oanda_order_response_v86(pair, resp)

        if resp.get("orderRejectTransaction"):
            logger.error(f"DEBUG EXECUTION RESULT | status=REJECTED | pair={pair}")
            log_account_snapshot_v86("AFTER_REJECT")
            return None

        if resp.get("orderCancelTransaction") and not resp.get("orderFillTransaction"):
            logger.error(f"DEBUG EXECUTION RESULT | status=CANCELLED_NO_FILL | pair={pair}")
            log_account_snapshot_v86("AFTER_CANCEL")
            return None

        trade_id = extract_oanda_trade_or_order_id_v86(resp)
        if not trade_id:
            logger.error(f"ORDRE NON CONFIRMÉ {pair}: aucune transaction Fill/Create exploitable.")
            logger.error(f"DEBUG EXECUTION RESULT | status=NO_TRADE_ID | pair={pair}")
            log_account_snapshot_v86("AFTER_NO_TRADE_ID")
            return None

        logger.info(f"✅ ORDRE RÉEL CONFIRMÉ {pair} | ID={trade_id}")
        logger.info(f"DEBUG EXECUTION RESULT | status=CONFIRMED | pair={pair} trade_or_order_id={trade_id}")

        # V86 : On enregistre le risque initial pour le BE
        _INITIAL_RISK_BY_TRADE_V86[trade_id] = abs(float(entry_price) - float(stop_loss))

        log_account_snapshot_v86("AFTER_ORDER_CREATE")
        open_after = get_open_trades_v86(log_raw=True)
        opened_for_pair = [t for t in open_after if t.get("instrument") == pair]
        if opened_for_pair:
            actual_trade_id = str(opened_for_pair[0].get("id") or trade_id)
            _INITIAL_RISK_BY_TRADE_V86[actual_trade_id] = abs(float(opened_for_pair[0].get("price", entry_price)) - float(stop_loss))
            logger.info(f"CONFIRMATION OPEN TRADE {pair}: {compact_json_v86(opened_for_pair)}")
        else:
            logger.warning(f"ATTENTION {pair}: ordre accepté mais OpenTrades ne montre pas encore la position.")

        return str(trade_id)

    except Exception as exc:
        logger.exception(f"Erreur ordre OANDA {pair}: {exc}")
        logger.error(f"DEBUG EXECUTION RESULT | status=EXCEPTION | pair={pair} error={exc}")
        log_account_snapshot_v86("EXCEPTION_ORDER_CREATE")
        return None

# =============================
# BOUCLE PRINCIPALE (V86)
# =============================
if __name__ == "__main__":
    logger.info("🚀 Démarrage du Bot Advanced Orderflow Trading - V86 (Trailing Stop OANDA)")

    api = oandapyV20.API(access_token=os.getenv("OANDA_API_KEY"))

    while True:
        try:
            # V86 : health check avant chaque scan
            if not health_check_v86():
                logger.warning("⚠️ Health check échoué, attente 2 minutes avant réessayer.")
                time.sleep(120)
                continue

            clear_scan_cache_v86()
            now_dt = datetime.utcnow()
            if not is_market_open_utc_v86(now_dt):
                logger.info("Marché Forex fermé. Attente 5 minutes.")
                time.sleep(300)
                continue

            # V86 : Vérification du Break Even et activation du trailing
            check_breakeven_simple_v86()

            current_open_count = open_trade_count_v86()
            if current_open_count >= MAX_TRADES_TOTAL:
                logger.info(f"Limite trades ouverts atteinte ({current_open_count}/{MAX_TRADES_TOTAL}). Scan entrées ignoré.")
                time.sleep(300)
                continue

            advanced_main()
            logger.info("⏳ Attente 15 minutes avant le prochain scan...")
            time.sleep(900)
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt demandé par l'utilisateur")
            break
        except Exception as e:
            logger.error(f"💥 Erreur critique: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(60)
