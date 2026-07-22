# ============================================================
# main(94).py - Version V94 "Audit ATR"
# 
# Modifications V94 :
# - Audit détaillé du calcul ATR (valeur brute, conversion, pips)
# - Logs de diagnostic ATR pour chaque paire
# - Seuils ATR provisoirement conservés (en attente de validation)
# ============================================================

import os
import sys
import time
import logging
import unicodedata
import requests
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import oandapyV20
from oandapyV20.endpoints import instruments, pricing, orders, accounts, trades, positions, transactions
import talib
import traceback
from ta.momentum import RSIIndicator
from typing import List, Dict, Tuple, Optional

# =========================
# CONFIGURATION V94
# =========================
load_dotenv()

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
DECISION_JOURNAL = os.getenv("DECISION_JOURNAL", "decision_journal.json")
TRACE_JOURNAL = os.getenv("TRACE_JOURNAL", "trade_trace.json")

# V94 - Scores minimum (conservés de V93)
MIN_CONFIDENCE_SCORE_BY_PAIR = {
    "EUR_USD": 10,
    "GBP_USD": 9,
    "USD_CAD": 8,
    "AUD_USD": 8,
    "AUD_CAD": 8,
    "XAU_USD": 9,
    "DEFAULT": 8
}

BREAKEVEN_TRIGGER_R = float(os.getenv("BREAKEVEN_TRIGGER_R", "0.6"))
TRAILING_STOP_DISTANCE_ATR_MULTIPLIER = float(os.getenv("TRAILING_STOP_DISTANCE_ATR_MULTIPLIER", "1.6"))
TRAILING_STOP_MIN_DISTANCE_PIPS = float(os.getenv("TRAILING_STOP_MIN_DISTANCE_PIPS", "5.0"))

ADX_MIN_THRESHOLD = float(os.getenv("ADX_MIN_THRESHOLD", "20.0"))
MOMENTUM_MIN_PERCENT = float(os.getenv("MOMENTUM_MIN_PERCENT", "0.15"))
VOLUME_MOMENTUM_MIN = float(os.getenv("VOLUME_MOMENTUM_MIN", "0.5"))

# V94 - Seuils ATR provisoirement conservés (en attente de l'audit)
MIN_ATR_PIPS_BY_PAIR = {
    "EUR_USD": 2.5,
    "GBP_USD": 4.0,
    "USD_CAD": 3.5,
    "AUD_USD": 2.5,
    "AUD_CAD": 3.5,
    "XAU_USD": 60.0,
    "USD_JPY": 3.0,
    "GBP_JPY": 3.0,
    "DEFAULT": 3.0
}

PULLBACK_MIN_PIPS_BY_PAIR = {
    "EUR_USD": 2.0,
    "GBP_USD": 3.0,
    "USD_CAD": 3.0,
    "AUD_USD": 2.0,
    "AUD_CAD": 3.0,
    "XAU_USD": 15.0,
    "USD_JPY": 4.0,
    "GBP_JPY": 6.0,
    "DEFAULT": 2.0
}

EQS_MIN_THRESHOLD = float(os.getenv("EQS_MIN_THRESHOLD", "60.0"))

# =========================
# TRACE JOURNAL
# =========================
class TradeTracer:
    def __init__(self):
        self.traces = []
        self._load()
    
    def _load(self):
        try:
            if os.path.exists(TRACE_JOURNAL):
                with open(TRACE_JOURNAL, 'r') as f:
                    data = json.load(f)
                    self.traces = data.get("traces", [])
        except Exception:
            pass
    
    def _save(self):
        try:
            data = {
                "traces": self.traces,
                "last_update": datetime.utcnow().isoformat()
            }
            with open(TRACE_JOURNAL, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder la trace: {e}")
    
    def log_step(self, trade_id: str, step: str, details: dict, response: dict = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "trade_id": trade_id,
            "step": step,
            "details": details,
            "response": response
        }
        self.traces.append(entry)
        if len(self.traces) > 200:
            self.traces = self.traces[-200:]
        self._save()
        if response:
            logger.info(f"[TRACE] {step} | trade={trade_id} | {details} | response={json.dumps(response, default=str)[:200]}")
        else:
            logger.info(f"[TRACE] {step} | trade={trade_id} | {details}")

tracer = TradeTracer()

# =========================
# LOG HELPERS
# =========================
_seen_log_keys_fvg_recent = set()
_seen_log_keys_fvg_added  = set()
_seen_log_keys_kept_entry = set()

def _reset_log_dedup():
    _seen_log_keys_fvg_recent.clear()
    _seen_log_keys_fvg_added.clear()
    _seen_log_keys_kept_entry.clear()

def _log_fvg_recent_once(pair: str, direction: str, level: float, msg: str, precision: int = 5):
    if not DEBUG_MODE:
        return
    key = (pair, (direction or "").upper(), round(float(level), precision))
    if key in _seen_log_keys_fvg_recent:
        return
    _seen_log_keys_fvg_recent.add(key)
    logger.debug(msg)

def _log_fvg_added_once(pair: str, direction: str, level: float, fvg_type: str, msg: str, precision: int = 5):
    if not DEBUG_MODE:
        return
    key = (pair, (direction or "").upper(), (fvg_type or "UNKNOWN").upper(), round(float(level), precision))
    if key in _seen_log_keys_fvg_added:
        return
    _seen_log_keys_fvg_added.add(key)
    logger.debug(msg)

def _log_narrative_list(entries: list, top_n: int = 10):
    if not DEBUG_MODE:
        return
    if not entries:
        logger.debug("🔎 AUCUNE ENTRÉE DÉTECTÉE")
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
        logger.debug(f" {i}. {entry.get('direction','?')} - {entry.get('type','?')} à {float(entry.get('entry_level',0)):.5f}")
    if other_count:
        logger.debug(f" … (+{other_count} autres entrées)")

logger = logging.getLogger("Advanced-Orderflow-Trading-Bot")
last_reset_time = datetime.utcnow()

# =============================
# CONFIGURATION
# =============================
load_dotenv()

PAIR_LIST = ["GBP_USD", "USD_CAD", "AUD_USD", "XAU_USD", "EUR_USD"]

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
    "MIN_CONFIDENCE_SCORE": 8,
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
        "MACD_DIVERGENCE": 2,
        "FAILURE_SWING": 2,
        "CRT_DETECTED": 1,
        "TBS_DETECTED": 2,
        "ERL_BONUS": 1,
        "IB_BONUS": 1,
        "STRUCTURE_OK": 2,
        "PULLBACK_OK": 2,
        "CLOSE_CONFIRMED": 1
    },
    "PENALTY": {
        "IB_PENALTY": 2,
        "NO_IB_PENALTY": 1,
        "IRL_PENALTY": 3
    }
}

# =============================
# STATISTIQUES ENRICHIES
# =============================
class TradingStats:
    def __init__(self):
        self.stats = defaultdict(lambda: {
            "total_signals": 0,
            "accepted": 0,
            "rejected": 0,
            "wins": 0,
            "losses": 0,
            "breakevens": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "trades": [],
            "entry_metrics": {
                "atr_values": [],
                "adx_values": [],
                "rsi_values": [],
                "eqs_values": [],
                "hours": [],
                "weekdays": [],
                "setup_types": []
            }
        })
        self._load()
    
    def _load(self):
        try:
            if os.path.exists(DECISION_JOURNAL):
                with open(DECISION_JOURNAL, 'r') as f:
                    data = json.load(f)
                    if "stats" in data:
                        for pair, stats_data in data["stats"].items():
                            self.stats[pair] = stats_data
        except Exception:
            pass
    
    def _save(self):
        try:
            data = {
                "stats": dict(self.stats),
                "last_update": datetime.utcnow().isoformat()
            }
            with open(DECISION_JOURNAL, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder les stats: {e}")
    
    def record_signal(self, pair: str, accepted: bool, reason: str = "", 
                      entry: float = 0, sl: float = 0, tp: float = 0,
                      score: int = 0, direction: str = "",
                      entry_metrics: dict = None):
        stats = self.stats[pair]
        stats["total_signals"] += 1
        decision = {
            "timestamp": datetime.utcnow().isoformat(),
            "pair": pair,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "score": score,
            "accepted": accepted,
            "reason": reason
        }
        if entry_metrics:
            decision.update(entry_metrics)
        stats["trades"].append(decision)
        if accepted:
            stats["accepted"] += 1
        else:
            stats["rejected"] += 1
        if len(stats["trades"]) > 200:
            stats["trades"] = stats["trades"][-200:]
        
        if entry_metrics:
            metrics = stats["entry_metrics"]
            if entry_metrics.get("atr"):
                metrics["atr_values"].append(entry_metrics["atr"])
            if entry_metrics.get("adx"):
                metrics["adx_values"].append(entry_metrics["adx"])
            if entry_metrics.get("rsi"):
                metrics["rsi_values"].append(entry_metrics["rsi"])
            if entry_metrics.get("eqs"):
                metrics["eqs_values"].append(entry_metrics["eqs"])
            if entry_metrics.get("hour"):
                metrics["hours"].append(entry_metrics["hour"])
            if entry_metrics.get("weekday"):
                metrics["weekdays"].append(entry_metrics["weekday"])
            if entry_metrics.get("setup_type"):
                metrics["setup_types"].append(entry_metrics["setup_type"])
        
        self._save()
    
    def get_summary(self, pair: str) -> dict:
        stats = self.stats.get(pair, {})
        total = stats.get("total_signals", 0)
        accepted = stats.get("accepted", 0)
        rejected = stats.get("rejected", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        breakevens = stats.get("breakevens", 0)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        profit_factor = stats.get("total_profit", 0) / stats.get("total_loss", 1) if stats.get("total_loss", 0) > 0 else 0
        expectancy = (stats.get("total_profit", 0) - stats.get("total_loss", 0)) / total if total > 0 else 0
        return {
            "pair": pair,
            "total_signals": total,
            "accepted": accepted,
            "rejected": rejected,
            "wins": wins,
            "losses": losses,
            "breakevens": breakevens,
            "win_rate": f"{win_rate*100:.1f}%",
            "profit_factor": f"{profit_factor:.2f}",
            "expectancy": f"${expectancy:.2f}"
        }
    
    def log_summary(self):
        logger.info("=" * 80)
        logger.info("📊 STATISTIQUES GLOBALES")
        logger.info("=" * 80)
        logger.info(f"{'Paire':10} | {'Signaux':>7} | {'Acceptés':>7} | {'Rejetés':>7} | {'Win Rate':>9} | {'PF':>6} | {'Espérance':>10}")
        logger.info("-" * 80)
        for pair in sorted(self.stats.keys()):
            summary = self.get_summary(pair)
            logger.info(
                f"{pair:10} | {summary['total_signals']:>7} | {summary['accepted']:>7} | {summary['rejected']:>7} | "
                f"{summary['win_rate']:>9} | {summary['profit_factor']:>6} | {summary['expectancy']:>10}"
            )
        logger.info("=" * 80)
        
        logger.info("📈 MÉTRIQUES D'ENTRÉE (moyennes par paire)")
        logger.info("-" * 80)
        for pair, stats in self.stats.items():
            metrics = stats.get("entry_metrics", {})
            if metrics.get("eqs_values"):
                avg_eqs = sum(metrics["eqs_values"]) / len(metrics["eqs_values"])
                avg_atr = sum(metrics["atr_values"]) / len(metrics["atr_values"]) if metrics.get("atr_values") else 0
                logger.info(f"{pair:10} | EQS moy: {avg_eqs:.1f} | ATR moy: {avg_atr:.3f} | Trades: {len(metrics['eqs_values'])}")
        logger.info("=" * 80)

stats = TradingStats()

# =============================
# FONCTIONS UTILITAIRES
# =============================
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
    return clusters

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
# LOGGING
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
    ALLOWED_TAGS_V83 = ("[START]", "[SCAN]", "[INFO]", "[SIGNAL]", "[ORDER]", "[RISK]", "[ERROR]", "[TRACE]", "[BE]", "[TSL]", "[CONFIRM]", "[DIAG]")
    def _clean_message_v83(self, message: str, levelname: str) -> str:
        text = repair_mojibake_v82(str(message))
        text = "".join(ch for ch in text if ord(ch) < 128)
        text = " ".join(text.split())
        upper = text.upper()
        if any(text.startswith(tag) for tag in self.ALLOWED_TAGS_V83):
            return text
        if levelname in ("ERROR", "CRITICAL"):
            tag = "[ERROR]"
        elif "DIAG" in upper:
            tag = "[DIAG]"
        elif "CONFIRM" in upper:
            tag = "[CONFIRM]"
        elif "TRACE" in upper:
            tag = "[TRACE]"
        elif "BE" in upper or "BREAKEVEN" in upper:
            tag = "[BE]"
        elif "TSL" in upper or "TRAILING" in upper:
            tag = "[TSL]"
        elif "SIGNAL" in upper:
            tag = "[SIGNAL]"
        elif "ORDER" in upper or "ORDRE" in upper or "EXECUTION" in upper or "/ORDERS" in upper:
            tag = "[ORDER]"
        elif "RISK" in upper or "MARGIN" in upper or "UNITS" in upper:
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
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
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

def mark_signal_sent(pair: str, direction: str, entry_level: float, zone_start: float, zone_end: float):
    key = (pair, direction, round(entry_level, 5), round(zone_start, 5), round(zone_end, 5))
    sent_signals[key] = time.time()
    logger.info(f"✅ Signal marqué comme envoyé : {key}")

# =============================
# INDICATEURS TECHNIQUES
# =============================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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

def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 14, divergence_type: str = "all") -> bool:
    if len(df) < lookback * 2 + 5:
        return False
    prices = df["close"].tail(lookback * 2).reset_index(drop=True)
    rsi_vals = calculate_rsi(df["close"]).tail(lookback * 2).reset_index(drop=True)
    price_peaks = []
    price_lows = []
    rsi_peaks = []
    rsi_lows = []
    for i in range(3, len(prices) - 3):
        if prices.iloc[i] > prices.iloc[i-3:i].max() and prices.iloc[i] > prices.iloc[i+1:i+4].max():
            price_peaks.append((i, prices.iloc[i]))
        if prices.iloc[i] < prices.iloc[i-3:i].min() and prices.iloc[i] < prices.iloc[i+1:i+4].min():
            price_lows.append((i, prices.iloc[i]))
        if rsi_vals.iloc[i] > rsi_vals.iloc[i-3:i].max() and rsi_vals.iloc[i] > rsi_vals.iloc[i+1:i+4].max():
            rsi_peaks.append((i, rsi_vals.iloc[i]))
        if rsi_vals.iloc[i] < rsi_vals.iloc[i-3:i].min() and rsi_vals.iloc[i] < rsi_vals.iloc[i+1:i+4].min():
            rsi_lows.append((i, rsi_vals.iloc[i]))
    if divergence_type in ["bullish", "all"] and len(price_lows) >= 2 and len(rsi_lows) >= 2:
        last_price_low = price_lows[-1][1]
        prev_price_low = price_lows[-2][1]
        last_rsi_low = rsi_lows[-1][1]
        prev_rsi_low = rsi_lows[-2][1]
        if last_price_low < prev_price_low and last_rsi_low > prev_rsi_low:
            return True
    if divergence_type in ["bearish", "all"] and len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
        last_price_peak = price_peaks[-1][1]
        prev_price_peak = price_peaks[-2][1]
        last_rsi_peak = rsi_peaks[-1][1]
        prev_rsi_peak = rsi_peaks[-2][1]
        if last_price_peak > prev_price_peak and last_rsi_peak < prev_rsi_peak:
            return True
    return False

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

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd_momentum(df: pd.DataFrame) -> pd.Series:
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram

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

def _directional_score(raw_score: int, direction: str) -> int:
    return int(raw_score) if (direction or "").upper() == "BUY" else -int(raw_score)

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
    if not DEBUG_MODE:
        return
    labels = [
        ("ICT", "ICT"),
        ("Structure_H1", "Structure H1"),
        ("HTF_Alignment", "HTF Alignment"),
        ("Risk_RR_Distance", "Risk/RR/Distance"),
        ("Secondary", "Secondary"),
        ("Momentum", "Momentum"),
        ("Structure", "Structure V94"),
        ("Pullback", "Pullback V94"),
    ]
    logger.debug("===== SCORE DETAIL =====")
    for key, label in labels:
        if key in score_components:
            logger.debug(f"{label:<19}: {int(score_components[key]):+d}")
    logger.debug(f"TOTAL = {int(total):+d}")
    logger.debug(f"Decision = {decision}")

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
# DÉTECTION SWING POINTS
# =============================
def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> tuple:
    swing_highs = []
    swing_lows = []
    for i in range(lookback, len(df) - lookback):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        if high == df["high"].iloc[i - lookback:i + lookback + 1].max() and df["close"].iloc[i] < df["open"].iloc[i]:
            swing_highs.append({"index": i, "time": df.index[i], "price": high})
        if low == df["low"].iloc[i - lookback:i + lookback + 1].min() and df["close"].iloc[i] > df["open"].iloc[i]:
            swing_lows.append({"index": i, "time": df.index[i], "price": low})
    return swing_highs, swing_lows

def detect_swing_points_advanced(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> tuple:
    swing_highs = []
    swing_lows = []
    for i in range(lookback, len(df) - lookback):
        try:
            if df["high"].iloc[i] == df["high"].iloc[i-lookback:i+lookback+1].max() and df["close"].iloc[i] < df["open"].iloc[i]:
                swing_highs.append({"index": i, "time": df.index[i], "price": df["high"].iloc[i], "type": "SWING_HIGH", "strength": "STRONG"})
        except Exception:
            pass
        try:
            if df["low"].iloc[i] == df["low"].iloc[i-lookback:i+lookback+1].min() and df["close"].iloc[i] > df["open"].iloc[i]:
                swing_lows.append({"index": i, "time": df.index[i], "price": df["low"].iloc[i], "type": "SWING_LOW", "strength": "STRONG"})
        except Exception:
            pass
    return swing_highs, swing_lows

# =============================
# DÉTECTION FVG
# =============================
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

def get_fvg_midpoint(fvg: dict) -> float:
    if not all(k in fvg for k in ["high_level", "low_level"]):
        return None
    high = float(fvg["high_level"])
    low = float(fvg["low_level"])
    if high == low:
        return None
    return round((high + low) / 2, 5)

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
        if (bias in ["BUY", "NEUTRAL"] and lower_wick >= body_size * min_wick_ratio and lower_wick >= upper_wick * 1.5 and lower_wick >= total_range * 0.4 and rsi_m15 < 60 and confirmation_candle["close"] > confirmation_candle["open"] and confirmation_candle["close"] > rejection_candle["high"] and (confirmation_candle["close"] - confirmation_candle["open"]) >= 0.7 * (confirmation_candle["high"] - confirmation_candle["low"])):
            if abs(current_price - rejection_candle["low"]) <= pip_tolerance:
                poi_list.append({"type": "WICK_REJECTION", "price_level": rejection_candle["low"], "wick_size": lower_wick, "body_size": body_size, "time": df.index[i], "direction": "BUY", "wick_ratio": lower_wick / total_range, "rsi_at_rejection": rsi_m15, "pair": pair})
        elif (bias in ["SELL", "NEUTRAL"] and upper_wick >= body_size * min_wick_ratio and upper_wick >= lower_wick * 1.5 and upper_wick >= total_range * 0.4 and rsi_m15 > 40 and confirmation_candle["close"] < confirmation_candle["open"] and confirmation_candle["close"] < rejection_candle["low"] and (confirmation_candle["open"] - confirmation_candle["close"]) >= 0.7 * (confirmation_candle["high"] - confirmation_candle["low"])):
            if abs(current_price - rejection_candle["high"]) <= pip_tolerance:
                poi_list.append({"type": "WICK_REJECTION", "price_level": rejection_candle["high"], "wick_size": upper_wick, "body_size": body_size, "time": df.index[i], "direction": "SELL", "wick_ratio": upper_wick / total_range, "rsi_at_rejection": rsi_m15, "pair": pair})
    return poi_list

# =============================
# DÉTECTION ORDER FLOW LEGS
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
            if fvg_idx <= sl_idx or fvg_low < sl_price:
                continue
            for swing_high in swing_highs:
                sh_idx = swing_high.get("index")
                sh_price = float(swing_high.get("price", 0))
                if sh_idx <= fvg_idx or sh_price <= float(fvg.get("high_level", 0)):
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
            if fvg_idx <= sh_idx or fvg_high > sh_price:
                continue
            for swing_low in swing_lows:
                sl_idx = swing_low.get("index")
                sl_price = float(swing_low.get("price", 0))
                if sl_idx <= fvg_idx or sl_price >= float(fvg.get("low_level", 0)):
                    continue
                ofls.append({"direction": "SELL", "start": swing_high, "fvg": fvg, "end": swing_low})
    for fvg in fvgs:
        ofls.append({"direction": fvg.get("direction"), "fvg": fvg})
    return ofls

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
# FONCTIONS DE PRIX ET CONVERSION
# =============================
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

# ============================================================
# V94 - FILTRES AVEC AUDIT ATR
# ============================================================

def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    try:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.zeros(len(df))
        for i in range(1, len(df)):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]),
                       abs(low[i] - close[i-1]))
        
        plus_dm = np.zeros(len(df))
        minus_dm = np.zeros(len(df))
        for i in range(1, len(df)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        atr = talib.ATR(high, low, close, timeperiod=period)
        plus_di = 100 * talib.SMA(plus_dm, period) / atr
        minus_di = 100 * talib.SMA(minus_dm, period) / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = talib.SMA(dx, period)
        
        return float(adx[-1]) if not np.isnan(adx[-1]) else 0.0
    except Exception:
        return 0.0

def detect_breakout_candle(df: pd.DataFrame, lookback: int = 5) -> dict:
    if len(df) < lookback + 2:
        return {"type": None, "level": None, "confirmed": False}
    
    last = df.iloc[-1]
    
    recent_high = df['high'].iloc[-lookback-1:-1].max()
    recent_low = df['low'].iloc[-lookback-1:-1].min()
    
    body = abs(last['close'] - last['open'])
    total_range = last['high'] - last['low']
    body_ratio = body / total_range if total_range > 0 else 0
    
    is_strong_close = body_ratio > 0.45
    
    if last['close'] > recent_high and last['close'] > last['open'] and is_strong_close:
        return {"type": "BUY", "level": recent_high, "confirmed": True}
    
    if last['close'] < recent_low and last['close'] < last['open'] and is_strong_close:
        return {"type": "SELL", "level": recent_low, "confirmed": True}
    
    if last['close'] > recent_high:
        return {"type": "BUY", "level": recent_high, "confirmed": False}
    if last['close'] < recent_low:
        return {"type": "SELL", "level": recent_low, "confirmed": False}
    
    return {"type": None, "level": None, "confirmed": False}

def calculate_momentum(df: pd.DataFrame, period: int = 5) -> float:
    if len(df) < period + 1:
        return 0.0
    try:
        close = df['close']
        roc = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period] * 100
        return float(roc)
    except Exception:
        return 0.0

def calculate_volume_momentum(df: pd.DataFrame, period: int = 3) -> float:
    if len(df) < period + 1 or 'volume' not in df.columns:
        return 1.0
    try:
        avg_volume = df['volume'].iloc[-period:].mean()
        prev_avg_volume = df['volume'].iloc[-period*2:-period].mean() if len(df) >= period*2 else avg_volume
        return (avg_volume / prev_avg_volume) if prev_avg_volume > 0 else 1.0
    except Exception:
        return 1.0

# ============================================================
# V94 - ENTRY QUALITY SCORE AVEC AUDIT ATR
# ============================================================

def calculate_entry_quality_score(
    pair: str,
    direction: str,
    df_m15: pd.DataFrame,
    entry_level: float,
    current_price: float,
    atr: float
) -> dict:
    direction = direction.upper()
    pip_value = get_pip_value_for_pair(pair)
    scores = {}
    total = 0
    logs = []
    
    # 1. Distance à la zone d'entrée
    entry_zone = abs(entry_level - current_price)
    entry_zone_pips = price_to_pips(entry_zone, pair)
    if entry_zone_pips <= 3:
        scores["distance_zone"] = 20
        logs.append(f"distance_zone: {entry_zone_pips:.1f}pips -> 20")
    elif entry_zone_pips <= 7:
        scores["distance_zone"] = 15
        logs.append(f"distance_zone: {entry_zone_pips:.1f}pips -> 15")
    elif entry_zone_pips <= 12:
        scores["distance_zone"] = 10
        logs.append(f"distance_zone: {entry_zone_pips:.1f}pips -> 10")
    elif entry_zone_pips <= 18:
        scores["distance_zone"] = 5
        logs.append(f"distance_zone: {entry_zone_pips:.1f}pips -> 5")
    else:
        scores["distance_zone"] = 0
        logs.append(f"distance_zone: {entry_zone_pips:.1f}pips -> 0")
    
    # 2. Proximité de l'EMA20
    try:
        ema20 = df_m15['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema_distance = abs(current_price - ema20)
        ema_distance_pips = price_to_pips(ema_distance, pair)
        if ema_distance_pips <= 4:
            scores["ema_proximity"] = 20
            logs.append(f"ema_proximity: {ema_distance_pips:.1f}pips -> 20")
        elif ema_distance_pips <= 8:
            scores["ema_proximity"] = 15
            logs.append(f"ema_proximity: {ema_distance_pips:.1f}pips -> 15")
        elif ema_distance_pips <= 13:
            scores["ema_proximity"] = 10
            logs.append(f"ema_proximity: {ema_distance_pips:.1f}pips -> 10")
        elif ema_distance_pips <= 20:
            scores["ema_proximity"] = 5
            logs.append(f"ema_proximity: {ema_distance_pips:.1f}pips -> 5")
        else:
            scores["ema_proximity"] = 0
            logs.append(f"ema_proximity: {ema_distance_pips:.1f}pips -> 0")
    except Exception:
        scores["ema_proximity"] = 10
        logs.append("ema_proximity: error -> 10")
    
    # 3. Positionnement par rapport au range récent
    try:
        recent_high = df_m15['high'].iloc[-10:].max()
        recent_low = df_m15['low'].iloc[-10:].min()
        range_size = recent_high - recent_low
        if range_size > 0:
            if direction == "BUY":
                position = (current_price - recent_low) / range_size
                if position < 0.4:
                    scores["range_position"] = 20
                    logs.append(f"range_position: {position:.2f} (bas) -> 20")
                elif position < 0.6:
                    scores["range_position"] = 15
                    logs.append(f"range_position: {position:.2f} -> 15")
                elif position < 0.8:
                    scores["range_position"] = 10
                    logs.append(f"range_position: {position:.2f} -> 10")
                else:
                    scores["range_position"] = 5
                    logs.append(f"range_position: {position:.2f} (haut) -> 5")
            else:
                position = (recent_high - current_price) / range_size
                if position < 0.4:
                    scores["range_position"] = 20
                    logs.append(f"range_position: {position:.2f} (haut) -> 20")
                elif position < 0.6:
                    scores["range_position"] = 15
                    logs.append(f"range_position: {position:.2f} -> 15")
                elif position < 0.8:
                    scores["range_position"] = 10
                    logs.append(f"range_position: {position:.2f} -> 10")
                else:
                    scores["range_position"] = 5
                    logs.append(f"range_position: {position:.2f} (bas) -> 5")
        else:
            scores["range_position"] = 10
            logs.append("range_position: range nul -> 10")
    except Exception:
        scores["range_position"] = 10
        logs.append("range_position: error -> 10")
    
    # 4. Retracement
    pullback_passed, _ = filter_pullback(df_m15, direction, entry_level, current_price, pair)
    if pullback_passed:
        scores["pullback_quality"] = 20
        logs.append("pullback_quality: OK -> 20")
    else:
        if len(df_m15) > 5:
            if direction == "BUY":
                recent_low = df_m15['low'].iloc[-5:].min()
                pullback_depth = current_price - recent_low
                if pullback_depth > 0:
                    pullback_pips = price_to_pips(pullback_depth, pair)
                    if pullback_pips >= 2:
                        scores["pullback_quality"] = 15
                        logs.append(f"pullback_quality: {pullback_pips:.1f}pips -> 15")
                    else:
                        scores["pullback_quality"] = 10
                        logs.append(f"pullback_quality: {pullback_pips:.1f}pips -> 10")
                else:
                    scores["pullback_quality"] = 5
                    logs.append("pullback_quality: pas de pullback -> 5")
            else:
                recent_high = df_m15['high'].iloc[-5:].max()
                pullback_depth = recent_high - current_price
                if pullback_depth > 0:
                    pullback_pips = price_to_pips(pullback_depth, pair)
                    if pullback_pips >= 2:
                        scores["pullback_quality"] = 15
                        logs.append(f"pullback_quality: {pullback_pips:.1f}pips -> 15")
                    else:
                        scores["pullback_quality"] = 10
                        logs.append(f"pullback_quality: {pullback_pips:.1f}pips -> 10")
                else:
                    scores["pullback_quality"] = 5
                    logs.append("pullback_quality: pas de pullback -> 5")
        else:
            scores["pullback_quality"] = 10
            logs.append("pullback_quality: données insuffisantes -> 10")
    
    # 5. StochRSI
    try:
        k, _ = calculate_stoch_rsi(df_m15['close'])
        if direction == "BUY":
            if 15 <= k <= 75:
                scores["stoch_position"] = 20
                logs.append(f"stoch_position: {k:.1f} (zone BUY) -> 20")
            elif k < 15:
                scores["stoch_position"] = 15
                logs.append(f"stoch_position: {k:.1f} (survendu) -> 15")
            elif k < 85:
                scores["stoch_position"] = 10
                logs.append(f"stoch_position: {k:.1f} -> 10")
            else:
                scores["stoch_position"] = 5
                logs.append(f"stoch_position: {k:.1f} (surachat) -> 5")
        else:
            if 25 <= k <= 85:
                scores["stoch_position"] = 20
                logs.append(f"stoch_position: {k:.1f} (zone SELL) -> 20")
            elif k > 85:
                scores["stoch_position"] = 15
                logs.append(f"stoch_position: {k:.1f} (surachat) -> 15")
            elif k > 15:
                scores["stoch_position"] = 10
                logs.append(f"stoch_position: {k:.1f} -> 10")
            else:
                scores["stoch_position"] = 5
                logs.append(f"stoch_position: {k:.1f} (survendu) -> 5")
    except Exception:
        scores["stoch_position"] = 10
        logs.append("stoch_position: error -> 10")
    
    total = sum(scores.values())
    
    details = {
        "distance_zone": scores["distance_zone"],
        "ema_proximity": scores["ema_proximity"],
        "range_position": scores["range_position"],
        "pullback_quality": scores["pullback_quality"],
        "stoch_position": scores["stoch_position"],
        "total": total,
        "passed": total >= EQS_MIN_THRESHOLD,
        "logs": logs
    }
    
    return details

# ============================================================
# V94 - FILTRES STRUCTURE / PULLBACK AVEC AUDIT ATR
# ============================================================

def filter_market_structure(df: pd.DataFrame, direction: str, lookback: int = 5) -> tuple:
    if len(df) < lookback * 2 + 2:
        return False, "Données insuffisantes"
    
    direction = direction.upper()
    swing_highs, swing_lows = detect_swing_points(df, lookback=3)
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False, "Pas assez de swing points"
    
    last_highs = sorted(swing_highs, key=lambda x: x['index'])[-2:]
    last_lows = sorted(swing_lows, key=lambda x: x['index'])[-2:]
    
    if direction == "BUY":
        hh = last_highs[-1]['price'] > last_highs[-2]['price']
        hl = last_lows[-1]['price'] > last_lows[-2]['price']
        if hh and hl:
            return True, "Structure haussière (HH/HL)"
        elif hh or hl:
            return True, "Structure partiellement haussière"
        else:
            return False, f"Structure non haussière (HH={hh}, HL={hl})"
    
    elif direction == "SELL":
        lh = last_highs[-1]['price'] < last_highs[-2]['price']
        ll = last_lows[-1]['price'] < last_lows[-2]['price']
        if lh and ll:
            return True, "Structure baissière (LH/LL)"
        elif lh or ll:
            return True, "Structure partiellement baissière"
        else:
            return False, f"Structure non baissière (LH={lh}, LL={ll})"
    
    return False, f"Direction {direction} invalide"

def filter_pullback(df: pd.DataFrame, direction: str, entry_level: float, current_price: float, pair: str) -> tuple:
    direction = direction.upper()
    pip_value = get_pip_value_for_pair(pair)
    min_pullback_pips = PULLBACK_MIN_PIPS_BY_PAIR.get(pair, PULLBACK_MIN_PIPS_BY_PAIR["DEFAULT"])
    min_pullback_price = min_pullback_pips * pip_value
    
    if len(df) < 8:
        return False, "Données insuffisantes pour pullback"
    
    recent = df.iloc[-8:]
    
    if direction == "BUY":
        recent_high = recent['high'].max()
        pullback_depth = recent_high - current_price
        pullback_pips = price_to_pips(pullback_depth, pair)
        if pullback_depth >= min_pullback_price:
            return True, f"Pullback OK ({pullback_pips:.1f} pips >= {min_pullback_pips})"
        else:
            return False, f"Pullback insuffisant ({pullback_pips:.1f} pips < {min_pullback_pips})"
    
    elif direction == "SELL":
        recent_low = recent['low'].min()
        pullback_depth = current_price - recent_low
        pullback_pips = price_to_pips(pullback_depth, pair)
        if pullback_depth >= min_pullback_price:
            return True, f"Pullback OK ({pullback_pips:.1f} pips >= {min_pullback_pips})"
        else:
            return False, f"Pullback insuffisant ({pullback_pips:.1f} pips < {min_pullback_pips})"
    
    return False, f"Direction {direction} invalide"

def filter_min_volatility(df: pd.DataFrame, pair: str) -> tuple:
    """
    V94 : Audit ATR détaillé.
    Affiche la valeur brute, la conversion en pips, et la comparaison.
    """
    if len(df) < ATR_PERIOD:
        return False, "Données insuffisantes pour ATR"
    
    # Calcul de l'ATR en prix
    atr_price = calculate_atr(df, period=ATR_PERIOD)
    # Conversion en pips
    atr_pips = price_to_pips(atr_price, pair)
    
    min_atr_pips = MIN_ATR_PIPS_BY_PAIR.get(pair, MIN_ATR_PIPS_BY_PAIR["DEFAULT"])
    
    # Log d'audit détaillé
    logger.info(f"[ATR_AUDIT] {pair} | ATR prix: {atr_price:.6f} | ATR pips: {atr_pips:.1f} | Seuil: {min_atr_pips:.1f} | Écart: {atr_pips - min_atr_pips:.1f}")
    
    if atr_pips < min_atr_pips:
        return False, f"ATR = {atr_pips:.1f} pips < seuil {min_atr_pips} pips (brut: {atr_price:.6f})"
    
    return True, f"ATR = {atr_pips:.1f} pips >= seuil {min_atr_pips} pips (brut: {atr_price:.6f})"

def filter_close_confirmation(df: pd.DataFrame, direction: str) -> tuple:
    if len(df) < 2:
        return False, "Données insuffisantes"
    
    last = df.iloc[-1]
    
    if direction == "BUY":
        if last['close'] > last['open']:
            return True, "Bougie M15 confirmée (close > open)"
        else:
            return False, "Bougie M15 non confirmée pour BUY (close < open)"
    elif direction == "SELL":
        if last['close'] < last['open']:
            return True, "Bougie M15 confirmée (close < open)"
        else:
            return False, "Bougie M15 non confirmée pour SELL (close > open)"
    
    return False, f"Direction {direction} invalide"

# ============================================================
# V89.5 - FILTRE ANTI-ESSOUFFLEMENT (conservé)
# ============================================================

def filter_momentum_exhaustion(
    pair: str,
    direction: str,
    df_m15: pd.DataFrame,
    df_h1: pd.DataFrame,
    entry_level: float,
    current_price: float,
    entry_type: str,
) -> tuple:
    direction = direction.upper()
    penalties = {"adx": 0, "momentum": 0, "breakout": 0, "volume": 0}
    messages = []
    passed = True

    breakout = detect_breakout_candle(df_m15, lookback=5)
    if not breakout.get("confirmed", False):
        if breakout.get("type") == direction:
            messages.append("Cassure non confirmée → -1")
            penalties["breakout"] = -1
        else:
            messages.append("Pas de cassure dans la direction → -2")
            penalties["breakout"] = -2
    elif breakout.get("type") != direction:
        messages.append(f"Cassure opposée → -3")
        penalties["breakout"] = -3
    else:
        messages.append("Cassure confirmée → +2")
        penalties["breakout"] = 2

    if breakout.get("level"):
        pip_value = get_pip_value_for_pair(pair)
        breakout_level = breakout["level"]
        if direction == "BUY" and entry_level > breakout_level + pip_value * 3:
            messages.append("Entrée trop loin de la cassure → -1")
            penalties["breakout"] -= 1
        elif direction == "SELL" and entry_level < breakout_level - pip_value * 3:
            messages.append("Entrée trop loin de la cassure → -1")
            penalties["breakout"] -= 1

    adx = calculate_adx(df_h1, period=14)
    if adx < 15:
        messages.append(f"ADX très faible ({adx:.1f}) → -3")
        penalties["adx"] = -3
        passed = False
    elif adx < ADX_MIN_THRESHOLD:
        messages.append(f"ADX modéré ({adx:.1f}) → -1")
        penalties["adx"] = -1
    else:
        messages.append(f"ADX OK ({adx:.1f}) → +1")
        penalties["adx"] = 1

    momentum = calculate_momentum(df_m15, period=5)
    if direction == "BUY":
        if momentum < -0.5:
            messages.append(f"Momentum baissier fort ({momentum:.2f}%) → -3")
            penalties["momentum"] = -3
            passed = False
        elif momentum < 0:
            messages.append(f"Momentum baissier léger ({momentum:.2f}%) → -1")
            penalties["momentum"] = -1
        elif momentum < MOMENTUM_MIN_PERCENT:
            messages.append(f"Momentum faible ({momentum:.2f}%) → 0")
            penalties["momentum"] = 0
        else:
            messages.append(f"Momentum haussier ({momentum:.2f}%) → +2")
            penalties["momentum"] = 2
    else:
        if momentum > 0.5:
            messages.append(f"Momentum haussier fort ({momentum:.2f}%) → -3")
            penalties["momentum"] = -3
            passed = False
        elif momentum > 0:
            messages.append(f"Momentum haussier léger ({momentum:.2f}%) → -1")
            penalties["momentum"] = -1
        elif momentum > -MOMENTUM_MIN_PERCENT:
            messages.append(f"Momentum faible ({momentum:.2f}%) → 0")
            penalties["momentum"] = 0
        else:
            messages.append(f"Momentum baissier ({momentum:.2f}%) → +2")
            penalties["momentum"] = 2

    roc_fast = calculate_momentum(df_m15, period=3)
    roc_slow = calculate_momentum(df_m15, period=10)
    if direction == "BUY":
        if roc_fast < roc_slow * 0.8:
            messages.append("Décélération haussière → -1")
            penalties["momentum"] -= 1
    else:
        if roc_fast > roc_slow * 0.8:
            messages.append("Décélération baissière → -1")
            penalties["momentum"] -= 1

    vol_momentum = calculate_volume_momentum(df_m15, period=3)
    if vol_momentum < VOLUME_MOMENTUM_MIN and vol_momentum > 0:
        messages.append(f"Volume en baisse ({vol_momentum:.2f}) → -1")
        penalties["volume"] = -1
    elif vol_momentum > 1.2:
        messages.append(f"Volume en hausse ({vol_momentum:.2f}) → +1")
        penalties["volume"] = 1
    else:
        messages.append("Volume neutre → 0")

    total_penalty = sum(penalties.values())
    message = " | ".join(messages)
    return passed, message, penalties, total_penalty

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
# GESTION DES ORDRES
# =============================
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
                       confidence_score: int = None, eqs_score: int = None):
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram désactivé")
        return
    if None in [pair, direction, entry_price, stop_loss, take_profit, rsi, entry_type]:
        logger.error(f"❌ Valeur manquante pour Telegram")
        return
    if direction == "BUY":
        if stop_loss > entry_price or take_profit < entry_price:
            logger.error(f"❌ INCOHÉRENCE SL/TP pour BUY")
            return
    elif direction == "SELL":
        if stop_loss < entry_price or take_profit > entry_price:
            logger.error(f"❌ INCOHÉRENCE SL/TP pour SELL")
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
    
    eqs_info = f" | <b>EQS:</b> {eqs_score}/100" if eqs_score else ""
    
    if confidence_score:
        score_info = (f"<b>Score:</b> {confidence_score}/{SCORING_CONFIG['MIN_CONFIDENCE_SCORE']}{eqs_info} | <b>Qualité:</b> {quality_label}\n<b>Win Rate estimé:</b> {win_rate}\n<b>R/R:</b> 1:{rr_display}\n")
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
    if score_details.get("Structure_V94", "").startswith("+"):
        confluence_tags.append("STRUCTURE")
    if score_details.get("Pullback_V94", "").startswith("+"):
        confluence_tags.append("PULLBACK")
    confluences_line = f"<b>Confluences:</b> {' · '.join(confluence_tags)}\n" if confluence_tags else ""
    
    message = f"""
<b>FVG ORDERFLOW TRADING SIGNAL V94</b>
<b>Paire:</b> {pair}
<b>Direction:</b> {direction}
<b>Type d'entrée:</b> {entry_type}
<b>Bias:</b> {bias_analysis.get('bias', 'N/A') if bias_analysis else 'N/A'}
{score_info}{confluences_line}
<b>Entrée:</b> {entry_price:.5f}
<b>Stop Loss:</b> {stop_loss:.5f}
<b>Take Profit:</b> {take_profit:.5f}
<b>RSI:</b> {rsi:.1f}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
        if response.status_code == 200:
            logger.info(f"✅ Telegram envoyé pour {pair}")
        else:
            logger.error(f"❌ Échec Telegram: {response.text}")
    except Exception as e:
        logger.error(f"💥 Erreur réseau Telegram: {e}")

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
        if d1_trend == direction:
            bonus = 2
            label = "+2 (D1 EMA50 aligné)"
            if len(df_d1) >= 200:
                ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
                if (direction == "BUY" and current > ema200) or (direction == "SELL" and current < ema200):
                    bonus += 1
                    label = "+3 (D1 EMA50+EMA200 alignés)"
            return bonus, label
        else:
            return -2, "-2 (Contre tendance D1)"
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
            if detect_rsi_divergence(df_h1, divergence_type="bullish"):
                return 3, "+3 (Divergence RSI H1 haussière)"
            if detect_rsi_divergence(df_m15, divergence_type="bullish"):
                return 2, "+2 (Divergence RSI M15 haussière)"
        elif direction == "SELL":
            if detect_rsi_divergence(df_h1, divergence_type="bearish"):
                return 3, "+3 (Divergence RSI H1 baissière)"
            if detect_rsi_divergence(df_m15, divergence_type="bearish"):
                return 2, "+2 (Divergence RSI M15 baissière)"
    except Exception:
        pass
    return 0, "0 (Pas de divergence RSI)"

def estimate_win_rate(score: int, eqs: int, confluences: dict) -> str:
    if score >= 14 and eqs >= 75:
        base = 80
    elif score >= 12 and eqs >= 65:
        base = 70
    elif score >= 10 and eqs >= 55:
        base = 60
    else:
        base = 50
    
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
    if confluences.get("structure_ok"):
        base += 3
    if confluences.get("pullback_ok"):
        base += 2
    
    return f"~{min(base, 92)}%"

def get_signal_quality_label(score: int, eqs: int) -> str:
    if score >= 15 and eqs >= 75:
        return "SNIPER"
    elif score >= 13 and eqs >= 65:
        return "A+"
    elif score >= 11 and eqs >= 55:
        return "A"
    elif score >= 9:
        return "B+"
    return "B"

# =============================
# SYSTÈME DE SCORING V94
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
        "Structure_H1": 0,
        "HTF_Alignment": 0,
        "Risk_RR_Distance": 0,
        "Secondary": 0,
        "Momentum": 0,
        "Structure": 0,
        "Pullback": 0,
    }
    details: dict = {}
    rejection_logs = []
    min_required = MIN_CONFIDENCE_SCORE_BY_PAIR.get(pair, MIN_CONFIDENCE_SCORE_BY_PAIR["DEFAULT"])
    direction = (direction or "").upper()
    entry_level = entry.get("entry_level")
    entry_type = str(entry.get("type", "FVG_RETEST")).upper()
    
    if entry_level is None or direction not in ["BUY", "SELL"]:
        return {"passed": False, "total_score": 0, "final_confidence": "LOW", "details": {"VETO": "Entrée/direction invalide"}}
    
    entry_level = float(entry_level)
    atr_value = calculate_atr(df_m15)
    fvg_data = entry.get("fvg") if "fvg" in entry else None
    
    stop_loss, take_profit = calculate_sl_tp(
        entry_price=entry_level, atr=atr_value, direction=direction,
        pair=pair, entry_type=entry_type, fvg_data=fvg_data,
    )
    
    # === V94 : ENTRY QUALITY SCORE (EQS) ===
    eqs_result = calculate_entry_quality_score(
        pair=pair,
        direction=direction,
        df_m15=df_m15,
        entry_level=entry_level,
        current_price=current_price,
        atr=atr_value
    )
    
    eqs_score = eqs_result["total"]
    eqs_passed = eqs_result["passed"]
    details["EQS_Details"] = eqs_result["logs"]
    
    if not eqs_passed:
        rejection_logs.append(f"EQS = {eqs_score}/100 < seuil {EQS_MIN_THRESHOLD}")
        details["VETO"] = f"❌ EQS insuffisant: {eqs_score}/100 < {EQS_MIN_THRESHOLD}"
        return {
            "passed": False,
            "total_score": 0,
            "final_confidence": "LOW",
            "details": details,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr_value,
            "eqs_score": eqs_score,
            "eqs_details": eqs_result,
            "rejection_logs": rejection_logs
        }
    
    details["EQS"] = f"{eqs_score}/100"
    
    # === FILTRES BLOQUANTS ===
    
    # 1. Volatilité minimum (avec audit ATR)
    vol_passed, vol_msg = filter_min_volatility(df_m15, pair)
    if not vol_passed:
        rejection_logs.append(vol_msg)
        details["VETO"] = f"❌ VOLATILITÉ: {vol_msg}"
        return {
            "passed": False,
            "total_score": 0,
            "final_confidence": "LOW",
            "details": details,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr_value,
            "eqs_score": eqs_score,
            "eqs_details": eqs_result,
            "rejection_logs": rejection_logs
        }
    details["Volatility"] = vol_msg
    
    # 2. Structure de marché
    struct_passed, struct_msg = filter_market_structure(df_h1, direction, lookback=5)
    if not struct_passed:
        rejection_logs.append(struct_msg)
        details["VETO"] = f"❌ STRUCTURE: {struct_msg}"
        return {
            "passed": False,
            "total_score": 0,
            "final_confidence": "LOW",
            "details": details,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr_value,
            "eqs_score": eqs_score,
            "eqs_details": eqs_result,
            "rejection_logs": rejection_logs
        }
    if "partiellement" in struct_msg:
        score_components["Structure"] += 1
        details["Structure_V94"] = f"+1 ({struct_msg})"
    else:
        score_components["Structure"] += 2
        details["Structure_V94"] = f"+2 ({struct_msg})"
    
    # 3. Pullback
    pullback_passed, pullback_msg = filter_pullback(df_m15, direction, entry_level, current_price, pair)
    if not pullback_passed:
        rejection_logs.append(pullback_msg)
        details["VETO"] = f"❌ PULLBACK: {pullback_msg}"
        return {
            "passed": False,
            "total_score": 0,
            "final_confidence": "LOW",
            "details": details,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr_value,
            "eqs_score": eqs_score,
            "eqs_details": eqs_result,
            "rejection_logs": rejection_logs
        }
    score_components["Pullback"] += 2
    details["Pullback_V94"] = f"+2 ({pullback_msg})"
    
    # 4. Confirmation de clôture
    close_passed, close_msg = filter_close_confirmation(df_m15, direction)
    if close_passed:
        score_components["Secondary"] += 1
        details["Close_Confirm"] = f"+1 ({close_msg})"
    else:
        details["Close_Confirm"] = close_msg
    
    # === FILTRE ANTI-ESSOUFFLEMENT (en pénalités) ===
    momentum_passed, momentum_msg, momentum_penalties, penalty_total = filter_momentum_exhaustion(
        pair=pair,
        direction=direction,
        df_m15=df_m15,
        df_h1=df_h1,
        entry_level=entry_level,
        current_price=current_price,
        entry_type=entry_type
    )
    
    score_components["Momentum"] += penalty_total
    details["Momentum"] = f"{penalty_total:+d} ({momentum_msg})"
    
    if not momentum_passed:
        score_components["Momentum"] -= 5
        details["Momentum"] += " | PASSAGE FORCÉ"
    
    momentum_filter_info = {"passed": momentum_passed, "message": momentum_msg, "penalties": momentum_penalties}
    
    # === SCORING PRINCIPAL ===
    if (direction == "BUY" and bias == "BUY") or (direction == "SELL" and bias == "SELL"):
        score_components["ICT"] += 3
        details["Trend_H4"] = "+3 (Aligné)"
    elif bias == "NEUTRAL":
        score_components["ICT"] += 1
        details["Trend_H4"] = "+1 (Neutre)"
    else:
        score_components["ICT"] -= 2
        details["Trend_H4"] = "-2 (H4 opposé)"
    
    try:
        distance = abs(float(current_price) - entry_level)
        pip = get_pip_value_for_pair(pair)
        entry_type_max_pips = {
            "FVG_RETEST_PERFECT": 15.0, "FVG_RETEST": 18.0,
            "NESTED_FVG": 18.0, "WICK_REJECTION": 15.0,
            "BISI": 18.0, "BREAKER": 15.0,
        }
        max_pips = entry_type_max_pips.get(entry_type, STRICT_MAX_DISTANCE_PIPS.get(pair, STRICT_MAX_DISTANCE_PIPS["DEFAULT"]))
        max_distance_price = max(float(atr_value) * 1.20, pip * max_pips)
        if distance <= max_distance_price * 0.50:
            score_components["Risk_RR_Distance"] += 2
            details["Distance"] = f"+2 proche ({distance:.5f})"
        elif distance <= max_distance_price:
            details["Distance"] = f"0 acceptable ({distance:.5f})"
        elif distance <= max_distance_price * 1.50:
            score_components["Risk_RR_Distance"] -= 2
            details["Distance"] = f"-2 un peu loin ({distance:.5f})"
        else:
            rejection_logs.append(f"Distance trop grande: {distance:.5f}")
            return {"passed": False, "total_score": 0, "final_confidence": "LOW",
                    "details": {"VETO": f"Prix vraiment trop loin ({distance:.5f})"},
                    "stop_loss": stop_loss, "take_profit": take_profit, "atr_value": atr_value,
                    "eqs_score": eqs_score, "eqs_details": eqs_result,
                    "rejection_logs": rejection_logs}
    except Exception as exc:
        details["Distance_Error"] = str(exc)
    
    try:
        ema_score = max(-2, min(2, _directional_score(score_ema_trend(df_h1), direction)))
        structure_score = _directional_score(score_market_structure(df_h1), direction)
        htf_score = score_higher_timeframe_alignment(direction, df_h1, df_h4)
        score_components["Structure_H1"] += ema_score + structure_score
        score_components["HTF_Alignment"] += htf_score
        details["EMA"] = f"{ema_score:+d} (EMA50 H1)"
        details["Structure_H1"] = f"{structure_score:+d} (HH/HL/LH/LL)"
        details["HTF_Alignment"] = f"{htf_score:+d} (alignement H1/H4)"
    except Exception as exc:
        details["Trend_H1_Error"] = str(exc)
    
    # Setup type
    if "LIQUIDITY" in entry_type:
        score_components["ICT"] += 2
        details["Setup_Type"] = "+2 Liquidity"
    elif any(x in entry_type for x in ["FVG", "BISI", "NESTED"]):
        score_components["ICT"] += 3 if "BISI" in entry_type else 2
        details["Setup_Type"] = f"+{3 if 'BISI' in entry_type else 2} ICT"
    elif "BREAKER" in entry_type:
        score_components["ICT"] += 2
        details["Setup_Type"] = "+2 Breaker"
    elif "WICK" in entry_type:
        score_components["ICT"] += 2
        details["Setup_Type"] = "+2 Wick rejection"
    else:
        score_components["ICT"] += 1
        details["Setup_Type"] = f"+1 ({entry_type})"
    
    # RR Ratio
    try:
        dist_sl = abs(entry_level - stop_loss)
        dist_tp = abs(take_profit - entry_level)
        rr_ratio = dist_tp / dist_sl if dist_sl > 0 else 0
        if rr_ratio >= 2.5:
            score_components["Risk_RR_Distance"] += 2
            details["RR"] = f"+2 (excellent {rr_ratio:.2f})"
        elif rr_ratio >= 2.0:
            score_components["Risk_RR_Distance"] += 1
            details["RR"] = f"+1 (correct {rr_ratio:.2f})"
        else:
            details["RR"] = f"0 (faible {rr_ratio:.2f})"
    except Exception:
        pass
    
    # D1 Trend
    try:
        d1_bonus, d1_label = get_d1_trend_bonus(df_d1, direction)
        if d1_bonus > 0:
            score_components["Secondary"] += 2
            details["D1_Trend"] = "+2 (D1 aligné)"
        else:
            details["D1_Trend"] = d1_label
    except Exception:
        pass
    
    # MACD
    try:
        macd_bonus, macd_label = get_macd_h1_bonus(df_h1, direction)
        if macd_bonus > 0:
            score_components["Secondary"] += 1
            details["MACD_H1"] = "+1 (confirme)"
        else:
            details["MACD_H1"] = macd_label
    except Exception:
        pass
    
    # Session
    try:
        session_bonus, session_label = get_session_quality_bonus(pair)
        if session_bonus > 0:
            score_components["Secondary"] += 1
            details["Session"] = "+1 (bonne session)"
        else:
            details["Session"] = session_label
    except Exception:
        pass
    
    # Score final
    score = compute_final_score(score_components)
    passed = score >= min_required
    
    if not passed:
        rejection_logs.append(f"Score = {score} < seuil {min_required}")
    
    final_confidence = "HIGH" if score >= min_required + 3 else "MEDIUM" if passed else "LOW"
    
    confluences = {
        "d1_aligned": details.get("D1_Trend", "").startswith("+"),
        "rsi_divergence": False,
        "session_active": details.get("Session", "").startswith("+"),
        "macd_confirmed": details.get("MACD_H1", "").startswith("+"),
        "bos_confirmed": "BOS" in str(details),
        "structure_ok": score_components.get("Structure", 0) >= 1,
        "pullback_ok": score_components.get("Pullback", 0) >= 2,
    }
    
    win_rate = estimate_win_rate(score, eqs_score, confluences)
    quality_label = get_signal_quality_label(score, eqs_score)
    
    log_score_detail(score_components, score, "PASSED" if passed else "REJECTED")
    
    if rejection_logs and not passed:
        logger.info(f"[REJECT] {pair} {direction}: " + " | ".join(rejection_logs))
    
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
        "momentum_filter": momentum_filter_info,
        "eqs_score": eqs_score,
        "eqs_details": eqs_result,
        "rejection_logs": rejection_logs
    }

# ============================================================
# V89.4 - DÉTECTION BIAS-FIRST (conservée)
# ============================================================
def detect_setups_aligned_with_bias(
    df_m15: pd.DataFrame,
    df_h1: pd.DataFrame,
    bias: str,
    pair: str = "XAU_USD",
    df_h4: pd.DataFrame = None
) -> List[Dict]:
    setups = []
    if bias not in ["BUY", "SELL"]:
        buy_setups = detect_setups_aligned_with_bias(df_m15, df_h1, "BUY", pair, df_h4)
        sell_setups = detect_setups_aligned_with_bias(df_m15, df_h1, "SELL", pair, df_h4)
        return buy_setups + sell_setups
    if DEBUG_MODE:
        logger.debug(f"🔍 Détection {bias} (biais H4) pour {pair}")
    all_fvgs = detect_fvg_advanced(df_m15, max_lookback_hours=36)
    fvgs = [f for f in all_fvgs if f.get("direction", "").upper() == bias]
    all_nested = detect_nested_fvg(df_m15, min_nesting=2)
    nested = [n for n in all_nested if n.get("direction", "").upper() == bias]
    all_wicks = detect_wick_rejection_poi(df_m15, bias)
    wicks = [w for w in all_wicks if w.get("direction", "").upper() == bias]
    bos = detect_bos(df_h1, lookback=50)
    choch = detect_choch(df_h1, lookback=50)
    current_price = float(df_m15["close"].iloc[-1])
    rsi_m15 = get_last_rsi(df_m15["close"])
    rsi_h4 = get_last_rsi(df_h4["close"]) if df_h4 is not None else 50
    for fvg in fvgs:
        entry_level = get_fvg_midpoint(fvg)
        if entry_level is None or abs(current_price - entry_level) > 0.0020:
            continue
        setups.append({
            "type": f"FVG_RETEST_{fvg.get('type', 'UNKNOWN')}",
            "direction": bias, "entry_level": round(entry_level, 5),
            "entry_zone": (round(entry_level - 0.0010, 5), round(entry_level + 0.0010, 5)),
            "confidence": "MEDIUM", "trigger": "FVG_RETEST",
            "rsi_m15": rsi_m15, "rsi_h4": rsi_h4,
            "fvg": fvg, "structure_analysis": {"bos": bos, "choch": choch},
            "bias_aligned": True
        })
    for nfvg in nested:
        entry_level = nfvg.get("midpoint")
        if entry_level is None or abs(current_price - entry_level) > 0.0020:
            continue
        setups.append({
            "type": "NESTED_FVG", "direction": bias,
            "entry_level": round(entry_level, 5),
            "entry_zone": (round(entry_level - 0.0015, 5), round(entry_level + 0.0015, 5)),
            "confidence": "HIGH", "trigger": "NESTED_FVG",
            "rsi_m15": rsi_m15, "rsi_h4": rsi_h4,
            "fvg": nfvg, "structure_analysis": {"bos": bos, "choch": choch},
            "bias_aligned": True
        })
    for wick in wicks:
        entry_level = wick.get("price_level")
        if entry_level is None or abs(current_price - entry_level) > 0.0020:
            continue
        setups.append({
            "type": "WICK_REJECTION", "direction": bias,
            "entry_level": round(entry_level, 5),
            "entry_zone": (round(entry_level - 0.0010, 5), round(entry_level + 0.0010, 5)),
            "confidence": "MEDIUM", "trigger": "WICK_REJECTION",
            "rsi_m15": rsi_m15, "rsi_h4": rsi_h4,
            "structure_analysis": {"bos": bos, "choch": choch},
            "bias_aligned": True
        })
    if bos.get("type") in ["BOS_BUY", "BOS_SELL"]:
        bos_direction = "BUY" if bos["type"] == "BOS_BUY" else "SELL"
        if bos_direction == bias:
            bos_level = bos["level"]
            for fvg in fvgs:
                fvg_level = get_fvg_midpoint(fvg)
                if fvg_level is None or abs(bos_level - fvg_level) > 0.00030:
                    continue
                setups.append({
                    "type": "BISI", "direction": bias,
                    "entry_level": round(fvg_level, 5),
                    "entry_zone": (round(fvg_level - 0.0010, 5), round(fvg_level + 0.0010, 5)),
                    "confidence": "VERY_HIGH", "trigger": "BISI",
                    "rsi_m15": rsi_m15, "rsi_h4": rsi_h4,
                    "bosis": {"level": bos_level, "type": bos["type"]},
                    "structure_analysis": {"bos": bos, "choch": choch},
                    "bias_aligned": True
                })
    if DEBUG_MODE:
        logger.debug(f"🎯 Setups {bias} pour {pair}: {len(setups)} détectés")
    return setups

# ============================================================
# V94 - FONCTION PRINCIPALE AVEC DIAGNOSTIC ATR
# ============================================================
def advanced_main_v94():
    try:
        api = oandapyV20.API(access_token=os.getenv("OANDA_API_KEY"))
        logger.info("✅ API OANDA initialisée avec succès")
        logger.info("✅ ENTRY QUALITY SCORE (EQS) V94 - Seuil: 60/100")
        logger.info(f"✅ Break Even: {BREAKEVEN_TRIGGER_R}R")
        logger.info("✅ AUDIT ATR ACTIVÉ (valeur brute + conversion en pips)")
    except Exception as e:
        logger.error(f"❌ Échec d'initialisation de l'API OANDA : {e}")
        return
    for pair in PAIR_LIST:
        _reset_log_dedup()
        try:
            df_h4 = get_candles_with_retry(api, pair, GRANULARITY_H4, 300)
            df_h1 = get_candles_with_retry(api, pair, GRANULARITY_H1, 200)
            df_m15 = get_candles_with_retry(api, pair, GRANULARITY_M15, 250)
            df_d1 = get_candles_with_retry(api, pair, "D", count=250)
            if any(df.empty for df in [df_h4, df_h1, df_m15]):
                logger.warning(f"⚠️ Données manquantes pour {pair}, analyse ignorée")
                continue
            
            # === DIAGNOSTIC ATR ===
            atr_price = calculate_atr(df_m15, period=ATR_PERIOD)
            atr_pips = price_to_pips(atr_price, pair)
            min_atr_pips = MIN_ATR_PIPS_BY_PAIR.get(pair, MIN_ATR_PIPS_BY_PAIR["DEFAULT"])
            logger.info(f"[ATR_DIAG] {pair} | ATR prix: {atr_price:.6f} | ATR pips: {atr_pips:.1f} | Seuil: {min_atr_pips:.1f} | Écart: {atr_pips - min_atr_pips:.1f}")
            
            current_price = float(df_m15["close"].iloc[-1])
            bias_analysis = determine_advanced_bias(df_h4)
            bias = bias_analysis.get("bias", "NEUTRAL")
            min_score = MIN_CONFIDENCE_SCORE_BY_PAIR.get(pair, MIN_CONFIDENCE_SCORE_BY_PAIR["DEFAULT"])
            
            if DEBUG_MODE:
                adx = calculate_adx(df_h1)
                momentum = calculate_momentum(df_m15)
                logger.debug(f"📊 {pair} | ADX={adx:.1f} | MOM={momentum:.2f}% | ATR_pips={atr_pips:.1f}")
            
            if bias == "NEUTRAL":
                buy_setups = detect_setups_aligned_with_bias(df_m15, df_h1, "BUY", pair, df_h4)
                sell_setups = detect_setups_aligned_with_bias(df_m15, df_h1, "SELL", pair, df_h4)
                setups = buy_setups + sell_setups
            else:
                setups = detect_setups_aligned_with_bias(df_m15, df_h1, bias, pair, df_h4)
            
            if DEBUG_MODE:
                logger.debug(f"📋 {pair}: {len(setups)} setups détectés (biais: {bias})")
            
            scored_entries = []
            rejected_reasons = defaultdict(int)
            rejected_details = []
            
            for entry in setups:
                direction = entry.get("direction", "").upper()
                entry_type = entry.get("type", "UNKNOWN")
                entry_level = entry.get("entry_level")
                if entry_level is None:
                    rejected_reasons["entry_level_none"] += 1
                    continue
                distance = abs(current_price - entry_level)
                max_distance = MAX_DISTANCE_PIPS.get(pair, MAX_DISTANCE_PIPS["DEFAULT"])
                if distance > max_distance * 3:
                    rejected_reasons["distance_too_far"] += 1
                    continue
                confidence_result = calculate_signal_confidence(
                    pair, direction, df_h4, df_h1, df_m15, entry, bias, current_price,
                    False, "", df_d1=df_d1
                )
                score = confidence_result.get("total_score", 0)
                eqs = confidence_result.get("eqs_score", 0)
                
                if DEBUG_MODE:
                    logger.debug(f"📊 {pair} {direction} | Score: {score} | EQS: {eqs}/100 | Passed: {confidence_result.get('passed', False)}")
                
                if confidence_result.get("passed", False):
                    scored_entries.append({"entry": entry, "confidence": confidence_result})
                    stats.record_signal(pair, True, "score_ok", entry_level, 0, 0, score, direction)
                else:
                    reason = confidence_result.get("details", {}).get("VETO", f"score_{score}")
                    rejected_reasons[reason[:30]] += 1
                    rejection_logs = confidence_result.get("rejection_logs", [])
                    if rejection_logs:
                        rejected_details.append(f"{pair} {direction}: " + " | ".join(rejection_logs))
                    stats.record_signal(pair, False, reason, entry_level, 0, 0, score, direction)
            
            if rejected_details:
                logger.info(f"[REJECT_DETAILS] {pair} - {len(rejected_details)} rejets détaillés")
                for detail in rejected_details[:5]:
                    logger.info(f"  {detail}")
                if len(rejected_details) > 5:
                    logger.info(f"  ... et {len(rejected_details)-5} autres")
            
            finalists = strict_keep_best_per_direction(scored_entries)
            log_line = (
                f"{pair:10} | Biais: {bias:6} | Setups: {len(setups):3} | "
                f"Scorés: {len(scored_entries):3} | Finalistes: {len(finalists):3}"
            )
            if rejected_reasons:
                reasons = ", ".join([f"{k}:{v}" for k, v in list(rejected_reasons.items())[:3] if v > 0])
                log_line += f" | Rejets: {reasons}"
            logger.info(log_line)
            
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
                    if DEBUG_MODE:
                        logger.debug(f"❌ {pair} {direction} déjà envoyé")
                    continue
                
                stop_loss, take_profit = calculate_sl_tp(
                    entry_price=entry_level,
                    atr=confidence_result["atr_value"],
                    direction=direction,
                    pair=pair,
                    entry_type=entry_type,
                    breaker_level=None
                )
                
                score = confidence_result.get("total_score", 0)
                eqs = confidence_result.get("eqs_score", 0)
                quality = confidence_result.get("quality_label", "B")
                
                logger.info(f"📊 TRADE {pair} {direction} {entry_type} @{entry_level:.5f} | Score: {score} | EQS: {eqs}/100 | Qualité: {quality}")
                
                entry_metrics = {
                    "atr": confidence_result.get("atr_value", 0),
                    "adx": calculate_adx(df_h1),
                    "rsi": get_last_rsi(df_m15["close"]),
                    "eqs": eqs,
                    "hour": datetime.utcnow().hour,
                    "weekday": datetime.utcnow().weekday(),
                    "setup_type": entry_type
                }
                
                if DEMO_MODE:
                    logger.info(f"🔬 DEMO: {pair} {direction} @ {entry_level:.5f} (SL: {stop_loss}, TP: {take_profit})")
                    stats.record_signal(pair, True, "demo_mode", entry_level, stop_loss, take_profit, score, direction, entry_metrics)
                    nb_envoyes += 1
                    continue
                
                trade_id = execute_oanda_trade_v893(
                    pair=pair,
                    direction=direction,
                    entry_price=entry_level,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    score=score,
                    entry_type=entry_type,
                )
                
                if trade_id:
                    enriched_bias = dict(bias_analysis) if bias_analysis else {}
                    enriched_bias["win_rate"] = confidence_result.get("win_rate", "~55%")
                    enriched_bias["quality_label"] = quality
                    enriched_bias["score_details"] = confidence_result.get("details", {})
                    send_telegram_alert(
                        pair=pair, direction=direction, entry_price=entry_level,
                        stop_loss=stop_loss, take_profit=take_profit,
                        narrative={}, bias_analysis=enriched_bias,
                        rsi=get_last_rsi(df_m15["close"]),
                        entry_type=entry_type, confidence_score=score,
                        eqs_score=eqs
                    )
                    mark_signal_sent(pair, direction, entry_level_key, zone_start, zone_end)
                    stats.record_signal(pair, True, "trade_opened", entry_level, stop_loss, take_profit, score, direction, entry_metrics)
                    nb_envoyes += 1
            
            if nb_envoyes > 0:
                logger.info(f"✅ {pair}: {nb_envoyes} trades envoyés")
                
        except Exception as e:
            logger.error(f"💥 Erreur sur {pair} : {str(e)}")
            logger.error(traceback.format_exc())
            continue
    stats.log_summary()

# ============================================================
# V89.3 - OANDA EXECUTION + API OFFICIELLE (conservée)
# ============================================================
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "101-004-31348578-001")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "true").lower() == "true"

RISK_PERCENTAGE = float(os.getenv("RISK_PERCENTAGE", "1.0"))
MAX_RISK_USD = float(os.getenv("MAX_RISK_USD", "1250"))
MAX_TRADES_TOTAL = int(os.getenv("MAX_TRADES_TOTAL", "9"))
ONE_TRADE_PER_PAIR = os.getenv("ONE_TRADE_PER_PAIR", "true").lower() == "true"

PIP_SIZE_V88 = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "USD_CAD": 0.0001, "AUD_CAD": 0.0001,
    "USD_JPY": 0.01, "AUD_JPY": 0.01, "GBP_JPY": 0.01,
    "XAU_USD": 0.01,
    "NAS100_USD": 0.1, "US30_USD": 1.0, "SPX500_USD": 0.1,
}
PRICE_DECIMALS_V88 = {
    "EUR_USD": 5, "GBP_USD": 5, "AUD_USD": 5,
    "USD_CAD": 5, "AUD_CAD": 5,
    "USD_JPY": 3, "AUD_JPY": 3, "GBP_JPY": 3,
    "XAU_USD": 3,
    "NAS100_USD": 1, "US30_USD": 1, "SPX500_USD": 1,
}

UNIT_STEP_BY_PAIR = {
    "XAU_USD": 1, "EUR_USD": 1000, "GBP_USD": 1000,
    "USD_JPY": 1000, "USD_CAD": 1000, "AUD_USD": 1000,
    "AUD_CAD": 1000, "AUD_JPY": 1000, "GBP_JPY": 1000,
    "NAS100_USD": 1, "US30_USD": 1, "SPX500_USD": 1,
    "DEFAULT": 1000,
}
MIN_UNITS_BY_PAIR = {
    "XAU_USD": 1, "NAS100_USD": 1, "US30_USD": 1,
    "SPX500_USD": 1, "DEFAULT": 1000,
}
MAX_UNITS_BY_PAIR = {
    "XAU_USD": 100, "EUR_USD": 200000, "GBP_USD": 200000,
    "USD_JPY": 200000, "USD_CAD": 200000, "AUD_USD": 200000,
    "AUD_CAD": 200000, "AUD_JPY": 200000, "GBP_JPY": 200000,
    "NAS100_USD": 50, "US30_USD": 20, "SPX500_USD": 50,
    "DEFAULT": 200000,
}

MAX_MARGIN_USAGE_PER_TRADE_PERCENT = float(os.getenv("MAX_MARGIN_USAGE_PER_TRADE_PERCENT", "5.0"))
OANDA_CACHE_TTL_SECONDS_V88 = float(os.getenv("OANDA_CACHE_TTL_SECONDS_V88", "3.0"))
_OANDA_CACHE_V88 = {}

def v88_client():
    token = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_ACCESS_TOKEN")
    return oandapyV20.API(access_token=token, environment=OANDA_ENVIRONMENT)

def _cache_get_v88(key: str, ttl_seconds: float = OANDA_CACHE_TTL_SECONDS_V88):
    item = _OANDA_CACHE_V88.get(key)
    if not item:
        return None
    ts, value = item
    if time.time() - ts > ttl_seconds:
        _OANDA_CACHE_V88.pop(key, None)
        return None
    return value

def _cache_set_v88(key: str, value):
    _OANDA_CACHE_V88[key] = (time.time(), value)
    return value

def clear_scan_cache_v88():
    _OANDA_CACHE_V88.clear()

def round_price_v88(pair: str, price: float) -> str:
    return f"{float(price):.{PRICE_DECIMALS_V88.get(pair, 5)}f}"

def is_market_open_utc_v88(now_dt: datetime) -> bool:
    wd = now_dt.weekday()
    t = now_dt.time()
    if wd == 5:
        return False
    if wd == 6 and t < datetime.strptime("21:00", "%H:%M").time():
        return False
    if wd == 4 and t >= datetime.strptime("21:00", "%H:%M").time():
        return False
    return True

def get_account_summary_v88() -> dict:
    cached = _cache_get_v88("account_summary")
    if cached is not None:
        return cached
    r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
    try:
        api = v88_client()
        resp = api.request(r)
        if not resp:
            return {}
        _cache_set_v88("account_summary", resp)
        return resp
    except Exception as e:
        logger.error(f"❌ AccountSummary error: {e}")
        return {}

def get_balance_v88() -> float:
    resp = get_account_summary_v88()
    try:
        return float(resp.get("account", {}).get("balance", 0))
    except Exception:
        return 0.0

def get_open_trades_v88(log_raw: bool = False) -> list:
    cache_key = "open_trades_raw"
    resp = _cache_get_v88(cache_key)
    if resp is None:
        r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
        try:
            api = v88_client()
            resp = api.request(r)
            if resp:
                _cache_set_v88(cache_key, resp)
        except Exception:
            return []
    if not resp:
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
    return open_trades

def open_trade_count_v88() -> int:
    return len(get_open_trades_v88(log_raw=True))

def has_open_trade_v88(pair: str) -> bool:
    for t in get_open_trades_v88():
        if t.get("instrument") == pair:
            return True
    return False

def quote_currency_v88(pair: str) -> str:
    return pair.split("_")[1]

def get_fx_rate_to_usd_v88(currency: str) -> float:
    if currency == "USD":
        return 1.0
    cached = _cache_get_v88(f"fx_to_usd:{currency}", ttl_seconds=60.0)
    if cached is not None:
        return float(cached)
    direct = f"{currency}_USD"
    inverse = f"USD_{currency}"
    try:
        if direct in PAIR_LIST:
            api = v88_client()
            df = get_candles_with_retry(api, direct, "M5", 10)
            if df is not None and not df.empty:
                val = float(df["close"].iloc[-1])
                _cache_set_v88(f"fx_to_usd:{currency}", val)
                return val
        if inverse in PAIR_LIST:
            api = v88_client()
            df = get_candles_with_retry(api, inverse, "M5", 10)
            if df is not None and not df.empty:
                val = 1.0 / float(df["close"].iloc[-1])
                _cache_set_v88(f"fx_to_usd:{currency}", val)
                return val
    except Exception:
        pass
    return 1.0

def get_oanda_margin_rate_v88(pair: str) -> float:
    cached = _cache_get_v88(f"instrument:{pair}", ttl_seconds=300.0)
    if cached is not None:
        return float(cached.get("marginRate", 0.0333) or 0.0333)
    try:
        api = v88_client()
        r = accounts.AccountInstruments(accountID=OANDA_ACCOUNT_ID, params={"instruments": pair})
        resp = api.request(r)
        instruments_data = resp.get("instruments", [])
        if instruments_data:
            _cache_set_v88(f"instrument:{pair}", instruments_data[0])
            margin_rate = float(instruments_data[0].get("marginRate", 0.0333))
            if margin_rate > 0:
                return margin_rate
    except Exception:
        pass
    return 0.0333

def get_available_margin_v88(account_summary: dict | None = None) -> float:
    account_summary = account_summary or get_account_summary_v88()
    account = account_summary.get("account", {}) if isinstance(account_summary, dict) else {}
    for key in ("marginAvailable", "NAV", "balance"):
        try:
            value = float(account.get(key, 0) or 0)
            if value > 0:
                return value
        except Exception:
            continue
    return 0.0

def calculate_margin_v88(pair: str, units: int, entry_price: float, account_summary: dict | None = None) -> dict:
    margin_required = estimate_margin_used_v88(pair, units, entry_price)
    available = get_available_margin_v88(account_summary)
    return {
        "pair": pair,
        "units": abs(int(units or 0)),
        "entry_price": float(entry_price or 0),
        "margin_required": float(margin_required),
        "margin_available": float(available),
        "sufficient": bool(available <= 0 or margin_required <= available),
    }

def estimate_margin_used_v88(pair: str, units: int, entry_price: float) -> float:
    units = abs(int(units))
    margin_rate = get_oanda_margin_rate_v88(pair)
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
        q_to_usd = get_fx_rate_to_usd_v88(quote)
        notional_usd = units * entry_price * q_to_usd
    return float(notional_usd * margin_rate)

def cap_units_absolute_v88(pair: str, units: int) -> int:
    max_units = MAX_UNITS_BY_PAIR.get(pair, MAX_UNITS_BY_PAIR["DEFAULT"])
    if units > max_units:
        logger.warning(f"ABS CAP {pair}: units {units} -> {max_units}")
        return max_units
    return units

def cap_units_by_margin_v88(pair: str, units: int, entry_price: float, balance: float) -> int:
    if units <= 0 or balance <= 0:
        return 0
    margin_info = calculate_margin_v88(pair, units, entry_price)
    account_available = margin_info["margin_available"]
    max_margin_usd = min(balance * (MAX_MARGIN_USAGE_PER_TRADE_PERCENT / 100.0), account_available) if account_available > 0 else balance * (MAX_MARGIN_USAGE_PER_TRADE_PERCENT / 100.0)
    estimated_margin = margin_info["margin_required"]
    if estimated_margin <= max_margin_usd:
        return units
    ratio = max_margin_usd / estimated_margin if estimated_margin > 0 else 0
    capped = int(units * ratio)
    step = UNIT_STEP_BY_PAIR.get(pair, UNIT_STEP_BY_PAIR["DEFAULT"])
    capped = int(capped // step * step)
    logger.warning(f"MARGIN CAP {pair}: units {units} -> {capped}")
    return max(capped, 0)

def calculate_units_v88(pair: str, entry: float, stop_loss: float, balance: float) -> float:
    try:
        balance = float(balance)
        entry = float(entry)
        stop_loss = float(stop_loss)
    except Exception:
        logger.error(f"paramètres sizing invalides pair={pair}")
        return 0
    risk_usd = min(balance * (RISK_PERCENTAGE / 100.0), MAX_RISK_USD)
    distance_quote = abs(entry - stop_loss)
    if balance <= 0 or risk_usd <= 0 or distance_quote <= 0:
        return 0
    quote = quote_currency_v88(pair)
    quote_to_usd = get_fx_rate_to_usd_v88(quote)
    if quote_to_usd <= 0:
        return 0
    risk_per_unit_usd = distance_quote * quote_to_usd
    if risk_per_unit_usd <= 0:
        return 0
    raw_units = risk_usd / risk_per_unit_usd
    step = UNIT_STEP_BY_PAIR.get(pair, UNIT_STEP_BY_PAIR["DEFAULT"])
    min_units = MIN_UNITS_BY_PAIR.get(pair, MIN_UNITS_BY_PAIR["DEFAULT"])
    units_before_caps = int(raw_units // step * step)
    units = cap_units_absolute_v88(pair, units_before_caps)
    units = cap_units_by_margin_v88(pair, units, entry, balance)
    if units < min_units:
        logger.warning(f"units trop faibles {pair}: {units} < min={min_units}")
        return 0
    return int(units)

def get_recent_m5_price_v88(pair: str) -> float:
    try:
        api = v88_client()
        df = get_candles_with_retry(api, pair, "M5", 10)
        if df is None or df.empty:
            return 0.0
        return float(df["close"].iloc[-1])
    except Exception:
        return 0.0

def get_price_spread_v88(pair: str) -> dict:
    cached = _cache_get_v88(f"pricing:{pair}", ttl_seconds=2.0)
    if cached is not None:
        return cached
    try:
        api = v88_client()
        r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params={"instruments": pair})
        resp = api.request(r)
        prices = resp.get("prices", []) or []
        if prices:
            item = prices[0]
            bid = float(item.get("bids", [{}])[0].get("price", 0) or 0)
            ask = float(item.get("asks", [{}])[0].get("price", 0) or 0)
            mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
            data = {"bid": bid, "ask": ask, "mid": mid, "spread": max(ask - bid, 0.0)}
            _cache_set_v88(f"pricing:{pair}", data)
            return data
    except Exception:
        pass
    fallback_price = get_recent_m5_price_v88(pair)
    return {"bid": fallback_price, "ask": fallback_price, "mid": fallback_price, "spread": 0.0}

def get_atr_m15_v88(pair: str) -> float:
    cached = _cache_get_v88(f"atr_m15:{pair}", ttl_seconds=60.0)
    if cached is not None:
        return float(cached)
    try:
        api = v88_client()
        df = get_candles_with_retry(api, pair, "M15", max(ATR_PERIOD + 10, 40))
        if df is None or df.empty:
            return 0.0
        atr = float(calculate_atr(df, ATR_PERIOD) or 0.0)
        _cache_set_v88(f"atr_m15:{pair}", atr)
        return atr
    except Exception:
        return 0.0

# ============================================================
# V94 - DIAGNOSTIC DE DÉMARRAGE
# ============================================================
def diagnostic_startup_v94():
    logger.info("=" * 60)
    logger.info("[DIAG] DIAGNOSTIC DE DÉMARRAGE V94")
    logger.info("=" * 60)
    logger.info(f"[DIAG] BREAKEVEN_TRIGGER_R = {BREAKEVEN_TRIGGER_R}")
    logger.info(f"[DIAG] EQS_MIN_THRESHOLD = {EQS_MIN_THRESHOLD}")
    logger.info(f"[DIAG] MIN_CONFIDENCE_SCORE_BY_PAIR = {MIN_CONFIDENCE_SCORE_BY_PAIR}")
    logger.info(f"[DIAG] MIN_ATR_PIPS = {MIN_ATR_PIPS_BY_PAIR}")
    logger.info(f"[DIAG] PULLBACK_MIN_PIPS = {PULLBACK_MIN_PIPS_BY_PAIR}")
    
    try:
        from oandapyV20.endpoints import trades
        logger.info("[DIAG] ✅ trades.TradeCRCDO disponible")
    except Exception as e:
        logger.error(f"[DIAG] ❌ trades.TradeCRCDO indisponible: {e}")
    
    try:
        from oandapyV20.endpoints import orders
        logger.info("[DIAG] ✅ orders.OrderCreate disponible")
    except Exception as e:
        logger.error(f"[DIAG] ❌ orders.OrderCreate indisponible: {e}")
    
    try:
        from oandapyV20.endpoints import trades
        logger.info("[DIAG] ✅ trades.TradeDetails disponible")
    except Exception as e:
        logger.error(f"[DIAG] ❌ trades.TradeDetails indisponible: {e}")
    logger.info("=" * 60)

# ============================================================
# V89.3 - CONFIRMATION DU TRADE
# ============================================================
def get_trade_details_v88(trade_id: str) -> dict:
    try:
        api = v88_client()
        r = trades.TradeDetails(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id)
        resp = api.request(r)
        return resp.get("trade", {})
    except Exception as e:
        logger.error(f"[TRADE] Erreur récupération trade {trade_id}: {e}")
        return {}

def has_trailing_stop_v88(trade: dict) -> bool:
    trailing_stop = trade.get("trailingStopLossOrder", {})
    return bool(trailing_stop and trailing_stop.get("id"))

def get_stop_loss_v88(trade: dict) -> float:
    sl_order = trade.get("stopLossOrder", {})
    return float(sl_order.get("price", 0)) if sl_order else 0.0

def modify_trade_sl_v893(trade_id: str, pair: str, new_sl: float) -> bool:
    try:
        api = v88_client()
        logger.info(f"[BE] Modification SL via TradeCRCDO pour trade {trade_id} -> {new_sl:.5f}")
        data = {
            "stopLoss": {
                "price": round_price_v88(pair, new_sl),
                "timeInForce": "GTC"
            }
        }
        r = trades.TradeCRCDO(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id, data=data)
        resp = api.request(r)
        tracer.log_step(trade_id, "BE_MODIFY_SL_VIA_TRADE_CRCDO", {"pair": pair, "new_sl": new_sl}, resp)
        if resp.get("orderRejectTransaction"):
            reject = resp.get("orderRejectTransaction")
            logger.error(f"[BE] Rejeté pour trade {trade_id}: {reject}")
            return False
        logger.info(f"[BE] SUCCESS: SL modifié pour trade {trade_id} -> {new_sl:.5f}")
        logger.info(f"[CONFIRM] Vérification du SL pour trade {trade_id}")
        time.sleep(1)
        _OANDA_CACHE_V88.pop("open_trades_raw", None)
        trade_details = get_trade_details_v88(trade_id)
        if not trade_details:
            logger.warning(f"[CONFIRM] Impossible de récupérer le trade {trade_id}")
            return False
        actual_sl = get_stop_loss_v88(trade_details)
        if abs(actual_sl - new_sl) > 0.000001:
            logger.warning(f"[CONFIRM] SL non confirmé: attendu {new_sl:.5f}, reçu {actual_sl:.5f}")
            tracer.log_step(trade_id, "BE_CONFIRM_FAIL", {"expected": new_sl, "received": actual_sl}, trade_details)
            return False
        logger.info(f"[CONFIRM] ✅ SL confirmé: {actual_sl:.5f}")
        tracer.log_step(trade_id, "BE_CONFIRM_OK", {"sl": actual_sl}, trade_details)
        return True
    except Exception as e:
        logger.error(f"[BE] Erreur modification SL trade {trade_id}: {e}")
        return False

def create_oanda_trailing_stop_v893(trade_id: str, pair: str, distance: float) -> bool:
    try:
        api = v88_client()
        logger.info(f"[TSL] Création trailing via OrderCreate pour trade {trade_id} -> distance={distance:.5f}")
        order_data = {
            "order": {
                "type": "TRAILING_STOP_LOSS",
                "tradeID": trade_id,
                "distance": str(distance),
                "timeInForce": "GTC"
            }
        }
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        resp = api.request(r)
        tracer.log_step(trade_id, "TSL_CREATE_VIA_ORDER_CREATE", {"pair": pair, "distance": distance}, resp)
        if resp.get("orderRejectTransaction"):
            reject = resp.get("orderRejectTransaction")
            logger.error(f"[TSL] Rejeté pour trade {trade_id}: {reject}")
            return False
        logger.info(f"[TSL] SUCCESS: Trailing stop créé pour trade {trade_id}, distance={distance:.5f}")
        logger.info(f"[CONFIRM] Vérification du trailing stop pour trade {trade_id}")
        time.sleep(1)
        _OANDA_CACHE_V88.pop("open_trades_raw", None)
        trade_details = get_trade_details_v88(trade_id)
        if not trade_details:
            logger.warning(f"[CONFIRM] Impossible de récupérer le trade {trade_id}")
            return False
        if not has_trailing_stop_v88(trade_details):
            logger.warning(f"[CONFIRM] Trailing stop non présent sur le trade {trade_id}")
            tracer.log_step(trade_id, "TSL_CONFIRM_FAIL", {"trade": trade_details}, None)
            return False
        trailing_id = trade_details.get("trailingStopLossOrder", {}).get("id", "unknown")
        logger.info(f"[CONFIRM] ✅ Trailing stop confirmé: ID={trailing_id}")
        tracer.log_step(trade_id, "TSL_CONFIRM_OK", {"trailing_id": trailing_id}, trade_details)
        return True
    except Exception as e:
        logger.error(f"[TSL] Erreur création trailing stop trade {trade_id}: {e}")
        return False

def extract_trade_id_v89(response: dict) -> str | None:
    if not response:
        return None
    oft = response.get("orderFillTransaction")
    if oft:
        if "tradeOpened" in oft and oft["tradeOpened"]:
            trade_id = oft["tradeOpened"].get("tradeID")
            if trade_id:
                return str(trade_id)
        if "tradeReduced" in oft and oft["tradeReduced"]:
            trade_id = oft["tradeReduced"].get("tradeID")
            if trade_id:
                return str(trade_id)
        if "tradesOpened" in oft and oft["tradesOpened"]:
            opened = oft["tradesOpened"]
            if opened and opened[0].get("tradeID"):
                return str(opened[0]["tradeID"])
    oct = response.get("orderCreateTransaction")
    if oct and "relatedTransactionIDs" in oct:
        related = oct.get("relatedTransactionIDs", [])
        if related:
            return str(related[-1])
    if response.get("tradeID"):
        return str(response["tradeID"])
    return None

def find_trade_by_instrument_v89(pair: str, entry_price: float, direction: str) -> str | None:
    pip_value = get_pip_value_for_pair(pair)
    atr = get_atr_m15_v88(pair)
    tolerance = max(5.0 * pip_value, 0.5 * atr, 0.0001)
    tolerance = round(tolerance, 6)
    logger.debug(f"[FALLBACK] Tolérance pour {pair}: {tolerance:.6f}")
    open_trades = get_open_trades_v88(log_raw=True)
    for t in open_trades:
        if t.get("instrument") != pair:
            continue
        t_dir = "BUY" if float(t.get("currentUnits", 0)) > 0 else "SELL"
        if t_dir != direction:
            continue
        t_entry = float(t.get("price", 0))
        if abs(t_entry - entry_price) <= tolerance:
            return str(t.get("id"))
    return None

def check_breakeven_v94():
    try:
        open_trades = get_open_trades_v88()
        logger.info(f"[BE] Scan de {len(open_trades)} trades ouverts (seuil: {BREAKEVEN_TRIGGER_R}R)")
        for t in open_trades:
            trade_id = str(t.get("id"))
            pair = t.get("instrument")
            direction = "BUY" if float(t.get("currentUnits", 0)) > 0 else "SELL"
            entry = float(t.get("price"))
            sl_order = t.get("stopLossOrder", {}) or {}
            current_sl = float(sl_order.get("price", 0))
            if current_sl <= 0:
                logger.debug(f"[BE] Trade {trade_id} sans SL, ignoré")
                continue
            pip = PIP_SIZE_V88.get(pair, get_pip_value_for_pair(pair))
            spread_data = get_price_spread_v88(pair)
            spread = spread_data.get("spread", 0)
            offset = max(spread, pip * 1.0)
            if direction == "BUY":
                be_price = entry + offset
                already_be = (current_sl >= be_price - 0.0001)
            else:
                be_price = entry - offset
                already_be = (current_sl <= be_price + 0.0001)
            if already_be:
                logger.debug(f"[BE] Trade {trade_id} SL déjà au BE ({current_sl:.5f}), saut")
                continue
            current_price = get_recent_m5_price_v88(pair)
            if current_price <= 0:
                logger.debug(f"[BE] Trade {trade_id} prix indisponible")
                continue
            if direction == "BUY":
                profit = current_price - entry
                risk = entry - current_sl
            else:
                profit = entry - current_price
                risk = current_sl - entry
            if risk <= 0:
                logger.debug(f"[BE] Trade {trade_id} risque invalide")
                continue
            r = profit / risk
            logger.info(f"[BE] Trade {trade_id} {pair} {direction} | R={r:.2f}")
            if r >= BREAKEVEN_TRIGGER_R:
                logger.info(f"[BE] 🎯 Condition R>={BREAKEVEN_TRIGGER_R} atteinte pour {trade_id}")
                if direction == "BUY":
                    be_sl = entry + offset
                else:
                    be_sl = entry - offset
                if (direction == "BUY" and be_sl > current_sl) or (direction == "SELL" and be_sl < current_sl):
                    logger.info(f"[BE] {pair} id={trade_id} R={r:.2f} => SL {current_sl:.5f} -> {be_sl:.5f}")
                    if modify_trade_sl_v893(trade_id, pair, be_sl):
                        logger.info(f"[BE] ✅ SL modifié avec succès pour {trade_id}")
                        time.sleep(1)
                        _OANDA_CACHE_V88.pop("open_trades_raw", None)
                        trade_details = get_trade_details_v88(trade_id)
                        if has_trailing_stop_v88(trade_details):
                            logger.info(f"[TSL] Trade {trade_id} a déjà un trailing, on saute")
                            continue
                        atr = get_atr_m15_v88(pair)
                        spread = spread_data.get("spread", 0)
                        pip_value = get_pip_value_for_pair(pair)
                        distance = max(atr * 1.6, spread * 2)
                        distance = min(distance, atr * 3)
                        distance = max(distance, pip_value * TRAILING_STOP_MIN_DISTANCE_PIPS)
                        distance = round(distance, PRICE_DECIMALS_V88.get(pair, 5))
                        if distance > 0:
                            logger.info(f"[TSL] Création du trailing stop pour trade {trade_id}")
                            if create_oanda_trailing_stop_v893(trade_id, pair, distance):
                                logger.info(f"[TSL] ✅ Trailing stop créé")
                            else:
                                logger.error(f"[TSL] ❌ ÉCHEC création trailing")
                        else:
                            logger.warning(f"[TSL] Distance invalide ({distance})")
                    else:
                        logger.error(f"[BE] ❌ ÉCHEC modification SL")
    except Exception as e:
        logger.error(f"Erreur check_breakeven_v94: {e}")
        logger.error(traceback.format_exc())

def execute_oanda_trade_v893(pair: str, direction: str, entry_price: float, stop_loss: float,
                             take_profit: float, score: int, entry_type: str) -> str | None:
    logger.info(f"[ORDER] V94 EXECUTION START {pair} {direction} type={entry_type} score={score}")
    if ONE_TRADE_PER_PAIR and has_open_trade_v88(pair):
        logger.info(f"{pair}: trade déjà ouvert")
        return None
    if open_trade_count_v88() >= MAX_TRADES_TOTAL:
        logger.info(f"Limite trades ouverts atteinte")
        return None
    balance = get_balance_v88()
    if balance <= 0:
        logger.error("Balance invalide")
        return None
    units = calculate_units_v88(pair, entry_price, stop_loss, balance)
    if not units or float(units) <= 0:
        logger.error(f"units invalides pour {pair}: {units}")
        return None
    margin_info = calculate_margin_v88(pair, units, entry_price)
    if not margin_info["sufficient"]:
        units = cap_units_by_margin_v88(pair, units, entry_price, balance)
        if not units or units <= 0:
            logger.error(f"[RISK] {pair} order blocked: insufficient margin")
            return None
    signed_units = units if direction == "BUY" else -units
    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": pair,
            "units": str(signed_units),
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": round_price_v88(pair, stop_loss), "timeInForce": "GTC"},
            "takeProfitOnFill": {"price": round_price_v88(pair, take_profit), "timeInForce": "GTC"}
        }
    }
    risk = abs(entry_price - stop_loss)
    rr = abs(take_profit - entry_price) / risk if risk > 0 else 0
    logger.info(f"[ORDER] SIGNAL V94 {pair} {direction} | RR={rr:.2f} score={score} units={units}")
    if not EXECUTE_TRADES:
        logger.info("[ORDER] EXECUTE_TRADES=false : ordre simulé")
        return "SIMULATION"
    try:
        api = v88_client()
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        resp = api.request(r)
        tracer.log_step("new", "ORDER_CREATE", {
            "pair": pair,
            "direction": direction,
            "entry": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "units": units
        }, resp)
        if resp.get("orderRejectTransaction"):
            reject = resp.get("orderRejectTransaction")
            logger.error(f"[ORDER] ORDRE REJETÉ {pair}: {reject}")
            tracer.log_step("new", "ORDER_REJECTED", {"reason": reject.get("rejectReason", "unknown")}, resp)
            return None
        trade_id = extract_trade_id_v89(resp)
        if not trade_id:
            logger.warning(f"[ORDER] tradeID non trouvé, recherche...")
            time.sleep(2)
            trade_id = find_trade_by_instrument_v89(pair, entry_price, direction)
            if trade_id:
                logger.info(f"[ORDER] tradeID retrouvé: {trade_id}")
            else:
                logger.error(f"[ORDER] ORDRE NON CONFIRMÉ {pair}")
                tracer.log_step("new", "ORDER_NO_TRADE_ID", {"response": str(resp)[:500]}, resp)
                return None
        logger.info(f"[ORDER] ✅ ORDRE CONFIRMÉ {pair} | ID={trade_id}")
        tracer.log_step(trade_id, "ORDER_CONFIRMED", {"pair": pair, "direction": direction}, resp)
        logger.info(f"[CONFIRM] Trade ouvert sans trailing")
        return str(trade_id)
    except Exception as exc:
        logger.exception(f"[ORDER] Erreur ordre OANDA {pair}: {exc}")
        tracer.log_step("new", "ORDER_EXCEPTION", {"error": str(exc)}, None)
        return None

# ============================================================
# STRICT FILTERS
# ============================================================
STRICT_ALLOWED_ENTRY_TYPES = {
    "FVG_RETEST_PERFECT", "FVG_RETEST", "BISI", "BREAKER",
    "NESTED_FVG", "WICK_REJECTION", "LIQUIDITY_DRAW",
}
STRICT_MAX_DISTANCE_PIPS = {
    "XAU_USD": 35.0, "USD_JPY": 18.0, "GBP_JPY": 22.0,
    "EUR_USD": 15.0, "GBP_USD": 18.0, "AUD_USD": 15.0,
    "USD_CAD": 15.0, "AUD_CAD": 15.0, "AUD_JPY": 18.0,
    "NAS100_USD": 50.0, "DEFAULT": 15.0,
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
        k_h1, _ = calculate_stoch_rsi(df_h1["close"])
        k_m15, _ = calculate_stoch_rsi(df_m15["close"])
        k_h1 = float(k_h1)
        k_m15 = float(k_m15)
        if direction == "BUY":
            if k_h1 >= 80:
                return True, f"StochRSI H1 surachat {k_h1:.1f}"
            if k_m15 >= 85:
                return True, f"StochRSI M15 surachat {k_m15:.1f}"
            return True, f"StochRSI OK H1={k_h1:.1f} M15={k_m15:.1f}"
        else:
            if k_h1 <= 20:
                return True, f"StochRSI H1 survendu {k_h1:.1f}"
            if k_m15 <= 15:
                return True, f"StochRSI M15 survendu {k_m15:.1f}"
            return True, f"StochRSI OK H1={k_h1:.1f} M15={k_m15:.1f}"
    except Exception:
        return True, "StochRSI indisponible"

def strict_trend_veto(direction: str, current_price: float, df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> tuple:
    try:
        ema50_h1 = df_h1["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        return True, f"EMA50 H1 scorée sans veto"
    except Exception:
        return True, "EMA50 H1 indisponible"

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
        "FVG_RETEST_PERFECT": 18.0, "FVG_RETEST": 20.0,
        "NESTED_FVG": 20.0, "WICK_REJECTION": 18.0,
        "BISI": 20.0, "BREAKER": 18.0,
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
        return True, f"distance acceptable={distance:.5f}"
    return False, f"trop loin distance={distance:.5f}"

def strict_keep_best_per_direction(scored_entries: list) -> list:
    best = {}
    for item in scored_entries:
        entry = item["entry"]
        direction = entry.get("direction", "").upper()
        score = item["confidence"].get("total_score", -999)
        eqs = item["confidence"].get("eqs_score", 0)
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
        key_score = (score, eqs, priority)
        if direction not in best or key_score > best[direction]["key_score"]:
            item["key_score"] = key_score
            best[direction] = item
    return sorted(best.values(), key=lambda x: x["key_score"], reverse=True)

def strict_direction_permission_v77(direction: str, bias: str, current_price: float, df_h1: pd.DataFrame, df_m15: pd.DataFrame, entry_type: str) -> tuple:
    try:
        direction = (direction or "").upper()
        bias = (bias or "NEUTRAL").upper()
        entry_type = (entry_type or "").upper()
        k_h1, _ = calculate_stoch_rsi(df_h1["close"])
        k_m15, _ = calculate_stoch_rsi(df_m15["close"])
        k_h1 = float(k_h1)
        k_m15 = float(k_m15)
        if bias not in {"BUY", "SELL"}:
            return True, f"Biais neutre: direction {direction} autorisée"
        if direction == bias:
            return True, f"Direction alignée H4 {bias}"
        allowed_counter_types = {"BREAKER", "BISI", "FVG_RETEST", "FVG_RETEST_PERFECT", "NESTED_FVG", "WICK_REJECTION"}
        is_allowed_counter_type = entry_type in allowed_counter_types or entry_type.startswith("FVG_RETEST")
        if direction == "SELL" and bias == "BUY":
            if k_h1 >= 75 and k_m15 <= 70 and is_allowed_counter_type:
                return True, f"SELL contre H4 BUY autorisé"
            return False, f"SELL contre H4 BUY refusé"
        if direction == "BUY" and bias == "SELL":
            if k_h1 <= 25 and k_m15 >= 30 and is_allowed_counter_type:
                return True, f"BUY contre H4 SELL autorisé"
            return False, f"BUY contre H4 SELL refusé"
        return False, f"Direction {direction} non autorisée contre biais {bias}"
    except Exception:
        return False, "permission direction indisponible"

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
    return list(seen.values())

# ============================================================
# V94 - BOUCLE PRINCIPALE
# ============================================================
if __name__ == "__main__":
    logger.info("🚀 Démarrage du Bot Advanced Orderflow Trading - V94 (Audit ATR)")
    logger.info("📋 Trace des trades activée dans trade_trace.json")
    logger.info("✅ Utilisation de TradeCRCDO pour la modification du SL")
    logger.info("✅ Utilisation de OrderCreate pour la création du Trailing Stop")
    logger.info(f"✅ Seuil Break Even: {BREAKEVEN_TRIGGER_R}R (0.6R)")
    logger.info(f"✅ Seuil EQS minimum: {EQS_MIN_THRESHOLD}/100 (60)")
    logger.info("🔄 DOUBLE BOUCLE : rapide (30s) pour BE/Trailing, lente (15min) pour les signaux")
    logger.info("")
    logger.info("🛡️ FILTRES V94 :")
    logger.info("   - Volatilité minimum ATR (seuils conservés, audit actif)")
    logger.info("   - Structure de marché (partielle acceptée)")
    logger.info("   - Pullback obligatoire (EUR=2, AUD=2, GBP=3)")
    logger.info("   - Confirmation par clôture M15")
    logger.info("   - Entry Quality Score (EQS) à 60/100")
    logger.info("   - Scores minimum ajustés (EUR=10, GBP=9, AUD=8, XAU=9)")
    logger.info("   - AUDIT ATR : affiche valeur brute, pips, seuil, écart")
    logger.info("")
    
    diagnostic_startup_v94()
    
    if DEMO_MODE:
        logger.info("🔬 MODE DEMO ACTIVÉ")
    if DEBUG_MODE:
        logger.info("🔍 MODE DEBUG ACTIVÉ")
    
    last_signal_scan = time.time()
    SIGNAL_SCAN_INTERVAL = 900
    
    while True:
        try:
            now = time.time()
            
            clear_scan_cache_v88()
            current_open_count = open_trade_count_v88()
            logger.info(f"[SCAN] Trades ouverts: {current_open_count}/{MAX_TRADES_TOTAL}")
            
            check_breakeven_v94()
            
            if now - last_signal_scan >= SIGNAL_SCAN_INTERVAL:
                logger.info(f"⏰ Scan des signaux V94")
                last_signal_scan = now
                
                now_dt = datetime.utcnow()
                if not is_market_open_utc_v88(now_dt):
                    logger.info("Marché fermé.")
                elif current_open_count >= MAX_TRADES_TOTAL:
                    logger.info(f"Limite trades atteinte")
                else:
                    advanced_main_v94()
            
            time.sleep(30)

        except KeyboardInterrupt:
            logger.info("🛑 Arrêt demandé")
            break
        except Exception as e:
            logger.error(f"💥 Erreur critique: {e}")
            traceback.print_exc()
            time.sleep(30)
