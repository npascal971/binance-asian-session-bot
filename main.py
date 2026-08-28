# ============================================================
# main.py - Version V111 avec stratégie simplifiée
# Refactorisation complète - Août 2026
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
from oandapyV20.endpoints import instruments, pricing, orders, accounts, trades, positions
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

BASE_ADX_MIN_THRESHOLD = float(os.getenv("ADX_MIN_THRESHOLD", "23.0"))
BASE_EQS_MIN_THRESHOLD = float(os.getenv("EQS_MIN_THRESHOLD", "55.0"))

# =========================
# CONFIGURATION GÉNÉRALE
# =========================
PAIR_LIST = ["GBP_USD", "USD_CAD", "AUD_USD", "XAU_USD", "EUR_USD", "USD_JPY", "AUD_JPY"]
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

BASE_MIN_CONFIDENCE_SCORE_BY_PAIR = {
    "EUR_USD": 10, "GBP_USD": 9, "USD_CAD": 8, "AUD_USD": 8,
    "AUD_CAD": 8, "XAU_USD": 9, "DEFAULT": 8
}
PULLBACK_MIN_PIPS_BY_PAIR = {
    "EUR_USD": 4.0, "GBP_USD": 4.0, "USD_CAD": 3.5, "AUD_USD": 4.5,
    "AUD_CAD": 4.0, "XAU_USD": 30.0, "USD_JPY": 5.0, "GBP_JPY": 5.0,
    "DEFAULT": 4.0
}
MIN_ATR_PIPS_BY_PAIR = {
    "EUR_USD": 3.5, "GBP_USD": 4.4, "USD_CAD": 3.9, "AUD_USD": 3.0,
    "AUD_CAD": 3.9, "XAU_USD": 34.0, "GBP_JPY": 7.0, "USD_JPY": 5.0,
    "DEFAULT": 4.0
}
MIN_ATR_PIPS_BY_PAIR_ASIA = {
    "EUR_USD": 2.5, "AUD_JPY": 3.5, "GBP_USD": 2.5, "USD_CAD": 3.0,
    "AUD_USD": 3.0, "AUD_CAD": 3.0, "XAU_USD": 24.0, "GBP_JPY": 5.0,
    "USD_JPY": 4.5, "DEFAULT": 3.0
}

MAX_DISTANCE_PIPS = {
    "XAU_USD": 500, "USD_JPY": 150, "NAS100_USD": 25.0,
    "AUD_USD": 0.0080, "EUR_USD": 0.0080, "GBP_USD": 0.0080,
    "USD_CAD": 0.0010, "GBP_JPY": 150, "DEFAULT": 0.0010
}

PAIR_SETTINGS = {
    "XAU_USD": {"atr_multiplier_sl": 1.8, "atr_multiplier_tp": 3.5, "max_volatility_ratio": 0.010, "risk_multiplier": 0.5, "required_confluence": "STRICT"},
    "NAS100_USD": {"atr_multiplier_sl": 1.6, "atr_multiplier_tp": 3.2, "max_volatility_ratio": 0.015, "risk_multiplier": 0.7, "required_confluence": "STRICT"},
    "GBP_JPY": {"atr_multiplier_sl": 1.8, "atr_multiplier_tp": 3.5, "max_volatility_ratio": 0.012, "risk_multiplier": 0.7, "required_confluence": "STRICT"},
    "DEFAULT": {"atr_multiplier_sl": 1.2, "atr_multiplier_tp": 3.0, "max_volatility_ratio": 0.02, "risk_multiplier": 1.0}
}

SIGNAL_RISK_SETTINGS = {
    "NESTED_FVG": {"sl_multiplier": 0.6, "tp_multiplier": 1.8},
    "FVG_RETEST": {"sl_multiplier": 0.8, "tp_multiplier": 2.0},
    "WICK_REJECTION": {"sl_multiplier": 0.9, "tp_multiplier": 2.7},
    "LIQUIDITY_DRAW": {"sl_multiplier": 1.0, "tp_multiplier": 2.5},
    "FVG_RETEST_PERFECT": {"sl_multiplier": 0.7, "tp_multiplier": 2.2},
    "BISI": {"sl_multiplier": 0.7, "tp_multiplier": 2.2},
    "BREAKER": {"sl_multiplier": 0.9, "tp_multiplier": 2.5},
}

BASE_BREAKEVEN_TRIGGER_R = float(os.getenv("BREAKEVEN_TRIGGER_R", "0.55"))
BASE_BREAKEVEN_EARLY_R = float(os.getenv("BREAKEVEN_EARLY_R", "0.25"))
BASE_TRAILING_STOP_DISTANCE_ATR_MULTIPLIER = float(os.getenv("TRAILING_STOP_DISTANCE_ATR_MULTIPLIER", "1.5"))
BASE_TRAILING_STOP_MIN_DISTANCE_PIPS = float(os.getenv("TRAILING_STOP_MIN_DISTANCE_PIPS", "8.0"))
BASE_TRAILING_ACTIVATION_R = float(os.getenv("TRAILING_ACTIVATION_R", "0.80"))

MAX_TRADES_TOTAL = int(os.getenv("MAX_TRADES_TOTAL", "10"))
ONE_TRADE_PER_PAIR = os.getenv("ONE_TRADE_PER_PAIR", "true").lower() == "true"
RISK_PERCENTAGE = float(os.getenv("RISK_PERCENTAGE", "0.75"))
MAX_RISK_USD = float(os.getenv("MAX_RISK_USD", "1250"))
MAX_MARGIN_USAGE_PER_TRADE_PERCENT = float(os.getenv("MAX_MARGIN_USAGE_PER_TRADE_PERCENT", "5.0"))

OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "101-004-31348578-001")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "true").lower() == "true"

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
EXECUTION_COOLDOWN_SECONDS = 60

# ============================================================
# LOGGING
# ============================================================
logger = logging.getLogger("Advanced-Orderflow-Trading-Bot")

def setup_logging():
    _log_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _file_handler = logging.FileHandler("advanced_orderflow_trading.log", encoding="utf-8")
    _file_handler.setFormatter(_log_formatter)
    _stream_handler = logging.StreamHandler(sys.stdout)
    _stream_handler.setFormatter(_log_formatter)
    logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO, handlers=[_file_handler, _stream_handler], force=True)
    for noisy in ("urllib3", "requests", "oandapyV20", "oandapy"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
        logging.getLogger(noisy).propagate = False

setup_logging()

# ============================================================
# CACHE OANDA
# ============================================================
_OANDA_CACHE_V88 = {}
OANDA_CACHE_TTL_SECONDS_V88 = 3.0

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

# ============================================================
# MAINTENANCE OANDA
# ============================================================
MAINTENANCE_DETECTED = False
MAINTENANCE_SUSPEND_TIME = 0
MAINTENANCE_RETRY_INTERVAL = 120
MAINTENANCE_ERROR_COUNT = 0
MAINTENANCE_MAX_ERRORS_BEFORE_SUSPEND = 3

def is_oanda_in_maintenance(error: Exception) -> bool:
    error_str = str(error).lower()
    patterns = ["system under maintenance", "maintenance", "temporarily unavailable", "service unavailable", "maintenance mode", "api is currently unavailable"]
    return any(p in error_str for p in patterns)

def handle_api_error(error: Exception) -> tuple:
    global MAINTENANCE_DETECTED, MAINTENANCE_SUSPEND_TIME, MAINTENANCE_ERROR_COUNT
    if is_oanda_in_maintenance(error):
        MAINTENANCE_ERROR_COUNT += 1
        if MAINTENANCE_ERROR_COUNT >= MAINTENANCE_MAX_ERRORS_BEFORE_SUSPEND:
            MAINTENANCE_DETECTED = True
            MAINTENANCE_SUSPEND_TIME = time.time() + MAINTENANCE_RETRY_INTERVAL
            logger.warning(f"🔧 OANDA en maintenance détecté ({MAINTENANCE_ERROR_COUNT} erreurs) - suspension {MAINTENANCE_RETRY_INTERVAL}s")
            return True, MAINTENANCE_RETRY_INTERVAL, "WARNING"
        else:
            logger.warning(f"⚠️ OANDA semble en maintenance (tentative {MAINTENANCE_ERROR_COUNT})")
            return False, 5, "WARNING"
    return False, 0, "ERROR"

def reset_maintenance_state():
    global MAINTENANCE_DETECTED, MAINTENANCE_SUSPEND_TIME, MAINTENANCE_ERROR_COUNT
    MAINTENANCE_DETECTED = False
    MAINTENANCE_SUSPEND_TIME = 0
    MAINTENANCE_ERROR_COUNT = 0

def is_maintenance_suspended() -> bool:
    global MAINTENANCE_DETECTED, MAINTENANCE_SUSPEND_TIME, MAINTENANCE_ERROR_COUNT
    if not MAINTENANCE_DETECTED:
        return False
    if time.time() < MAINTENANCE_SUSPEND_TIME:
        return True
    MAINTENANCE_DETECTED = False
    MAINTENANCE_SUSPEND_TIME = 0
    MAINTENANCE_ERROR_COUNT = 0
    logger.info("🔧 Fin de la suspension OANDA - reprise")
    return False

# ============================================================
# CLASSES PRINCIPALES
# ============================================================

class AdaptiveState:
    """État adaptatif du bot : paramètres dynamiques, qualité du marché, apprentissage"""
    def __init__(self):
        self.pair_params = {}
        self.setup_weights = {}
        self.suspended_pairs = {}
        self.consecutive_losses = defaultdict(int)
        self.last_adaptation = time.time()
        self.adaptation_history = []
        self.adaptation_counters = defaultdict(lambda: {"good": 0, "bad": 0})
        self.last_loss_time = defaultdict(float)
        self.loss_cooldown = 3600  # 1 heure
        self.setup_performance = defaultdict(lambda: {"wins": 0, "losses": 0, "total_r": 0.0})

    def get_pair_params(self, pair: str) -> dict:
        if pair not in self.pair_params:
            self.pair_params[pair] = {
                "adx_min": BASE_ADX_MIN_THRESHOLD,
                "eqs_min": BASE_EQS_MIN_THRESHOLD,
                "be_trigger_r": BASE_BREAKEVEN_TRIGGER_R,
                "be_early_r": BASE_BREAKEVEN_EARLY_R,
                "trailing_atr_mult": BASE_TRAILING_STOP_DISTANCE_ATR_MULTIPLIER,
                "trailing_min_pips": BASE_TRAILING_STOP_MIN_DISTANCE_PIPS,
                "trailing_activation_r": BASE_TRAILING_ACTIVATION_R,
                "confidence_min": BASE_MIN_CONFIDENCE_SCORE_BY_PAIR.get(pair, BASE_MIN_CONFIDENCE_SCORE_BY_PAIR["DEFAULT"])
            }
        return self.pair_params[pair]

    def get_setup_weight(self, pair: str, setup_type: str) -> float:
        key = f"{pair}_{setup_type}"
        if key not in self.setup_weights:
            return SETUP_WEIGHTS_DEFAULT.get(setup_type, 1.0)
        return self.setup_weights[key]

    def update_setup_weight(self, pair: str, setup_type: str, new_weight: float):
        key = f"{pair}_{setup_type}"
        self.setup_weights[key] = max(0.2, min(2.0, new_weight))
        logger.info(f"[SETUP_WEIGHT] {pair} | {setup_type} | Nouveau poids: {new_weight:.2f}")

    def is_pair_suspended(self, pair: str) -> bool:
        if pair not in self.suspended_pairs:
            return False
        suspend_time = self.suspended_pairs[pair]
        if time.time() - suspend_time > 3600:
            del self.suspended_pairs[pair]
            return False
        return True

    def suspend_pair(self, pair: str, reason: str):
        self.suspended_pairs[pair] = time.time()
        logger.warning(f"[SUSPEND] {pair} suspendu pour 1 heure | Raison: {reason}")

    def can_trade(self, pair: str) -> bool:
        if pair in self.last_loss_time:
            elapsed = time.time() - self.last_loss_time[pair]
            if elapsed < self.loss_cooldown:
                logger.debug(f"[COOLDOWN] {pair} en cooldown pour encore {int(self.loss_cooldown - elapsed)}s")
                return False
        return True

    def record_loss(self, pair: str):
        self.consecutive_losses[pair] += 1
        self.last_loss_time[pair] = time.time()
        # On pourrait ajouter une logique de suspension ici, mais simplifiée
        if self.consecutive_losses[pair] >= 4:
            self.suspend_pair(pair, f"{self.consecutive_losses[pair]} pertes consécutives")
            self.consecutive_losses[pair] = 0

    def record_win(self, pair: str):
        self.consecutive_losses[pair] = 0
        if pair in self.last_loss_time:
            del self.last_loss_time[pair]

    def adapt_parameters(self, pair: str, stats: dict):
        # Version simplifiée, conservée pour compatibilité
        pass

# ============================================================
# TRADE TRACKER (MFE/MAE)
# ============================================================
class TradeTracker:
    def __init__(self):
        self.trades = {}

    def add_trade(self, trade_id: str, pair: str, direction: str, entry_price: float, sl: float, tp: float, setup_type: str, eqs: int):
        self.trades[trade_id] = {
            "pair": pair,
            "direction": direction,
            "entry": entry_price,
            "sl": sl,
            "tp": tp,
            "setup_type": setup_type,
            "eqs": eqs,
            "highest_price": entry_price,
            "lowest_price": entry_price,
            "mfe": 0.0,
            "mae": 0.0,
            "max_favorable_pips": 0.0,
            "max_adverse_pips": 0.0,
            "last_update": time.time(),
            "closed": False,
            "exit_price": None,
            "exit_r": None,
            "entry_time": datetime.utcnow().isoformat(),
            "exit_time": None
        }

    def update_price(self, trade_id: str, current_price: float):
        if trade_id not in self.trades or self.trades[trade_id]["closed"]:
            return
        trade = self.trades[trade_id]
        direction = trade["direction"]
        entry = trade["entry"]
        pip_value = get_pip_value_for_pair(trade["pair"])
        if direction == "BUY":
            price_move = current_price - entry
            if current_price > trade["highest_price"]:
                trade["highest_price"] = current_price
            if current_price < trade["lowest_price"]:
                trade["lowest_price"] = current_price
        else:
            price_move = entry - current_price
            if current_price < trade["lowest_price"]:
                trade["lowest_price"] = current_price
            if current_price > trade["highest_price"]:
                trade["highest_price"] = current_price
        mfe_pips = max(price_move, trade["mfe"]) / pip_value if pip_value > 0 else 0
        mae_pips = min(price_move, trade["mae"]) / pip_value if pip_value > 0 else 0
        trade["mfe"] = max(price_move, trade["mfe"])
        trade["mae"] = min(price_move, trade["mae"])
        trade["max_favorable_pips"] = max(trade["max_favorable_pips"], mfe_pips)
        trade["max_adverse_pips"] = min(trade["max_adverse_pips"], mae_pips)
        trade["last_update"] = time.time()

    def close_trade(self, trade_id: str, exit_price: float, r_multiple: float):
        if trade_id not in self.trades:
            return
        trade = self.trades[trade_id]
        trade["closed"] = True
        trade["exit_price"] = exit_price
        trade["exit_r"] = r_multiple
        trade["exit_time"] = datetime.utcnow().isoformat()
        logger.info(f"[MFE/MAE] {trade['pair']} | {trade['setup_type']} | MFE={trade['max_favorable_pips']:.1f}pips | MAE={trade['max_adverse_pips']:.1f}pips | R={r_multiple:.2f} | EQS={trade['eqs']}")
        stats.record_mfe_mae(trade["pair"], trade["setup_type"], trade["eqs"], trade["max_favorable_pips"], trade["max_adverse_pips"], r_multiple)

    def get_trade(self, trade_id: str):
        return self.trades.get(trade_id)

# ============================================================
# STATISTIQUES
# ============================================================
class TradingStatsV101:
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
                "setup_types": [],
                "momentum_values": [],
                "spread_values": [],
                "volatility_values": [],
                "session_labels": [],
                "h1_trend": [],
                "h4_trend": []
            },
            "by_setup": defaultdict(lambda: {"wins": 0, "losses": 0, "total_r": 0.0, "trades": []}),
            "by_eqs_range": defaultdict(lambda: {"wins": 0, "losses": 0, "total_r": 0.0, "trades": []}),
            "by_hour": defaultdict(lambda: {"wins": 0, "losses": 0, "total_r": 0.0}),
            "by_weekday": defaultdict(lambda: {"wins": 0, "losses": 0, "total_r": 0.0}),
            "by_session": defaultdict(lambda: {"wins": 0, "losses": 0, "total_r": 0.0}),
            "by_adx_range": defaultdict(lambda: {"wins": 0, "losses": 0, "total_r": 0.0}),
            "mfe_mae": [],
            "setup_performance": defaultdict(lambda: {"wins": 0, "losses": 0, "total_r": 0.0, "trades": []})
        })
        self.last_transaction_id = None
        self.adaptive_state = AdaptiveState()  # <-- AJOUT
        self._load_last_id()
        self.last_daily_summary = time.time()
        self.rejection_stats = defaultdict(lambda: defaultdict(int))

    def _load_last_id(self):
        try:
            if os.path.exists("last_transaction_id.txt"):
                with open("last_transaction_id.txt", "r") as f:
                    self.last_transaction_id = f.read().strip()
        except:
            pass

    def _save_last_id(self):
        try:
            with open("last_transaction_id.txt", "w") as f:
                f.write(str(self.last_transaction_id))
        except:
            pass

    def record_rejection(self, pair: str, filter_name: str, reason: str):
        self.rejection_stats[pair][filter_name] += 1
        if DEBUG_MODE:
            logger.debug(f"[REJECT] {pair} | {filter_name}: {reason}")

    def get_rejection_summary(self, pair: str) -> dict:
        return dict(self.rejection_stats.get(pair, {}))

    def log_rejection_summary(self):
        logger.info("=" * 60)
        logger.info("📊 RÉSUMÉ DES REJETS")
        logger.info("=" * 60)
        for pair, filters in self.rejection_stats.items():
            total_rejections = sum(filters.values())
            logger.info(f"{pair:10} | Total rejets: {total_rejections}")
            for filter_name, count in sorted(filters.items(), key=lambda x: -x[1])[:5]:
                pct = count / total_rejections * 100 if total_rejections > 0 else 0
                logger.info(f"  {filter_name:20} | {count:4} ({pct:.1f}%)")
        logger.info("=" * 60)

    def record_signal(self, pair: str, accepted: bool, reason: str = "",
                      entry: float = 0, sl: float = 0, tp: float = 0,
                      score: int = 0, direction: str = "",
                      entry_metrics: dict = None):
        stats = self.stats[pair]
        stats["total_signals"] += 1
        if accepted:
            stats["accepted"] += 1
            logger.info(f"[SIGNAL_ACCEPTED] {pair} | {direction} | Score={score} | EQS={entry_metrics.get('eqs', 0) if entry_metrics else 0} | {reason}")
        else:
            stats["rejected"] += 1
            self.record_rejection(pair, reason.split(":")[0] if ":" in reason else "UNKNOWN", reason)
            logger.info(f"[SIGNAL_REJECTED] {pair} | {direction} | {reason}")
        if accepted and entry_metrics:
            trade_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "pair": pair,
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "score": score,
                "eqs": entry_metrics.get("eqs", 0),
                "setup_type": entry_metrics.get("setup_type", "UNKNOWN"),
                "atr": entry_metrics.get("atr", 0),
                "adx": entry_metrics.get("adx", 0),
                "rsi": entry_metrics.get("rsi", 0),
                "momentum": entry_metrics.get("momentum", 0),
                "hour": entry_metrics.get("hour", 0),
                "weekday": entry_metrics.get("weekday", 0),
                "spread": entry_metrics.get("spread", 0),
                "volatility": entry_metrics.get("volatility", 0),
                "session": entry_metrics.get("session", "UNKNOWN"),
                "h1_trend": entry_metrics.get("h1_trend", 0),
                "h4_trend": entry_metrics.get("h4_trend", 0),
                "result": None,
                "close_price": None,
                "close_pl": None,
                "mfe": None,
                "mae": None
            }
            stats["trades"].append(trade_record)

    def record_close(self, trade_id: str, pair: str, setup_type: str, eqs: int, r: float, profit_loss: float,
                     close_price: float = None, is_estimate: bool = False, trade_info: dict = None):
        stats = self.stats[pair]
        
        if profit_loss > 0:
            stats["wins"] += 1
            stats["total_profit"] += profit_loss
            result = "WIN"
            self.adaptive_state.record_win(pair)
        elif profit_loss < 0:
            stats["losses"] += 1
            stats["total_loss"] += abs(profit_loss)
            result = "LOSS"
            self.adaptive_state.record_loss(pair)
        else:
            stats["breakevens"] += 1
            result = "BREAKEVEN"
            if r > 0.02:
                logger.warning(f"[CLOSE_AMBIGUOUS] {pair} | R={r:.2f} | P&L={profit_loss:+.2f} | Frais ont mangé le profit")

        # Setup performance
        setup_stats = stats["by_setup"][setup_type]
        setup_stats["wins"] += 1 if result == "WIN" else 0
        setup_stats["losses"] += 1 if result == "LOSS" else 0
        setup_stats["total_r"] += r

        setup_perf = stats["setup_performance"][setup_type]
        setup_perf["wins"] += 1 if result == "WIN" else 0
        setup_perf["losses"] += 1 if result == "LOSS" else 0
        setup_perf["total_r"] += r

        # Enregistrement MFE/MAE
        tracker_trade = trade_tracker.get_trade(trade_id)
        if tracker_trade:
            mfe = tracker_trade.get("max_favorable_pips", 0)
            mae = tracker_trade.get("max_adverse_pips", 0)
            self.record_mfe_mae(pair, setup_type, eqs, mfe, mae, r)

        logger.info(f"[CLOSE] {pair} | {setup_type} | {result} | R={r:.2f} | P&L={profit_loss:+.2f} | EQS={eqs}")

        # Mise à jour des trades ouverts
        for trade in stats["trades"]:
            if trade.get("close_price") is None and trade.get("entry") == trade_info.get("entry"):
                trade["result"] = result
                trade["close_price"] = close_price
                trade["close_pl"] = profit_loss
                trade["mfe"] = mfe
                trade["mae"] = mae
                break

    def record_mfe_mae(self, pair: str, setup_type: str, eqs: int, mfe: float, mae: float, r: float):
        self.stats[pair]["mfe_mae"].append({
            "setup_type": setup_type,
            "eqs": eqs,
            "mfe": mfe,
            "mae": mae,
            "r": r
        })

    def get_summary(self, pair: str) -> dict:
        stats = self.stats.get(pair, {})
        total_signals = stats.get("total_signals", 0)
        accepted = stats.get("accepted", 0)
        rejected = stats.get("rejected", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        breakevens = stats.get("breakevens", 0)
        total_profit = stats.get("total_profit", 0.0)
        total_loss = stats.get("total_loss", 0.0)
        total_closed = wins + losses + breakevens
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        expectancy = (total_profit - total_loss) / total_closed if total_closed > 0 else 0
        return {
            "pair": pair,
            "total_signals": total_signals,
            "accepted": accepted,
            "rejected": rejected,
            "wins": wins,
            "losses": losses,
            "breakevens": breakevens,
            "total_closed": total_closed,
            "win_rate": f"{win_rate*100:.1f}%",
            "profit_factor": f"{profit_factor:.2f}",
            "expectancy": f"${expectancy:.2f}"
        }

    def log_daily_summary(self):
        logger.info("=" * 60)
        logger.info("📊 RÉSUMÉ QUOTIDIEN")
        logger.info("=" * 60)
        total_signals = 0
        total_accepted = 0
        total_rejected = 0
        total_wins = 0
        total_losses = 0
        total_be = 0
        for pair in sorted(self.stats.keys()):
            summary = self.get_summary(pair)
            total_signals += summary["total_signals"]
            total_accepted += summary["accepted"]
            total_rejected += summary["rejected"]
            total_wins += summary["wins"]
            total_losses += summary["losses"]
            total_be += summary["breakevens"]
            logger.info(f"{pair:10} | Signaux:{summary['total_signals']:3} | Acceptés:{summary['accepted']:3} | Rejetés:{summary['rejected']:3} | Clôturés:{summary['total_closed']:3} | WR:{summary['win_rate']:>6} | PF:{summary['profit_factor']:>6} | Esp:{summary['expectancy']:>8}")
        if total_signals > 0:
            logger.info("-" * 60)
            logger.info(f"TOTAL      | Signaux:{total_signals:3} | Acceptés:{total_accepted:3} | Rejetés:{total_rejected:3} | Clôturés:{total_wins + total_losses + total_be:3}")
            if total_wins + total_losses > 0:
                global_wr = total_wins / (total_wins + total_losses) * 100
                logger.info(f"Win Rate global: {global_wr:.1f}% | Wins:{total_wins} | Losses:{total_losses} | BE:{total_be}")
        self.log_rejection_summary()
        logger.info("=" * 60)

    def log_summary(self):
        if time.time() - self.last_daily_summary >= 86400:
            self.log_daily_summary()
            self.last_daily_summary = time.time()
        logger.info("=" * 80)
        logger.info("📊 STATISTIQUES GLOBALES")
        logger.info("=" * 80)
        logger.info(f"{'Paire':10} | {'Signaux':>7} | {'Acceptés':>7} | {'Rejetés':>7} | {'Clôturés':>7} | {'Win Rate':>9} | {'PF':>6} | {'Espérance':>10}")
        logger.info("-" * 80)
        for pair in sorted(self.stats.keys()):
            summary = self.get_summary(pair)
            logger.info(
                f"{pair:10} | {summary['total_signals']:>7} | {summary['accepted']:>7} | {summary['rejected']:>7} | "
                f"{summary['total_closed']:>7} | {summary['win_rate']:>9} | {summary['profit_factor']:>6} | {summary['expectancy']:>10}"
            )
        logger.info("=" * 80)

# ============================================================
# FONCTIONS OANDA
# ============================================================
def v88_client():
    token = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_ACCESS_TOKEN")
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")
    return oandapyV20.API(access_token=token, environment=environment)

def get_candles_with_retry(api, instrument: str, granularity: str, count: int = 500, retries: int = 3) -> pd.DataFrame:
    valid_granularities = ["S5","S10","S15","S30","M1","M2","M4","M5","M10","M15","M30","H1","H2","H3","H4","H6","H8","H12","D","W","M"]
    if granularity not in valid_granularities:
        logger.error(f"❌ Granularité invalide: {granularity}")
        return pd.DataFrame()
    if is_maintenance_suspended():
        logger.debug(f"⏳ OANDA en maintenance - get_candles {instrument} suspendu")
        return pd.DataFrame()
    for attempt in range(retries):
        try:
            params = {"granularity": granularity, "count": min(count, 500), "price": "M"}
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            api.request(r)
            resp = getattr(r, "response", {}) or {}
            candles = resp.get("candles", [])
            if not candles:
                logger.warning(f"⚠️ Aucune candle reçue pour {instrument} {granularity} (tentative {attempt+1})")
                time.sleep(2)
                continue
            data = []
            for c in candles:
                mid = c.get("mid")
                if not mid:
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
                    continue
            if len(data) < max(10, count // 10):
                logger.warning(f"⚠️ Données insuffisantes pour {instrument} {granularity}: {len(data)}")
                time.sleep(2)
                continue
            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df.attrs['instrument'] = instrument
            logger.info(f"✅ {instrument} {granularity}: {len(df)} candles")
            return df
        except oandapyV20.exceptions.V20Error as e:
            if is_oanda_in_maintenance(e):
                should_suspend, duration, log_level = handle_api_error(e)
                if should_suspend:
                    logger.warning(f"🔧 Maintenance OANDA - suspension {duration}s")
                    time.sleep(duration)
                    return pd.DataFrame()
                else:
                    logger.warning(f"⚠️ OANDA en maintenance (tentative {attempt+1}) - attente 5s")
                    time.sleep(5)
                    continue
            logger.warning(f"❌ Erreur OANDA {attempt+1}/{retries} pour {instrument}: {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"❌ Tentative {attempt+1}/{retries} pour {instrument}: {e}")
            time.sleep(2 ** attempt)
    logger.error(f"❌ Échec après {retries} tentatives pour {instrument} {granularity}")
    return pd.DataFrame()

def get_price_spread_v88(pair: str) -> dict:
    cached = _cache_get_v88(f"pricing:{pair}", ttl_seconds=2.0)
    if cached is not None:
        return cached
    try:
        if is_maintenance_suspended():
            fallback = get_recent_m5_price_v88(pair)
            return {"bid": fallback, "ask": fallback, "mid": fallback, "spread": 0.0}
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
    except Exception as e:
        logger.debug(f"Erreur pricing {pair}: {e}")
        if is_oanda_in_maintenance(e):
            handle_api_error(e)
    fallback = get_recent_m5_price_v88(pair)
    return {"bid": fallback, "ask": fallback, "mid": fallback, "spread": 0.0}

def get_recent_m5_price_v88(pair: str) -> float:
    try:
        if is_maintenance_suspended():
            return 0.0
        api = v88_client()
        df = get_candles_with_retry(api, pair, "M5", 10)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
    except Exception:
        pass
    return 0.0

def get_atr_m15_v88(pair: str) -> float:
    cached = _cache_get_v88(f"atr_m15:{pair}", ttl_seconds=60.0)
    if cached is not None:
        return float(cached)
    try:
        if is_maintenance_suspended():
            return 0.0
        api = v88_client()
        df = get_candles_with_retry(api, pair, "M15", max(ATR_PERIOD + 10, 40))
        if df is None or df.empty:
            return 0.0
        atr = float(calculate_atr(df, ATR_PERIOD) or 0.0)
        _cache_set_v88(f"atr_m15:{pair}", atr)
        return atr
    except Exception:
        return 0.0

def get_open_trades_v88(log_raw: bool = False, skip_maintenance_check: bool = False, force_refresh: bool = False) -> list:
    cache_key = "open_trades_raw"
    if force_refresh:
        _OANDA_CACHE_V88.pop(cache_key, None)
        logger.debug("[CACHE] Force refresh - cache invalidé")
    if not skip_maintenance_check and is_maintenance_suspended():
        logger.debug("⏳ OANDA en maintenance - appel OpenTrades suspendu")
        return []
    resp = _cache_get_v88(cache_key)
    if resp is None:
        try:
            api = v88_client()
            r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
            resp = api.request(r)
            reset_maintenance_state()
            if resp:
                _cache_set_v88(cache_key, resp)
        except oandapyV20.exceptions.V20Error as e:
            should_suspend, duration, log_level = handle_api_error(e)
            if log_level == "WARNING":
                logger.warning(f"⚠️ Erreur OpenTrades: {e}")
            else:
                logger.error(f"❌ Erreur OpenTrades: {e}")
            cached_resp = _cache_get_v88(cache_key, ttl_seconds=10.0)
            if cached_resp is not None:
                logger.debug("📦 Utilisation du cache pour OpenTrades")
                resp = cached_resp
            else:
                return []
        except Exception as e:
            logger.error(f"Erreur OpenTrades: {e}")
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

def get_account_summary_v88() -> dict:
    cached = _cache_get_v88("account_summary")
    if cached is not None:
        return cached
    try:
        if is_maintenance_suspended():
            return {}
        api = v88_client()
        r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
        resp = api.request(r)
        if resp:
            _cache_set_v88("account_summary", resp)
            return resp
    except Exception as e:
        logger.error(f"AccountSummary error: {e}")
        if is_oanda_in_maintenance(e):
            handle_api_error(e)
    return {}

def get_balance_v88() -> float:
    resp = get_account_summary_v88()
    try:
        return float(resp.get("account", {}).get("balance", 0))
    except Exception:
        return 0.0

def get_oanda_margin_rate_v88(pair: str) -> float:
    cached = _cache_get_v88(f"instrument:{pair}", ttl_seconds=300.0)
    if cached is not None:
        return float(cached.get("marginRate", 0.0333) or 0.0333)
    try:
        if is_maintenance_suspended():
            return 0.0333
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

def get_fx_rate_to_usd_v88(currency: str) -> float:
    if currency == "USD":
        return 1.0
    cached = _cache_get_v88(f"fx_to_usd:{currency}", ttl_seconds=60.0)
    if cached is not None:
        return float(cached)
    direct = f"{currency}_USD"
    inverse = f"USD_{currency}"
    try:
        if is_maintenance_suspended():
            return 1.0
        api = v88_client()
        if direct in PAIR_LIST:
            df = get_candles_with_retry(api, direct, "M5", 10)
            if df is not None and not df.empty:
                val = float(df["close"].iloc[-1])
                _cache_set_v88(f"fx_to_usd:{currency}", val)
                return val
        if inverse in PAIR_LIST:
            df = get_candles_with_retry(api, inverse, "M5", 10)
            if df is not None and not df.empty:
                val = 1.0 / float(df["close"].iloc[-1])
                _cache_set_v88(f"fx_to_usd:{currency}", val)
                return val
    except Exception:
        pass
    return 1.0

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

def calculate_units_v88(pair: str, entry: float, stop_loss: float, balance: float, risk_pct: float = None) -> float:
    try:
        balance = float(balance)
        entry = float(entry)
        stop_loss = float(stop_loss)
    except Exception:
        logger.error(f"paramètres sizing invalides pair={pair}")
        return 0
    if risk_pct is None:
        risk_pct = RISK_PERCENTAGE
    risk_usd = min(balance * (risk_pct / 100.0), MAX_RISK_USD)
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

def quote_currency_v88(pair: str) -> str:
    return pair.split("_")[1]

def round_price_v88(pair: str, price: float) -> str:
    decimals = PRICE_DECIMALS_V88.get(pair, 5)
    return f"{float(price):.{decimals}f}"

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

def open_trade_count_v88() -> int:
    return len(get_open_trades_v88(log_raw=True))

def has_open_trade_v88(pair: str) -> bool:
    for t in get_open_trades_v88():
        if t.get("instrument") == pair:
            return True
    return False

def get_trade_details_v88(trade_id: str) -> dict:
    try:
        if is_maintenance_suspended():
            return {}
        api = v88_client()
        r = trades.TradeDetails(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id)
        resp = api.request(r)
        return resp.get("trade", {})
    except oandapyV20.exceptions.V20Error as e:
        if "404" in str(e):
            logger.debug(f"[TRADE] Trade {trade_id} non trouvé (probablement fermé)")
        else:
            logger.error(f"[TRADE] Erreur récupération trade {trade_id}: {e}")
        return {}
    except Exception as e:
        logger.error(f"[TRADE] Erreur récupération trade {trade_id}: {e}")
        return {}

def has_trailing_stop_v88(trade: dict) -> bool:
    trailing_stop = trade.get("trailingStopLossOrder", {})
    return bool(trailing_stop and trailing_stop.get("id"))

def get_stop_loss_v88(trade: dict) -> float:
    sl_order = trade.get("stopLossOrder", {})
    return float(sl_order.get("price", 0)) if sl_order else 0.0

# ============================================================
# INDICATEURS TECHNIQUES
# ============================================================
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

def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    try:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        tr = np.zeros(len(df))
        for i in range(1, len(df)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
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

def calculate_momentum(df: pd.DataFrame, period: int = 5) -> float:
    if len(df) < period + 1:
        return 0.0
    try:
        close = df['close']
        roc = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period] * 100
        return float(roc)
    except Exception:
        return 0.0

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
    min_gap_size = 0.00015 if "JPY" in pair_name else 0.0002  # simplifié
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
        if prev_high < next_low and next_low - prev_high >= min_gap_size:
            fvgs.append({
                "index": i,
                "direction": "BUY",
                "type": "PERFECT",
                "high_level": next_low,
                "low_level": prev_high,
                "gap_size": next_low - prev_high,
                "time": candle_time,
                "midpoint": (prev_high + next_low) / 2
            })
        if prev_low > next_high and prev_low - next_high >= min_gap_size:
            fvgs.append({
                "index": i,
                "direction": "SELL",
                "type": "PERFECT",
                "high_level": prev_low,
                "low_level": next_high,
                "gap_size": prev_low - next_high,
                "time": candle_time,
                "midpoint": (next_high + prev_low) / 2
            })
    return fvgs

def get_fvg_midpoint(fvg: dict) -> float:
    if "high_level" not in fvg or "low_level" not in fvg:
        return None
    high = float(fvg["high_level"])
    low = float(fvg["low_level"])
    if high == low:
        return None
    return round((high + low) / 2, 5)

def detect_wick_rejection_poi(df: pd.DataFrame, bias: str, min_wick_ratio: float = 0.7) -> list:
    poi_list = []
    pair = df.attrs.get("instrument", "DEFAULT")
    pip_tolerance_map = {"XAU_USD": 20, "USD_JPY": 0.50, "AUD_USD": 0.0050, "EUR_USD": 0.0020, "USD_CAD": 0.0050, "GBP_USD": 0.0050, "DEFAULT": 0.0010}
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
        if bias in ["BUY", "NEUTRAL"] and lower_wick >= body_size * min_wick_ratio and lower_wick >= upper_wick * 1.5 and lower_wick >= total_range * 0.4 and rsi_m15 < 60 and confirmation_candle["close"] > confirmation_candle["open"] and confirmation_candle["close"] > rejection_candle["high"]:
            if abs(current_price - rejection_candle["low"]) <= pip_tolerance:
                poi_list.append({"type": "WICK_REJECTION", "price_level": rejection_candle["low"], "wick_size": lower_wick, "body_size": body_size, "time": df.index[i], "direction": "BUY", "wick_ratio": lower_wick / total_range, "rsi_at_rejection": rsi_m15, "pair": pair})
        elif bias in ["SELL", "NEUTRAL"] and upper_wick >= body_size * min_wick_ratio and upper_wick >= lower_wick * 1.5 and upper_wick >= total_range * 0.4 and rsi_m15 > 40 and confirmation_candle["close"] < confirmation_candle["open"] and confirmation_candle["close"] < rejection_candle["low"]:
            if abs(current_price - rejection_candle["high"]) <= pip_tolerance:
                poi_list.append({"type": "WICK_REJECTION", "price_level": rejection_candle["high"], "wick_size": upper_wick, "body_size": body_size, "time": df.index[i], "direction": "SELL", "wick_ratio": upper_wick / total_range, "rsi_at_rejection": rsi_m15, "pair": pair})
    return poi_list

def detect_bos(df: pd.DataFrame, lookback: int = 50) -> dict:
    if len(df) < lookback + 10:
        return {"type": None, "level": None, "time": None}
    swing_highs, swing_lows = detect_swing_points_advanced(df, lookback=5)
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
    swing_highs, swing_lows = detect_swing_points_advanced(df, lookback=5)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"type": None, "level": None, "time": None}
    current_price = df["close"].iloc[-1]
    current_time = df.index[-1]
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-2]["price"]
        lh = swing_highs[-1]["price"]
        hl = swing_lows[-2]["price"]
        ll = swing_lows[-1]["price"]
        is_uptrend = (hh > (swing_highs[-3]["price"] if len(swing_highs) >= 3 else 0) and hl > (swing_lows[-3]["price"] if len(swing_lows) >= 3 else 0))
        if is_uptrend and ll < hl and current_price < ll:
            return {"type": "CHOCH_SELL", "level": ll, "time": current_time}
    if len(swing_lows) >= 2 and len(swing_highs) >= 2:
        ll = swing_lows[-2]["price"]
        hl = swing_lows[-1]["price"]
        lh = swing_highs[-2]["price"]
        hh = swing_highs[-1]["price"]
        is_downtrend = (ll < (swing_lows[-3]["price"] if len(swing_lows) >= 3 else float('inf')) and lh < (swing_highs[-3]["price"] if len(swing_highs) >= 3 else float('inf')))
        if is_downtrend and hh > lh and current_price > hh:
            return {"type": "CHOCH_BUY", "level": hh, "time": current_time}
    return {"type": None, "level": None, "time": None}

def detect_nested_fvg(df: pd.DataFrame, min_nesting: int = 2) -> list:
    fvgs = detect_fvg_advanced(df)
    nested_fvgs = []
    for i in range(len(fvgs) - min_nesting + 1):
        current_fvg = fvgs[i]
        next_fvg = fvgs[i + 1]
        if current_fvg.get("direction") == next_fvg.get("direction"):
            direction = current_fvg["direction"]
            if direction == "BUY":
                entry_zone = (min(float(current_fvg["high_level"]), float(next_fvg["low_level"])), max(float(current_fvg["high_level"]), float(next_fvg["low_level"])))
            else:
                entry_zone = (min(float(current_fvg["low_level"]), float(next_fvg["high_level"])), max(float(current_fvg["low_level"]), float(next_fvg["high_level"])))
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

def detect_setups_aligned_with_bias(df_m15: pd.DataFrame, df_h1: pd.DataFrame, bias: str, pair: str = "XAU_USD", df_h4: pd.DataFrame = None) -> List[Dict]:
    setups = []
    if bias not in ["BUY", "SELL"]:
        buy_setups = detect_setups_aligned_with_bias(df_m15, df_h1, "BUY", pair, df_h4)
        sell_setups = detect_setups_aligned_with_bias(df_m15, df_h1, "SELL", pair, df_h4)
        return buy_setups + sell_setups
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
            "direction": bias,
            "entry_level": round(entry_level, 5),
            "entry_zone": (round(entry_level - 0.0010, 5), round(entry_level + 0.0010, 5)),
            "confidence": "MEDIUM",
            "trigger": "FVG_RETEST",
            "rsi_m15": rsi_m15,
            "rsi_h4": rsi_h4,
            "fvg": fvg,
            "structure_analysis": {"bos": bos, "choch": choch},
            "bias_aligned": True
        })
    for nfvg in nested:
        entry_level = nfvg.get("midpoint")
        if entry_level is None or abs(current_price - entry_level) > 0.0020:
            continue
        setups.append({
            "type": "NESTED_FVG",
            "direction": bias,
            "entry_level": round(entry_level, 5),
            "entry_zone": (round(entry_level - 0.0015, 5), round(entry_level + 0.0015, 5)),
            "confidence": "HIGH",
            "trigger": "NESTED_FVG",
            "rsi_m15": rsi_m15,
            "rsi_h4": rsi_h4,
            "fvg": nfvg,
            "structure_analysis": {"bos": bos, "choch": choch},
            "bias_aligned": True
        })
    for wick in wicks:
        entry_level = wick.get("price_level")
        if entry_level is None or abs(current_price - entry_level) > 0.0020:
            continue
        setups.append({
            "type": "WICK_REJECTION",
            "direction": bias,
            "entry_level": round(entry_level, 5),
            "entry_zone": (round(entry_level - 0.0010, 5), round(entry_level + 0.0010, 5)),
            "confidence": "MEDIUM",
            "trigger": "WICK_REJECTION",
            "rsi_m15": rsi_m15,
            "rsi_h4": rsi_h4,
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
                    "type": "BISI",
                    "direction": bias,
                    "entry_level": round(fvg_level, 5),
                    "entry_zone": (round(fvg_level - 0.0010, 5), round(fvg_level + 0.0010, 5)),
                    "confidence": "VERY_HIGH",
                    "trigger": "BISI",
                    "rsi_m15": rsi_m15,
                    "rsi_h4": rsi_h4,
                    "bosis": {"level": bos_level, "type": bos["type"]},
                    "structure_analysis": {"bos": bos, "choch": choch},
                    "bias_aligned": True
                })
    return setups

# ============================================================
# STRATÉGIE SIMPLIFIÉE : FONCTIONS DE DÉCISION
# ============================================================
def get_directional_bias(df_h4: pd.DataFrame, df_h1: pd.DataFrame) -> str:
    def get_bias_from_structure(df: pd.DataFrame) -> str:
        swing_highs, swing_lows = detect_swing_points_advanced(df, lookback=5)
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "NEUTRAL"
        last_h = (swing_highs[-1]['price'], swing_highs[-2]['price'])
        last_l = (swing_lows[-1]['price'], swing_lows[-2]['price'])
        hh = last_h[0] > last_h[1]
        hl = last_l[0] > last_l[1]
        lh = last_h[0] < last_h[1]
        ll = last_l[0] < last_l[1]
        if hh and hl:
            return "BUY"
        elif lh and ll:
            return "SELL"
        else:
            return "NEUTRAL"
    bias_h4 = get_bias_from_structure(df_h4)
    bias_h1 = get_bias_from_structure(df_h1)
    if bias_h4 == "BUY" and bias_h1 == "BUY":
        return "BUY"
    elif bias_h4 == "SELL" and bias_h1 == "SELL":
        return "SELL"
    else:
        return "NEUTRAL"

def get_confirmation_signal(df_m15: pd.DataFrame, direction: str) -> Tuple[bool, str]:
    if len(df_m15) < 3:
        return False, "données insuffisantes"
    last = df_m15.iloc[-1]
    prev = df_m15.iloc[-2]
    total_range = last['high'] - last['low']
    if total_range == 0:
        return False, "range nul"
    upper_wick = last['high'] - max(last['open'], last['close'])
    lower_wick = min(last['open'], last['close']) - last['low']
    if direction == "BUY":
        rejet = lower_wick > total_range * 0.4
        micro_break = last['close'] > prev['high']
    else:
        rejet = upper_wick > total_range * 0.4
        micro_break = last['close'] < prev['low']
    if rejet and micro_break:
        return True, f"confirmation {direction}: rejet + micro-break"
    else:
        reasons = []
        if not rejet:
            reasons.append("pas de rejet")
        if not micro_break:
            reasons.append("pas de micro-break")
        return False, ", ".join(reasons)

def calculate_sl_tp_structural(df_m15: pd.DataFrame, direction: str, entry_price: float, pair: str) -> Tuple[float, float, float, float]:
    swing_highs, swing_lows = detect_swing_points_advanced(df_m15, lookback=5)
    pip = get_pip_value_for_pair(pair)
    if direction == "BUY":
        if swing_lows:
            sl = min(swing_lows[-1]['price'], entry_price - pip*5)
        else:
            atr = calculate_atr(df_m15)
            sl = entry_price - atr * 1.5
    else:
        if swing_highs:
            sl = max(swing_highs[-1]['price'], entry_price + pip*5)
        else:
            atr = calculate_atr(df_m15)
            sl = entry_price + atr * 1.5
    risk = abs(entry_price - sl)
    tp = entry_price + (2 * risk) if direction == "BUY" else entry_price - (2 * risk)
    sl = float(round_price_v88(pair, sl))
    tp = float(round_price_v88(pair, tp))
    risk_pips = price_to_pips(risk, pair)
    atr_pips = price_to_pips(calculate_atr(df_m15), pair)
    return sl, tp, risk_pips, atr_pips

def has_enough_room_to_tp(df_h1: pd.DataFrame, direction: str, entry: float, tp: float) -> bool:
    swing_highs, swing_lows = detect_swing_points_advanced(df_h1, lookback=5)
    if direction == "BUY":
        for sh in swing_highs:
            if entry < sh['price'] < tp:
                return False
    else:
        for slw in swing_lows:
            if tp < slw['price'] < entry:
                return False
    return True

def get_session_label() -> str:
    hour = datetime.utcnow().hour
    if 7 <= hour < 16:
        return "LONDON"
    elif 12 <= hour < 21:
        return "NY"
    elif hour >= 21 or hour < 7:
        return "ASIA"
    else:
        return "OTHER"

def evaluate_simple_setup(pair: str, direction: str, entry: dict, df_h4: pd.DataFrame, df_h1: pd.DataFrame, df_m15: pd.DataFrame, current_price: float) -> dict:
    entry_level = float(entry["entry_level"])
    atr_price = calculate_atr(df_m15)
    if abs(current_price - entry_level) > atr_price * 1.5:
        return {"passed": False, "reason": "prix hors zone", "entry_score": 0}
    confirm_ok, msg = get_confirmation_signal(df_m15, direction)
    if not confirm_ok:
        return {"passed": False, "reason": f"confirmation: {msg}", "entry_score": 0}
    sl, tp, risk_pips, atr_pips = calculate_sl_tp_structural(df_m15, direction, entry_level, pair)
    if not has_enough_room_to_tp(df_h1, direction, entry_level, tp):
        return {"passed": False, "reason": "RR réel impossible: niveau entre entry et TP", "entry_score": 0}
    rr = abs(tp - entry_level) / abs(sl - entry_level)
    if rr < 2.0:
        return {"passed": False, "reason": f"RR={rr:.2f} < 2.0", "entry_score": 0}
    return {
        "passed": True,
        "entry_level": entry_level,
        "sl": sl,
        "tp": tp,
        "risk_pips": risk_pips,
        "atr_pips": atr_pips,
        "rr": rr,
        "reason": "OK",
        "metrics": {
            "atr": atr_pips,
            "adx": calculate_adx(df_h1),
            "momentum": calculate_momentum(df_m15),
            "session": get_session_label(),
            "rsi": get_last_rsi(df_m15["close"])
        }
    }

# ============================================================
# UTILITAIRES DE PRIX
# ============================================================
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
        return 1.0
    elif pair == "NAS100_USD":
        return 0.1
    elif "JPY" in pair:
        return 0.01
    else:
        return 0.0001

# ============================================================
# INSTANCIATION DES CLASSES
# ============================================================
stats = TradingStatsV101()
trade_tracker = TradeTracker()
open_trade_details = {}
stagnant_trade_tracker = {}
last_execution_attempt = {}

# ============================================================
# EXÉCUTION ORDRE
# ============================================================
def execute_oanda_trade_v981(pair: str, direction: str, entry_price: float, stop_loss: float, take_profit: float, score: int, entry_type: str, eqs: int, setup_type: str, metrics: dict, rr: float = None) -> str | None:
    global last_execution_attempt
    pair_upper = pair.upper()
    direction = direction.upper()
    if rr is None:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr = reward / risk if risk > 0 else 0.0
    if rr < 1.30:
        logger.warning(f"[ORDER_REJECT] {pair} {direction} RR={rr:.2f} < 1.30 → refusé")
        return None
    now = time.time()
    if pair_upper in last_execution_attempt and now - last_execution_attempt[pair_upper] < EXECUTION_COOLDOWN_SECONDS:
        logger.warning(f"[ORDER] Cooldown actif pour {pair_upper}")
        return None
    last_execution_attempt[pair_upper] = now
    expected_entry = float(entry_price)
    logger.info(f"[ORDER] V110 EXECUTION START {pair} {direction} type={entry_type} score={score}")
    risk = abs(expected_entry - stop_loss)
    reward = abs(take_profit - expected_entry)
    if risk <= 0:
        logger.error("[ORDER] Risque nul")
        return None
    rr_computed = reward / risk
    if rr_computed < 1.80:
        if direction == "BUY":
            take_profit = expected_entry + risk * 2.0
        else:
            take_profit = expected_entry - risk * 2.0
        logger.info(f"[ORDER] TP ajusté pour RR=2.0: {take_profit:.5f}")
    if ONE_TRADE_PER_PAIR and has_open_trade_v88(pair):
        logger.info(f"{pair}: trade déjà ouvert")
        return None
    if open_trade_count_v88() >= MAX_TRADES_TOTAL:
        logger.info(f"Limite trades atteinte")
        return None
    if is_maintenance_suspended():
        logger.warning(f"[ORDER] OANDA maintenance")
        return None
    balance = get_balance_v88()
    if balance <= 0:
        logger.error("Balance invalide")
        return None
    pip_value = get_pip_value_for_pair(pair)
    min_sl_distance = pip_value * 10
    risk = abs(expected_entry - stop_loss)
    if risk < min_sl_distance:
        logger.warning(f"[ORDER] SL trop proche: {risk:.5f} < {min_sl_distance:.5f}")
        if direction == "BUY":
            stop_loss = expected_entry - min_sl_distance
        else:
            stop_loss = expected_entry + min_sl_distance
        risk = min_sl_distance
        if direction == "BUY":
            take_profit = expected_entry + risk * 2.0
        else:
            take_profit = expected_entry - risk * 2.0
    hour = datetime.utcnow().hour
    is_asia = (21 <= hour or hour < 7)
    risk_pct = 0.5 if is_asia else RISK_PERCENTAGE
    units = calculate_units_v88(pair, expected_entry, stop_loss, balance, risk_pct=risk_pct)
    if not units or float(units) <= 0:
        logger.error(f"Units invalides: {units}")
        return None
    margin_info = calculate_margin_v88(pair, units, expected_entry)
    if not margin_info["sufficient"]:
        units = cap_units_by_margin_v88(pair, units, expected_entry, balance)
        if not units or units <= 0:
            logger.error(f"[RISK] Marge insuffisante")
            return None
    signed_units = units if direction == "BUY" else -units
    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": pair,
            "units": str(int(signed_units)),
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": round_price_v88(pair, stop_loss), "timeInForce": "GTC"},
            "takeProfitOnFill": {"price": round_price_v88(pair, take_profit), "timeInForce": "GTC"}
        }
    }
    logger.info(f"[ORDER_EXPECTED] {pair} | {direction} | ENTRY_EXPECTED={expected_entry:.5f} | SL={stop_loss:.5f} | TP={take_profit:.5f} | RR={abs(take_profit-expected_entry)/risk:.2f}")
    if not EXECUTE_TRADES:
        logger.info("[ORDER] EXECUTE_TRADES=false")
        return "SIMULATION"
    try:
        api = v88_client()
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        resp = api.request(r)
        if resp.get("orderRejectTransaction"):
            reject = resp["orderRejectTransaction"]
            logger.error(f"[ORDER] REJECT {pair}: {reject.get('rejectReason')}")
            return None
        fill = resp.get("orderFillTransaction", {})
        trade_id = None
        if fill.get("tradeOpened"):
            trade_id = fill["tradeOpened"].get("tradeID")
        actual_entry = None
        for key in ["fullPrice", "price"]:
            if fill.get(key) is not None:
                try:
                    actual_entry = float(fill[key])
                    break
                except Exception:
                    pass
        if not trade_id:
            time.sleep(1.0)
            for attempt in range(8):
                open_trades = get_open_trades_v88(skip_maintenance_check=True, force_refresh=True)
                candidates = []
                for t in open_trades:
                    if t.get("instrument") != pair:
                        continue
                    t_direction = "BUY" if float(t.get("currentUnits", 0)) > 0 else "SELL"
                    if t_direction != direction:
                        continue
                    t_entry = float(t.get("price", 0))
                    candidates.append((abs(t_entry - expected_entry), t))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    _, best_trade = candidates[0]
                    trade_id = best_trade.get("id")
                    actual_entry = float(best_trade.get("price"))
                    break
                time.sleep(0.8)
        if not trade_id:
            logger.error(f"[ORDER] Trade non confirmé {pair}")
            return None
        if actual_entry is None:
            try:
                trade_details = get_trade_details_v88(trade_id)
                if trade_details:
                    actual_entry = float(trade_details.get("price", expected_entry))
            except Exception as e:
                logger.warning(f"[ORDER] Impossible récupérer fill réel: {e}")
        if actual_entry is None:
            actual_entry = expected_entry
        slippage_price = actual_entry - expected_entry
        slippage_pips = slippage_price / pip_value
        logger.info(f"[FILL_REAL] {pair} | TRADE_ID={trade_id} | ENTRY_EXPECTED={expected_entry:.5f} | ENTRY_FILLED={actual_entry:.5f} | SLIPPAGE={slippage_pips:+.1f} pips")
        MAX_ENTRY_SLIPPAGE_PIPS = {"GBP_USD": 2.5, "EUR_USD": 2.5, "USD_CAD": 2.5, "AUD_USD": 2.5, "DEFAULT": 3.0}
        max_slippage = MAX_ENTRY_SLIPPAGE_PIPS.get(pair, MAX_ENTRY_SLIPPAGE_PIPS["DEFAULT"])
        if abs(slippage_pips) > max_slippage:
            logger.warning(f"[SLIPPAGE] {pair}: {slippage_pips:+.1f} pips > max {max_slippage:.1f}")
            metrics = dict(metrics or {})
            metrics["slippage_pips"] = slippage_pips
            metrics["slippage_warning"] = True
        else:
            metrics = dict(metrics or {})
            metrics["slippage_pips"] = slippage_pips
            metrics["slippage_warning"] = False
        trade_tracker.add_trade(trade_id, pair, direction, actual_entry, stop_loss, take_profit, setup_type, eqs)
        open_trade_details[str(trade_id)] = {
            "entry": actual_entry,
            "expected_entry": expected_entry,
            "actual_entry": actual_entry,
            "sl": stop_loss,
            "tp": take_profit,
            "direction": direction,
            "setup_type": setup_type,
            "eqs": eqs,
            "pair": pair,
            "units": units,
            **metrics
        }
        logger.info(f"[ORDER_CONFIRMED] {pair} | {direction} | ID={trade_id} | EXPECTED={expected_entry:.5f} | FILLED={actual_entry:.5f} | SL={stop_loss:.5f} | TP={take_profit:.5f}")
        return str(trade_id)
    except oandapyV20.exceptions.V20Error as e:
        logger.error(f"[ORDER] V20Error {pair}: {e}")
        if is_oanda_in_maintenance(e):
            handle_api_error(e)
        return None
    except Exception as e:
        logger.exception(f"[ORDER] Erreur OANDA {pair}: {e}")
        if is_oanda_in_maintenance(e):
            handle_api_error(e)
        return None

# ============================================================
# MODIFICATIONS SL / TRAILING
# ============================================================
def modify_trade_sl_v981(trade_id: str, pair: str, new_sl: float, adjust_tp: bool = True) -> bool:
    try:
        if is_maintenance_suspended():
            logger.warning(f"[BE] OANDA en maintenance - modification SL suspendue pour {trade_id}")
            return False
        trade_details = get_trade_details_v88(trade_id)
        if not trade_details:
            logger.error(f"[BE] Impossible de récupérer le trade {trade_id}")
            return False
        entry = float(trade_details.get("price", 0))
        current_tp = float(trade_details.get("takeProfitOrder", {}).get("price", 0))
        current_units = float(trade_details.get("currentUnits", 0))
        if entry == 0 or current_tp == 0:
            logger.warning(f"[BE] Pas de TP pour trade {trade_id}, on modifie SL seulement")
            adjust_tp = False
        api = v88_client()
        logger.info(f"[BE] Modification SL via TradeCRCDO pour trade {trade_id} -> {new_sl:.5f}")
        data = {"stopLoss": {"price": round_price_v88(pair, new_sl), "timeInForce": "GTC"}}
        if adjust_tp and current_tp > 0:
            risk = abs(entry - new_sl)
            if risk > 0:
                new_tp = entry + (2.0 * risk) if current_units > 0 else entry - (2.0 * risk)
                if (current_units > 0 and new_tp > entry) or (current_units < 0 and new_tp < entry):
                    data["takeProfit"] = {"price": round_price_v88(pair, new_tp), "timeInForce": "GTC"}
                    logger.info(f"[BE] TP ajusté à {new_tp:.5f} pour maintenir RR=2.0")
                else:
                    logger.warning(f"[BE] TP ajusté invalide, on le laisse inchangé")
        r = trades.TradeCRCDO(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id, data=data)
        resp = api.request(r)
        if resp.get("orderRejectTransaction"):
            reject = resp.get("orderRejectTransaction")
            logger.error(f"[BE] Rejeté pour trade {trade_id}: {reject.get('rejectReason', 'unknown')}")
            return False
        logger.info(f"[BE] SUCCESS: SL modifié pour trade {trade_id} -> {new_sl:.5f}")
        time.sleep(1)
        _OANDA_CACHE_V88.pop("open_trades_raw", None)
        return True
    except Exception as e:
        logger.error(f"[BE] Erreur modification SL trade {trade_id}: {e}")
        if is_oanda_in_maintenance(e):
            handle_api_error(e)
        return False

def create_oanda_trailing_stop_v981(trade_id: str, pair: str, distance: float) -> bool:
    try:
        if is_maintenance_suspended():
            logger.warning(f"[TSL] OANDA en maintenance - trailing suspendu pour {trade_id}")
            return False
        api = v88_client()
        logger.info(f"[TSL] Création trailing via OrderCreate pour trade {trade_id} -> distance={distance:.5f}")
        order_data = {"order": {"type": "TRAILING_STOP_LOSS", "tradeID": trade_id, "distance": str(distance), "timeInForce": "GTC"}}
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        resp = api.request(r)
        if resp.get("orderRejectTransaction"):
            reject = resp.get("orderRejectTransaction")
            logger.error(f"[TSL] Rejeté pour trade {trade_id}: {reject.get('rejectReason', 'unknown')}")
            return False
        logger.info(f"[TSL] SUCCESS: Trailing stop créé pour trade {trade_id}, distance={distance:.5f}")
        time.sleep(1)
        _OANDA_CACHE_V88.pop("open_trades_raw", None)
        return True
    except Exception as e:
        logger.error(f"[TSL] Erreur création trailing stop trade {trade_id}: {e}")
        if is_oanda_in_maintenance(e):
            handle_api_error(e)
        return False

# ============================================================
# SUIVI DES TRADES STAGNANTS
# ============================================================
def check_stagnant_trades(trade_id: str, pair: str, direction: str, entry: float, current_r: float):
    global stagnant_trade_tracker
    if trade_id not in stagnant_trade_tracker:
        stagnant_trade_tracker[trade_id] = {"first_seen": time.time(), "last_r": current_r, "r_stable_count": 0, "action_taken": False}
    tracker = stagnant_trade_tracker[trade_id]
    if -0.15 <= current_r <= 0.15:
        if time.time() - tracker["first_seen"] > 3600:
            if abs(current_r - tracker["last_r"]) < 0.02:
                tracker["r_stable_count"] += 1
                if tracker["r_stable_count"] > 120 and not tracker["action_taken"]:
                    logger.info(f"[STAGNANT] Trade {trade_id} {pair} stagne à R={current_r:.2f} depuis {int((time.time() - tracker['first_seen'])/60)}min")
                    tracker["action_taken"] = True
    else:
        tracker["first_seen"] = time.time()
        tracker["r_stable_count"] = 0
        tracker["action_taken"] = False
    tracker["last_r"] = current_r

# ============================================================
# CHECK BREAK EVEN ET TRAILING
# ============================================================
def check_breakeven_v981():
    try:
        if is_maintenance_suspended():
            logger.debug("⏳ OANDA en maintenance - BE suspendu")
            return
        open_trades = get_open_trades_v88()
        logger.info(f"[BE] Scan de {len(open_trades)} trades ouverts (seuil adaptatif)")
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
            current_price = get_recent_m5_price_v88(pair)
            if current_price <= 0:
                continue
            trade_tracker.update_price(trade_id, current_price)
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
            check_stagnant_trades(trade_id, pair, direction, entry, r)
            pair_params = stats.adaptive_state.get_pair_params(pair)
            effective_threshold = pair_params["be_trigger_r"]
            is_already_be = (direction == "BUY" and current_sl >= entry) or (direction == "SELL" and current_sl <= entry)
            if not is_already_be and r >= effective_threshold:
                logger.info(f"[BE] 🎯 Condition R>={effective_threshold:.2f} atteinte pour {trade_id}")
                if direction == "BUY":
                    be_sl = entry + offset
                else:
                    be_sl = entry - offset
                if (direction == "BUY" and be_sl > current_sl) or (direction == "SELL" and be_sl < current_sl):
                    logger.info(f"[BE] {pair} id={trade_id} R={r:.2f} => SL {current_sl:.5f} -> {be_sl:.5f}")
                    if modify_trade_sl_v981(trade_id, pair, be_sl, adjust_tp=True):
                        logger.info(f"[BE] ✅ SL et TP ajustés avec succès pour {trade_id}")
                        time.sleep(1)
                        _OANDA_CACHE_V88.pop("open_trades_raw", None)
                        current_sl = be_sl
                    else:
                        logger.error(f"[BE] ❌ ÉCHEC modification SL")
                        continue
            trade_details = get_trade_details_v88(trade_id)
            if has_trailing_stop_v88(trade_details):
                logger.debug(f"[TSL] Trade {trade_id} a déjà un trailing, on saute")
                continue
            trailing_activation = pair_params.get("trailing_activation_r", 0.80)
            if r >= trailing_activation:
                logger.info(f"[TSL] R={r:.2f} >= seuil d'activation {trailing_activation:.2f}R - création du trailing")
                atr = get_atr_m15_v88(pair)
                pip_value = get_pip_value_for_pair(pair)
                trailing_mult = pair_params["trailing_atr_mult"]
                trailing_min_pips = pair_params["trailing_min_pips"]
                base_distance = atr * trailing_mult
                r_factor = max(0.6, min(1.4, 1.0 / (1.0 + abs(r) * 0.3)))
                distance = base_distance * r_factor
                distance = max(distance, atr * 0.8)
                distance = min(distance, atr * 2.8)
                distance = max(distance, pip_value * trailing_min_pips)
                distance = round(distance, PRICE_DECIMALS_V88.get(pair, 5))
                if distance > 0:
                    logger.info(f"[TSL] Création du trailing stop pour trade {trade_id}, distance={distance:.5f}")
                    if create_oanda_trailing_stop_v981(trade_id, pair, distance):
                        logger.info(f"[TSL] ✅ Trailing stop créé")
                    else:
                        logger.error(f"[TSL] ❌ ÉCHEC création trailing")
                else:
                    logger.warning(f"[TSL] Distance invalide ({distance})")
            else:
                logger.debug(f"[TSL] R={r:.2f} < seuil d'activation {trailing_activation:.2f}R - pas de trailing pour l'instant")
    except Exception as e:
        logger.error(f"Erreur check_breakeven_v981: {e}")
        logger.error(traceback.format_exc())

# ============================================================
# SUIVI DES TRADES FERMÉS
# ============================================================
def get_closed_trade_details_v110(trade_id: str) -> dict | None:
    api = None
    try:
        api = v88_client()
        try:
            r = trades.TradeDetails(accountID=OANDA_ACCOUNT_ID, tradeID=str(trade_id))
            resp = api.request(r)
            trade_data = resp.get("trade", {}) or {}
            if trade_data:
                logger.info(f"[CLOSE_API] Trade {trade_id} récupéré via TradeDetails")
                return trade_data
        except Exception as e:
            logger.debug(f"[CLOSE_API] TradeDetails indisponible {trade_id}: {e}")
        try:
            params = {"state": "CLOSED", "count": 100}
            r = trades.TradesList(accountID=OANDA_ACCOUNT_ID, params=params)
            resp = api.request(r)
            closed_trades = resp.get("trades", []) or []
            for trade in closed_trades:
                if str(trade.get("id")) == str(trade_id):
                    logger.info(f"[CLOSE_API] Trade {trade_id} récupéré via TradesList CLOSED")
                    return trade
        except Exception as e:
            logger.debug(f"[CLOSE_API] TradesList CLOSED indisponible: {e}")
        logger.warning(f"[CLOSE_API] Trade {trade_id} non trouvé chez OANDA")
        return None
    except Exception as e:
        logger.error(f"[CLOSE_API] Erreur récupération trade {trade_id}: {e}")
        return None

def check_closed_trades():
    global stats, open_trade_details
    try:
        if is_maintenance_suspended():
            logger.debug("OANDA en maintenance - check_closed_trades suspendu")
            return
        current_open_trades = get_open_trades_v88(skip_maintenance_check=True, force_refresh=True)
        current_open_ids = {str(t.get("id")) for t in current_open_trades if t.get("id") is not None}
        for trade_id in list(open_trade_details.keys()):
            trade_id = str(trade_id)
            if trade_id in current_open_ids:
                continue
            trade_info = open_trade_details.pop(trade_id, None)
            if not trade_info:
                continue
            pair = trade_info.get("pair", "UNKNOWN")
            setup_type = trade_info.get("setup_type", "UNKNOWN")
            eqs = trade_info.get("eqs", 0)
            direction = trade_info.get("direction", "").upper()
            entry = trade_info.get("entry")
            sl = trade_info.get("sl")
            units = trade_info.get("units", 0)
            try:
                entry = float(entry)
            except Exception:
                logger.error(f"[CLOSE] Trade {trade_id}: entry invalide={entry}")
                continue
            try:
                sl = float(sl)
            except Exception:
                sl = 0.0
            try:
                units = abs(float(units))
            except Exception:
                units = 0.0
            pip_val = get_pip_value_for_pair(pair)
            risk = abs(entry - sl) if entry > 0 and sl > 0 else 0.0
            risk_pips = risk / pip_val if risk > 0 and pip_val > 0 else 0.0
            close_price = None
            pl = 0.0
            is_estimate = True
            trade_data = {}
            try:
                api = v88_client()
                r = trades.TradeDetails(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id)
                resp = api.request(r)
                trade_data = resp.get("trade", {}) or {}
                if trade_data:
                    average_close_price = trade_data.get("averageClosePrice")
                    realized_pl = trade_data.get("realizedPL")
                    if average_close_price is not None:
                        try:
                            candidate_close = float(average_close_price)
                            if candidate_close > 0:
                                close_price = candidate_close
                                is_estimate = False
                        except (TypeError, ValueError):
                            pass
                    if realized_pl is not None:
                        try:
                            pl = float(realized_pl)
                        except (TypeError, ValueError):
                            logger.warning(f"[CLOSE] realizedPL invalide pour {trade_id}: {realized_pl}")
            except Exception as e:
                logger.warning(f"[CLOSE_API] Impossible de récupérer les détails du trade {trade_id}: {e}")
            if close_price is None or close_price <= 0:
                try:
                    current_price = get_recent_m5_price_v88(pair)
                except Exception as e:
                    logger.error(f"[CLOSE] Impossible de récupérer prix M5 {pair}: {e}")
                    current_price = 0.0
                if current_price is None or current_price <= 0:
                    logger.error(f"[CLOSE] Trade {trade_id} fermé mais impossible de déterminer le prix de sortie")
                    continue
                close_price = float(current_price)
                if direction == "BUY":
                    price_move = close_price - entry
                else:
                    price_move = entry - close_price
                pl = price_move * units if units > 0 else 0.0
                is_estimate = True
                logger.warning(f"[CLOSE_ESTIMATED] {pair} | Trade={trade_id} | Prix M5={close_price:.5f} | PL_ESTIME={pl:.2f}")
            if pip_val <= 0 or risk_pips <= 0:
                r_multiple = 0.0
                logger.warning(f"[CLOSE] Trade {trade_id}: R impossible (risk_pips={risk_pips}, pip={pip_val})")
            else:
                if direction == "BUY":
                    price_move_pips = (close_price - entry) / pip_val
                else:
                    price_move_pips = (entry - close_price) / pip_val
                r_multiple = price_move_pips / risk_pips
            exit_type = "UNKNOWN"
            try:
                if trade_data:
                    if trade_data.get("trailingStopLossOrder"):
                        exit_type = "TRAILING"
                    elif trade_data.get("stopLossOrder"):
                        sl_order = trade_data.get("stopLossOrder", {}) or {}
                        current_sl_order = sl_order.get("price")
                        initial_sl_order = trade_info.get("sl")
                        if current_sl_order is not None and initial_sl_order is not None:
                            try:
                                current_sl_order = float(current_sl_order)
                                initial_sl_order = float(initial_sl_order)
                                tolerance = max(pip_val * 0.5, 0.00001)
                                if abs(current_sl_order - initial_sl_order) > tolerance:
                                    exit_type = "BE_TRAILING"
                                else:
                                    exit_type = "SL_INITIAL"
                            except (TypeError, ValueError):
                                exit_type = "SL_INITIAL"
                        else:
                            exit_type = "SL_INITIAL"
                    else:
                        exit_type = "CLOSE_MANUAL"
            except Exception as e:
                logger.debug(f"[CLOSE] Impossible de déterminer le type de sortie {trade_id}: {e}")
                exit_type = "UNKNOWN"
            logger.info(f"[CLOSE] {pair} | {direction} | Trade={trade_id} | ENTRY={entry:.5f} | CLOSE={close_price:.5f} | R={r_multiple:.2f} | PL={pl:.2f} | Exit={exit_type} | EQS={eqs} | {'ESTIMATED' if is_estimate else 'CONFIRMED'}")
            try:
                stats.record_close(trade_id, pair, setup_type, eqs, r_multiple, pl, close_price, is_estimate, trade_info)
            except Exception as e:
                logger.error(f"[CLOSE] Erreur stats.record_close {trade_id}: {e}")
            try:
                trade_tracker.close_trade(trade_id, close_price, r_multiple)
            except Exception as e:
                logger.error(f"[CLOSE] Erreur trade_tracker {trade_id}: {e}")
    except Exception as e:
        logger.error(f"Erreur lors du check des trades fermés: {e}")
        logger.error(traceback.format_exc())

# ============================================================
# SIGNALS SENT TRACKING
# ============================================================
sent_signals = {}

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

# ============================================================
# LOGGING HELPERS
# ============================================================
_seen_log_keys_fvg_recent = set()
_seen_log_keys_fvg_added = set()
_seen_log_keys_kept_entry = set()

def _reset_log_dedup():
    _seen_log_keys_fvg_recent.clear()
    _seen_log_keys_fvg_added.clear()
    _seen_log_keys_kept_entry.clear()

# ============================================================
# TELEGRAM
# ============================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = TELEGRAM_BOT_TOKEN is not None and TELEGRAM_CHAT_ID is not None

def send_telegram_alert(pair: str, direction: str, entry_price: float, stop_loss: float, take_profit: float, narrative: dict, bias_analysis: dict, rsi: float = 50, entry_type: str = "UNKNOWN", confidence_score: int = 0, eqs_score: int = 0):
    if not TELEGRAM_ENABLED:
        return
    try:
        direction_emoji = "🟢" if direction == "BUY" else "🔴"
        entry_type_display = entry_type if entry_type else "UNKNOWN"
        msg = f"{direction_emoji} TRADE OPPORTUNITY\nPair: {pair}\nDirection: {direction}\nEntry: {entry_price:.5f}\nSL: {stop_loss:.5f}\nTP: {take_profit:.5f}\nSetup: {entry_type_display}\nRSI: {rsi:.1f}\n"
        if confidence_score > 0:
            msg += f"Confiance: {confidence_score}%\n"
        if eqs_score > 0:
            msg += f"EQS: {eqs_score}/100\n"
        if bias_analysis:
            bias = bias_analysis.get("bias", "NEUTRAL")
            msg += f"Biais: {bias}\n"
            if "win_rate" in bias_analysis:
                msg += f"Win Rate estimé: {bias_analysis['win_rate']}\n"
            if "quality_label" in bias_analysis:
                msg += f"Qualité: {bias_analysis['quality_label']}\n"
        rr = abs((take_profit - entry_price) / (entry_price - stop_loss)) if entry_price != stop_loss else 0
        msg += f"RR: {rr:.2f}\nTrade ID: {narrative.get('trade_id', 'N/A')}" if narrative else f"RR: {rr:.2f}\nTrade ID: N/A"
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
        response = requests.post(url, data=data, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Telegram erreur: {response.status_code}")
    except Exception as e:
        logger.error(f"Erreur envoi Telegram: {e}")

# ============================================================
# ADVANCED MAIN SIMPLIFIÉE
# ============================================================
def advanced_main_v981():
    try:
        api = v88_client()
        logger.info("✅ API OANDA initialisée avec succès")
        logger.info("🎯 NOUVEAU MODE SIMPLIFIÉ : Biais → Retracement → Confirmation → 2R")
        logger.info(f"📊 MAX TRADES: {MAX_TRADES_TOTAL}")
        logger.info(f"🔒 VERROUILLAGE PAR PAIRE: {EXECUTION_COOLDOWN_SECONDS}s")
    except Exception as e:
        logger.error(f"❌ Échec d'initialisation de l'API OANDA : {e}")
        return
    for pair in PAIR_LIST:
        _reset_log_dedup()
        if not stats.adaptive_state.can_trade(pair):
            logger.info(f"[COOLDOWN] {pair} - en cooldown après une perte, scan ignoré")
            continue
        if stats.adaptive_state.is_pair_suspended(pair):
            logger.info(f"[SUSPEND] {pair} est suspendue - scan ignoré")
            continue
        if has_open_trade_v88(pair):
            logger.info(f"[INFO] {pair}: trade déjà ouvert - scan ignoré")
            continue
        try:
            df_h4 = get_candles_with_retry(api, pair, GRANULARITY_H4, 300)
            df_h1 = get_candles_with_retry(api, pair, GRANULARITY_H1, 200)
            df_m15 = get_candles_with_retry(api, pair, GRANULARITY_M15, 250)
            df_d1 = get_candles_with_retry(api, pair, "D", count=250)  # non utilisé
            if any(df.empty for df in [df_h4, df_h1, df_m15]):
                logger.warning(f"⚠️ Données manquantes pour {pair}, analyse ignorée")
                continue
            current_price = float(df_m15["close"].iloc[-1])
            bias = get_directional_bias(df_h4, df_h1)
            if bias == "NEUTRAL":
                logger.info(f"{pair} | Régime NEUTRAL → pas de trade")
                continue
            setups = detect_setups_aligned_with_bias(df_m15, df_h1, bias, pair, df_h4)
            if not setups:
                logger.info(f"{pair} | Aucun setup {bias} détecté")
                continue
            valid_trades = []
            for entry in setups:
                result = evaluate_simple_setup(pair, bias, entry, df_h4, df_h1, df_m15, current_price)
                if result["passed"]:
                    valid_trades.append({
                        "entry": entry,
                        "sl": result["sl"],
                        "tp": result["tp"],
                        "risk_pips": result["risk_pips"],
                        "rr": result["rr"],
                        "atr_pips": result["atr_pips"],
                        "metrics": result["metrics"]
                    })
                else:
                    logger.info(f"[REJECT] {pair} {bias} {entry.get('type')} : {result['reason']}")
            if not valid_trades:
                logger.info(f"{pair} | Aucun trade valide après filtres")
                continue
            valid_trades.sort(key=lambda x: x["rr"], reverse=True)
            best = valid_trades[0]
            entry_level = float(best["entry"]["entry_level"])
            stop_loss = best["sl"]
            take_profit = best["tp"]
            rr = best["rr"]
            metrics = best["metrics"]
            if ONE_TRADE_PER_PAIR and has_open_trade_v88(pair):
                logger.info(f"{pair}: trade déjà ouvert - annulation")
                continue
            if open_trade_count_v88() >= MAX_TRADES_TOTAL:
                logger.info(f"Limite trades atteinte ({MAX_TRADES_TOTAL})")
                continue
            if is_maintenance_suspended():
                logger.warning(f"[ORDER] OANDA en maintenance - exécution annulée")
                continue
            trade_id = execute_oanda_trade_v981(
                pair=pair, direction=bias, entry_price=entry_level,
                stop_loss=stop_loss, take_profit=take_profit,
                score=0, entry_type=best["entry"]["type"], eqs=0,
                setup_type=best["entry"]["type"], metrics=metrics, rr=rr
            )
            if trade_id:
                logger.info(f"[DECISION_EXECUTED] {pair} | {bias} | {best['entry']['type']} | ENTRY={entry_level:.5f} | SL={stop_loss:.5f} | TP={take_profit:.5f} | RR={rr:.2f} | TRADE_ID={trade_id} | ACTION=EXECUTED")
                bias_analysis = {"bias": bias, "win_rate": "~55% (estimé)", "quality_label": "SIMPLE_SETUP", "score_details": {"rr": rr, "risk_pips": best["risk_pips"]}}
                send_telegram_alert(pair, bias, entry_level, stop_loss, take_profit, {}, bias_analysis, metrics.get("rsi", 50), best["entry"]["type"], 0, 0)
                zone_start, zone_end = best["entry"].get("entry_zone", (entry_level, entry_level))
                mark_signal_sent(pair, bias, entry_level, zone_start, zone_end)
                entry_metrics = {
                    "atr": best["atr_pips"],
                    "adx": metrics.get("adx", 0),
                    "rsi": metrics.get("rsi", 0),
                    "eqs": 0,
                    "hour": datetime.utcnow().hour,
                    "weekday": datetime.utcnow().weekday(),
                    "setup_type": best["entry"]["type"],
                    "momentum": metrics.get("momentum", 0),
                    "spread": 0,
                    "volatility": 0,
                    "session": metrics.get("session", "UNKNOWN"),
                    "h1_trend": 0,
                    "h4_trend": 0
                }
                stats.record_signal(pair, True, "trade_opened", entry_level, stop_loss, take_profit, 0, bias, entry_metrics)
                logger.info(f"✅ {pair}: trade exécuté (ID {trade_id})")
            else:
                logger.error(f"❌ {pair}: échec d'exécution du trade")
        except Exception as e:
            logger.error(f"💥 Erreur sur {pair} : {str(e)}")
            logger.error(traceback.format_exc())
            continue
    stats.log_summary()

# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
if __name__ == "__main__":
    logger.info("🚀 Démarrage du Bot Advanced Orderflow Trading - V111 (STRATÉGIE SIMPLIFIÉE)")
    logger.info("✅ Utilisation de TradeCRCDO pour la modification du SL")
    logger.info("✅ Utilisation de OrderCreate pour la création du Trailing Stop")
    logger.info(f"✅ Seuil Break Even adaptatif (base: {BASE_BREAKEVEN_TRIGGER_R}R)")
    logger.info(f"✅ Seuil Break Even anticipé adaptatif (base: {BASE_BREAKEVEN_EARLY_R}R)")
    logger.info(f"✅ Trailing stop optimisé (activation {BASE_TRAILING_ACTIVATION_R}R, distance {BASE_TRAILING_STOP_DISTANCE_ATR_MULTIPLIER}R)")
    logger.info(f"✅ MAX TRADES: {MAX_TRADES_TOTAL}")
    logger.info("🔄 DOUBLE BOUCLE : rapide (30s) pour BE/Trailing, lente (15min) pour les signaux")
    logger.info("📈 SUIVI DES CLÔTURES : tentative de récupération via TradeDetails + fallback")
    logger.info("📊 ESPÉRANCE CALCULÉE SUR LES TRADES CLÔTURÉS")
    logger.info("📈 SUIVI MFE/MAE ACTIVÉ AVEC LOGS ENRICHIS")
    logger.info("🔧 APPELS OANDA CORRIGÉS")
    if DEMO_MODE:
        logger.info("🔬 MODE DEMO ACTIVÉ")
    if DEBUG_MODE:
        logger.info("🔍 MODE DEBUG ACTIVÉ")

    last_signal_scan = time.time()
    SIGNAL_SCAN_INTERVAL = 900
    FAST_LOOP_INTERVAL = 30
    maintenance_mode = False

    while True:
        try:
            now = time.time()
            if is_maintenance_suspended():
                if not maintenance_mode:
                    maintenance_mode = True
                    logger.warning("🔧 BOT EN MODE MAINTENANCE - appels suspendus")
                time.sleep(10)
                continue
            if maintenance_mode:
                maintenance_mode = False
                logger.info("🔧 FIN DU MODE MAINTENANCE - reprise normale")
            clear_scan_cache_v88()
            current_open_count = open_trade_count_v88()
            logger.info(f"[SCAN] Trades ouverts: {current_open_count}/{MAX_TRADES_TOTAL}")
            check_closed_trades()
            check_breakeven_v981()
            if now - last_signal_scan >= SIGNAL_SCAN_INTERVAL:
                logger.info(f"⏰ Scan des signaux V111")
                last_signal_scan = now
                now_dt = datetime.utcnow()
                if not is_market_open_utc_v88(now_dt):
                    logger.info("Marché fermé.")
                elif current_open_count >= MAX_TRADES_TOTAL:
                    logger.info(f"Limite trades atteinte ({MAX_TRADES_TOTAL})")
                else:
                    advanced_main_v981()
            time.sleep(FAST_LOOP_INTERVAL)
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt demandé")
            break
        except Exception as e:
            if is_oanda_in_maintenance(e):
                logger.warning(f"🔧 Maintenance OANDA détectée: {e}")
                handle_api_error(e)
                time.sleep(5)
                continue
            logger.error(f"💥 Erreur critique: {e}")
            traceback.print_exc()
            time.sleep(30)
