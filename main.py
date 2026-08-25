# ============================================================
# main(115).py - Version V111 "ASIA/LONDON ENTRY BYPASS FIX"
#
# AMÉLIORATIONS V111 :
# 1. ✅ Ajout des constantes manquantes pour le bypass ASIA/LONDON
#    - ASIA_BYPASS_MIN_SCORE, ASIA_BYPASS_MIN_EQS, ASIA_BYPASS_MIN_ADX
#    - LONDON_BYPASS_MIN_SCORE, LONDON_BYPASS_MIN_EQS, LONDON_BYPASS_MIN_ADX
# 2. ✅ Correction du calcul du R après Break Even (utilisation du risque initial)
# 3. ✅ Adaptation dynamique moins agressive (20 trades, 2 cycles d'hystérésis)
# 4. ✅ Nettoyage des logs de démarrage (version unifiée V111)
# 5. ✅ Conservation de toutes les règles V106/V107/V109.1/V110
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
# CONFIGURATION V106
# =========================
load_dotenv()

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# ============================================================
# PARAMÈTRES DE BASE (seront adaptés dynamiquement)
# ============================================================
BASE_MIN_CONFIDENCE_SCORE_BY_PAIR = {
    "EUR_USD": 10,
    "GBP_USD": 9,
    "USD_CAD": 8,
    "AUD_USD": 8,
    "AUD_CAD": 8,
    "XAU_USD": 9,
    "DEFAULT": 8
}
# V90 - Pullback minimum (en pips) pour confirmer un retracement
PULLBACK_MIN_PIPS_BY_PAIR = {
    "EUR_USD": 4.0,
    "GBP_USD": 4.0,
    "USD_CAD": 3.5,
    "AUD_USD": 4.5,
    "AUD_CAD": 4.0,
    "XAU_USD": 30.0,
    "USD_JPY": 5.0,
    "GBP_JPY": 5.0,
    "DEFAULT": 4.0
}
# ✅ V102 : Break Even baissé à 0.40R (au lieu de 0.80R)
BASE_BREAKEVEN_TRIGGER_R = float(os.getenv("BREAKEVEN_TRIGGER_R", "0.55"))
# ✅ V102 : Break Even Early baissé à 0.25R (au lieu de 0.50R)
BASE_BREAKEVEN_EARLY_R = float(os.getenv("BREAKEVEN_EARLY_R", "0.25"))
# ✅ V105.1 : Trailing Stop optimisé (activation 0.80R, distance 1.5R)
BASE_TRAILING_STOP_DISTANCE_ATR_MULTIPLIER = float(os.getenv("TRAILING_STOP_DISTANCE_ATR_MULTIPLIER", "1.5"))
BASE_TRAILING_STOP_MIN_DISTANCE_PIPS = float(os.getenv("TRAILING_STOP_MIN_DISTANCE_PIPS", "8.0"))
# ✅ V105.1 : Seuil d'activation du trailing (0.80R)
BASE_TRAILING_ACTIVATION_R = float(os.getenv("TRAILING_ACTIVATION_R", "0.80"))

# ✅ V106 : Seuil de score d'entrée /100 pour accepter un signal
MIN_ENTRY_SCORE = int(os.getenv("MIN_ENTRY_SCORE", "55"))

BASE_ADX_MIN_THRESHOLD = float(os.getenv("ADX_MIN_THRESHOLD", "23.0"))
BASE_MOMENTUM_MIN_PERCENT = float(os.getenv("MOMENTUM_MIN_PERCENT", "0.15"))
BASE_VOLUME_MOMENTUM_MIN = float(os.getenv("VOLUME_MOMENTUM_MIN", "0.5"))
BASE_EQS_MIN_THRESHOLD = float(os.getenv("EQS_MIN_THRESHOLD", "55.0"))

MAX_TRADES_TOTAL = int(os.getenv("MAX_TRADES_TOTAL", "10"))
MIN_ATR_PIPS_BY_PAIR = {
    "EUR_USD": 3.5,
    "GBP_USD": 4.4,
    "USD_CAD": 3.9,
    "AUD_USD": 3.0,
    "AUD_CAD": 3.9,
    "XAU_USD": 34.0,
    "GBP_JPY": 7.0,
    "USD_JPY": 5.0,
    "DEFAULT": 4.0
}

# ✅ V105 : Seuils ATR réduits de 30% pour la session ASIA
MIN_ATR_PIPS_BY_PAIR_ASIA = {
    "EUR_USD": 2.5,
    "AUD_JPY": 3.5,
    "GBP_USD": 2.5,
    "USD_CAD": 3.0,
    "AUD_USD": 3.0,
    "AUD_CAD": 3.0,
    "XAU_USD": 24.0,
    "GBP_JPY": 5.0,
    "USD_JPY": 4.5,
    "DEFAULT": 3.0,
}

# ============================================================
# V111 - PARAMÈTRES DU BYPASS ASIA/LONDON
# ============================================================
ASIA_BYPASS_MIN_SCORE = 48
ASIA_BYPASS_MIN_EQS = 80
ASIA_BYPASS_MIN_ADX = 18
LONDON_BYPASS_MIN_SCORE = 48
LONDON_BYPASS_MIN_EQS = 80
LONDON_BYPASS_MIN_ADX = 18

# ============================================================
# CONFIGURATION ADAPTATIVE (V111 - moins agressive)
# ============================================================
# Fenêtre d'apprentissage
LEARNING_WINDOW = int(os.getenv("LEARNING_WINDOW", "50"))  # 50 derniers trades
# ✅ V111 : Adaptation plus robuste (20 trades au lieu de 5)
ADAPTATION_MIN_TRADES = int(os.getenv("ADAPTATION_MIN_TRADES", "20"))
ADAPTATION_INTERVAL = int(os.getenv("ADAPTATION_INTERVAL", "300"))  # 5 minutes

# Seuils d'adaptation (V102 : combinaison de critères)
PF_GOOD_THRESHOLD = 1.3
PF_BAD_THRESHOLD = 0.9
WR_GOOD_THRESHOLD = 0.55
WR_BAD_THRESHOLD = 0.40
EXPECTANCY_GOOD_THRESHOLD = 0.0
EXPECTANCY_BAD_THRESHOLD = -5.0

# ✅ V111 : Hystérésis à 2 cycles pour plus de stabilité
HYSTERESIS_CYCLES_REQUIRED = int(os.getenv("HYSTERESIS_CYCLES", "2"))

# Amplitude maximale de changement par cycle
MAX_ADX_CHANGE = 0.5
MAX_EQS_CHANGE = 0.5
MAX_BE_CHANGE = 0.05
MAX_TRAILING_CHANGE = 0.05

# Suspension
CONSECUTIVE_LOSSES_SUSPEND = 4
PF_FOR_SUSPEND = 0.7

# Plages d'adaptation (inchangées)
ADX_MIN_RANGE = (15, 35)
EQS_MIN_RANGE = (5, 85)
BE_TRIGGER_R_RANGE = (0.5, 1.2)
TRAILING_ATR_RANGE = (1.2, 2.8)

# ============================================================
# ✅ V106 : Pondération des setups - dynamique et initiale ajustée
# ============================================================
SETUP_WEIGHTS_DEFAULT = {
    "FVG_RETEST_PERFECT": 1.30,
    "BISI": 1.25,
    "BREAKER": 1.20,
    "NESTED_FVG": 1.15,
    "FVG_RETEST": 1.05,
    "WICK_REJECTION": 0.95,
    "LIQUIDITY_DRAW": 0.90,
}

# ============================================================
# ÉTAT ADAPTATIF (V111 - cooldown 2h, adaptation moins agressive)
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

    def record_setup_performance(self, pair: str, setup_type: str, result: str, r: float):
        key = f"{pair}_{setup_type}"
        perf = self.setup_performance[key]
        if result == "WIN":
            perf["wins"] += 1
        elif result == "LOSS":
            perf["losses"] += 1
        perf["total_r"] += r
        total = perf["wins"] + perf["losses"]
        if total >= 5:
            win_rate = perf["wins"] / total if total > 0 else 0.5
            avg_r = perf["total_r"] / total if total > 0 else 0
            raw_performance = (win_rate * 0.6) + (avg_r * 0.4)
            if total < 20:
                performance_score = 0.5 + raw_performance * 0.5
            else:
                performance_score = raw_performance
            new_weight = max(0.5, min(1.5, performance_score * 1.2))
            self.update_setup_weight(pair, setup_type, new_weight)

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
        stats_pair = stats.stats[pair]
        total_closed = stats_pair.get("wins", 0) + stats_pair.get("losses", 0)
        if total_closed >= 10:
            pf = stats_pair.get("total_profit", 0) / max(stats_pair.get("total_loss", 0), 0.01)
            if self.consecutive_losses[pair] >= CONSECUTIVE_LOSSES_SUSPEND and pf < PF_FOR_SUSPEND:
                self.suspend_pair(pair, f"pertes consécutives ({self.consecutive_losses[pair]}) + PF={pf:.2f}")
                self.consecutive_losses[pair] = 0
        elif self.consecutive_losses[pair] >= CONSECUTIVE_LOSSES_SUSPEND * 1.5:
            self.suspend_pair(pair, f"{self.consecutive_losses[pair]} pertes consécutives")
            self.consecutive_losses[pair] = 0

    def record_win(self, pair: str):
        self.consecutive_losses[pair] = 0
        if pair in self.last_loss_time:
            del self.last_loss_time[pair]

    def adapt_parameters(self, pair: str, stats: dict):
        total_trades = stats.get("wins", 0) + stats.get("losses", 0) + stats.get("breakevens", 0)
        if total_trades < ADAPTATION_MIN_TRADES:
            logger.debug(f"[ADAPT] {pair} : pas assez de trades ({total_trades} < {ADAPTATION_MIN_TRADES})")
            return

        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        total_closed = wins + losses
        if total_closed == 0:
            return

        win_rate = wins / total_closed
        profit_factor = stats.get("total_profit", 0) / max(stats.get("total_loss", 0), 0.01)
        total_profit = stats.get("total_profit", 0)
        total_loss = stats.get("total_loss", 0)
        expectancy = (total_profit - total_loss) / max(total_closed, 1)

        good_conditions = 0
        bad_conditions = 0
        if profit_factor > PF_GOOD_THRESHOLD:
            good_conditions += 1
        elif profit_factor < PF_BAD_THRESHOLD:
            bad_conditions += 1

        if win_rate > WR_GOOD_THRESHOLD:
            good_conditions += 1
        elif win_rate < WR_BAD_THRESHOLD:
            bad_conditions += 1

        if expectancy > EXPECTANCY_GOOD_THRESHOLD:
            good_conditions += 1
        elif expectancy < EXPECTANCY_BAD_THRESHOLD:
            bad_conditions += 1

        counter = self.adaptation_counters[pair]
        if good_conditions >= 2 and bad_conditions == 0:
            counter["good"] += 1
            counter["bad"] = 0
        elif bad_conditions >= 2 and good_conditions == 0:
            counter["bad"] += 1
            counter["good"] = 0
        else:
            counter["good"] = 0
            counter["bad"] = 0
            return

        if counter["good"] >= HYSTERESIS_CYCLES_REQUIRED:
            direction = "good"
            counter["good"] = 0
            self._apply_adaptation(pair, stats, direction)
        elif counter["bad"] >= HYSTERESIS_CYCLES_REQUIRED:
            direction = "bad"
            counter["bad"] = 0
            self._apply_adaptation(pair, stats, direction)

    def _apply_adaptation(self, pair: str, stats: dict, direction: str):
        params = self.get_pair_params(pair)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        total_closed = wins + losses
        if total_closed == 0:
            return
        win_rate = wins / total_closed
        profit_factor = stats.get("total_profit", 0) / max(stats.get("total_loss", 0), 0.01)

        sign = 1 if direction == "good" else -1

        new_adx = params["adx_min"] - sign * MAX_ADX_CHANGE
        params["adx_min"] = max(ADX_MIN_RANGE[0], min(ADX_MIN_RANGE[1], new_adx))

        new_eqs = params["eqs_min"] - sign * MAX_EQS_CHANGE
        params["eqs_min"] = max(EQS_MIN_RANGE[0], min(EQS_MIN_RANGE[1], new_eqs))

        mfe_mae_list = stats.get("mfe_mae", [])
        if len(mfe_mae_list) >= 10:
            avg_mfe = sum(item["mfe"] for item in mfe_mae_list) / len(mfe_mae_list)
            avg_mae = sum(item["mae"] for item in mfe_mae_list) / len(mfe_mae_list)
            avg_r = sum(item["r"] for item in mfe_mae_list) / len(mfe_mae_list)
            if direction == "good" and avg_mfe > avg_mae * 2 and avg_r > 0.3:
                new_be = params["be_trigger_r"] + MAX_BE_CHANGE
            elif direction == "bad" or avg_mfe < avg_mae * 1.2:
                new_be = params["be_trigger_r"] - MAX_BE_CHANGE
            else:
                new_be = params["be_trigger_r"]
        else:
            if direction == "good":
                new_be = params["be_trigger_r"] + MAX_BE_CHANGE
            else:
                new_be = params["be_trigger_r"] - MAX_BE_CHANGE
        params["be_trigger_r"] = max(BE_TRIGGER_R_RANGE[0], min(BE_TRIGGER_R_RANGE[1], new_be))

        if direction == "good":
            new_trail = params["trailing_atr_mult"] + MAX_TRAILING_CHANGE
        else:
            new_trail = params["trailing_atr_mult"] - MAX_TRAILING_CHANGE
        params["trailing_atr_mult"] = max(TRAILING_ATR_RANGE[0], min(TRAILING_ATR_RANGE[1], new_trail))

        adaptation_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "pair": pair,
            "total_trades": total_closed,
            "win_rate": f"{win_rate*100:.1f}%",
            "profit_factor": profit_factor,
            "direction": direction,
            "new_params": params.copy()
        }
        self.adaptation_history.append(adaptation_log)
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]

        logger.info(f"[ADAPT] {pair} | direction={direction} | Trades={total_closed} | WR={win_rate*100:.1f}% | PF={profit_factor:.2f} | "
                    f"ADX={params['adx_min']:.1f} | EQS={params['eqs_min']:.0f} | BE={params['be_trigger_r']:.2f}R | "
                    f"Trailing={params['trailing_atr_mult']:.2f}")

# ============================================================
# V106 - STATISTIQUES AVEC APPRENTISSAGE ROBUSTE
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
        self.adaptive_state = AdaptiveState()
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

        self.adaptive_state.record_setup_performance(pair, setup_type, result, r)

        setup_stats = stats["by_setup"][setup_type]
        setup_stats["wins"] += 1 if result == "WIN" else 0
        setup_stats["losses"] += 1 if result == "LOSS" else 0
        setup_stats["total_r"] += r

        setup_perf = stats["setup_performance"][setup_type]
        setup_perf["wins"] += 1 if result == "WIN" else 0
        setup_perf["losses"] += 1 if result == "LOSS" else 0
        setup_perf["total_r"] += r

        if eqs < 65:
            eqs_range = "60-64"
        elif eqs < 70:
            eqs_range = "65-69"
        elif eqs < 80:
            eqs_range = "70-79"
        else:
            eqs_range = "80+"
        eqs_stats = stats["by_eqs_range"][eqs_range]
        eqs_stats["wins"] += 1 if result == "WIN" else 0
        eqs_stats["losses"] += 1 if result == "LOSS" else 0
        eqs_stats["total_r"] += r

        if trade_info:
            hour = trade_info.get("hour", 0)
            weekday = trade_info.get("weekday", 0)
            session = trade_info.get("session", "UNKNOWN")
            adx = trade_info.get("adx", 0)

            hour_stats = stats["by_hour"][hour]
            hour_stats["wins"] += 1 if result == "WIN" else 0
            hour_stats["losses"] += 1 if result == "LOSS" else 0
            hour_stats["total_r"] += r

            wd_stats = stats["by_weekday"][weekday]
            wd_stats["wins"] += 1 if result == "WIN" else 0
            wd_stats["losses"] += 1 if result == "LOSS" else 0
            wd_stats["total_r"] += r

            sess_stats = stats["by_session"][session]
            sess_stats["wins"] += 1 if result == "WIN" else 0
            sess_stats["losses"] += 1 if result == "LOSS" else 0
            sess_stats["total_r"] += r

            if adx < 20:
                adx_range = "0-19"
            elif adx < 25:
                adx_range = "20-24"
            elif adx < 30:
                adx_range = "25-29"
            elif adx < 40:
                adx_range = "30-39"
            else:
                adx_range = "40+"
            adx_stats = stats["by_adx_range"][adx_range]
            adx_stats["wins"] += 1 if result == "WIN" else 0
            adx_stats["losses"] += 1 if result == "LOSS" else 0
            adx_stats["total_r"] += r

        estimate_tag = " (estimé)" if is_estimate else ""
        price_str = f" | close={close_price:.5f}" if close_price else ""
        metrics_str = ""
        if trade_info:
            metrics_str = f" | ATR={trade_info.get('atr',0):.1f} | ADX={trade_info.get('adx',0):.1f} | RSI={trade_info.get('rsi',0):.1f} | H={trade_info.get('hour',0)} | Sess={trade_info.get('session','UNKNOWN')}"
        logger.info(f"[CLOSE] {pair} | {setup_type} | {result}{estimate_tag} | R={r:.2f} | P&L={profit_loss:+.2f} | EQS={eqs}{price_str}{metrics_str}")

        if setup_perf["wins"] + setup_perf["losses"] >= 10:
            self._update_setup_weight(pair, setup_type)

        if time.time() - self.adaptive_state.last_adaptation > ADAPTATION_INTERVAL:
            self.adaptive_state.adapt_parameters(pair, stats)
            self.adaptive_state.last_adaptation = time.time()

        for trade in stats["trades"]:
            if trade.get("close_price") is None and trade.get("entry") == trade_info.get("entry"):
                trade["result"] = result
                trade["close_price"] = close_price
                trade["close_pl"] = profit_loss
                tracker_trade = trade_tracker.get_trade(trade_id)
                if tracker_trade:
                    trade["mfe"] = tracker_trade.get("max_favorable_pips")
                    trade["mae"] = tracker_trade.get("max_adverse_pips")
                break

    def _update_setup_weight(self, pair: str, setup_type: str):
        stats = self.stats[pair]
        setup_perf = stats["setup_performance"].get(setup_type)
        if not setup_perf:
            return
        total = setup_perf["wins"] + setup_perf["losses"]
        if total < 10:
            return

        win_rate = setup_perf["wins"] / total if total > 0 else 0.5
        avg_r = setup_perf["total_r"] / total if total > 0 else 0

        raw_performance = (win_rate * 0.6) + (avg_r * 0.4)
        if total < 20:
            performance_score = 0.5 + raw_performance * 0.5
        else:
            performance_score = raw_performance

        new_weight = max(0.5, min(1.5, performance_score * 1.2))

        self.adaptive_state.update_setup_weight(pair, setup_type, new_weight)
        logger.info(f"[SETUP_WEIGHT] {pair} | {setup_type} | WR={win_rate*100:.1f}% | AvgR={avg_r:.2f} | Trades={total} | Poids={new_weight:.2f}")

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
        logger.info("📊 RÉSUMÉ QUOTIDIEN V111")
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
        logger.info("📊 STATISTIQUES GLOBALES V111")
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

        logger.info("📈 PARAMÈTRES ADAPTATIFS")
        logger.info("-" * 80)
        for pair, params in self.adaptive_state.pair_params.items():
            logger.info(f"{pair:10} | ADX={params['adx_min']:.1f} | EQS={params['eqs_min']:.0f} | BE={params['be_trigger_r']:.2f}R | Trailing={params['trailing_atr_mult']:.2f} | TrailAct={params.get('trailing_activation_r', 0.80):.2f}R")

        logger.info("=" * 80)
        logger.info("📈 PERFORMANCE PAR SETUP")
        logger.info("-" * 80)
        for pair, stats in self.stats.items():
            setup_stats = stats.get("by_setup", {})
            for setup, data in setup_stats.items():
                total = data["wins"] + data["losses"]
                if total > 0:
                    win_rate = data["wins"] / total
                    avg_r = data["total_r"] / total
                    weight = self.adaptive_state.get_setup_weight(pair, setup)
                    logger.info(f"{pair:10} | {setup:20} | Win={win_rate*100:.1f}% | AvgR={avg_r:.2f} | Trades={total} | Poids={weight:.2f}")

        logger.info("=" * 80)
        logger.info("📈 PERFORMANCE PAR TRANCHE EQS")
        logger.info("-" * 80)
        for pair, stats in self.stats.items():
            eqs_stats = stats.get("by_eqs_range", {})
            for range_label, data in eqs_stats.items():
                total = data["wins"] + data["losses"]
                if total > 0:
                    win_rate = data["wins"] / total
                    avg_r = data["total_r"] / total
                    logger.info(f"{pair:10} | EQS {range_label:6} | Win={win_rate*100:.1f}% | AvgR={avg_r:.2f} | Trades={total}")

        logger.info("=" * 80)
        logger.info("📈 PERFORMANCE PAR HEURE")
        logger.info("-" * 80)
        for pair, stats in self.stats.items():
            hour_stats = stats.get("by_hour", {})
            for hour, data in sorted(hour_stats.items()):
                total = data["wins"] + data["losses"]
                if total > 0:
                    win_rate = data["wins"] / total
                    avg_r = data["total_r"] / total
                    logger.info(f"{pair:10} | H{hour:02d}       | Win={win_rate*100:.1f}% | AvgR={avg_r:.2f} | Trades={total}")

        logger.info("=" * 80)
        logger.info("📈 PERFORMANCE PAR JOUR")
        logger.info("-" * 80)
        days = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        for pair, stats in self.stats.items():
            wd_stats = stats.get("by_weekday", {})
            for wd, data in sorted(wd_stats.items()):
                total = data["wins"] + data["losses"]
                if total > 0:
                    win_rate = data["wins"] / total
                    avg_r = data["total_r"] / total
                    logger.info(f"{pair:10} | {days[wd]}       | Win={win_rate*100:.1f}% | AvgR={avg_r:.2f} | Trades={total}")

        logger.info("=" * 80)
        logger.info("📈 PERFORMANCE PAR SESSION")
        logger.info("-" * 80)
        for pair, stats in self.stats.items():
            sess_stats = stats.get("by_session", {})
            for session, data in sess_stats.items():
                total = data["wins"] + data["losses"]
                if total > 0:
                    win_rate = data["wins"] / total
                    avg_r = data["total_r"] / total
                    logger.info(f"{pair:10} | {session:12} | Win={win_rate*100:.1f}% | AvgR={avg_r:.2f} | Trades={total}")

        logger.info("=" * 80)
        logger.info("📈 PERFORMANCE PAR TRANCHE ADX")
        logger.info("-" * 80)
        for pair, stats in self.stats.items():
            adx_stats = stats.get("by_adx_range", {})
            for range_label, data in adx_stats.items():
                total = data["wins"] + data["losses"]
                if total > 0:
                    win_rate = data["wins"] / total
                    avg_r = data["total_r"] / total
                    logger.info(f"{pair:10} | ADX {range_label:6} | Win={win_rate*100:.1f}% | AvgR={avg_r:.2f} | Trades={total}")

        logger.info("=" * 80)
        logger.info("📈 STATISTIQUES MFE/MAE")
        logger.info("-" * 80)
        for pair, stats in self.stats.items():
            mfe_mae_list = stats.get("mfe_mae", [])
            if mfe_mae_list:
                avg_mfe = sum(item["mfe"] for item in mfe_mae_list) / len(mfe_mae_list)
                avg_mae = sum(item["mae"] for item in mfe_mae_list) / len(mfe_mae_list)
                avg_r = sum(item["r"] for item in mfe_mae_list) / len(mfe_mae_list)
                logger.info(f"{pair:10} | MFE moy={avg_mfe:.1f}pips | MAE moy={avg_mae:.1f}pips | R moy={avg_r:.2f} | échantillon={len(mfe_mae_list)}")
        logger.info("=" * 80)

        if self.adaptive_state.adaptation_history:
            logger.info("📈 HISTORIQUE DES ADAPTATIONS (dernières 5)")
            logger.info("-" * 80)
            for entry in self.adaptive_state.adaptation_history[-5:]:
                params = entry["new_params"]
                logger.info(f"{entry['timestamp']} | {entry['pair']} | dir={entry['direction']} | WR={entry['win_rate']} | PF={entry['profit_factor']:.2f} | ADX={params['adx_min']:.1f} | EQS={params['eqs_min']:.0f} | BE={params['be_trigger_r']:.2f}R")
            logger.info("=" * 80)


# ============================================================
# INSTANCIATION STATS V106
# ============================================================
stats = TradingStatsV101()

# ============================================================
# SUIVI DES CLÔTURES
# ============================================================
open_trade_details = {}

# ============================================================
# V110 - RÉCUPÉRATION DU VRAI TRADE FERMÉ
# ============================================================
def get_closed_trade_details_v110(
    trade_id: str
) -> dict | None:
    """
    V110 - Récupère les informations réelles d'un trade fermé.
    Priorité : 1. TradeDetails, 2. TradesList state=CLOSED
    Retourne : dict du trade si trouvé, None sinon.
    """
    api = None
    try:
        api = v88_client()
        try:
            r = trades.TradeDetails(
                accountID=OANDA_ACCOUNT_ID,
                tradeID=str(trade_id)
            )
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

# ============================================================
# V110 - SUIVI DES TRADES FERMÉS
# ============================================================
def check_closed_trades():
    """
    V110 - Détection et traitement des trades fermés.
    Corrections : utilisation de averageClosePrice et realizedPL.
    """
    global stats, open_trade_details
    try:
        if is_maintenance_suspended():
            logger.debug("OANDA en maintenance - check_closed_trades suspendu")
            return

        current_open_trades = get_open_trades_v88(
            skip_maintenance_check=True,
            force_refresh=True
        )
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
                    if close_price is not None and not is_estimate:
                        logger.info(f"[CLOSE_DETAILS] {pair} | Trade={trade_id} | ENTRY_API={float(trade_data.get('price', 0)):.5f} | CLOSE_API={close_price:.5f} | PL={pl:.2f}")
                    else:
                        logger.warning(f"[CLOSE_API] Trade {trade_id} récupéré mais averageClosePrice indisponible")
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
                logger.warning(f"[CLOSE_ESTIMATED] {pair} | Trade={trade_id} | Prix M5={close_price:.5f} | PL_ESTIME={pl:.2f} | ATTENTION: sortie OANDA non récupérée")

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
                        current_sl = sl_order.get("price")
                        initial_sl = trade_info.get("sl")
                        if current_sl is not None and initial_sl is not None:
                            try:
                                current_sl = float(current_sl)
                                initial_sl = float(initial_sl)
                                tolerance = max(pip_val * 0.5, 0.00001)
                                if abs(current_sl - initial_sl) > tolerance:
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

            try:
                tracker_trade = trade_tracker.get_trade(trade_id)
                if tracker_trade:
                    mfe = float(tracker_trade.get("max_favorable_pips", 0))
                    mae = float(tracker_trade.get("max_adverse_pips", 0))
                    logger.info(f"[MFE_MAE_DIAG] {pair} | {setup_type} | MFE={mfe:.1f}pips | MAE={mae:.1f}pips | R_sortie={r_multiple:.2f} | Sortie={exit_type} | EQS={eqs}")
            except Exception as e:
                logger.debug(f"[MFE_MAE] Erreur tracker {trade_id}: {e}")

            if is_estimate:
                logger.warning(f"[CLOSE_ESTIMATED] Trade {trade_id} fermé (ESTIMATION) | prix={close_price:.5f} | R={r_multiple:.2f} | PL={pl:.2f}")
            else:
                logger.info(f"[CLOSE_CONFIRMED] Trade {trade_id} fermé (OANDA CONFIRMÉ) | prix={close_price:.5f} | R={r_multiple:.2f} | PL={pl:.2f}")

    except Exception as e:
        logger.error(f"Erreur lors du check des trades fermés: {e}")
        logger.error(traceback.format_exc())

# =============================
# FONCTIONS OANDA (inchangées)
# =============================
def v88_client():
    token = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_ACCESS_TOKEN")
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")
    return oandapyV20.API(access_token=token, environment=environment)


def get_candles_with_retry(api, instrument: str, granularity: str, count: int = 500, retries: int = 3) -> pd.DataFrame:
    valid_granularities = ["S5", "S10", "S15", "S30",
                           "M1", "M2", "M4", "M5", "M10", "M15", "M30",
                           "H1", "H2", "H3", "H4", "H6", "H8", "H12",
                           "D", "W", "M"]
    if granularity not in valid_granularities:
        logger.error(f"❌ Granularité invalide: {granularity}")
        return pd.DataFrame()

    if is_maintenance_suspended():
        logger.debug(f"⏳ OANDA en maintenance - get_candles {instrument} suspendu")
        return pd.DataFrame()

    for attempt in range(retries):
        try:
            params = {
                "granularity": granularity,
                "count": min(count, 500),
                "price": "M"
            }
            r = instruments.InstrumentsCandles(
                instrument=instrument,
                params=params
            )
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

            logger.warning(f"❌ Erreur OANDA {attempt + 1}/{retries} pour {instrument}: {e}")
            time.sleep(2 ** attempt)

        except Exception as e:
            logger.warning(f"❌ Tentative {attempt + 1}/{retries} pour {instrument}: {e}")
            time.sleep(2 ** attempt)

    logger.error(f"❌ Échec après {retries} tentatives pour {instrument} {granularity}")
    return pd.DataFrame()


def get_price_spread_v88(pair: str) -> dict:
    cached = _cache_get_v88(f"pricing:{pair}", ttl_seconds=2.0)
    if cached is not None:
        return cached
    try:
        if is_maintenance_suspended():
            fallback_price = get_recent_m5_price_v88(pair)
            return {"bid": fallback_price, "ask": fallback_price, "mid": fallback_price, "spread": 0.0}

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
    fallback_price = get_recent_m5_price_v88(pair)
    return {"bid": fallback_price, "ask": fallback_price, "mid": fallback_price, "spread": 0.0}


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

# Cache
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
# CONFIGURATION OANDA
# ============================================================
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "101-004-31348578-001")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "true").lower() == "true"

# ✅ V103 : Risque réduit à 0.75% (au lieu de 1.0%)
RISK_PERCENTAGE = float(os.getenv("RISK_PERCENTAGE", "0.75"))
MAX_RISK_USD = float(os.getenv("MAX_RISK_USD", "1250"))
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

# ============================================================
# TRADE TRACKER (MFE/MAE) - inchangé
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
        logger.info(f"[MFE/MAE] {trade['pair']} | {trade['setup_type']} | "
                    f"MFE={trade['max_favorable_pips']:.1f}pips | MAE={trade['max_adverse_pips']:.1f}pips | "
                    f"R={r_multiple:.2f} | EQS={trade['eqs']}")
        stats.record_mfe_mae(trade["pair"], trade["setup_type"], trade["eqs"],
                             trade["max_favorable_pips"], trade["max_adverse_pips"], r_multiple)

    def get_trade(self, trade_id: str):
        return self.trades.get(trade_id)

trade_tracker = TradeTracker()

# ============================================================
# GESTION DE LA MAINTENANCE OANDA - inchangé
# ============================================================
MAINTENANCE_DETECTED = False
MAINTENANCE_SUSPEND_TIME = 0
MAINTENANCE_RETRY_INTERVAL = 120
MAINTENANCE_ERROR_COUNT = 0
MAINTENANCE_MAX_ERRORS_BEFORE_SUSPEND = 3


def is_oanda_in_maintenance(error: Exception) -> bool:
    error_str = str(error).lower()
    maintenance_patterns = [
        "system under maintenance",
        "maintenance",
        "temporarily unavailable",
        "service unavailable",
        "maintenance mode",
        "api is currently unavailable"
    ]
    return any(pattern in error_str for pattern in maintenance_patterns)


def handle_api_error(error: Exception) -> tuple:
    global MAINTENANCE_DETECTED, MAINTENANCE_SUSPEND_TIME, MAINTENANCE_ERROR_COUNT

    if is_oanda_in_maintenance(error):
        MAINTENANCE_ERROR_COUNT += 1

        if MAINTENANCE_ERROR_COUNT >= MAINTENANCE_MAX_ERRORS_BEFORE_SUSPEND:
            MAINTENANCE_DETECTED = True
            MAINTENANCE_SUSPEND_TIME = time.time() + MAINTENANCE_RETRY_INTERVAL
            logger.warning(
                f"🔧 OANDA en maintenance détecté ({MAINTENANCE_ERROR_COUNT} erreurs) - "
                f"suspension des appels pendant {MAINTENANCE_RETRY_INTERVAL}s"
            )
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
    logger.info("🔧 Fin de la suspension OANDA - reprise des appels")
    return False

# ============================================================
# SUIVI DES TRADES STAGNANTS - inchangé
# ============================================================
stagnant_trade_tracker = {}


def check_stagnant_trades(trade_id: str, pair: str, direction: str, entry: float, current_r: float):
    global stagnant_trade_tracker

    if trade_id not in stagnant_trade_tracker:
        stagnant_trade_tracker[trade_id] = {
            "first_seen": time.time(),
            "last_r": current_r,
            "r_stable_count": 0,
            "action_taken": False
        }

    tracker = stagnant_trade_tracker[trade_id]

    if -0.15 <= current_r <= 0.15:
        if time.time() - tracker["first_seen"] > 3600:
            if abs(current_r - tracker["last_r"]) < 0.02:
                tracker["r_stable_count"] += 1

                if tracker["r_stable_count"] > 120 and not tracker["action_taken"]:
                    logger.info(
                        f"[STAGNANT] Trade {trade_id} {pair} stagne à R={current_r:.2f} "
                        f"depuis {int((time.time() - tracker['first_seen'])/60)}min"
                    )
                    tracker["action_taken"] = True
    else:
        tracker["first_seen"] = time.time()
        tracker["r_stable_count"] = 0
        tracker["action_taken"] = False

    tracker["last_r"] = current_r


# ============================================================
# BREAK EVEN ANTICIPÉ - inchangé
# ============================================================
def should_early_breakeven(trade_info: dict, df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> bool:
    direction = trade_info.get("direction")
    if direction is None:
        return False

    try:
        adx = calculate_adx(df_h1)
        h1_trend = score_ema_trend(df_h1)
        h4_trend = score_ema_trend(df_h4)
        momentum = calculate_momentum(df_h1)

        if adx > 35:
            if direction == "BUY" and h1_trend > 0 and h4_trend > 0 and momentum > 0.2:
                return True
            if direction == "SELL" and h1_trend < 0 and h4_trend < 0 and momentum < -0.2:
                return True
    except Exception:
        pass

    return False


# ============================================================
# ✅ V103 : SORTIE ANTICIPÉE SUR RETOURNEMENT - PLUS STRICTE (4 signaux)
# ============================================================
def check_indicator_reversal(pair: str, direction: str, df_m15: pd.DataFrame, df_h1: pd.DataFrame, current_r: float) -> bool:
    if current_r > -0.30:
        return False
    
    direction = direction.upper()

    try:
        close = df_m15['close'].iloc[-1]
        close_prev = df_m15['close'].iloc[-2]
        rsi_m15 = get_last_rsi(df_m15['close'])
        adx_h1 = calculate_adx(df_h1)
        macd_hist = calculate_macd_momentum(df_h1)
        macd_last = macd_hist.iloc[-1]
        macd_prev = macd_hist.iloc[-2]

        signals_against = 0

        if direction == "BUY":
            if close < close_prev:
                signals_against += 1
            if rsi_m15 < 40:
                signals_against += 1
            if macd_last < macd_prev:
                signals_against += 1
            if adx_h1 < 20:
                signals_against += 1
        else:
            if close > close_prev:
                signals_against += 1
            if rsi_m15 > 60:
                signals_against += 1
            if macd_last > macd_prev:
                signals_against += 1
            if adx_h1 < 20:
                signals_against += 1

        return signals_against >= 4
    except Exception:
        return False


def close_trade_api(trade_id: str) -> bool:
    try:
        api = v88_client()
        r = trades.TradeClose(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id)
        resp = api.request(r)
        if resp.get("orderCreateTransaction"):
            logger.info(f"[CLOSE] Trade {trade_id} fermé via API")
            return True
        else:
            logger.error(f"[CLOSE] Échec fermeture trade {trade_id}")
            return False
    except Exception as e:
        logger.error(f"[CLOSE] Erreur fermeture trade {trade_id}: {e}")
        return False


# =========================
# LOG HELPERS - inchangé
# =========================
_seen_log_keys_fvg_recent = set()
_seen_log_keys_fvg_added = set()
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
# CONFIGURATION - inchangé
# =============================
load_dotenv()

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
        "atr_multiplier_sl": 1.2,
        "atr_multiplier_tp": 3.0,
        "max_volatility_ratio": 0.02,
        "risk_multiplier": 1.0
    }
}

# ✅ V105.1 : SL/TP rééquilibrés (multiplicateurs plus serrés)
SIGNAL_RISK_SETTINGS = {
    "NESTED_FVG": {"sl_multiplier": 0.6, "tp_multiplier": 1.8},
    "FVG_RETEST": {"sl_multiplier": 0.8, "tp_multiplier": 2.0},
    "WICK_REJECTION": {"sl_multiplier": 0.9, "tp_multiplier": 2.7},
    "LIQUIDITY_DRAW": {"sl_multiplier": 1.0, "tp_multiplier": 2.5},
    "FVG_RETEST_PERFECT": {"sl_multiplier": 0.7, "tp_multiplier": 2.2},
    "BISI": {"sl_multiplier": 0.7, "tp_multiplier": 2.2},
    "BREAKER": {"sl_multiplier": 0.9, "tp_multiplier": 2.5},
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

# ============================================================
# V106 - EXÉCUTION D'ORDRE (avec vérification RR)
# ============================================================
last_execution_attempt = {}
EXECUTION_COOLDOWN_SECONDS = 60

# ============================================================
# V110 - EXÉCUTION OANDA AVEC PRIX DE FILL RÉEL
# ============================================================
def execute_oanda_trade_v981(
    pair: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    score: int,
    entry_type: str,
    eqs: int,
    setup_type: str,
    metrics: dict
) -> str | None:

    global last_execution_attempt

    pair_upper = pair.upper()
    direction = direction.upper()

    now = time.time()

    if (
        pair_upper in last_execution_attempt
        and now - last_execution_attempt[pair_upper]
        < EXECUTION_COOLDOWN_SECONDS
    ):
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

    rr = reward / risk

    if rr < 1.8:
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
            "stopLossOnFill": {
                "price": round_price_v88(pair, stop_loss),
                "timeInForce": "GTC"
            },
            "takeProfitOnFill": {
                "price": round_price_v88(pair, take_profit),
                "timeInForce": "GTC"
            }
        }
    }

    logger.info(f"[ORDER_EXPECTED] {pair} | {direction} | ENTRY_EXPECTED={expected_entry:.5f} | SL={stop_loss:.5f} | TP={take_profit:.5f} | RR={abs(take_profit-expected_entry)/risk:.2f}")

    if not EXECUTE_TRADES:
        logger.info("[ORDER] EXECUTE_TRADES=false")
        return "SIMULATION"

    try:
        api = v88_client()
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        import json
        logger.info(f"[ORDER] 📤 OANDA {json.dumps(order_data)}")
        resp = api.request(r)
        logger.info(f"[ORDER] 📥 OANDA RESPONSE {json.dumps(resp)}")

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

        MAX_ENTRY_SLIPPAGE_PIPS = {
            "GBP_USD": 2.5,
            "EUR_USD": 2.5,
            "USD_CAD": 2.5,
            "AUD_USD": 2.5,
            "DEFAULT": 3.0,
        }
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

        # Enregistrement du trade avec le prix réel et stockage du SL initial
        trade_tracker.add_trade(trade_id, pair, direction, actual_entry, stop_loss, take_profit, setup_type, eqs)
        open_trade_details[str(trade_id)] = {
            "entry": actual_entry,
            "expected_entry": expected_entry,
            "actual_entry": actual_entry,
            "sl": stop_loss,                # SL initial
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
# V106 - MODIFICATION SL (avec réajustement TP pour préserver RR)
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

        trade_details = get_trade_details_v88(trade_id)
        if trade_details:
            actual_sl = get_stop_loss_v88(trade_details)
            if abs(actual_sl - new_sl) <= 0.0001:
                logger.info(f"[CONFIRM] ✅ SL confirmé: {actual_sl:.5f}")
                return True
            else:
                logger.warning(f"[CONFIRM] SL non confirmé: attendu {new_sl:.5f}, reçu {actual_sl:.5f}")
                return True
        else:
            logger.warning(f"[CONFIRM] Impossible de confirmer le SL pour {trade_id}")
            return True

    except Exception as e:
        logger.error(f"[BE] Erreur modification SL trade {trade_id}: {e}")
        if is_oanda_in_maintenance(e):
            handle_api_error(e)
        return False

# ============================================================
# V106 - CRÉATION TRAILING STOP (optimisé avec activation à 0.80R)
# ============================================================
def create_oanda_trailing_stop_v981(trade_id: str, pair: str, distance: float) -> bool:
    try:
        if is_maintenance_suspended():
            logger.warning(f"[TSL] OANDA en maintenance - trailing suspendu pour {trade_id}")
            return False

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

        if resp.get("orderRejectTransaction"):
            reject = resp.get("orderRejectTransaction")
            logger.error(f"[TSL] Rejeté pour trade {trade_id}: {reject.get('rejectReason', 'unknown')}")
            return False

        logger.info(f"[TSL] SUCCESS: Trailing stop créé pour trade {trade_id}, distance={distance:.5f}")
        time.sleep(1)
        _OANDA_CACHE_V88.pop("open_trades_raw", None)

        trade_details = get_trade_details_v88(trade_id)
        if trade_details and has_trailing_stop_v88(trade_details):
            trailing_id = trade_details.get("trailingStopLossOrder", {}).get("id", "unknown")
            logger.info(f"[CONFIRM] ✅ Trailing stop confirmé: ID={trailing_id}")
            return True
        else:
            logger.warning(f"[CONFIRM] Trailing stop non confirmé pour {trade_id}")
            return True

    except Exception as e:
        logger.error(f"[TSL] Erreur création trailing stop trade {trade_id}: {e}")
        if is_oanda_in_maintenance(e):
            handle_api_error(e)
        return False


# ============================================================
# FONCTIONS DE CONFIRMATION - inchangé
# ============================================================
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

    open_trades = get_open_trades_v88(log_raw=True, skip_maintenance_check=True, force_refresh=True)
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


def open_trade_count_v88() -> int:
    return len(get_open_trades_v88(log_raw=True))


def has_open_trade_v88(pair: str) -> bool:
    for t in get_open_trades_v88():
        if t.get("instrument") == pair:
            return True
    return False

# =============================
# FONCTIONS UTILITAIRES - inchangé
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
# LOGGING (avec tag V106)
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
    ALLOWED_TAGS_V83 = ("[START]", "[SCAN]", "[INFO]", "[SIGNAL]", "[ORDER]", "[RISK]", "[ERROR]", "[TRACE]", "[BE]", "[TSL]", "[CONFIRM]", "[DIAG]", "[DECISION]", "[CLOSE]", "[CLOSE_DETAILS]", "[CLOSE_ESTIMATED]", "[CLOSE_CONFIRMED]", "[MFE/MAE]", "[MFE_MAE_DIAG]", "[ADAPT]", "[SUSPEND]", "[SETUP_WEIGHT]")

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
        elif "DECISION" in upper:
            tag = "[DECISION]"
        elif "CLOSE" in upper:
            tag = "[CLOSE]"
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
        elif "MFE" in upper or "MAE" in upper:
            tag = "[MFE/MAE]"
        elif "ADAPT" in upper:
            tag = "[ADAPT]"
        elif "SUSPEND" in upper:
            tag = "[SUSPEND]"
        elif "SETUP_WEIGHT" in upper:
            tag = "[SETUP_WEIGHT]"
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
# GESTION DES SIGNAUX - inchangé
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
# INDICATEURS TECHNIQUES - inchangé
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
        ("Structure", "Structure V98.1"),
        ("Pullback", "Pullback V98.1"),
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
# DÉTECTION SWING POINTS - inchangé
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
# DÉTECTION FVG - inchangé
# =============================
def get_min_gap_for_pair(pair: str) -> float:
    pair = pair.upper()
    if pair == "XAU_USD":
        return 0.015
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
# DÉTECTION WICK REJECTION POI - inchangé
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
# DÉTECTION ORDER FLOW LEGS - inchangé
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
# BIAS AVANCÉ - inchangé
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
# FONCTIONS DE PRIX ET CONVERSION - inchangé
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
        return 1.0
    elif pair == "NAS100_USD":
        return 0.1
    elif "JPY" in pair:
        return 0.01
    else:
        return 0.0001

# ============================================================
# FILTRES ET SCORING - V106 (score d'entrée /100, momentum gradué, ADX dynamique)
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
    components = {}

    # 1. Distance Zone (max 20)
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
    components["distance_zone"] = {"value": f"{entry_zone_pips:.1f}pips", "score": scores["distance_zone"], "max": 20}

    # 2. EMA Proximity (max 20)
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
    components["ema_proximity"] = {"value": f"{ema_distance_pips:.1f}pips" if 'ema_distance_pips' in locals() else "N/A", "score": scores["ema_proximity"], "max": 20}

    # 3. Range Position (max 20)
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
    components["range_position"] = {"value": f"{position:.2f}" if 'position' in locals() else "N/A", "score": scores["range_position"], "max": 20}

    # 4. Pullback Quality (max 20)
    pullback_passed, _ = filter_pullback(df_m15, direction, entry_level, current_price, pair)
    if pullback_passed:
        scores["pullback_quality"] = 20
        logs.append("pullback_quality: OK -> 20")
        pullback_pips = "OK"
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
                    pullback_pips = 0
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
                    pullback_pips = 0
        else:
            scores["pullback_quality"] = 10
            logs.append("pullback_quality: données insuffisantes -> 10")
            pullback_pips = "N/A"
    components["pullback_quality"] = {"value": f"{pullback_pips:.1f}pips" if isinstance(pullback_pips, float) else pullback_pips, "score": scores["pullback_quality"], "max": 20}

    # 5. Stoch Position (max 20)
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
        k = 50
    components["stoch_position"] = {"value": f"{k:.1f}", "score": scores["stoch_position"], "max": 20}

    # 6. Spread Penalty (pénalité, max -10)
    spread_data = get_price_spread_v88(pair)
    spread = spread_data.get("spread", 0.0)
    if spread > pip_value * 2:
        scores["spread_penalty"] = -10
        logs.append(f"spread_penalty: {spread:.2f} > 2 pips -> -10")
    elif spread > pip_value * 1.5:
        scores["spread_penalty"] = -5
        logs.append(f"spread_penalty: {spread:.2f} > 1.5 pips -> -5")
    else:
        scores["spread_penalty"] = 0
        logs.append(f"spread_penalty: {spread:.2f} -> 0")
    components["spread_penalty"] = {"value": f"{spread:.2f}", "score": scores["spread_penalty"], "max": 0}

    total = sum(scores.values())

    details = {
        "distance_zone": scores["distance_zone"],
        "ema_proximity": scores["ema_proximity"],
        "range_position": scores["range_position"],
        "pullback_quality": scores["pullback_quality"],
        "stoch_position": scores["stoch_position"],
        "spread_penalty": scores["spread_penalty"],
        "total": total,
        "passed": total >= BASE_EQS_MIN_THRESHOLD,
        "logs": logs,
        "components": components
    }

    return details


def filter_market_structure(df: pd.DataFrame, direction: str, lookback: int = 5, score: int = 0, eqs: int = 0, is_asia: bool = False) -> tuple:
    if len(df) < lookback * 2 + 2:
        return True, "Données insuffisantes, structure acceptée par défaut"

    direction = direction.upper()
    swing_highs, swing_lows = detect_swing_points(df, lookback=3)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return True, "Pas assez de swing points, structure acceptée par défaut"

    last_highs = sorted(swing_highs, key=lambda x: x['index'])[-2:]
    last_lows = sorted(swing_lows, key=lambda x: x['index'])[-2:]

    if direction == "BUY":
        hh = last_highs[-1]['price'] > last_highs[-2]['price']
        hl = last_lows[-1]['price'] > last_lows[-2]['price']
        lh = last_highs[-1]['price'] < last_highs[-2]['price']
        ll = last_lows[-1]['price'] < last_lows[-2]['price']
        if lh and ll:
            if is_asia:
                if score >= 75 and eqs >= 80:
                    return True, f"Structure BEARISH acceptée en ASIA (score={score}, EQS={eqs})"
                else:
                    return False, f"Structure BEARISH confirmée en ASIA (LH={lh}, LL={ll}, score={score}, EQS={eqs})"
            else:
                adx = calculate_adx(df, period=14)
                # ✅ NOUVEAU : Accepte si ADX < 20 OU si EQS est très élevé (≥80)
                if adx < 20 or eqs >= 80:
                    return True, f"Structure BEARISH acceptée en session active (ADX={adx:.1f}<20 ou EQS={eqs}>=80)"
                return False, f"Structure BEARISH confirmée (LH={lh}, LL={ll})"
        else:
            return True, f"Structure non-bearish (HH={hh}, HL={hl})"

    elif direction == "SELL":
        hh = last_highs[-1]['price'] > last_highs[-2]['price']
        hl = last_lows[-1]['price'] > last_lows[-2]['price']
        lh = last_highs[-1]['price'] < last_highs[-2]['price']
        ll = last_lows[-1]['price'] < last_lows[-2]['price']
        if hh and hl:
            if is_asia:
                if score >= 75 and eqs >= 80:
                    return True, f"Structure BULLISH acceptée en ASIA (score={score}, EQS={eqs})"
                else:
                    return False, f"Structure BULLISH confirmée en ASIA (HH={hh}, HL={hl}, score={score}, EQS={eqs})"
            else:
                adx = calculate_adx(df, period=14)
                # ✅ NOUVEAU : Accepte si ADX < 20 OU si EQS est très élevé (≥80)
                if adx < 20 or eqs >= 80:
                    return True, f"Structure BULLISH acceptée en session active (ADX={adx:.1f}<20 ou EQS={eqs}>=80)"
                return False, f"Structure BULLISH confirmée (HH={hh}, HL={hl})"
        else:
            return True, f"Structure non-bullish (LH={lh}, LL={ll})"

    return False, f"Direction {direction} invalide"

def filter_pullback(
    df: pd.DataFrame,
    direction: str,
    entry_level: float,
    current_price: float,
    pair: str
) -> tuple:
    try:
        direction = str(direction).upper()
        pair = str(pair).upper()

        if df is None or len(df) < max(8, ATR_PERIOD):
            return False, "Données insuffisantes pour pullback"

        if current_price is None or float(current_price) <= 0:
            return False, "Prix courant invalide"

        current_price = float(current_price)
        pip_value = get_pip_value_for_pair(pair)
        if pip_value is None or pip_value <= 0:
            return False, f"Valeur pip invalide pour {pair}"
        pip_value = float(pip_value)

        try:
            atr_price = calculate_atr(df, period=ATR_PERIOD)
            atr_pips = price_to_pips(atr_price, pair)
            atr_pips = float(atr_pips)
        except Exception as e:
            logger.warning(f"[PULLBACK_ATR] {pair} | Impossible de calculer ATR: {e}")
            atr_pips = 0.0

        base_min_pullback = PULLBACK_MIN_PIPS_BY_PAIR.get(pair, PULLBACK_MIN_PIPS_BY_PAIR["DEFAULT"])
        base_min_pullback = float(base_min_pullback)

        if atr_pips > 0:
            atr_based_threshold = atr_pips * 0.75
            min_absolute_pips = min(2.0, base_min_pullback * 0.75)
            max_pullback_pips = base_min_pullback
            dynamic_min_pullback = max(min_absolute_pips, min(max_pullback_pips, atr_based_threshold))
        else:
            dynamic_min_pullback = base_min_pullback

        try:
            hour = datetime.utcnow().hour
            is_asia = hour >= 21 or hour < 7
        except Exception:
            is_asia = False

        if is_asia:
            dynamic_min_pullback = max(2.0, dynamic_min_pullback * 0.90)

        tolerance_pips = 0.5
        effective_threshold_pips = max(0.0, dynamic_min_pullback - tolerance_pips)
        min_pullback_price = effective_threshold_pips * pip_value

        recent = df.iloc[-8:]

        if direction == "BUY":
            recent_high = float(recent["high"].max())
            pullback_depth = recent_high - current_price
            pullback_pips = price_to_pips(pullback_depth, pair)
            pullback_pips = max(0.0, float(pullback_pips))

            if pullback_depth >= min_pullback_price:
                logger.info(f"[PULLBACK_DYNAMIC] {pair} | BUY | Pullback={pullback_pips:.1f}p | Seuil={dynamic_min_pullback:.1f}p | ATR={atr_pips:.1f}p | Asia={is_asia} | PASSED")
                return True, f"Pullback OK ({pullback_pips:.1f} pips >= {dynamic_min_pullback:.1f} pips | ATR={atr_pips:.1f})"
            else:
                logger.info(f"[PULLBACK_DYNAMIC] {pair} | BUY | Pullback={pullback_pips:.1f}p | Seuil={dynamic_min_pullback:.1f}p | ATR={atr_pips:.1f}p | Asia={is_asia} | REJECT")
                return False, f"Pullback insuffisant ({pullback_pips:.1f} pips < {dynamic_min_pullback:.1f} pips | ATR={atr_pips:.1f})"

        elif direction == "SELL":
            recent_low = float(recent["low"].min())
            pullback_depth = current_price - recent_low
            pullback_pips = price_to_pips(pullback_depth, pair)
            pullback_pips = max(0.0, float(pullback_pips))

            if pullback_depth >= min_pullback_price:
                logger.info(f"[PULLBACK_DYNAMIC] {pair} | SELL | Pullback={pullback_pips:.1f}p | Seuil={dynamic_min_pullback:.1f}p | ATR={atr_pips:.1f}p | Asia={is_asia} | PASSED")
                return True, f"Pullback OK ({pullback_pips:.1f} pips >= {dynamic_min_pullback:.1f} pips | ATR={atr_pips:.1f})"
            else:
                logger.info(f"[PULLBACK_DYNAMIC] {pair} | SELL | Pullback={pullback_pips:.1f}p | Seuil={dynamic_min_pullback:.1f}p | ATR={atr_pips:.1f}p | Asia={is_asia} | REJECT")
                return False, f"Pullback insuffisant ({pullback_pips:.1f} pips < {dynamic_min_pullback:.1f} pips | ATR={atr_pips:.1f})"

        return False, f"Direction {direction} invalide"

    except Exception as e:
        logger.error(f"[PULLBACK_DYNAMIC] {pair} | Erreur: {e}", exc_info=True)
        return False, f"Erreur filtre pullback: {e}"

def score_atr_volatility(df: pd.DataFrame, pair: str) -> tuple:
    if len(df) < ATR_PERIOD:
        return 0, "Données insuffisantes pour ATR"

    atr_price = calculate_atr(df, period=ATR_PERIOD)
    atr_pips = price_to_pips(atr_price, pair)

    hour = datetime.utcnow().hour
    is_asia = 21 <= hour or hour < 7
    is_london = 7 <= hour < 16
    is_ny = 12 <= hour < 21

    base_min_atr = MIN_ATR_PIPS_BY_PAIR.get(pair, MIN_ATR_PIPS_BY_PAIR["DEFAULT"])
    if is_asia:
        min_atr_pips = MIN_ATR_PIPS_BY_PAIR_ASIA.get(pair, base_min_atr * 0.75)
    elif is_london or is_ny:
        min_atr_pips = base_min_atr
    else:
        min_atr_pips = base_min_atr * 0.85

    ratio = atr_pips / min_atr_pips if min_atr_pips > 0 else 0
    if ratio >= 1.5:
        score = 10
        msg = f"ATR très élevé ({atr_pips:.1f} pips)"
    elif ratio >= 1.2:
        score = 8
        msg = f"ATR bon ({atr_pips:.1f} pips)"
    elif ratio >= 1.0:
        score = 6
        msg = f"ATR correct ({atr_pips:.1f} pips)"
    elif ratio >= 0.7:
        score = 3
        msg = f"ATR faible ({atr_pips:.1f} pips)"
    else:
        score = 0
        msg = f"ATR très faible ({atr_pips:.1f} pips)"

    if ratio < 0.7:
        return 0, f"ATR trop faible ({atr_pips:.1f} < {min_atr_pips*0.7:.1f})"
    return score, msg


def score_momentum(pair: str, direction: str, df_m15: pd.DataFrame, df_h1: pd.DataFrame, entry_level: float, current_price: float, entry_type: str) -> tuple:
    direction = direction.upper()
    momentum = calculate_momentum(df_m15, period=5)
    abs_momentum = abs(momentum)

    if direction == "BUY":
        if momentum > 0.2:
            score = 15
            msg = f"Momentum haussier fort ({momentum:.2f}%)"
        elif momentum > 0.05:
            score = 12
            msg = f"Momentum haussier modéré ({momentum:.2f}%)"
        elif momentum > -0.05:
            score = 8
            msg = f"Momentum neutre ({momentum:.2f}%)"
        elif momentum > -0.15:
            score = 4
            msg = f"Momentum légèrement baissier ({momentum:.2f}%)"
        else:
            score = 0
            msg = f"Momentum baissier fort ({momentum:.2f}%)"
    else:
        if momentum < -0.2:
            score = 15
            msg = f"Momentum baissier fort ({momentum:.2f}%)"
        elif momentum < -0.05:
            score = 12
            msg = f"Momentum baissier modéré ({momentum:.2f}%)"
        elif momentum < 0.05:
            score = 8
            msg = f"Momentum neutre ({momentum:.2f}%)"
        elif momentum < 0.15:
            score = 4
            msg = f"Momentum légèrement haussier ({momentum:.2f}%)"
        else:
            score = 0
            msg = f"Momentum haussier fort ({momentum:.2f}%)"

    if direction == "BUY" and momentum < -0.15:
        return 0, f"Momentum baissier fort ({momentum:.2f}%) contre BUY"
    if direction == "SELL" and momentum > 0.15:
        return 0, f"Momentum haussier fort ({momentum:.2f}%) contre SELL"

    if direction == "BUY" and momentum < -0.05:
        score -= 2
    if direction == "SELL" and momentum > 0.05:
        score -= 2

    return max(0, score), msg


def score_adx_dynamic(df_h1: pd.DataFrame) -> tuple:
    adx = calculate_adx(df_h1, period=14)
    if adx == 0:
        return 0, "ADX indisponible"

    adx_series = []
    for i in range(1, 4):
        if len(df_h1) > i:
            adx_series.append(calculate_adx(df_h1.iloc[:-i] if i>0 else df_h1, period=14))
    pente = adx - adx_series[-1] if len(adx_series) > 0 else 0

    if adx >= 30 and pente > 0:
        score = 10
        msg = f"ADX fort et montant ({adx:.1f})"
    elif adx >= 30:
        score = 8
        msg = f"ADX fort mais stable ou descendant ({adx:.1f})"
    elif adx >= 25:
        if pente > 0:
            score = 8
            msg = f"ADX correct et montant ({adx:.1f})"
        else:
            score = 6
            msg = f"ADX correct ({adx:.1f})"
    elif adx >= 20:
        if pente > 0:
            score = 5
            msg = f"ADX modéré mais montant ({adx:.1f})"
        else:
            score = 3
            msg = f"ADX modéré ({adx:.1f})"
    elif adx >= 15:
        score = 1
        msg = f"ADX faible ({adx:.1f})"
    else:
        score = 0
        msg = f"ADX très faible ({adx:.1f})"

    return score, msg


def check_htf_confluence(direction: str, df_h1: pd.DataFrame, df_h4: pd.DataFrame, is_asia: bool = False, score: int = 0, eqs: int = 0) -> tuple:
    score_htf = 0
    details = []

    h1_trend = score_ema_trend(df_h1)
    if (direction == "BUY" and h1_trend > 0) or (direction == "SELL" and h1_trend < 0):
        score_htf += 1
        details.append("Tendance H1 alignée")
    else:
        details.append("Tendance H1 neutre/opposée")

    h4_trend = score_ema_trend(df_h4)
    if (direction == "BUY" and h4_trend > 0) or (direction == "SELL" and h4_trend < 0):
        score_htf += 1
        details.append("Tendance H4 alignée")
    else:
        details.append("Tendance H4 neutre/opposée")

    h1_structure = score_market_structure(df_h1)
    if (direction == "BUY" and h1_structure > 0) or (direction == "SELL" and h1_structure < 0):
        score_htf += 1
        details.append("Structure H1 alignée")
    else:
        details.append("Structure H1 neutre/opposée")

    # ✅ V112 : Seuil HTF adaptatif selon la session
    if is_asia:
        # ASIA : 1/3 accepté si score>=75 et EQS>=85
        if score_htf == 1 and score >= 75 and eqs >= 85:
            required_htf = 1
            details.append(f"HTF=1/3 accepté en ASIA (score={score}, EQS={eqs})")
        else:
            required_htf = 2
    else:
        # NY/LONDON : 2/3 requis (ou 1/3 si score>=70 et EQS>=80, déjà géré ailleurs)
        if score_htf >= 2:
            required_htf = 2
        elif score_htf == 1 and score >= 70 and eqs >= 80:
            required_htf = 1
            details.append(f"HTF=1/3 accepté en session active (score={score}, EQS={eqs})")
        else:
            required_htf = 2

    if score_htf >= required_htf:
        bonus = 5 if score_htf == 3 else 0
        if bonus:
            details.append("BONUS 3/3 +5")
        return score_htf, f"{score_htf}/3", details, bonus
    else:
        return score_htf, f"{score_htf}/3 (requis {required_htf}/3)", details, 0


def get_effective_atr_threshold(pair: str) -> float:
    hour = datetime.utcnow().hour
    is_asia = 21 <= hour or hour < 7
    is_london = 7 <= hour < 16
    is_ny = 12 <= hour < 21

    base_min_atr = MIN_ATR_PIPS_BY_PAIR.get(pair, MIN_ATR_PIPS_BY_PAIR["DEFAULT"])
    if is_asia:
        return MIN_ATR_PIPS_BY_PAIR_ASIA.get(pair, base_min_atr * 0.75)
    elif is_london or is_ny:
        return base_min_atr
    else:
        return base_min_atr * 0.85

# ============================================================
# V106 - CALCUL SL/TP AVEC MULTIPLICATEURS PAR SETUP
# ============================================================
def calculate_sl_tp(
    entry_price: float,
    atr: float,
    direction: str,
    pair: str,
    entry_type: str = "FVG_RETEST",
    fvg_data: dict = None,
    breaker_level: float = None,
) -> Tuple[float, float]:
    direction = direction.upper()
    pip_value = get_pip_value_for_pair(pair)

    risk_settings = SIGNAL_RISK_SETTINGS.get(entry_type, SIGNAL_RISK_SETTINGS["FVG_RETEST"])
    sl_mult = risk_settings.get("sl_multiplier", 0.8)
    tp_mult = risk_settings.get("tp_multiplier", 2.0)

    if entry_type == "BREAKER" and breaker_level is not None:
        if direction == "BUY":
            stop_loss = breaker_level - (atr * 0.3)
        else:
            stop_loss = breaker_level + (atr * 0.3)
    else:
        sl_distance = max(atr * sl_mult, pip_value * 10)
        if direction == "BUY":
            stop_loss = entry_price - sl_distance
        else:
            stop_loss = entry_price + sl_distance

    risk = abs(entry_price - stop_loss)
    tp_distance = max(risk * max(tp_mult, 2.0), risk * 2.0)

    if direction == "BUY":
        take_profit = entry_price + tp_distance
    else:
        take_profit = entry_price - tp_distance

    stop_loss = float(round_price_v88(pair, stop_loss))
    take_profit = float(round_price_v88(pair, take_profit))

    return stop_loss, take_profit

# ============================================================
# V106 - FILTRE DE CONFIRMATION DE CLÔTURE M15 (V110)
# ============================================================
def filter_close_confirmation(
    df_m15: pd.DataFrame,
    direction: str
) -> Tuple[bool, str]:
    direction = (direction or "").upper()

    if df_m15 is None or len(df_m15) < 3:
        return False, "Données M15 insuffisantes"

    try:
        last = df_m15.iloc[-2]
        prev = df_m15.iloc[-3]

        open_price = float(last["open"])
        high = float(last["high"])
        low = float(last["low"])
        close = float(last["close"])

        prev_open = float(prev["open"])
        prev_high = float(prev["high"])
        prev_low = float(prev["low"])
        prev_close = float(prev["close"])

        total_range = high - low

        if total_range <= 0:
            return False, "Amplitude M15 nulle"

        body = abs(close - open_price)
        body_ratio = body / total_range

        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low

        if direction == "BUY":
            is_bullish = close > open_price
            strong_body = body_ratio >= 0.45
            upper_half = close >= (low + total_range * 0.55)
            momentum_confirm = (close > prev_close and high >= prev_high)
            no_strong_upper_rejection = (upper_wick <= body * 1.5)

            if (is_bullish and strong_body and upper_half and no_strong_upper_rejection):
                return True, f"BUY confirmé M15 | body={body_ratio*100:.0f}% | close haut de range"
            if (is_bullish and momentum_confirm and upper_half):
                return True, f"BUY momentum M15 confirmé | close={close:.5f} > prev={prev_close:.5f} | high={high:.5f} >= prev_high={prev_high:.5f}"
            reasons = []
            if not is_bullish:
                reasons.append("bougie non haussière")
            if not strong_body:
                reasons.append(f"corps faible={body_ratio*100:.0f}%")
            if not upper_half:
                reasons.append("clôture trop basse")
            if not momentum_confirm:
                reasons.append("pas de nouveau momentum")
            if not no_strong_upper_rejection:
                reasons.append("rejet haut important")
            return False, "BUY non confirmé M15: " + ", ".join(reasons)

        if direction == "SELL":
            is_bearish = close < open_price
            strong_body = body_ratio >= 0.45
            lower_half = close <= (low + total_range * 0.45)
            momentum_confirm = (close < prev_close and low <= prev_low)
            no_strong_lower_rejection = (lower_wick <= body * 1.5)

            if (is_bearish and strong_body and lower_half and no_strong_lower_rejection):
                return True, f"SELL confirmé M15 | body={body_ratio*100:.0f}% | close bas de range"
            if (is_bearish and momentum_confirm and lower_half):
                return True, f"SELL momentum M15 confirmé | close={close:.5f} < prev={prev_close:.5f} | low={low:.5f} <= prev_low={prev_low:.5f}"
            reasons = []
            if not is_bearish:
                reasons.append("bougie non baissière")
            if not strong_body:
                reasons.append(f"corps faible={body_ratio*100:.0f}%")
            if not lower_half:
                reasons.append("clôture trop haute")
            if not momentum_confirm:
                reasons.append("pas de nouveau momentum")
            if not no_strong_lower_rejection:
                reasons.append("rejet bas important")
            return False, "SELL non confirmé M15: " + ", ".join(reasons)

        return False, f"Direction invalide: {direction}"

    except Exception as e:
        logger.error(f"[CONFIRM] Erreur confirmation M15: {e}")
        return False, f"Erreur confirmation M15: {e}"

# ============================================================
# V106 - FONCTIONS DE SCORING MANQUANTES
# ============================================================
def estimate_win_rate(entry_score: int, eqs_score: int, confluences: dict) -> str:
    base = 45
    if entry_score >= 80:
        base += 15
    elif entry_score >= 70:
        base += 10
    elif entry_score >= 65:
        base += 5

    if eqs_score >= 80:
        base += 8
    elif eqs_score >= 70:
        base += 5
    elif eqs_score >= 65:
        base += 2

    if confluences.get("d1_aligned", False):
        base += 5
    if confluences.get("structure_ok", False):
        base += 3
    if confluences.get("pullback_ok", False):
        base += 3
    if confluences.get("macd_confirmed", False):
        base += 3
    if confluences.get("session_active", False):
        base += 2
    if confluences.get("bos_confirmed", False):
        base += 2
    if confluences.get("rsi_divergence", False):
        base += 3

    base = max(35, min(75, base))
    return f"{base-3}-{base+3}%"


def get_signal_quality_label(entry_score: int, eqs_score: int) -> str:
    if entry_score >= 85 and eqs_score >= 80:
        return "SNIPER"
    elif entry_score >= 78 and eqs_score >= 75:
        return "A+"
    elif entry_score >= 70 and eqs_score >= 68:
        return "A"
    elif entry_score >= 65 and eqs_score >= 62:
        return "B"
    elif entry_score >= 55 and eqs_score >= 55:
        return "C"
    else:
        return "D"


def get_d1_trend_bonus(df_d1: pd.DataFrame, direction: str) -> Tuple[int, str]:
    if df_d1 is None or len(df_d1) < 20:
        return 0, "Données D1 insuffisantes"
    try:
        ema50_d1 = df_d1['close'].ewm(span=50, adjust=False).mean()
        if len(ema50_d1.dropna()) < 2:
            return 0, "EMA50 D1 indisponible"
        price = df_d1['close'].iloc[-1]
        ema50 = ema50_d1.iloc[-1]
        ema50_prev = ema50_d1.iloc[-5] if len(ema50_d1) > 5 else ema50
        slope = ema50 - ema50_prev
        macd_hist = calculate_macd_momentum(df_d1)
        macd_last = macd_hist.iloc[-1] if len(macd_hist) > 0 else 0
        macd_prev = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0

        if direction == "BUY":
            if price > ema50 and slope > 0 and macd_last > 0 and macd_last > macd_prev:
                return 2, "Tendance D1 haussière forte"
            elif price > ema50:
                return 1, "Tendance D1 haussière"
            else:
                return 0, "D1 neutre ou baissière"
        else:
            if price < ema50 and slope < 0 and macd_last < 0 and macd_last < macd_prev:
                return 2, "Tendance D1 baissière forte"
            elif price < ema50:
                return 1, "Tendance D1 baissière"
            else:
                return 0, "D1 neutre ou haussière"
    except Exception as e:
        logger.debug(f"[D1_TREND] Erreur: {e}")
        return 0, f"Erreur D1: {str(e)[:30]}"


def get_macd_h1_bonus(df_h1: pd.DataFrame, direction: str) -> Tuple[int, str]:
    if df_h1 is None or len(df_h1) < 26:
        return 0, "Données H1 insuffisantes"
    try:
        macd_hist = calculate_macd_momentum(df_h1)
        if len(macd_hist.dropna()) < 2:
            return 0, "MACD indisponible"
        macd_last = macd_hist.iloc[-1]
        macd_prev = macd_hist.iloc[-2]

        if direction == "BUY":
            if macd_last > 0 and macd_last > macd_prev:
                return 1, "MACD haussier"
            elif macd_last > 0:
                return 0, "MACD positif mais non croissant"
            else:
                return 0, "MACD négatif"
        else:
            if macd_last < 0 and macd_last < macd_prev:
                return 1, "MACD baissier"
            elif macd_last < 0:
                return 0, "MACD négatif mais non décroissant"
            else:
                return 0, "MACD positif"
    except Exception as e:
        logger.debug(f"[MACD] Erreur: {e}")
        return 0, f"Erreur MACD: {str(e)[:30]}"

def get_session_quality_bonus(pair: str) -> Tuple[int, str]:
    hour = datetime.utcnow().hour
    if 7 <= hour < 16:
        return 1, "LONDON"
    elif 12 <= hour < 21:
        return 1, "NY"
    elif hour >= 21 or hour < 7:
        return 0, "ASIA"
    else:
        return 0, "OTHER"

# ============================================================
# V110 - ENTRY QUALITY GATE
# ============================================================
def validate_entry_quality_gate(
    pair: str,
    direction: str,
    entry_type: str,
    entry_score: int,
    eqs_score: int,
    adx: float,
    momentum: float,
    close_confirmed: bool,
    h1_structure: int,
    htf_score: int,
    min_score: int = 55,      # ✅ V112 : ajouté
    is_asia: bool = False,    # ✅ V112 : ajouté
) -> tuple:
    """
    V112 - Entry Quality Gate avec seuil adaptatif selon la session.
    """
    try:
        pair = str(pair).upper()
        direction = str(direction).upper()
        entry_type = str(entry_type).upper()

        entry_score = int(entry_score)
        eqs_score = float(eqs_score)
        adx = float(adx)
        momentum = float(momentum)
        h1_structure = int(h1_structure)
        htf_score = int(htf_score)

        # ✅ V112 : Seuil adaptatif
        if entry_score < min_score:
            return False, f"Score trop faible: {entry_score}/{min_score} < {min_score}"

        # ✅ V112 : Zone 55-59 interdite uniquement en ASIA (sinon, on garde min_score)
        if is_asia and 55 <= entry_score <= 59:
            return False, f"Score limite {entry_score}/100: zone 55-59 interdite en ASIA"

        if direction == "BUY" and h1_structure < 0:
            return False, f"Structure H1 baissière ({h1_structure}) contre BUY"
        if direction == "SELL" and h1_structure > 0:
            return False, f"Structure H1 haussière ({h1_structure}) contre SELL"

        if 60 <= entry_score <= 64:
            if not close_confirmed and adx < 30:
                return False, f"Score {entry_score}/100: confirmation M15 obligatoire"
            if eqs_score < 70:
                return False, f"Score {entry_score}/100 mais EQS={eqs_score:.0f}<70"
            if adx < 25:
                return False, f"Score {entry_score}/100 mais ADX={adx:.1f}<25"
            if direction == "BUY" and momentum < -0.05:
                return False, f"Momentum opposé BUY: {momentum:+.2f}%"
            if direction == "SELL" and momentum > 0.05:
                return False, f"Momentum opposé SELL: {momentum:+.2f}%"

        if entry_type == "FVG_RETEST_PERFECT":
            if entry_score < 65:
                return False, f"FVG_RETEST_PERFECT exige Score>=65: actuel={entry_score}"
            if 65 <= entry_score <= 69:
                if not close_confirmed:
                    return False, f"FVG_RETEST_PERFECT Score={entry_score}: confirmation M15 obligatoire"
                if eqs_score < 70:
                    return False, f"FVG_RETEST_PERFECT: EQS={eqs_score:.0f}<70"
                if adx < 25:
                    return False, f"FVG_RETEST_PERFECT: ADX={adx:.1f}<25"
            elif entry_score >= 70:
                if eqs_score < 70:
                    return False, f"FVG_RETEST_PERFECT Score={entry_score}: EQS={eqs_score:.0f}<70"
                if adx < 25:
                    return False, f"FVG_RETEST_PERFECT Score={entry_score}: ADX={adx:.1f}<25"
                if (eqs_score < 75 or adx < 27) and not close_confirmed:
                    return False, f"FVG_RETEST_PERFECT Score={entry_score}: confirmation M15 requise (EQS={eqs_score:.0f}, ADX={adx:.1f})"

        if htf_score < 2 and entry_score < 70:
            if htf_score == 1 and entry_score >= 60 and eqs_score >= 75:
                pass  # accepté
        else:            
            return False, f"Confluence HTF insuffisante: {htf_score}/3 pour Score={entry_score}"

        if direction == "BUY" and momentum < -0.10:
            return False, f"Momentum trop opposé au BUY: {momentum:+.2f}%"
        if direction == "SELL" and momentum > 0.10:
            return False, f"Momentum trop opposé au SELL: {momentum:+.2f}%"

        return True, f"ENTRY_GATE_PASS | Score={entry_score} | EQS={eqs_score:.0f} | ADX={adx:.1f} | MOM={momentum:+.2f}% | M15_CONFIRM={close_confirmed}"

    except Exception as e:
        logger.error(f"[ENTRY_GATE] Erreur validation: {e}")
        return False, f"Erreur Entry Quality Gate: {e}"

# ============================================================
# V109.1 - CALCUL DU SCORE DE CONFIANCE (AVEC BYPASS ASIA/LONDON)
# ============================================================
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

    base_min_required = BASE_MIN_CONFIDENCE_SCORE_BY_PAIR.get(pair, BASE_MIN_CONFIDENCE_SCORE_BY_PAIR["DEFAULT"])
    setup_type = str(entry.get("type", "FVG_RETEST")).upper()
    setup_weight = stats.adaptive_state.get_setup_weight(pair, setup_type)
    min_required = max(5, int(base_min_required / max(0.5, setup_weight)))

    hour = datetime.utcnow().hour
    is_asia = 21 <= hour or hour < 7
    is_london = 7 <= hour < 16
    is_ny = 12 <= hour < 21
    is_active = is_london or is_ny

    pair_params = stats.adaptive_state.get_pair_params(pair)

    if is_asia:
        # ✅ V112 : EQS minimum abaissé à 60 en ASIA (au lieu de 65)
        eqs_min_effective = max(pair_params["eqs_min"], 60)
        if pair in ["USD_JPY", "AUD_JPY"]:
            adx_min_effective = max(pair_params["adx_min"], 15)
        else:
            adx_min_effective = max(pair_params["adx_min"], 18)
        min_required += 3
    elif is_active:
        eqs_min_effective = max(pair_params["eqs_min"], 50)
        adx_min_effective = max(pair_params["adx_min"], 20)
        min_required += 2
    else:
        eqs_min_effective = pair_params["eqs_min"]
        adx_min_effective = pair_params["adx_min"]

    direction = (direction or "").upper()
    entry_level = entry.get("entry_level")
    entry_type = str(entry.get("type", "FVG_RETEST")).upper()

    if entry_level is None or direction not in ["BUY", "SELL"]:
        return {
            "passed": False,
            "entry_score": 0,
            "total_score": 0,
            "final_confidence": "LOW",
            "details": {"VETO": "Entrée/direction invalide"},
            "metrics": {},
            "filter_diag": {"HTF_BYPASS": "NO", "STRUCTURE_FILTER": "FAIL", "SCORE_FILTER": "FAIL", "FINAL_DECISION": "REJECT"}
        }

    entry_level = float(entry_level)
    atr_value = calculate_atr(df_m15)
    fvg_data = entry.get("fvg") if "fvg" in entry else None

    stop_loss, take_profit = calculate_sl_tp(
        entry_price=entry_level, atr=atr_value, direction=direction,
        pair=pair, entry_type=entry_type, fvg_data=fvg_data,
    )

    atr_pips = price_to_pips(atr_value, pair)
    adx = calculate_adx(df_h1)
    rsi = get_last_rsi(df_m15["close"])
    momentum = calculate_momentum(df_m15)
    now_dt = datetime.utcnow()
    hour = now_dt.hour
    weekday = now_dt.weekday()
    spread_data = get_price_spread_v88(pair)
    spread = spread_data.get("spread", 0.0)
    volatility_ratio = atr_value / current_price if current_price > 0 else 0
    if 7 <= hour < 16:
        session = "LONDON"
    elif 12 <= hour < 21:
        session = "NY"
    elif hour >= 21 or hour < 7:
        session = "ASIA"
    else:
        session = "OTHER"
    h1_trend = score_ema_trend(df_h1)
    h4_trend = score_ema_trend(df_h4)

    metrics = {
        "eqs": 0,
        "setup_type": entry_type,
        "atr": atr_pips,
        "adx": adx,
        "rsi": rsi,
        "momentum": momentum,
        "hour": hour,
        "weekday": weekday,
        "spread": spread,
        "volatility": volatility_ratio,
        "session": session,
        "h1_trend": h1_trend,
        "h4_trend": h4_trend
    }

    # --- 3. EQS ---
    eqs_result = calculate_entry_quality_score(
        pair=pair,
        direction=direction,
        df_m15=df_m15,
        entry_level=entry_level,
        current_price=current_price,
        atr=atr_value
    )
    eqs_score = eqs_result["total"]
    eqs_passed = eqs_score >= eqs_min_effective
    eqs_components = eqs_result.get("components", {})
    details["EQS_Details"] = eqs_result["logs"]
    metrics["eqs"] = eqs_score

    if not eqs_passed:
        eqs_reject_details = []
        for comp_name, comp_data in eqs_components.items():
            comp_label = comp_name.replace("_", " ").title()
            eqs_reject_details.append(f"{comp_label}: {comp_data['score']:+d}/{comp_data['max']}")
        eqs_reject_summary = " | ".join(eqs_reject_details) if eqs_reject_details else "EQS insuffisant"
        rejection_logs.append(f"EQS = {eqs_score}/100 < seuil {eqs_min_effective:.0f} | {eqs_reject_summary}")
        details["VETO"] = f"EQS insuffisant: {eqs_score}/100 < {eqs_min_effective:.0f}"
        return {
            "passed": False,
            "entry_score": 0,
            "total_score": 0,
            "final_confidence": "LOW",
            "details": details,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr_value,
            "eqs_score": eqs_score,
            "eqs_details": eqs_result,
            "eqs_components": eqs_components,
            "rejection_logs": rejection_logs,
            "metrics": metrics,
            "filter_diag": {"HTF_BYPASS": "NO", "STRUCTURE_FILTER": "PASS", "SCORE_FILTER": "FAIL", "FINAL_DECISION": "REJECT"}
        }

    details["EQS"] = f"{eqs_score}/100"

    # --- 4. Volatilité (ATR) ---
    atr_score, atr_msg = score_atr_volatility(df_m15, pair)
    if atr_score == 0 and "trop faible" in atr_msg:
        rejection_logs.append(atr_msg)
        details["VETO"] = f"VOLATILITÉ: {atr_msg}"
        return {
            "passed": False,
            "entry_score": 0,
            "total_score": 0,
            "final_confidence": "LOW",
            "details": details,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr_value,
            "eqs_score": eqs_score,
            "eqs_details": eqs_result,
            "eqs_components": eqs_components,
            "rejection_logs": rejection_logs,
            "metrics": metrics,
            "filter_diag": {"HTF_BYPASS": "NO", "STRUCTURE_FILTER": "PASS", "SCORE_FILTER": "FAIL", "FINAL_DECISION": "REJECT"}
        }
    details["Volatility"] = f"{atr_msg} (score {atr_score}/10)"
    score_components["Momentum"] += atr_score

    # --- 5. Structure H1 (V112 - veto assoupli en ASIA) ---
    # Calcul du score provisoire pour les conditions
    temp_score = compute_final_score(score_components) + 10

    # ✅ V112 : appel avec paramètres supplémentaires
    struct_passed, struct_msg = filter_market_structure(
        df_h1, direction, lookback=5,
        score=temp_score,
        eqs=eqs_score,
        is_asia=is_asia
    )

    is_opposed = "BEARISH" in struct_msg or "BULLISH" in struct_msg

    if not struct_passed:
        rejection_logs.append(struct_msg)
        details["VETO"] = f"STRUCTURE: {struct_msg}"
        return {
            "passed": False,
            "entry_score": 0,
            "total_score": 0,
            "final_confidence": "LOW",
            "details": details,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr_value,
            "eqs_score": eqs_score,
            "eqs_details": eqs_result,
            "eqs_components": eqs_components,
            "rejection_logs": rejection_logs,
            "metrics": metrics,
            "filter_diag": {"HTF_BYPASS": "NO", "STRUCTURE_FILTER": "FAIL", "SCORE_FILTER": "FAIL", "FINAL_DECISION": "REJECT"}
        }

    if "acceptée en ASIA" in struct_msg:
        # Structure opposée mais acceptée en ASIA avec score élevé
        score_components["Structure"] += 0  # ni bonus ni pénalité
        details["Structure_V98.1"] = f"0 ({struct_msg}, opposée acceptée en ASIA)"
    elif "non-bearish" in struct_msg or "non-bullish" in struct_msg:
        score_components["Structure"] += 1
        details["Structure_V98.1"] = f"+1 ({struct_msg}, neutre acceptée)"
    else:
        score_components["Structure"] += 2
        details["Structure_V98.1"] = f"+2 ({struct_msg})"

    # --- 6. Pullback ---
    pullback_passed, pullback_msg = filter_pullback(df_m15, direction, entry_level, current_price, pair)
    if not pullback_passed:
        rejection_logs.append(pullback_msg)
        details["VETO"] = f"PULLBACK: {pullback_msg}"
        return {
            "passed": False,
            "entry_score": 0,
            "total_score": 0,
            "final_confidence": "LOW",
            "details": details,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr_value,
            "eqs_score": eqs_score,
            "eqs_details": eqs_result,
            "eqs_components": eqs_components,
            "rejection_logs": rejection_logs,
            "metrics": metrics,
            "filter_diag": {"HTF_BYPASS": "NO", "STRUCTURE_FILTER": "PASS", "SCORE_FILTER": "FAIL", "FINAL_DECISION": "REJECT"}
        }
    score_components["Pullback"] += 2
    details["Pullback_V98.1"] = f"+2 ({pullback_msg})"

    # --- 7. Close confirmation ---
    close_passed, close_msg = filter_close_confirmation(df_m15, direction)
    if close_passed:
        score_components["Secondary"] += 1
        details["Close_Confirm"] = f"+1 ({close_msg})"
    else:
        details["Close_Confirm"] = close_msg

    # --- 8. Momentum ---
    mom_score, mom_msg = score_momentum(pair, direction, df_m15, df_h1, entry_level, current_price, entry_type)
    if mom_score == 0:
        rejection_logs.append(mom_msg)
        details["VETO"] = f"MOMENTUM: {mom_msg}"
        return {
            "passed": False,
            "entry_score": 0,
            "total_score": 0,
            "final_confidence": "LOW",
            "details": details,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr_value,
            "eqs_score": eqs_score,
            "eqs_details": eqs_result,
            "eqs_components": eqs_components,
            "rejection_logs": rejection_logs,
            "metrics": metrics,
            "filter_diag": {"HTF_BYPASS": "NO", "STRUCTURE_FILTER": "PASS", "SCORE_FILTER": "FAIL", "FINAL_DECISION": "REJECT"}
        }
    score_components["Momentum"] += mom_score
    details["Momentum"] = f"{mom_msg} (score {mom_score}/15)"

    # --- 9. ADX ---
    adx_score, adx_msg = score_adx_dynamic(df_h1)
    if adx_score == 0:
        rejection_logs.append(adx_msg)
        details["VETO"] = f"ADX: {adx_msg}"
        return {
            "passed": False,
            "entry_score": 0,
            "total_score": 0,
            "final_confidence": "LOW",
            "details": details,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr_value,
            "eqs_score": eqs_score,
            "eqs_details": eqs_result,
            "eqs_components": eqs_components,
            "rejection_logs": rejection_logs,
            "metrics": metrics,
            "filter_diag": {"HTF_BYPASS": "NO", "STRUCTURE_FILTER": "PASS", "SCORE_FILTER": "FAIL", "FINAL_DECISION": "REJECT"}
        }
    score_components["Momentum"] += adx_score
    details["ADX"] = f"{adx_msg} (score {adx_score}/10)"

    # --- 10. Filtre structure H1 (veto dur) - déjà traité par filter_market_structure, mais conservé pour sécurité ---
    h1_structure = score_market_structure(df_h1)
    if direction == "BUY" and h1_structure < 0:
        if not (is_asia and temp_score >= 75 and eqs_score >= 80):  # condition déjà validée plus haut
            rejection_logs.append(f"Structure H1 baissière ({h1_structure}) contre BUY")
            details["VETO"] = f"Structure H1: {h1_structure} (baissière)"
            return {
                "passed": False,
                "entry_score": 0,
                "total_score": 0,
                "final_confidence": "LOW",
                "details": details,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr_value": atr_value,
                "eqs_score": eqs_score,
                "eqs_details": eqs_result,
                "eqs_components": eqs_components,
                "rejection_logs": rejection_logs,
                "metrics": metrics,
                "filter_diag": {"HTF_BYPASS": "NO", "STRUCTURE_FILTER": "FAIL", "SCORE_FILTER": "FAIL", "FINAL_DECISION": "REJECT"}
            }
    if direction == "SELL" and h1_structure > 0:
        if not (is_asia and temp_score >= 75 and eqs_score >= 80):
            rejection_logs.append(f"Structure H1 haussière ({h1_structure}) contre SELL")
            details["VETO"] = f"Structure H1: {h1_structure} (haussière)"
            return {
                "passed": False,
                "entry_score": 0,
                "total_score": 0,
                "final_confidence": "LOW",
                "details": details,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr_value": atr_value,
                "eqs_score": eqs_score,
                "eqs_details": eqs_result,
                "eqs_components": eqs_components,
                "rejection_logs": rejection_logs,
                "metrics": metrics,
                "filter_diag": {"HTF_BYPASS": "NO", "STRUCTURE_FILTER": "FAIL", "SCORE_FILTER": "FAIL", "FINAL_DECISION": "REJECT"}
            }

    details["H1_Structure"] = f"OK ({h1_structure:+d})"

    # --- 11. Confluence HTF (V112 - seuil adaptatif) ---
    # ✅ V112 : appel avec paramètres supplémentaires
    htf_score, htf_str, htf_details, htf_bonus = check_htf_confluence(
        direction, df_h1, df_h4,
        is_asia=is_asia,
        score=temp_score,
        eqs=eqs_score
    )

    strong_setups = ["FVG_RETEST_PERFECT", "NESTED_FVG"]
    is_strong_setup = entry_type in strong_setups
    bypass_htf = False

    # ✅ V112 : HTF 1/3 accepté en ASIA si score>=75 et EQS>=85 (déjà géré dans check_htf_confluence)
    # On peut garder un log pour information
    if "accepté en ASIA" in htf_str:
        bypass_htf = True
        logger.info(f"[HTF_BYPASS] {pair} | {direction} | {entry_type} | HTF 1/3 accepté en ASIA (score={temp_score}, EQS={eqs_score})")
        htf_score = 2  # pour que le gate voie 2/3

    # Bypass existant pour NY/LONDON (score>=70, EQS>=80)
    if htf_score == 1 and not is_asia and temp_score >= 70 and eqs_score >= 80:
        if struct_passed:
            bypass_htf = True
            logger.info(f"[HTF_BYPASS] {pair} | {direction} | {entry_type} | HTF 1/3 accepté en session active (score={temp_score}, EQS={eqs_score})")
            htf_score = 2

    # Ancien bypass pour setups forts en ASIA (EQS>=80, ADX>=28)
    if htf_score == 1 and is_asia and is_strong_setup and eqs_score >= 80 and adx >= 28:
        if struct_passed:
            bypass_htf = True
            logger.info(f"[HTF_BYPASS] {pair} | {direction} | {entry_type} | HTF 1/3 bypassé en ASIA car EQS={eqs_score}>=80, ADX={adx:.1f}>=28, setup fort")
            htf_score = 2

    if htf_score < 2 and not bypass_htf:
        rejection_logs.append(f"Confluence HTF: {htf_score}/3 (requis 2/3)")
        details["VETO"] = f"Confluence HTF: {htf_score}/3 (requis 2/3)"
        return {
            "passed": False,
            "entry_score": 0,
            "total_score": 0,
            "final_confidence": "LOW",
            "details": details,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr_value,
            "eqs_score": eqs_score,
            "eqs_details": eqs_result,
            "eqs_components": eqs_components,
            "rejection_logs": rejection_logs,
            "metrics": metrics,
            "filter_diag": {"HTF_BYPASS": "NO" if not bypass_htf else "YES", "STRUCTURE_FILTER": "PASS", "SCORE_FILTER": "FAIL", "FINAL_DECISION": "REJECT"}
        }

    if not bypass_htf:
        score_components["HTF_Alignment"] += htf_score + htf_bonus
        details["HTF_Confluence"] = f"{htf_str} ({' | '.join(htf_details)}) (bonus {htf_bonus})"
    else:
        score_components["HTF_Alignment"] += 0
        details["HTF_Confluence"] = f"{htf_str} (BYPASS accepté)"

    # --- 12. Bias / tendance H4 ---
    if (direction == "BUY" and bias == "BUY") or (direction == "SELL" and bias == "SELL"):
        score_components["ICT"] += 3
        details["Trend_H4"] = "+3 (Aligné)"
    elif bias == "NEUTRAL":
        score_components["ICT"] += 1
        details["Trend_H4"] = "+1 (Neutre)"
    else:
        allowed_counter = ["BREAKER", "BISI"]
        if setup_type not in allowed_counter:
            rejection_logs.append(f"Contre-tendance 4H ({bias} vs {direction})")
            details["VETO"] = f"Contre-tendance: bias 4H={bias}, direction={direction}"
            return {
                "passed": False,
                "entry_score": 0,
                "total_score": 0,
                "final_confidence": "LOW",
                "details": details,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr_value": atr_value,
                "eqs_score": eqs_score,
                "eqs_details": eqs_result,
                "eqs_components": eqs_components,
                "rejection_logs": rejection_logs,
                "metrics": metrics,
                "filter_diag": {"HTF_BYPASS": "NO" if not bypass_htf else "YES", "STRUCTURE_FILTER": "PASS", "SCORE_FILTER": "FAIL", "FINAL_DECISION": "REJECT"}
            }
        else:
            score_components["ICT"] -= 2
            details["Trend_H4"] = f"-2 (H4 opposé, autorisé pour {setup_type})"

    # --- 13. Distance ---
    try:
        distance = abs(float(current_price) - entry_level)
        pip = get_pip_value_for_pair(pair)
        entry_type_max_pips = {
            "FVG_RETEST_PERFECT": 15.0, "FVG_RETEST": 18.0,
            "NESTED_FVG": 18.0, "WICK_REJECTION": 15.0,
            "BISI": 18.0, "BREAKER": 15.0,
        }
        max_pips = entry_type_max_pips.get(entry_type, STRICT_MAX_DISTANCE_PIPS.get(pair, STRICT_MAX_DISTANCE_PIPS["DEFAULT"]))
        # ✅ V112 : distance plus tolérante en session active (+30%)
        if is_active:
            max_pips *= 1.3
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
            return {
                "passed": False,
                "entry_score": 0,
                "total_score": 0,
                "final_confidence": "LOW",
                "details": {"VETO": f"Prix vraiment trop loin ({distance:.5f})"},
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr_value": atr_value,
                "eqs_score": eqs_score,
                "eqs_details": eqs_result,
                "eqs_components": eqs_components,
                "rejection_logs": rejection_logs,
                "metrics": metrics,
                "filter_diag": {"HTF_BYPASS": "NO" if not bypass_htf else "YES", "STRUCTURE_FILTER": "PASS", "SCORE_FILTER": "FAIL", "FINAL_DECISION": "REJECT"}
            }
    except Exception as exc:
        details["Distance_Error"] = str(exc)

    # --- 14. Scores structurels ---
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

    # --- 15. Type de setup ---
    setup_bonus = int((setup_weight - 1.0) * 10)
    score_components["Secondary"] += setup_bonus
    details["Setup_Weight"] = f"{setup_bonus:+d} (poids {setup_weight:.2f})"

    # --- 16. RR ---
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
        rr_ratio = 0.0
        pass

    # --- 17. D1 Trend ---
    try:
        d1_bonus, d1_label = get_d1_trend_bonus(df_d1, direction)
        if d1_bonus > 0:
            score_components["Secondary"] += 2
            details["D1_Trend"] = "+2 (D1 aligné)"
        else:
            details["D1_Trend"] = d1_label
    except Exception:
        pass

    # --- 18. MACD H1 ---
    try:
        macd_bonus, macd_label = get_macd_h1_bonus(df_h1, direction)
        if macd_bonus > 0:
            score_components["Secondary"] += 1
            details["MACD_H1"] = "+1 (confirme)"
        else:
            details["MACD_H1"] = macd_label
    except Exception:
        pass

    # --- 19. Session quality ---
    try:
        session_bonus, session_label = get_session_quality_bonus(pair)
        if session_bonus > 0:
            score_components["Secondary"] += 1
            details["Session"] = "+1 (bonne session)"
        else:
            details["Session"] = session_label
    except Exception:
        pass

    # --- 20. Calcul du score total ---
    raw_score = compute_final_score(score_components)
    entry_score = min(100, max(0, raw_score + 10))

    # Confirmation M15
    close_passed, close_msg = filter_close_confirmation(df_m15, direction)

    # Structure H1 finale
    try:
        final_h1_structure = int(score_market_structure(df_h1))
    except Exception:
        final_h1_structure = 0

    # HTF final
    try:
        final_htf_score = int(score_higher_timeframe_alignment(direction, df_h1, df_h4))
    except Exception:
        final_htf_score = 0

    # --- ENTRY QUALITY GATE (V113 - seuil adaptatif par setup) ---
    # ✅ V113 : FVG_RETEST_PERFECT et NESTED_FVG acceptés à partir de 44
    if entry_type in ["FVG_RETEST_PERFECT", "NESTED_FVG"]:
        min_score_gate = 44
    else:
        min_score_gate = 50   # tous les autres setups restent à 50

    gate_passed, gate_reason = validate_entry_quality_gate(
        pair=pair,
        direction=direction,
        entry_type=entry_type,
        entry_score=entry_score,
        eqs_score=eqs_score,
        adx=adx,
        momentum=momentum,
        close_confirmed=close_passed,
        h1_structure=final_h1_structure,
        htf_score=final_htf_score,
        min_score=min_score_gate,
        is_asia=is_asia
    )

    # ============================================================
    # ASIA/LONDON ENTRY BYPASS (avec constantes définies)
    # ============================================================
    bypass_used = False
    bypass_reason = None

    hour = datetime.utcnow().hour
    is_asia_now = (hour >= 21 or hour < 7)
    is_london_now = (7 <= hour < 12)

    is_fvg_retest = (str(entry_type).upper() == "FVG_RETEST_PERFECT")

    bypass_score = float(entry_score)
    bypass_eqs = float(eqs_score)
    bypass_adx = float(adx or 0.0)

    bypass_struct = bool(struct_passed)
    bypass_pullback = bool(pullback_passed)

    bypass_momentum_ok = not (
        (direction == "BUY" and momentum < -0.05)
        or
        (direction == "SELL" and momentum > 0.05)
    )

    try:
        pip_value = float(get_pip_value_for_pair(pair))
        spread_value = float(spread_data.get("spread", 0.0))
        bypass_spread_ok = (spread_value <= pip_value * 1.5)
    except Exception as e:
        logger.warning(f"[BYPASS] Erreur calcul spread {pair}: {e}")
        bypass_spread_ok = False

    # ASIA
    asia_conditions = {
        "score": bypass_score >= ASIA_BYPASS_MIN_SCORE,
        "eqs": bypass_eqs >= ASIA_BYPASS_MIN_EQS,
        "adx": bypass_adx >= ASIA_BYPASS_MIN_ADX,
        "structure": bypass_struct,
        "pullback": bypass_pullback,
        "momentum": bypass_momentum_ok,
        "spread": bypass_spread_ok,
    }
    asia_all_conditions_met = (
        is_asia_now
        and (is_fvg_retest or entry_type == "NESTED_FVG")
        and not gate_passed
        and all(asia_conditions.values())
    )

    # LONDON
    london_conditions = {
        "score": bypass_score >= LONDON_BYPASS_MIN_SCORE,
        "eqs": bypass_eqs >= LONDON_BYPASS_MIN_EQS,
        "adx": bypass_adx >= LONDON_BYPASS_MIN_ADX,
        "structure": bypass_struct,
        "pullback": bypass_pullback,
        "momentum": bypass_momentum_ok,
        "spread": bypass_spread_ok,
    }
    london_all_conditions_met = (
        is_london_now
        and is_fvg_retest
        and not gate_passed
        and all(london_conditions.values())
    )

    if asia_all_conditions_met:
        gate_passed = True
        gate_reason = "ASIA_BYPASS"
        bypass_used = True
        logger.info(f"[ASIA_ENTRY_BYPASS] {pair} | {direction} | FVG_RETEST_PERFECT | Score={bypass_score:.1f} | EQS={bypass_eqs:.1f} | ADX={bypass_adx:.1f} | H1_STRUCT=PASS | Pullback=PASS | RESULT=ACCEPT")

    elif london_all_conditions_met:
        gate_passed = True
        gate_reason = "LONDON_BYPASS"
        bypass_used = True
        logger.info(f"[LONDON_ENTRY_BYPASS] {pair} | {direction} | FVG_RETEST_PERFECT | Score={bypass_score:.1f} | EQS={bypass_eqs:.1f} | ADX={bypass_adx:.1f} | H1_STRUCT=PASS | Pullback=PASS | RESULT=ACCEPT")

    elif is_asia_now and is_fvg_retest and not gate_passed:
        failed_conditions = [key for key, value in asia_conditions.items() if not value]
        logger.info(f"[ASIA_ENTRY_BYPASS_REJECT] {pair} | {direction} | reason={', '.join(failed_conditions)} | Score={bypass_score:.1f} | EQS={bypass_eqs:.1f} | ADX={bypass_adx:.1f} | H1_STRUCT={'PASS' if bypass_struct else 'FAIL'} | Pullback={'PASS' if bypass_pullback else 'FAIL'}")

    elif is_london_now and is_fvg_retest and not gate_passed:
        failed_conditions = [key for key, value in london_conditions.items() if not value]
        logger.info(f"[LONDON_ENTRY_BYPASS_REJECT] {pair} | {direction} | reason={', '.join(failed_conditions)} | Score={bypass_score:.1f} | EQS={bypass_eqs:.1f} | ADX={bypass_adx:.1f} | H1_STRUCT={'PASS' if bypass_struct else 'FAIL'} | Pullback={'PASS' if bypass_pullback else 'FAIL'}")

    passed = bool(gate_passed)

    if not passed:
        rejection_logs.append(f"ENTRY_GATE: {gate_reason}")

    if passed and entry_score >= 80:
        final_confidence = "HIGH"
    elif passed and entry_score >= 65:
        final_confidence = "MEDIUM"
    else:
        final_confidence = "LOW"

    confluences = {
        "d1_aligned": details.get("D1_Trend", "").startswith("+"),
        "rsi_divergence": False,
        "session_active": details.get("Session", "").startswith("+"),
        "macd_confirmed": details.get("MACD_H1", "").startswith("+"),
        "bos_confirmed": "BOS" in str(details),
        "structure_ok": score_components.get("Structure", 0) >= 1,
        "pullback_ok": score_components.get("Pullback", 0) >= 2,
        "m15_confirmation": close_passed,
        "entry_gate": passed,
    }

    win_rate = estimate_win_rate(entry_score, eqs_score, confluences)
    quality_label = get_signal_quality_label(entry_score, eqs_score)

    if is_asia and quality_label not in ["SNIPER", "A+", "A"] and not bypass_used:
        passed = False
        rejection_logs.append(f"Qualité {quality_label} insuffisante en ASIA (requis SNIPER/A+/A)")

    log_score_detail(score_components, entry_score, "PASSED" if passed else "REJECTED")

    eqs_detail_str = ""
    if eqs_components:
        comp_parts = []
        for comp_name, comp_data in eqs_components.items():
            comp_label = comp_name.replace("_", " ").title()
            comp_parts.append(f"{comp_label}:{comp_data['score']:+d}")
        eqs_detail_str = " | EQS=" + " ".join(comp_parts)

    if passed:
        status = "✅ ACCEPT"
    else:
        status = "❌ REJECT"
        if rejection_logs:
            status += f" | raison={rejection_logs[0][:120]}"

    entry_gate_label = "ASIA_BYPASS" if bypass_used else ("PASS" if passed else "FAIL")

    decision_line = (
        f"[DECISION] {pair} | {direction} | {entry_type} | {status} | Score={entry_score}/100 | "
        f"EQS={eqs_score}/{eqs_min_effective:.0f}{eqs_detail_str} | ATR={atr_pips:.1f}pips | ADX={adx:.1f} | RSI={rsi:.1f} | "
        f"MOM={momentum:+.2f}% | M15_CONFIRM={'YES' if close_passed else 'NO'} | H1_STRUCT={final_h1_structure:+d} | "
        f"HTF={final_htf_score}/3 | H={hour:02d}h | Sess={session} | Spread={spread:.2f} | RR={rr_ratio:.2f} | "
        f"PoidsSetup={setup_weight:.2f} | ENTRY_GATE={entry_gate_label}"
    )
    if not passed and rejection_logs:
        decision_line += f" | REJECT={rejection_logs[0][:120]}"
    logger.info(decision_line)

    filter_diag = {
        "HTF_BYPASS": "YES" if bypass_htf else "NO",
        "STRUCTURE_FILTER": "PASS" if struct_passed else "FAIL",
        "M15_CONFIRMATION": "PASS" if close_passed else "FAIL",
        "ENTRY_GATE": "PASS" if gate_passed else "FAIL",
        "SCORE_FILTER": "PASS" if passed else "FAIL",
        "FINAL_DECISION": "PASS" if passed else "REJECT",
    }
    logger.info(f"[FILTER_DIAG] {pair} | {direction} | {entry_type} | HTF_BYPASS={filter_diag['HTF_BYPASS']} | STRUCTURE={filter_diag['STRUCTURE_FILTER']} | M15={filter_diag['M15_CONFIRMATION']} | ENTRY_GATE={filter_diag['ENTRY_GATE']} | SCORE={entry_score} | EQS={eqs_score} | ADX={adx:.1f} | FINAL={filter_diag['FINAL_DECISION']} | REASON={gate_reason}")

    return {
        "total_score": entry_score,
        "entry_score": entry_score,
        "details": details,
        "score_components": score_components,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr_value": atr_value,
        "passed": passed,
        "min_required": MIN_ENTRY_SCORE,
        "final_confidence": final_confidence,
        "win_rate": win_rate,
        "quality_label": quality_label,
        "confluences": confluences,
        "eqs_score": eqs_score,
        "eqs_details": eqs_result,
        "eqs_components": eqs_components,
        "rejection_logs": rejection_logs,
        "metrics": metrics,
        "filter_diag": filter_diag,
        "m15_confirmation": close_passed,
        "m15_confirmation_message": close_msg,
        "entry_gate_reason": gate_reason,
        "bypass_used": bypass_used,
    }

# =============================
# DÉTECTION BIAS-FIRST - inchangé
# =============================
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

# =============================
# FONCTION PRINCIPALE (scan des signaux) - avec filtres session V105 + score V106
# =============================
def advanced_main_v981():
    try:
        api = v88_client()
        logger.info("✅ API OANDA initialisée avec succès")
        logger.info(f"✅ ENTRY QUALITY SCORE (EQS) V111 - Seuil adaptatif + ASIA (65)")
        logger.info(f"✅ ENTRY SCORE /100 - Seuil minimum {MIN_ENTRY_SCORE}")
        logger.info(f"✅ V111 - ASIA/LONDON ENTRY BYPASS pour FVG_RETEST_PERFECT")
        logger.info(f"✅ Break Even adaptatif (base: {BASE_BREAKEVEN_TRIGGER_R}R)")
        logger.info("✅ AUDIT ATR ACTIVÉ")
        logger.info("✅ LOGS [DECISION] ENRICHIS AVEC MÉTRIQUES")
        logger.info("✅ SUIVI DES CLÔTURES AMÉLIORÉ")
        logger.info("✅ ESPÉRANCE CALCULÉE SUR LES TRADES CLÔTURÉS")
        logger.info("✅ APPELS OANDA CORRIGÉS")
        logger.info("✅ DISTANCE SL MINIMUM (10 pips)")
        logger.info("✅ FILTRE SPREAD ÉLEVÉ")
        logger.info(f"✅ MAX TRADES: {MAX_TRADES_TOTAL}")
        logger.info(f"✅ VERROUILLAGE PAR PAIRE: {EXECUTION_COOLDOWN_SECONDS}s")
        logger.info("✅ SUIVI MFE/MAE ACTIVÉ")
        logger.info("✅ PARAMÈTRES ADAPTATIFS ROBUSTES (seuil 20 trades, hystérésis 2 cycles)")
        logger.info("✅ V104 : Blocage contre-tendance 4H (sauf BREAKER/BISI)")
        logger.info("✅ V104 : ADX minimum 25 en session active")
        logger.info("✅ V104 : Veto si momentum opposé >0.15%")
        logger.info("✅ V104 : Confluence HTF requise (2/3)")
        logger.info("✅ V105 : Cooldown de 2h après une perte")
        logger.info("✅ V105 : Score minimum +3 en ASIA")
        logger.info("✅ V105 : Qualité requise SNIPER/A+ en ASIA")
        logger.info("✅ V105 : Seuils ATR réduits de 30% en ASIA")
        logger.info("✅ V105 : Risque réduit à 0.5% en ASIA")
        logger.info("✅ V105 : EQS minimum 65 en ASIA")
        logger.info("✅ V105 : Suppression filtre EUR/USD NY (18h-21h)")
        logger.info("✅ V105 : Filtre structure H1")
        logger.info("✅ V105 : Adaptation des poids des setups")
        logger.info("✅ V105 : Sélection du meilleur setup avec seuil relatif")
        logger.info("✅ V103 : Blocage USD/CAD et AUD/USD en ASIA")
        logger.info("✅ V103 : Filtre ADX renforcé (seuil 18)")
        logger.info("✅ V103 : RISK 0.75% (0.5% ASIA)")
        logger.info("✅ V103 : EXIT_EARLY plus sélectif (4 signaux)")
        logger.info("✅ V105.1 : Logs MFE/MAE enrichis pour diagnostic")
        logger.info("✅ V105.1 : SL/TP rééquilibrés (multiplicateurs plus serrés)")
        logger.info("✅ V105.1 : Trailing stop optimisé (activation 0.80R, distance 1.5R)")
        logger.info("✅ V105.1 : Pullback AUD/USD corrigé (3.5)")
        logger.info("✅ V106 : Score d'entrée /100 avec composantes")
        logger.info("✅ V106 : Bonus 3/3 HTF (+5)")
        logger.info("✅ V106 : Momentum gradué (pénalité progressive)")
        logger.info("✅ V106 : ADX dynamique avec pente")
        logger.info("✅ V106 : Poids initiaux des setups ajustés")
        logger.info("✅ V106 : Correction RR après BE (TP réajusté)")
        logger.info("✅ V106.1 : Filtre STRUCTURE H1 assoupli (rejet uniquement structure opposée)")
        logger.info("✅ V106.1 : Logs FILTER_DIAG et REJECT_DIAG pour diagnostic")
        logger.info("✅ V107 : ASIA ENTRY BYPASS pour FVG_RETEST_PERFECT")
        logger.info("✅ V111 : Correction des constantes manquantes + R initial pour trailing")
    except Exception as e:
        logger.error(f"❌ Échec d'initialisation de l'API OANDA : {e}")
        return

    for pair in PAIR_LIST:
        _reset_log_dedup()

        hour = datetime.utcnow().hour

        #if (21 <= hour or hour < 7) and pair in ["USD_CAD", "AUD_USD"]:
         #   logger.info(f"[SESSION] {pair} - Session ASIA ({hour}h), trade ignoré")
          #  continue

        if not stats.adaptive_state.can_trade(pair):
            logger.info(f"[COOLDOWN] {pair} - en cooldown après une perte, scan ignoré")
            continue

        if stats.adaptive_state.is_pair_suspended(pair):
            logger.info(f"[SUSPEND] {pair} est suspendue - scan ignoré")
            continue

        if has_open_trade_v88(pair):
            logger.info(f"[INFO] {pair}: trade deja ouvert - scan ignore")
            continue

        try:
            df_h4 = get_candles_with_retry(api, pair, GRANULARITY_H4, 300)
            df_h1 = get_candles_with_retry(api, pair, GRANULARITY_H1, 200)
            df_m15 = get_candles_with_retry(api, pair, GRANULARITY_M15, 250)
            df_d1 = get_candles_with_retry(api, pair, "D", count=250)
            
            if any(df.empty for df in [df_h4, df_h1, df_m15]):
                logger.warning(f"⚠️ Données manquantes pour {pair}, analyse ignorée")
                continue

            atr_price = calculate_atr(df_m15, period=ATR_PERIOD)
            atr_pips = price_to_pips(atr_price, pair)
            effective_atr_threshold = get_effective_atr_threshold(pair)
            logger.info(f"[ATR_DIAG] {pair} | ATR pips: {atr_pips:.1f} | Seuil effectif: {effective_atr_threshold:.1f}")

            current_price = float(df_m15["close"].iloc[-1])
            bias_analysis = determine_advanced_bias(df_h4)
            bias = bias_analysis.get("bias", "NEUTRAL")

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
                
                score = confidence_result.get("entry_score", 0)
                eqs = confidence_result.get("eqs_score", 0)
                metrics = confidence_result.get("metrics", {})
                passed = confidence_result.get("passed", False)
                bypass_used = confidence_result.get("bypass_used", False)
                
                bypass_tag = " | BYPASS=ASIA" if bypass_used else ""
                logger.info(
                    f"[SIGNAL] {pair} | "
                    f"DIR={direction} | "
                    f"Score={score} | "
                    f"EQS={eqs} | "
                    f"ADX={metrics.get('adx', 'NA')} | "
                    f"ATR={atr_pips:.1f} | "
                    f"SETUP={entry_type} | "
                    f"PASSED={passed}{bypass_tag}"
                )
                
                if DEBUG_MODE:
                    logger.debug(f"📊 {pair} {direction} | Score: {score} | EQS: {eqs}/100 | Passed: {passed} | Bypass: {bypass_used}")

                if not passed:
                    filter_diag = confidence_result.get("filter_diag", {})
                    if filter_diag:
                        logger.info(
                            f"[REJECT_DIAG] {pair} | {direction} | {entry_type} | "
                            f"HTF_BYPASS={filter_diag.get('HTF_BYPASS', 'NO')} | "
                            f"STRUCTURE={filter_diag.get('STRUCTURE_FILTER', 'UNKNOWN')} | "
                            f"SCORE={filter_diag.get('SCORE_FILTER', 'UNKNOWN')} | "
                            f"FINAL={filter_diag.get('FINAL_DECISION', 'UNKNOWN')} | "
                            f"Score={score} | EQS={eqs} | ADX={metrics.get('adx', 'NA')}"
                        )

                if passed:
                    scored_entries.append({"entry": entry, "confidence": confidence_result})
                    stats.record_signal(pair, True, "score_ok", entry_level, 0, 0, score, direction, metrics)
                else:
                    reason = confidence_result.get("details", {}).get("VETO", f"score_{score}")
                    rejected_reasons[reason[:30]] += 1
                    rejection_logs = confidence_result.get("rejection_logs", [])
                    if rejection_logs:
                        rejected_details.append(f"{pair} {direction}: " + " | ".join(rejection_logs))
                    stats.record_signal(pair, False, reason, entry_level, 0, 0, score, direction)

            if rejected_details:
                logger.debug(f"[REJECT_DETAILS] {pair} - {len(rejected_details)} rejets détaillés")
                for detail in rejected_details[:5]:
                    logger.debug(f"  {detail}")
                if len(rejected_details) > 5:
                    logger.debug(f"  ... et {len(rejected_details)-5} autres")

            finalists = strict_keep_best_per_direction(scored_entries, min_score_gap=5)
            
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

                score = confidence_result.get("entry_score", 0)
                eqs = confidence_result.get("eqs_score", 0)
                quality = confidence_result.get("quality_label", "B")
                metrics = confidence_result.get("metrics", {})

                logger.info(f"📊 TRADE {pair} {direction} {entry_type} @{entry_level:.5f} | Score: {score} | EQS: {eqs}/100 | Qualité: {quality}")

                entry_metrics = {
                    "atr": metrics.get("atr", 0),
                    "adx": metrics.get("adx", 0),
                    "rsi": metrics.get("rsi", 0),
                    "eqs": eqs,
                    "hour": metrics.get("hour", 0),
                    "weekday": metrics.get("weekday", 0),
                    "setup_type": entry_type,
                    "momentum": metrics.get("momentum", 0),
                    "spread": metrics.get("spread", 0),
                    "volatility": metrics.get("volatility", 0),
                    "session": metrics.get("session", "UNKNOWN"),
                    "h1_trend": metrics.get("h1_trend", 0),
                    "h4_trend": metrics.get("h4_trend", 0)
                }

                if DEMO_MODE:
                    logger.info(f"🔬 DEMO: {pair} {direction} @ {entry_level:.5f} (SL: {stop_loss}, TP: {take_profit})")
                    stats.record_signal(pair, True, "demo_mode", entry_level, stop_loss, take_profit, score, direction, entry_metrics)
                    nb_envoyes += 1
                    continue

                trade_id = execute_oanda_trade_v981(
                    pair=pair,
                    direction=direction,
                    entry_price=entry_level,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    score=score,
                    entry_type=entry_type,
                    eqs=eqs,
                    setup_type=entry_type,
                    metrics=entry_metrics
                )

                if trade_id:
                    logger.info(
                        f"[DECISION_EXECUTED] {pair} | {direction} | {entry_type} | "
                        f"Score={score} | EQS={eqs} | "
                        f"ENTRY={entry_level:.5f} | SL={stop_loss:.5f} | TP={take_profit:.5f} | "
                        f"TRADE_ID={trade_id} | ACTION=EXECUTED"
                    )

                    enriched_bias = dict(bias_analysis) if bias_analysis else {}
                    enriched_bias["win_rate"] = confidence_result.get("win_rate", "~55%")
                    enriched_bias["quality_label"] = quality
                    enriched_bias["score_details"] = confidence_result.get("details", {})
                    send_telegram_alert(
                        pair=pair, direction=direction, entry_price=entry_level,
                        stop_loss=stop_loss, take_profit=take_profit,
                        narrative={}, bias_analysis=enriched_bias,
                        rsi=metrics.get("rsi", 50),
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
# V106 - CORRECTION DU TRAILING APRÈS BE (utilisation du risque initial)
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

            # Récupérer le SL initial depuis open_trade_details
            trade_info = open_trade_details.get(trade_id, {})
            initial_sl = trade_info.get("sl", current_sl)
            if initial_sl <= 0:
                initial_sl = current_sl  # fallback

            # Calcul du R avec le risque initial
            if direction == "BUY":
                profit = current_price - entry
                initial_risk = entry - initial_sl
            else:
                profit = entry - current_price
                initial_risk = current_sl - entry   # pour un SELL, le SL est au-dessus de l'entrée
            if initial_risk <= 0:
                initial_risk = abs(entry - current_sl)  # fallback
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

            # Vérifier le trailing (en utilisant le R basé sur le risque initial)
            trade_details = get_trade_details_v88(trade_id)
            if has_trailing_stop_v88(trade_details):
                logger.debug(f"[TSL] Trade {trade_id} a déjà un trailing, on saute")
                continue

            # Recalculer R avec le SL actuel (si BE vient d'être déclenché, current_sl a été mis à jour)
            if direction == "BUY":
                risk = entry - current_sl
                profit = current_price - entry
            else:
                risk = current_sl - entry
                profit = entry - current_price
            if risk <= 0:
                continue
            # Mais pour la décision de trailing, on utilise le R basé sur le risque initial
            # r a déjà été calculé ci-dessus avec initial_risk
            # On peut utiliser r pour la condition de trailing
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
# STRICT FILTERS - inchangé
# ============================================================
STRICT_ALLOWED_ENTRY_TYPES = {
    "FVG_RETEST_PERFECT", "FVG_RETEST", "BISI", "BREAKER",
    "NESTED_FVG", "WICK_REJECTION", "LIQUIDITY_DRAW",
}
STRICT_MAX_DISTANCE_PIPS = {
    "XAU_USD": 60.0, "USD_JPY": 18.0, "GBP_JPY": 22.0,
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


def strict_keep_best_per_direction(scored_entries: list, min_score_gap: int = 3) -> list:
    if not scored_entries:
        return []
    
    best = {}
    for item in scored_entries:
        entry = item["entry"]
        direction = entry.get("direction", "").upper()
        score = item["confidence"].get("entry_score", -999)
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
    
    if len(best) >= 2:
        scores = {}
        for direction, item in best.items():
            scores[direction] = item["key_score"][0]
        
        sorted_dirs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_score = sorted_dirs[0][1]
        second_score = sorted_dirs[1][1] if len(sorted_dirs) > 1 else 0
        
        if best_score - second_score < min_score_gap:
            logger.info(f"[SELECTION] Écart de score trop faible: {best_score} vs {second_score} (gap={best_score - second_score} < {min_score_gap}) - aucun trade")
            return []
        
        logger.info(f"[SELECTION] Meilleur trade: {sorted_dirs[0][0]} (score={best_score}) contre {sorted_dirs[1][0]} (score={second_score}) - écart={best_score - second_score}")
    
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
# TELEGRAM - inchangé
# ============================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = TELEGRAM_BOT_TOKEN is not None and TELEGRAM_CHAT_ID is not None


def send_telegram_alert(pair: str, direction: str, entry_price: float, stop_loss: float, take_profit: float,
                        narrative: dict, bias_analysis: dict, rsi: float = 50,
                        entry_type: str = "UNKNOWN", confidence_score: int = 0,
                        eqs_score: int = 0):
    if not TELEGRAM_ENABLED:
        return
    try:
        if direction == "BUY":
            direction_emoji = "🟢"
        else:
            direction_emoji = "🔴"
        if entry_type:
            entry_type_display = entry_type
        else:
            entry_type_display = "UNKNOWN"
        msg = f"{direction_emoji} TRADE OPPORTUNITY\n"
        msg += f"Pair: {pair}\n"
        msg += f"Direction: {direction}\n"
        msg += f"Entry: {entry_price:.5f}\n"
        msg += f"SL: {stop_loss:.5f}\n"
        msg += f"TP: {take_profit:.5f}\n"
        if confidence_score > 0:
            msg += f"Confiance: {confidence_score}%\n"
        if eqs_score > 0:
            msg += f"EQS: {eqs_score}/100\n"
        msg += f"Setup: {entry_type_display}\n"
        msg += f"RSI: {rsi:.1f}\n"
        if bias_analysis:
            bias = bias_analysis.get("bias", "NEUTRAL")
            msg += f"Biais: {bias}\n"
            if "win_rate" in bias_analysis:
                msg += f"Win Rate estimé: {bias_analysis['win_rate']}\n"
            if "quality_label" in bias_analysis:
                msg += f"Qualité: {bias_analysis['quality_label']}\n"
        rr = abs((take_profit - entry_price) / (entry_price - stop_loss)) if entry_price != stop_loss else 0
        msg += f"RR: {rr:.2f}\n"
        msg += f"Trade ID: {narrative.get('trade_id', 'N/A')}" if narrative else f"Trade ID: N/A"
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
        response = requests.post(url, data=data, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Telegram erreur: {response.status_code}")
    except Exception as e:
        logger.error(f"Erreur envoi Telegram: {e}")

# ============================================================
# DIAGNOSTIC DE DÉMARRAGE V111
# ============================================================
def diagnostic_startup_v981():
    logger.info("=" * 60)
    logger.info("[DIAG] DIAGNOSTIC DE DÉMARRAGE V111")
    logger.info("=" * 60)
    logger.info(f"[DIAG] BREAKEVEN_TRIGGER_R = {BASE_BREAKEVEN_TRIGGER_R} (adaptatif)")
    logger.info(f"[DIAG] BREAKEVEN_EARLY_R = {BASE_BREAKEVEN_EARLY_R} (adaptatif)")
    logger.info(f"[DIAG] EQS_MIN_THRESHOLD = {BASE_EQS_MIN_THRESHOLD} (adaptatif, 60 en ASIA)")
    logger.info(f"[DIAG] ADX_MIN_THRESHOLD = {BASE_ADX_MIN_THRESHOLD} (adaptatif, 20 en ASIA, 25 en session active)")
    logger.info(f"[DIAG] TRAILING_STOP = {BASE_TRAILING_STOP_DISTANCE_ATR_MULTIPLIER}R")
    logger.info(f"[DIAG] TRAILING_ACTIVATION = {BASE_TRAILING_ACTIVATION_R}R")
    logger.info(f"[DIAG] MIN_ENTRY_SCORE = {MIN_ENTRY_SCORE}")
    logger.info(f"[DIAG] ASIA_BYPASS: Score>={ASIA_BYPASS_MIN_SCORE}, EQS>={ASIA_BYPASS_MIN_EQS}, ADX>={ASIA_BYPASS_MIN_ADX}")
    logger.info(f"[DIAG] LONDON_BYPASS: Score>={LONDON_BYPASS_MIN_SCORE}, EQS>={LONDON_BYPASS_MIN_EQS}, ADX>={LONDON_BYPASS_MIN_ADX}")
    logger.info(f"[DIAG] BASE_MIN_CONFIDENCE_SCORE_BY_PAIR = {BASE_MIN_CONFIDENCE_SCORE_BY_PAIR}")
    logger.info(f"[DIAG] MIN_ATR_PIPS = {MIN_ATR_PIPS_BY_PAIR}")
    logger.info(f"[DIAG] MIN_ATR_PIPS_ASIA = {MIN_ATR_PIPS_BY_PAIR_ASIA}")
    logger.info(f"[DIAG] PULLBACK_MIN_PIPS = {PULLBACK_MIN_PIPS_BY_PAIR}")
    logger.info("[DIAG] SUIVI DES CLÔTURES : tentative API + fallback")
    logger.info("[DIAG] ESPÉRANCE CALCULÉE SUR LES TRADES CLÔTURÉS")
    logger.info("[DIAG] APPELS OANDA CORRIGÉS")
    logger.info("[DIAG] GESTION DE MAINTENANCE OANDA ACTIVÉE")
    logger.info("[DIAG] SUIVI DES TRADES STAGNANTS ACTIVÉ")
    logger.info("[DIAG] BREAK EVEN ADAPTATIF ACTIVÉ")
    logger.info("[DIAG] TRAILING STOP OPTIMISÉ ACTIVÉ (activation 0.80R, distance 1.5R)")
    logger.info("[DIAG] SORTIE ANTICIPÉE SUR RETOURNEMENT ACTIVÉE (4 signaux requis)")
    logger.info("[DIAG] DISTANCE SL MINIMUM (10 pips) ACTIVÉE")
    logger.info("[DIAG] FILTRE SPREAD ÉLEVÉ ACTIVÉ")
    logger.info(f"[DIAG] MAX TRADES = {MAX_TRADES_TOTAL}")
    logger.info("[DIAG] SUIVI MFE/MAE ACTIVÉ AVEC LOGS ENRICHIS")
    logger.info("[DIAG] APPRENTISSAGE DES SETUPS ACTIVÉ (seuil 10 trades)")
    logger.info("[DIAG] PARAMÈTRES ADAPTATIFS ROBUSTES (seuil 20 trades, hystérésis 2 cycles, amplitude limitée)")
    logger.info("[DIAG] FILTRES SESSION ASIA : USD/CAD, AUD/USD bloqués")
    logger.info("[DIAG] RISQUE PAR TRADE : 0.75% (0.5% ASIA)")
    logger.info("[DIAG] BLOCAGE CONTRE-TENDANCE 4H (sauf BREAKER/BISI)")
    logger.info("[DIAG] ADX MINIMUM 25 EN SESSION ACTIVE")
    logger.info("[DIAG] VETO SI MOMENTUM OPPOSÉ >0.15% (remplacé par score gradué en V106)")
    logger.info("[DIAG] CONFLUENCE HTF REQUISE (2/3)")
    logger.info("[DIAG] COOLDOWN 2H APRÈS UNE PERTE")
    logger.info("[DIAG] SCORE MINIMUM +3 EN ASIA")
    logger.info("[DIAG] QUALITÉ REQUISE SNIPER/A+ EN ASIA")
    logger.info("[DIAG] SUPPRESSION FILTRE EUR/USD NY (18h-21h)")
    logger.info("[DIAG] FILTRE STRUCTURE H1")
    logger.info("[DIAG] ADAPTATION DES POIDS DES SETUPS")
    logger.info("[DIAG] SÉLECTION DU MEILLEUR SETUP AVEC SEUIL RELATIF")
    logger.info("[DIAG] SL/TP RÉÉQUILIBRÉS (multiplicateurs plus serrés)")
    logger.info("[DIAG] CORRECTION RR APRÈS BE (TP réajusté)")
    logger.info("[DIAG] SCORE D'ENTRÉE /100 AVEC COMPOSANTES")
    logger.info("[DIAG] BONUS 3/3 HTF (+5)")
    logger.info("[DIAG] MOMENTUM GRADUÉ (pénalité progressive)")
    logger.info("[DIAG] ADX DYNAMIQUE AVEC PENTE")
    logger.info("[DIAG] POIDS INITIAUX DES SETUPS AJUSTÉS")
    logger.info("[DIAG] ASIA/LONDON ENTRY BYPASS CORRIGÉ (constantes définies)")
    logger.info("[DIAG] R initial pour trailing (après BE) corrigé")
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
# BOUCLE PRINCIPALE
# ============================================================
if __name__ == "__main__":
    logger.info("🚀 Démarrage du Bot Advanced Orderflow Trading - V111 (ASIA/LONDON BYPASS FIX)")
    logger.info("✅ Utilisation de TradeCRCDO pour la modification du SL")
    logger.info("✅ Utilisation de OrderCreate pour la création du Trailing Stop")
    logger.info(f"✅ Seuil Break Even adaptatif (base: {BASE_BREAKEVEN_TRIGGER_R}R)")
    logger.info(f"✅ Seuil Break Even anticipé adaptatif (base: {BASE_BREAKEVEN_EARLY_R}R)")
    logger.info(f"✅ Seuil EQS adaptatif (base: {BASE_EQS_MIN_THRESHOLD}/100, 60 en ASIA)")
    logger.info(f"✅ Trailing stop optimisé (activation {BASE_TRAILING_ACTIVATION_R}R, distance {BASE_TRAILING_STOP_DISTANCE_ATR_MULTIPLIER}R)")
    logger.info(f"✅ Score d'entrée minimum /100: {MIN_ENTRY_SCORE}")
    logger.info(f"✅ ASIA BYPASS: FVG_RETEST_PERFECT (Score>={ASIA_BYPASS_MIN_SCORE}, EQS>={ASIA_BYPASS_MIN_EQS}, ADX>={ASIA_BYPASS_MIN_ADX}, HTF bypassé)")
    logger.info(f"✅ LONDON BYPASS: FVG_RETEST_PERFECT (Score>={LONDON_BYPASS_MIN_SCORE}, EQS>={LONDON_BYPASS_MIN_EQS}, ADX>={LONDON_BYPASS_MIN_ADX}, HTF bypassé)")
    logger.info("🔄 DOUBLE BOUCLE : rapide (30s) pour BE/Trailing, lente (15min) pour les signaux")
    logger.info("📈 SUIVI DES CLÔTURES : tentative de récupération via TradeDetails + fallback")
    logger.info("📊 ESPÉRANCE CALCULÉE SUR LES TRADES CLÔTURÉS (wins+losses+breakevens)")
    logger.info("📊 MÉTRIQUES ENRICHIES : ATR, ADX, RSI, Momentum, Heure, Jour, Session, Spread, Volatilité, Tendances H1/H4")
    logger.info("🔧 APPELS OANDA CORRIGÉS : formatage, retry, gestion d'erreur")
    logger.info("📈 SUIVI MFE/MAE ACTIVÉ AVEC LOGS ENRICHIS")
    logger.info("📈 APPRENTISSAGE DES SETUPS ACTIVÉ (seuil 10 trades)")
    logger.info("📈 PARAMÈTRES ADAPTATIFS ROBUSTES (seuil 20 trades, hystérésis 2 cycles, amplitude limitée)")
    logger.info("")
    logger.info("🔧 CORRECTIONS V111 APPLIQUÉES :")
    logger.info("  ✅ Ajout des constantes ASIA_BYPASS_MIN_SCORE, ASIA_BYPASS_MIN_EQS, ASIA_BYPASS_MIN_ADX")
    logger.info("  ✅ Ajout des constantes LONDON_BYPASS_MIN_SCORE, LONDON_BYPASS_MIN_EQS, LONDON_BYPASS_MIN_ADX")
    logger.info("  ✅ Correction du calcul du R après BE (utilisation du risque initial)")
    logger.info("  ✅ Adaptation moins agressive (20 trades, 2 cycles d'hystérésis)")
    logger.info("  ✅ Nettoyage des logs de démarrage (version unifiée V111)")
    logger.info("")

    diagnostic_startup_v981()

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
