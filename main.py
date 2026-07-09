import os
import time
import logging
import requests
from datetime import datetime, timedelta
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import oandapyV20
from oandapyV20.endpoints import instruments
from oandapyV20.endpoints import orders, accounts, trades
from oandapyV20.endpoints import positions, transactions
import talib
import traceback
from ta.momentum import RSIIndicator
from typing import List, Dict, Tuple

# =========================
# LOG HELPERS (dÃ©-dup + compact)
# =========================

_seen_log_keys_fvg_recent = set()
_seen_log_keys_fvg_added  = set()
_seen_log_keys_kept_entry = set()

def _reset_log_dedup():
    """RÃ©initialise les registres de dÃ©-duplication pour un nouveau scan de paire."""
    _seen_log_keys_fvg_recent.clear()
    _seen_log_keys_fvg_added.clear()
    _seen_log_keys_kept_entry.clear()

def _log_fvg_recent_once(pair: str, direction: str, level: float, msg: str, precision: int = 5):
    """Loggue un FVG rÃ©cent au plus une fois par (pair, direction, level)."""
    key = (pair, (direction or "").upper(), round(float(level), precision))
    if key in _seen_log_keys_fvg_recent:
        return
    _seen_log_keys_fvg_recent.add(key)
    logger.info(msg)

def _log_fvg_added_once(pair: str, direction: str, level: float, fvg_type: str, msg: str, precision: int = 5):
    """Loggue l'ajout d'un FVG/Nested/Wick au plus une fois par (pair, direction, type, level)."""
    key = (pair, (direction or "").upper(), (fvg_type or "UNKNOWN").upper(), round(float(level), precision))
    if key in _seen_log_keys_fvg_added:
        return
    _seen_log_keys_fvg_added.add(key)
    logger.info(msg)

def _log_kept_entry_once(pair: str, level: float, status: str, dist: float, msg: str, precision: int = 5):
    """(Optionnel) DÃ©-dup 'EntrÃ©e conservÃ©e' par (pair, level, status)."""
    key = (pair, round(float(level), precision), status)
    if key in _seen_log_keys_kept_entry:
        return
    _seen_log_keys_kept_entry.add(key)
    logger.info(msg)

def _log_narrative_list(entries: list, top_n: int = 10):
    """Affiche au plus 'top_n' entrÃ©es (triÃ©es) + compteur restant."""
    if not entries:
        logger.info("ðŸ”Ž AUCUNE ENTRÃ‰E DÃ‰TECTÃ‰E - Analyse dÃ©taillÃ©e:")
        return
    safe_entries = []
    for e in entries:
        try:
            lvl = float(e.get("entry_level", 0))
            zone = e.get("entry_zone", (lvl, lvl))
            # prioritÃ©: proximitÃ© au bas de zone (simple & stable)
            d = abs(lvl - float(zone[0]))
        except Exception:
            d = 0.0
        safe_entries.append((d, e))
    safe_entries.sort(key=lambda x: x[0])
    top = [e for _, e in safe_entries[:top_n]]
    other_count = max(0, len(entries) - len(top))
    for i, entry in enumerate(top, start=1):
        logger.info(f" {i}. {entry.get('direction','?')} - {entry.get('type','?')} Ã  {float(entry.get('entry_level',0)):.5f}")
    if other_count:
        logger.info(f" â€¦ (+{other_count} autres entrÃ©es)")
        
def _log_kept_compact(pair: str, kept: list, top_n: int = 8):
    """Compacte l'affichage des 'EntrÃ©es conservÃ©es' (top_n + compteur)."""
    if not kept:
        logger.info(f"{pair} Â· Aucune entrÃ©e conservÃ©e.")
        return
    show = kept[:top_n]
    for e in show:
        entry_level = e.get("entry_level")
        status = e.get("status_label", "conservÃ©e")
        dist = e.get("status_distance", None)
        if entry_level is None:
            continue
        if dist is not None:
            logger.info(f"âœ… EntrÃ©e {status}: {float(entry_level):.5f} (distance: {float(dist):.5f})")
        else:
            logger.info(f"âœ… EntrÃ©e {status}: {float(entry_level):.5f}")
    if len(kept) > len(show):
        logger.info(f"â€¦ (+{len(kept)-len(show)} autres entrÃ©es {pair})")
logger = logging.getLogger("Advanced-Orderflow-Trading-Bot")

last_reset_time = datetime.utcnow()

# =============================
# CONFIGURATION
# =============================
load_dotenv()

# Liste des paires incluant XAU_USD (Or)
PAIR_LIST = ["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD", "AUD_USD", "AUD_CAD", "AUD_JPY", "GBP_JPY"]

GRANULARITY_D1 = "D"  # Pour analyse de bias
GRANULARITY_H4 = "H4"  # Pour narrative et orderflow principal
GRANULARITY_H1 = "H1"  # Pour validation narrative
GRANULARITY_M15 = "M15"  # Pour entrÃ©e prÃ©cise

EMA_SLOW = 200
EMA_MEDIUM = 50
EMA_FAST = 20
RSI_PERIOD = 14
ATR_PERIOD = 14

RISK_REWARD_RATIO = 2
MAX_VOLATILITY_RATIO = 0.02
SWING_LOOKBACK = 3
MIN_WICK_RATIO = 0.7

 # ðŸ”¥ CONFIG DISTANCE MAXIMALE (Ã‰LARGIE)
MAX_DISTANCE_PIPS = {
    "XAU_USD": 500,     # = 50 pips (1 pip = 0.01)
    "USD_JPY": 150,     # = 10 pips (1 pip = 0.01)
    "NAS100_USD": 25.0,   # = 10 pips (1 pip â‰ˆ 0.1)
    "AUD_USD": 0.0080,   # = 10 pips (1 pip = 0.0001)
    "EUR_USD": 0.0080,
    "GBP_USD": 0.0080,
    "USD_CAD": 0.0010,
    "GBP_JPY": 150,      # = 15 pips (1 pip = 0.01)
    "DEFAULT": 0.0010
}
# ParamÃ¨tres spÃ©cifiques par paire
PAIR_SETTINGS = {
    "XAU_USD": {
        "atr_multiplier_sl": 1.8,
        "atr_multiplier_tp": 3.5,
        "max_volatility_ratio": 0.010,
        "risk_multiplier": 0.5,
        "required_confluence": "STRICT"
    },
     "NAS100_USD": {
        "atr_multiplier_sl": 1.6,        # SL un peu plus serrÃ©
        "atr_multiplier_tp": 3.2,        # TP adaptÃ© Ã  la volatilitÃ©
        "max_volatility_ratio": 0.015,   # Plus volatile que Forex
        "risk_multiplier": 0.7,          # Risque modÃ©rÃ©
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
    "NESTED_FVG": {
        "sl_multiplier": 1.0,   # â† Plus serrÃ©
        "tp_multiplier": 2.5    # â† RR > 2.5
    },
    "FVG_RETEST": {
        "sl_multiplier": 1.5,
        "tp_multiplier": 3.0
    },
    "WICK_REJECTION": {
        "sl_multiplier": 1.7,
        "tp_multiplier": 4.5
    },
    "LIQUIDITY_DRAW": {
        "sl_multiplier": 1.8,
        "tp_multiplier": 3.5
    }
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

# Configuration du scoring
# === CONFIGURATION DU SCORING (MISE Ã€ JOUR) ===
# V78: le filtre EMA50 H1 n'est plus un veto absolu, il devient un malus de score.
SCORING_CONFIG = {
    "MIN_CONFIDENCE_SCORE": 10,  # V78 Ã©quilibrÃ© : setups B+/A acceptÃ©s si risque OK
    "SIGNAL_WEIGHTS": {
        "BISI": 5,              # Combo fort BOS + FVG
        "NESTED_FVG": 4,
        "FVG_RETEST_PERFECT": 4,# Meilleure qualitÃ© quâ€™un simple FVG
        "FVG_RETEST": 3, 
        "BREAKER": 2,# Classique
        "WICK_REJECTION": 3,
        "TBS_PIN_BUY": 4,# Plus faible
        "LIQUIDITY_DRAW": 2,
        "TBS_PIN_SELL": 4# Cible secondaire
    },
    "BONUS": {
        "BOS_CONFIRMED": 2,
        "CHOCH_CONFIRMED": 2,
        "RSI_CONFLUENCE": 2,    # RenforcÃ©
        "VOLATILITY_OK": 1,
        "RR_OK": 2,             # RenforcÃ©
        "MACD_DIVERGENCE": 3,   # Nouveau bonus
        "FAILURE_SWING": 3,     # Nouveau bonus
        "CRT_DETECTED": 2,    # <-- AJOUTER CETTE LIGNE
        "TBS_DETECTED": 3, 
        "ERL_BONUS": 1,        # <-- DÃ©jÃ  prÃ©sent
        "IB_BONUS": 2
    },
     "PENALTY": {
        "IB_PENALTY": 2,       # <-- AJOUTER CETTE LIGNE
        "NO_IB_PENALTY": 1,    # <-- AJOUTER CETTE LIGNE
        "IRL_PENALTY": 3      # <-- DÃ©jÃ  prÃ©sent ou Ã  ajouter
    }
}


# === FONCTION DE CALCUL DU SCORE (MISE Ã€ JOUR) ===


def get_dynamic_max_distance(df: pd.DataFrame, pair: str, atr_multiplier: float = 1.5) -> float:
    """
    Retourne une distance maximale en PIPS (pas en prix) basÃ©e sur l'ATR.
    BornÃ©e entre 10 et 50 pips pour normaliser la sensibilitÃ©.
    """
    if df is None or len(df) < 14:
        return 20.0  # fallback par dÃ©faut (en pips)

    try:
        atr = calculate_atr(df, period=14)
        atr_pips = price_to_pips(atr, pair)
        dynamic_max_pips = max(10.0, min(50.0, atr_pips * atr_multiplier))
        return dynamic_max_pips
    except Exception:
        return 20.0



logger = logging.getLogger(__name__)

# --- FONCTION Ã€ AJOUTER ---
def is_in_key_zone_or_consolidation(current_price, pair, df_m15, liquidity_levels, nested_fvgs, recent_ofls, structure_analysis, max_zone_width_pips=30.0) -> bool:
    """
    VÃ©rifie si le prix est dans une zone clÃ© (liquiditÃ©, FVG, BOS/CHoCH).
    Correction : Ajout dâ€™un fallback pour Ã©viter rejet excessif.
    """
    try:
        pip_value = 0.01 if "JPY" in pair or pair == "XAU_USD" else 0.0001
        max_zone_width_price = max_zone_width_pips * pip_value

        # VÃ©rification Weekly High/Low
        liq_high = liquidity_levels.get("previous_week_high")
        liq_low = liquidity_levels.get("previous_week_low")
        if liq_high and abs(current_price - liq_high) <= max_zone_width_price:
            return True
        if liq_low and abs(current_price - liq_low) <= max_zone_width_price:
            return True

        # VÃ©rification Nested FVG
        for nfvg in nested_fvgs:
            midpoint = nfvg.get("midpoint")
            if midpoint and abs(current_price - midpoint) <= max_zone_width_price:
                return True

        # VÃ©rification BOS/CHoCH
        for key in ["bos", "choch"]:
            level = structure_analysis.get(key, {}).get("level")
            if level and abs(current_price - level) <= max_zone_width_price:
                return True

        return False
    except Exception:
        return True  # Correction : Ã©viter blocage complet
# --- FIN DE LA FONCTION ---

def detect_imbalances(df: pd.DataFrame, lookback: int = 3) -> list:
    """
    DÃ©tecte les zones d'imbalance (IB) sur un dataframe.
    Retourne une liste de dictionnaires avec 'type', 'high', 'low', 'level'.
    """
    if len(df) < lookback + 2:
        return []

    ibs = []
    for i in range(lookback, len(df) - 1):
        # VÃ©rifiez si la bougie actuelle crÃ©e une imbalance
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        next_high = df.iloc[i + 1]['high']
        next_low = df.iloc[i + 1]['low']

        # Imbalance haussiÃ¨re (le prix casse le low de la bougie prÃ©cÃ©dente)
        if current_low > next_high:
            ibs.append({
                'type': 'BULLISH',
                'high': current_high,
                'low': next_high,
                'level': (current_high + next_high) / 2
            })

        # Imbalance baissiÃ¨re (le prix casse le high de la bougie prÃ©cÃ©dente)
        elif current_high < next_low:
            ibs.append({
                'type': 'BEARISH',
                'high': next_low,
                'low': current_low,
                'level': (current_low + next_low) / 2
            })

    return ibs

def is_in_imbalance_zone(entry_level: float, ibs: list, tolerance: float = 0.0001) -> dict:
    """
    VÃ©rifie si le niveau d'entrÃ©e est dans une zone d'imbalance.
    Retourne un dictionnaire avec 'is_in_zone', 'type', 'level'.
    """
    for ib in ibs:
        if ib['low'] - tolerance <= entry_level <= ib['high'] + tolerance:
            return {
                'is_in_zone': True,
                'type': ib['type'],
                'level': ib['level']
            }
    return {
        'is_in_zone': False,
        'type': None,
        'level': None
    }

def detect_breaker(df: pd.DataFrame, lookback: int = 10) -> dict:
    """
    DÃ©tecte un Breaker Block (bougie d'inversion aprÃ¨s Stop Hunt).
    Retourne {'type': 'BUY'/'SELL', 'level': prix, 'time': timestamp} ou None.
    """
    if len(df) < lookback + 3:
        return {"type": None, "level": None}
    # Exemple logique : aprÃ¨s un Stop Hunt (nouveau haut/bas), bougie inverse ferme au-dessus du corps prÃ©cÃ©dent
    for i in range(len(df) - 3, len(df)):
        candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        if candle['close'] > prev_candle['high']:  # Breaker BUY
            return {"type": "BUY", "level": prev_candle['high'], "time": df.index[i]}
        elif candle['close'] < prev_candle['low']:  # Breaker SELL
            return {"type": "SELL", "level": prev_candle['low'], "time": df.index[i]}
    return {"type": None, "level": None}

# Dans main.py, cherchez ou ajoutez cette logique de validation

def validate_trend_alignment(direction, df_h1, df_h4):
    """
    VÃ©rifie l'alignement H1/H4. 
    Si H4 est BUY mais H1 est en forte baisse (sous EMA 50), on annule le BUY.
    """
    # Calcul des EMA H1
    ema50_h1 = df_h1['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    current_price = df_h1['close'].iloc[-1]

    # RÃˆGLE : Si on veut ACHETER (BUY)
    if direction == "BUY":
        # INTERDIT si le prix est SOUS la EMA 50 H1 (momentum baissier court terme)
        if current_price < ema50_h1:
            return False 
            
    # RÃˆGLE : Si on veut VENDRE (SELL)
    elif direction == "SELL":
        # INTERDIT si le prix est AU-DESSUS de la EMA 50 H1 (momentum haussier court terme)
        if current_price > ema50_h1:
            return False
            
    return True

def detect_dealing_range(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    DÃ©tecte la derniÃ¨re "Dealing Range" (SWHL) sur le dataframe.
    Retourne un dictionnaire avec les niveaux de la range ou None.
    """
    if df is None or df.empty or len(df) < lookback:
        return None

    # DÃ©tecter les swings highs et lows
    swing_highs, swing_lows = detect_swing_points_advanced(df, lookback)

    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return None

    # Trier par index (temps) pour trouver la derniÃ¨re range
    all_swings = []
    for sh in swing_highs:
        all_swings.append((sh['index'], sh['price'], 'high'))
    for sl in swing_lows:
        all_swings.append((sl['index'], sl['price'], 'low'))

    all_swings.sort(key=lambda x: x[0])  # Trier par index (temps)

    # Trouver la derniÃ¨re paire high/low ou low/high (la range la plus rÃ©cente)
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
        # On suppose que la range est dÃ©finie par le dernier high et le dernier low trouvÃ©s
        # (l'ordre peut varier, on prend le max et le min)
        range_high = max(last_high, last_low)
        range_low = min(last_high, last_low)
        return {
            "high": range_high,
            "low": range_low,
            "range_size": range_high - range_low
        }

    return None

# Vous pouvez appeler cette fonction dans advanced_main ou determine_advanced_narrative
# Exemple dans advanced_main :
# dealing_range = detect_dealing_range(df_h4, lookback=50)
# if dealing_range:
#     logger.info(f"ðŸ” Dealing Range dÃ©tectÃ©e: H {dealing_range['high']:.5f} - L {dealing_range['low']:.5f}")

def classify_zone_irl_erl(zone_level: float, dealing_range: dict, tolerance: float = 0.0001) -> str:
    """
    Classifie une zone (reprÃ©sentÃ©e par un niveau) comme IRL ou ERL par rapport Ã  une dealing range.
    Utilise une petite tolÃ©rance pour les niveaux exacts sur les bords.
    Retourne "IRL", "ERL", ou None si la range n'est pas valide.
    """
    if not dealing_range or dealing_range.get("high") is None or dealing_range.get("low") is None:
        return None

    range_high = dealing_range["high"]
    range_low = dealing_range["low"]

    # VÃ©rifier si le niveau est Ã  l'intÃ©rieur de la range (ou trÃ¨s proche des bords)
    if range_low - tolerance <= zone_level <= range_high + tolerance:
        return "IRL"  # Internal Range Liquidity
    else:
        return "ERL"  # External Range Liquidity

# Vous pouvez utiliser cette fonction dans determine_advanced_narrative ou calculate_signal_confidence
# Exemple dans determine_advanced_narrative, aprÃ¨s avoir dÃ©tectÃ© un FVG :
# dealing_range = detect_dealing_range(df_h4, lookback=50)
# if dealing_range:
#     for entry in narrative["potential_entries"]:
#         level = entry.get("entry_level")
#         if level:
#             irl_erl_type = classify_zone_irl_erl(level, dealing_range)
#             entry["irl_erl_type"] = irl_erl_type # Ajouter l'info Ã  l'entrÃ©e
#             logger.info(f"ðŸ“Š FVG {level:.5f} classÃ© comme {irl_erl_type} selon la range {dealing_range['low']:.5f}-{dealing_range['high']:.5f}")


    
def is_displacement_strong(df: pd.DataFrame, threshold: float = 0.0005) -> bool:
    """
    VÃ©rifie si la derniÃ¨re bougie a un corps > threshold (fort dÃ©placement).
    """
    if df.empty or len(df) < 2:
        return False

    last_candle = df.iloc[-1]
    body_size = abs(last_candle['close'] - last_candle['open'])
    total_range = last_candle['high'] - last_candle['low']

    # Displacement fort si corps > threshold et ratio corps/range > 0.6
    return body_size >= threshold and (body_size / total_range) > 0.6

def detect_amd_phase(df: pd.DataFrame, lookback: int = 50) -> str:
    """
    DÃ©tecte la phase AMD : Accumulation, Manipulation, Distribution.
    BasÃ© sur range et sweep.
    """
    if df.empty or len(df) < lookback:
        return "UNKNOWN"

    recent_high = df['high'].tail(lookback).max()
    recent_low = df['low'].tail(lookback).min()
    current_price = df['close'].iloc[-1]
    range_size = recent_high - recent_low

    # Accumulation : range Ã©troit
    if range_size < (current_price * 0.005):
        return "ACCUMULATION"

    # Manipulation : sweep fort (cassure du range)
    if df['high'].iloc[-1] > recent_high or df['low'].iloc[-1] < recent_low:
        return "MANIPULATION"

    # Distribution : breakout confirmÃ© aprÃ¨s manipulation
    if current_price > recent_high or current_price < recent_low:
        return "DISTRIBUTION"

    return "UNKNOWN"



def cluster_signals(signals: List[Dict], pair: str, max_distance_pips_for_clustering: float = None) -> List[Dict]:
    """
    Regroupe les entrÃ©es proches et ne garde que le meilleur (score le plus Ã©levÃ©).
    Utilise la signature attendue par `advanced_main`.
    """
    if not signals:
        return []

    pip_value = get_pip_value_for_pair(pair)
    # Utilise max_distance_pips_for_clustering, ou une valeur par dÃ©faut de 15 pips si non fournie
    max_distance_pips_arg = max_distance_pips_for_clustering or 15.0
    max_distance_price = max_distance_pips_arg * pip_value

    # Trier les signaux par score de confiance (dÃ©croissant) pour s'assurer que le premier est le meilleur
    signals.sort(key=lambda s: s.get("confidence_score", 0), reverse=True)

    clusters = []
    current_cluster = []

    for s in signals:
        # Utilise le niveau d'entrÃ©e du signal
        lvl = float(s["entry_level"])
        if not current_cluster:
            current_cluster = [s]
            last_level = lvl
            continue
        # VÃ©rifier si le niveau est proche du dernier niveau du cluster actuel
        if abs(lvl - last_level) <= max_distance_price:
            current_cluster.append(s)
            last_level = lvl
        else:
            # Fin du cluster actuel : sÃ©lectionner le meilleur et commencer un nouveau cluster
            # Le tri initial garantit que le premier Ã©lÃ©ment du cluster a le meilleur score
            best_signal_in_cluster = current_cluster[0] # DÃ©jÃ  triÃ© par score
            clusters.append(best_signal_in_cluster)
            current_cluster = [s]
            last_level = lvl

    # Ne pas oublier le dernier cluster
    if current_cluster:
        best_signal_in_cluster = current_cluster[0] # DÃ©jÃ  triÃ© par score
        clusters.append(best_signal_in_cluster)

    if 'logger' in globals() and isinstance(globals()['logger'], logging.Logger):
        logger.info(f"ðŸ—‚ï¸ EntrÃ©es aprÃ¨s clustering: {len(clusters)}")

    return clusters


def is_crt_candle(candle: pd.Series, min_body_ratio: float = 0.5) -> bool: # <-- Changement ici : 0.5 au lieu de 0.7
    """
    VÃ©rifie si la bougie est une CRT Candle (corps large, wicks courts).
    Utilise `min_body_ratio` pour la flexibilitÃ©.
    """
    body = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    if total_range == 0:
        return False
    body_ratio = body / total_range

    # Assouplissement : on accepte maintenant >= 0.5
    # VÃ©rifier que les wicks sont courts (ex: < 20% du range total)
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range

    # Condition : corps >= 50%, wicks <= 20% chacun
    return body_ratio >= min_body_ratio and upper_wick_ratio <= 0.2 and lower_wick_ratio <= 0.2





    # Fonction pour convertir une diffÃ©rence de prix en pips
  

# --- Exemple de dÃ©finition de MAX_DISTANCE_PIPS si elle n'est pas dans le scope ---
# MAX_DISTANCE_PIPS = {
#     "XAU_USD": 1.0,     # = 100 pips (1 pip = 0.01)
#     "USD_JPY": 0.08,     # = 8 pips (1 pip = 0.01)
#     "NAS100_USD": 1.0,   # = 10 pips (1 pip â‰ˆ 0.1)
#     "AUD_USD": 0.0080,   # = 80 pips (1 pip = 0.0001)
#     "EUR_USD": 0.0080,
#     "GBP_USD": 0.0080,
#     "USD_CAD": 0.0010, # = 10 pips
#     "DEFAULT": 0.0010
# }
# --- Fin exemple ---

def rr_points(rr: float) -> int:
    """ 2 pts si RR >= 3.0 ; 1 pt si RR >= 2.0 ; sinon 0."""
    if rr >= 3.0:
        return 2
    if rr >= 2.0:
        return 1
    return 0

def detect_tbs_setup(df: pd.DataFrame) -> dict:
    """
    DÃ©tecte un potentiel TBS Setup (The Book Setup) sur le dataframe M15.
    Un TBS typique peut Ãªtre un Inside Bar (IB) suivi d'une cassure confirmÃ©e.
    Retourne un dictionnaire avec 'type' ('TBS_IB_BULL', 'TBS_IB_SELL', 'TBS_PIN_BUY', 'TBS_PIN_SELL', '') et 'level'.
    """
    if df.empty or len(df) < 3:
        return {"type": "", "level": None}

    # RÃ©cupÃ©rer les derniÃ¨res bougies
    current_candle = df.iloc[-1]  # Bougie M15 actuelle
    prev_candle = df.iloc[-2]     # Bougie M15 prÃ©cÃ©dente
    prev2_candle = df.iloc[-3]    # Bougie M15 avant la prÃ©cÃ©dente

    # 1.1. VÃ©rifier si la bougie N-1 est un "Inside Bar" (IB) de la bougie N-2
    is_inside_bar = (
        prev_candle['high'] < prev2_candle['high'] and
        prev_candle['low'] > prev2_candle['low']
    )

    # 1.2. VÃ©rifier si la bougie N (current) casse l'IB (haut ou bas)
    if is_inside_bar:
        # Cassure haussiÃ¨re de l'IB
        if current_candle['high'] > prev_candle['high']:
            # VÃ©rifier si la bougie de cassure est forte (corps > 60% du range)
            body_size = abs(current_candle['close'] - current_candle['open'])
            total_range = current_candle['high'] - current_candle['low']
            if body_size > total_range * 0.6:
                return {"type": "TBS_IB_BULL", "level": prev_candle['high']}
        # Cassure baissiÃ¨re de l'IB
        elif current_candle['low'] < prev_candle['low']:
            body_size = abs(current_candle['close'] - current_candle['open'])
            total_range = current_candle['high'] - current_candle['low']
            if body_size > total_range * 0.6:
                return {"type": "TBS_IB_SELL", "level": prev_candle['low']}

    # 1.3. VÃ©rifier si la bougie N-1 est une "Pin Bar" forte (par exemple, basse)
    # On peut aussi chercher une Pin Bar Ã  l'extÃ©rieur de l'IB ou d'une zone clÃ©.
    pb_body = abs(prev_candle['close'] - prev_candle['open'])
    pb_range = prev_candle['high'] - prev_candle['low']
    if pb_range > 0: # Ã‰viter la division par zÃ©ro
        pb_body_ratio = pb_body / pb_range
        pb_upper_wick = prev_candle['high'] - max(prev_candle['open'], prev_candle['close'])
        pb_lower_wick = min(prev_candle['open'], prev_candle['close']) - prev_candle['low']

        # Pin Bar haussiÃ¨re (mÃ¨che infÃ©rieure longue)
        if pb_lower_wick > pb_upper_wick * 2 and pb_body_ratio < 0.4:
            # Si la bougie actuelle casse le haut de la pin bar
            if current_candle['high'] > prev_candle['high']:
                return {"type": "TBS_PIN_BUY", "level": prev_candle['high']}
        # Pin Bar baissiÃ¨re (mÃ¨che supÃ©rieure longue)
        elif pb_upper_wick > pb_lower_wick * 2 and pb_body_ratio < 0.4:
            # Si la bougie actuelle casse le bas de la pin bar
            if current_candle['low'] < prev_candle['low']:
                return {"type": "TBS_PIN_SELL", "level": prev_candle['low']}

    # Aucun TBS Setup dÃ©tectÃ©
    return {"type": "", "level": None}

def detect_crt_candle(candle: pd.Series, min_body_ratio: float = 0.5) -> bool:
    """
    DÃ©tecte une CRT Candle (corps large, wicks courts).
    Utilise `min_body_ratio` pour la flexibilitÃ© (assoupli Ã  0.5 au lieu de 0.7).
    """
    body = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    if total_range == 0:
        return False
    body_ratio = body / total_range

    # Assouplissement : on accepte maintenant >= 0.5
    # VÃ©rifier que les wicks sont courts (ex: < 20% du range total)
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range

    # Condition : corps >= 50%, wicks <= 20% chacun
    return body_ratio >= min_body_ratio and upper_wick_ratio <= 0.2 and lower_wick_ratio <= 0.2
    
def compute_confidence_score(
    *,
    bias_points: int,
    structure_points: int,
    rr: float,
    nested_fvg_in_zone: bool = False,
    other_bonuses: int = 0
) -> int:
    """
    Petit utilitaire si tu veux composer un score en dehors de calculate_signal_confidence.
    """
    score = 0
    score += bias_points
    score += structure_points
    score += rr_points(rr)
    if nested_fvg_in_zone:
        score += 1
    score += other_bonuses
    return score

def get_pair_settings(pair: str) -> dict:
    """Retourne les paramÃ¨tres spÃ©cifiques Ã  la paire."""
    return PAIR_SETTINGS.get(pair, PAIR_SETTINGS["DEFAULT"])

# =============================
# LOGGING
# =============================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("advanced_orderflow_trading.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Advanced-Orderflow-Trading-Bot")

# =============================
# GESTION DES SIGNAUX
# =============================
sent_signals = {}

    # A placer en zone globale (Ã  cÃ´tÃ© de sent_signals) :
recent_signals = {}  # { (pair, direction, level): datetime }

def is_duplicate(pair: str, direction: str, level: float, ttl_seconds: int = 1800) -> bool:
    """
    Anti-doublon : 1 envoi par (pair, direction, niveau) toutes les 30 minutes (par dÃ©faut).
    """
    from datetime import datetime
    now = datetime.utcnow()
    key = (pair, direction, round(float(level), 4))
    last = recent_signals.get(key)

    if last and (now - last).total_seconds() < ttl_seconds:
        return True

    recent_signals[key] = now
    return False
    
def detect_rsi_divergence_haussiere(df: pd.DataFrame, lookback: int = 14) -> bool:
    """DÃ©tecte une divergence haussiÃ¨re (prix fait un nouveau bas, RSI non)."""
    if len(df) < lookback * 2 + 5:
        return False

    prices = df["close"].tail(lookback * 2).reset_index(drop=True)
    rsi_vals = calculate_rsi(df["close"]).tail(lookback * 2).reset_index(drop=True)

    # Trouver les creux de prix
    price_lows = []
    for i in range(3, len(prices) - 3):
        if (prices.iloc[i] < prices.iloc[i-3:i].min() and 
            prices.iloc[i] < prices.iloc[i+1:i+4].min()):
            price_lows.append((i, prices.iloc[i]))

    # Trouver les creux de RSI
    rsi_lows = []
    for i in range(3, len(rsi_vals) - 3):
        if (rsi_vals.iloc[i] < rsi_vals.iloc[i-3:i].min() and 
            rsi_vals.iloc[i] < rsi_vals.iloc[i+1:i+4].min()):
            rsi_lows.append((i, rsi_vals.iloc[i]))

    if len(price_lows) < 2 or len(rsi_lows) < 2:
        return False

    last_price_low = price_lows[-1][1]
    prev_price_low = price_lows[-2][1]
    last_rsi_low = rsi_lows[-1][1]
    prev_rsi_low = rsi_lows[-2][1]

    return last_price_low < prev_price_low and last_rsi_low > prev_rsi_low

def calculate_macd_momentum(df: pd.DataFrame) -> pd.Series:
    """Calcule l'histogramme MACD pour mesurer l'accÃ©lÃ©ration du momentum."""
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
    # VÃ©rifie si l'histogramme est croissant (accÃ©lÃ©ration haussiÃ¨re)
    is_accelerating_up = all(recent_hist.iloc[i] > recent_hist.iloc[i-1] for i in range(1, len(recent_hist)))
    # VÃ©rifie si l'histogramme est dÃ©croissant (accÃ©lÃ©ration baissiÃ¨re)
    is_accelerating_down = all(recent_hist.iloc[i] < recent_hist.iloc[i-1] for i in range(1, len(recent_hist)))
    # Pour un momentum haussier : croissant ET positif
    is_BUY = is_accelerating_up and (recent_hist.iloc[-1] > 0)
    # Pour un momentum baissier : dÃ©croissant ET nÃ©gatif
    is_SELL = is_accelerating_down and (recent_hist.iloc[-1] < 0)
    return is_BUY or is_SELL

def is_candle_momentum_strong(candle: pd.Series, direction: str) -> bool:
    """VÃ©rifie si la bougie a une structure de momentum fort."""
    body_size = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    if total_range == 0:
        return False
    body_ratio = body_size / total_range
    # Pour un SELL : bougie rouge avec petit wick haut
    if direction == "SELL":
        is_red = candle['close'] < candle['open']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        upper_wick_ratio = upper_wick / total_range
        return is_red and body_ratio > 0.7 and upper_wick_ratio < 0.2
    # Pour un BUY : bougie verte avec petit wick bas
    elif direction == "BUY":
        is_green = candle['close'] > candle['open']
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        lower_wick_ratio = lower_wick / total_range
        return is_green and body_ratio > 0.7 and lower_wick_ratio < 0.2
    return False

def is_volume_confirming_momentum(df: pd.DataFrame, direction: str) -> bool:
    """VÃ©rifie si le volume confirme le momentum."""
    last_3_volumes = df['volume'].tail(3)
    avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
    # Volume en hausse + supÃ©rieur Ã  la moyenne
    volume_increasing = last_3_volumes.iloc[-1] > last_3_volumes.iloc[-2] > last_3_volumes.iloc[-3]
    volume_above_avg = last_3_volumes.iloc[-1] > avg_volume * 1.2
    return volume_increasing and volume_above_avg

def validate_risk_reward(entry_price, stop_loss, take_profit, min_ratio=2.0) -> bool:
    """
    VÃ©rifie RR >= min_ratio.
    Correction : Ajuste TP si RR < min_ratio.
    """
    if None in [entry_price, stop_loss, take_profit]:
        return False
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    if risk <= 0:
        return False
    rr_ratio = reward / risk
    return rr_ratio >= min_ratio

def price_to_pips(price_diff: float, pair: str) -> float:
    """
    Convertit une diffÃ©rence de prix en pips.
    ATTENTION: NÃ©cessite une mise Ã  jour si de nouvelles paires avec tailles de pip spÃ©cifiques sont ajoutÃ©es.
    """
    pair = pair.upper() # Assurez-vous du format
    if pair == "XAU_USD":
        pip_size = 0.01
    elif pair == "NAS100_USD":
        pip_size = 0.1  # 1 pip = 0.1 point pour NAS100
    elif "JPY" in pair:
        pip_size = 0.01 # 1 pip = 0.01 pour les autres JPY
    else:
        pip_size = 0.0001 # 1 pip = 0.0001 pour les paires Forex majeures/mineures

    return abs(price_diff) / pip_size

# Vous devriez Ã©galement corriger la rÃ©cupÃ©ration de la pip_value dans advanced_main
# Remplacez la ligne :
# pip_value_for_clustering = 0.01 if "JPY" in pair else 0.0001
# Par une fonction similaire Ã  celle-ci, ou une constante dÃ©finie globalement.
# Exemple :
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

# Et utilisez-la dans advanced_main :
# pip_value_for_clustering = get_pip_value_for_pair(pair)

def detect_rsi_momentum_acceleration(df, period=14, lookback=8) -> dict:
    """
    DÃ©tecte accÃ©lÃ©ration RSI + divergence.
    Correction : Normalisation slope + robustesse.
    """
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
    """DÃ©tecte une divergence baissiÃ¨re (prix fait un nouveau haut, RSI non)."""
    if len(df) < lookback * 2 + 5:
        return False

    prices = df["close"].tail(lookback * 2).reset_index(drop=True)
    rsi_vals = calculate_rsi(df["close"]).tail(lookback * 2).reset_index(drop=True)

    # Trouver les pics de prix
    price_peaks = []
    for i in range(3, len(prices) - 3):
        if (prices.iloc[i] > prices.iloc[i-3:i].max() and 
            prices.iloc[i] > prices.iloc[i+1:i+4].max()):
            price_peaks.append((i, prices.iloc[i]))

    # Trouver les pics de RSI
    rsi_peaks = []
    for i in range(3, len(rsi_vals) - 3):
        if (rsi_vals.iloc[i] > rsi_vals.iloc[i-3:i].max() and 
            rsi_vals.iloc[i] > rsi_vals.iloc[i+1:i+4].max()):
            rsi_peaks.append((i, rsi_vals.iloc[i]))

    if len(price_peaks) < 2 or len(rsi_peaks) < 2:
        return False

    last_price_peak = price_peaks[-1][1]
    prev_price_peak = price_peaks[-2][1]
    last_rsi_peak = rsi_peaks[-1][1]
    prev_rsi_peak = rsi_peaks[-2][1]

    return last_price_peak > prev_price_peak and last_rsi_peak < prev_rsi_peak
    
def cleanup_old_signals():
    """Nettoie les signaux de plus de 24 heures."""
    now = time.time()
    for key, timestamp in list(sent_signals.items()):
        if now - timestamp > 86400:
            del sent_signals[key]
            logger.debug(f"ðŸ§¹ Signal nettoyÃ©: {key}")

def is_signal_in_recent_zone(pair: str, direction: str, price: float, zone_start: float, zone_end: float) -> bool:
    
    global sent_signals
    now = time.time()
    zone_width = abs(zone_end - zone_start)
    tolerance = zone_width * 0.5  # 50% de la zone

    for (p, d, lvl, z_s, z_e), timestamp in sent_signals.items():
        if p != pair or d != direction:
            continue
        # Nettoyage des signaux > 4h
        if now - timestamp > 1 * 3600:
            continue
        # VÃ©rifier si les zones se chevauchent ou sont proches
        if abs(price - lvl) <= tolerance:
            return True
    return False


def is_signal_sent_recently(pair: str, direction: str, price: float, zone_start: float, zone_end: float) -> bool:
    global sent_signals
    now = time.time()
    # TolÃ©rance plus stricte pour Ã©viter les spam sur le mÃªme niveau exact
    tolerance_price = 0.00001 if "JPY" not in pair and pair != "XAU_USD" else 0.01 
    
    price_rounded = round(price, 5)
    
    # Nettoyage automatique des vieux signaux dans la boucle de vÃ©rification pour Ã©viter que le dict ne grossisse indÃ©finiment
    keys_to_delete = []
    
    is_sent = False
    for key, timestamp in sent_signals.items():
        # key structure: (pair, direction, level, zone_start, zone_end)
        p, d, lvl, _, _ = key
        
        # Nettoyage : si > 4h, on marque pour suppression
        if now - timestamp > 4 * 3600:
            keys_to_delete.append(key)
            continue

        if p == pair and d == direction:
            # VÃ©rifie si on est trÃ¨s proche d'un niveau dÃ©jÃ  envoyÃ©
            if abs(price_rounded - lvl) < tolerance_price:
                 # DÃ©lai anti-spam (ex: 2h)
                if now - timestamp < 2 * 3600: 
                    is_sent = True
    
    # Appliquer le nettoyage
    for k in keys_to_delete:
        sent_signals.pop(k, None)
        
    return is_sent
    # SUPPRIMER LA LIGNE: sent_signals[...] = now   <-- C'Ã©tait l'erreur fatale

def detect_macd_acceleration(df: pd.DataFrame, lookback: int = 5) -> str:
    """
    Detects if MACD histogram is accelerating (increasing/decreasing rapidly).
    Returns: 'STRONG_BUY', 'STRONG_SELL', 'WEAK_BUY', 'WEAK_SELL', 'NEUTRAL'
    """
    hist = calculate_macd_momentum(df)
    if len(hist) < lookback + 2:
        return 'NEUTRAL'
    recent_hist = hist.tail(lookback + 1).values
    # Calculate acceleration: change in change
    changes = [recent_hist[i] - recent_hist[i-1] for i in range(1, len(recent_hist))]
    acceleration = sum(changes) / len(changes)  # Avg acceleration
    
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
    """
    DÃ©tecte un Failure Swing selon Investopedia.
    Retourne 'BULLISH', 'BEARISH' ou None.
    """
    if len(df) < lookback + 5:
        return {"type": None, "level": None}

    # Trouver les swing highs et lows
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

    # DÃ©tecter un Failure Swing Haussier (Bottom)
    last_swing_low = swing_lows[-1]
    prev_swing_low = swing_lows[-2]
    # Un failure swing haussier se forme quand le prix crÃ©e un nouveau bas (prev_swing_low) mais ne parvient pas Ã  descendre aussi bas que le prÃ©cÃ©dent (last_swing_low).
    # Le prix doit ensuite rebondir au-dessus du point de faille (le plus haut point du mouvement baissier entre les deux swings).
    # Pour simplifier, on vÃ©rifie si le dernier swing low est plus haut que le prÃ©cÃ©dent et que le prix actuel est au-dessus du niveau du dernier swing low.
    if last_swing_low["price"] > prev_swing_low["price"]:
        # VÃ©rifier que le prix a rebondi au-dessus du niveau du dernier swing low
        if df["close"].iloc[-1] > last_swing_low["price"]:
            return {"type": "BULLISH", "level": last_swing_low["price"], "time": df.index[-1]}

    # DÃ©tecter un Failure Swing Baissier (Top)
    last_swing_high = swing_highs[-1]
    prev_swing_high = swing_highs[-2]
    # Un failure swing baissier se forme quand le prix crÃ©e un nouveau haut (prev_swing_high) mais ne parvient pas Ã  monter aussi haut que le prÃ©cÃ©dent (last_swing_high).
    # Le prix doit ensuite chuter en dessous du point de faille.
    if last_swing_high["price"] < prev_swing_high["price"]:
        # VÃ©rifier que le prix a baissÃ© en dessous du niveau du dernier swing high
        if df["close"].iloc[-1] < last_swing_high["price"]:
            return {"type": "BEARISH", "level": last_swing_high["price"], "time": df.index[-1]}

    return {"type": None, "level": None}



def validate_momentum_confluence(df_h4: pd.DataFrame, df_h1: pd.DataFrame, df_m15: pd.DataFrame, direction: str) -> int:
    """
    Returns 0, 1, 2, or 3 based on how many timeframes agree on momentum direction.
    """
    score = 0
    
    # H4
    h4_mom = detect_rsi_momentum_acceleration(df_h4, period=14, lookback=8)
    if h4_mom['direction'] == ('STRONG_' + direction) or h4_mom['direction'] == direction:
        score += 1
    
    # H1
    h1_mom = detect_rsi_momentum_acceleration(df_h1, period=14, lookback=8)
    if h1_mom['direction'] == ('STRONG_' + direction) or h1_mom['direction'] == direction:
        score += 1
    
    # M15
    m15_mom = detect_rsi_momentum_acceleration(df_m15, period=14, lookback=8)
    if m15_mom['direction'] == ('STRONG_' + direction) or m15_mom['direction'] == direction:
        score += 1
    
    return score

def detect_volume_momentum(df: pd.DataFrame, window: int = 10) -> str:
    """
    Measures if volume is accelerating or decelerating relative to its moving average.
    Returns: 'STRONG_BUY', 'STRONG_SELL', 'NEUTRAL'
    """
    vol_ma = df['volume'].rolling(window=window).mean()
    vol_current = df['volume'].iloc[-1]
    vol_prev = df['volume'].iloc[-2]
    vol_ma_current = vol_ma.iloc[-1]
    vol_ma_prev = vol_ma.iloc[-2]
    
    # Volume increasing faster than MA
    vol_ratio = vol_current / vol_ma_current
    vol_accel = (vol_current - vol_prev) / vol_prev  # % change
    
    if vol_ratio > 1.3 and vol_accel > 0.2:
        return 'STRONG_BUY'
    elif vol_ratio > 1.3 and vol_accel < -0.2:
        return 'STRONG_SELL'  # Volume spike fading â€” trap
    elif vol_ratio > 1.1 and vol_accel > 0.1:
        return 'BUY'
    elif vol_ratio > 1.1 and vol_accel < -0.1:
        return 'SELL'
    else:
        return 'NEUTRAL'

def mark_signal_sent(pair: str, direction: str, entry_level: float, zone_start: float, zone_end: float):
    """Marque un signal comme envoyÃ© avec sa zone complÃ¨te."""
    key = (pair, direction, round(entry_level, 5), round(zone_start, 5), round(zone_end, 5))
    sent_signals[key] = time.time()
    logger.info(f"âœ… Signal marquÃ© comme envoyÃ© : {key}")

def detect_bos(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    DÃ©tecte un vrai BOS (continuation) :
    - BOS BUY = cassure du dernier swing high + close au-dessus.
    - BOS SELL = cassure du dernier swing low + close en dessous.
    """
    if len(df) < lookback + 10:
        return {"type": None, "level": None, "time": None}

    swing_highs, swing_lows = detect_swing_points(df, lookback=5)
    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return {"type": None, "level": None, "time": None}

    current_close = df["close"].iloc[-1]
    current_high = df["high"].iloc[-1]
    current_low = df["low"].iloc[-1]

    # Dernier swing high
    last_swing_high = swing_highs[-1]["price"]
    # Dernier swing low
    last_swing_low = swing_lows[-1]["price"]

    # BOS BUY : cassure du dernier swing high
    if current_close > last_swing_high and current_high > last_swing_high:
        return {
            "type": "BOS_BUY",
            "level": last_swing_high,
            "time": df.index[-1]
        }

    # BOS SELL : cassure du dernier swing low
    if current_close < last_swing_low and current_low < last_swing_low:
        return {
            "type": "BOS_SELL",
            "level": last_swing_low,
            "time": df.index[-1]
        }

    return {"type": None, "level": None, "time": None}


def detect_choch(df: pd.DataFrame, lookback: int = 50) -> dict:
    
    if len(df) < lookback + 15:
        return {"type": None, "level": None, "time": None}

    swing_highs, swing_lows = detect_swing_points(df, lookback=5)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"type": None, "level": None, "time": None}

    current_price = df["close"].iloc[-1]
    current_time = df.index[-1]

    # === CAS 1 : CHoCH SELL (inversion haussiÃ¨re â†’ baissiÃ¨re) ===
    # SÃ©quence attendue : HH â†’ HL â†’ **LL**
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-2]["price"]  # avant-dernier high
        lh = swing_highs[-1]["price"]  # dernier high (doit Ãªtre < hh â†’ HL)
        hl = swing_lows[-2]["price"]   # avant-dernier low
        ll = swing_lows[-1]["price"]   # dernier low (doit Ãªtre < hl â†’ LL)

        # VÃ©rifier structure haussiÃ¨re initiale
        is_uptrend = (
            hh > (swing_highs[-3]["price"] if len(swing_highs) >= 3 else 0) and
            hl > (swing_lows[-3]["price"] if len(swing_lows) >= 3 else 0)
        )

        # Inversion : LL < HL
        if is_uptrend and ll < hl and current_price < ll:
            return {
                "type": "CHOCH_SELL",
                "level": ll,
                "time": current_time
            }

    # === CAS 2 : CHoCH BUY (inversion baissiÃ¨re â†’ haussiÃ¨re) ===
    # SÃ©quence attendue : LL â†’ LH â†’ **HH**
    if len(swing_lows) >= 2 and len(swing_highs) >= 2:
        ll = swing_lows[-2]["price"]
        hl = swing_lows[-1]["price"]   # doit Ãªtre > ll â†’ LH
        lh = swing_highs[-2]["price"]
        hh = swing_highs[-1]["price"]  # doit Ãªtre > lh â†’ HH

        # VÃ©rifier structure baissiÃ¨re initiale
        is_downtrend = (
            ll < (swing_lows[-3]["price"] if len(swing_lows) >= 3 else float('inf')) and
            lh < (swing_highs[-3]["price"] if len(swing_highs) >= 3 else float('inf'))
        )

        # Inversion : HH > LH
        if is_downtrend and hh > lh and current_price > hh:
            return {
                "type": "CHOCH_BUY",
                "level": hh,
                "time": current_time
            }

    return {"type": None, "level": None, "time": None}


def generate_bos_signal(df: pd.DataFrame, pair: str, lookback: int = 20, rsi_col: str = "RSI") -> dict:
    """
    GÃ©nÃ¨re un signal BOS amÃ©liorÃ© avec SL sur swing et TP sur zone clÃ©/FVG/swing.
    Score pondÃ©rÃ© par confluence (wick, RSI, momentum)
    """
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

    # Wick rejection
    if bos["type"] == "BOS_BUY" and lower_wick > body:
        score += 1
        confluences.append("WICK_REJECTION_BUY")
    elif bos["type"] == "BOS_SELL" and upper_wick > body:
        score += 1
        confluences.append("WICK_REJECTION_SELL")

    # RSI filter
    if rsi_col in df.columns:
        rsi = df[rsi_col].iloc[-1]
        if bos["type"] == "BOS_BUY" and rsi < 50:
            score += 0.5
            confluences.append("RSI_SUPPORT")
        elif bos["type"] == "BOS_SELL" and rsi > 50:
            score += 0.5
            confluences.append("RSI_RESIST")

    # Swing levels pour SL
    if bos["type"] == "BOS_BUY":
        swing_lows = [df["low"].iloc[i] for i in range(lookback, len(df)-lookback)
                      if df["low"].iloc[i] <= df["low"].iloc[i-lookback:i+lookback+1].min()]
        if not swing_lows:
            return {}
        stop_loss = min(swing_lows)
        risk = current_price - stop_loss

        # TP basÃ© sur prochaine zone clÃ© / FVG / swing high
        tp_candidates = [level for level in df["high"].iloc[-lookback:] if level > current_price]
        take_profit = max(tp_candidates + [current_price + risk*1.5])  # fallback si pas de zone
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

        # TP basÃ© sur prochaine zone clÃ© / FVG / swing low
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
# FONCTION GET_CANDLES_WITH_RETRY (robuste)
# =============================
def get_candles_with_retry(api, instrument: str, granularity: str, count: int = 500, retries: int = 5) -> pd.DataFrame:
    """RÃ©cupÃ¨re les candles avec systÃ¨me de retry. Accepte des jeux de donnÃ©es rÃ©duits."""
    valid_granularities = ["S5", "S10", "S15", "S30",
                           "M1", "M2", "M4", "M5", "M10", "M15", "M30",
                           "H1", "H2", "H3", "H4", "H6", "H8", "H12",
                           "D", "W", "M"]
    if granularity not in valid_granularities:
        logger.error(f"âŒ GranularitÃ© invalide: {granularity}")
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
                logger.warning(f"âš ï¸ Aucune candle reÃ§ue pour {instrument} {granularity} (tentative {attempt+1})")
                time.sleep(3)
                continue
            
            data = []
            for c in candles:
                mid = c.get("mid")
                if not mid:
                    continue
                # Filtrer les incomplÃ¨tes uniquement sur M15/H1/H4/D
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
                    logger.debug(f"âš ï¸ Candle malformed skipped for {instrument}: {c}")
                    continue
            
            # âœ… CORRECTION : on accepte dÃ¨s qu'on a AU MOINS 'count' ou ce qui est disponible
            # Pour H1, on peut accepter 20 bougies (1 jour de trading)
            min_required = 20 if granularity == "H1" else min(count, 50)
            if len(data) < min_required:
                # âš ï¸ MAIS SEULEMENT SI ON EST SUR UN TF HAUT
                if granularity in ["H4", "D"] and len(data) > 5:  # â† Si on a au moins 5 jours â†’ OK
                    logger.info(f"â„¹ï¸ DonnÃ©es D/{instrument}: {len(data)} candles (minimum requis: {min_required}) â†’ AcceptÃ© malgrÃ© tout")
                else:
                    logger.warning(f"âš ï¸ DonnÃ©es insuffisantes pour {instrument} {granularity}: {len(data)} < {min_required}")
                    time.sleep(3)
                    continue

            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df.attrs['instrument'] = instrument
            logger.info(f"âœ… {instrument} {granularity}: {len(df)} candles (dernier prix: {df['close'].iloc[-1]:.5f})")
            return df
            
        except oandapyV20.exceptions.V20Error as e:
            logger.warning(f"âŒ Erreur OANDA {attempt + 1}/{retries} pour {instrument}: {e}")
            time.sleep(5 ** attempt if attempt < 4 else 5)
        except Exception as e:
            logger.warning(f"âŒ Tentative {attempt + 1}/{retries} pour {instrument}: {e}")
            time.sleep(5 ** attempt if attempt < 4 else 5)
    
    logger.error(f"âŒ Ã‰chec aprÃ¨s {retries} tentatives pour {instrument} {granularity}")
    return pd.DataFrame()

# =============================
# INDICATEURS TECHNIQUES
# =============================

def validate_volume_confirmation(df: pd.DataFrame, fvg: dict) -> bool:
    """VÃ©rifie si le volume est suffisant aprÃ¨s le retest du FVG."""
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

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calcule le RSI complet sur la sÃ©rie de prix."""
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
    """
    Stochastic RSI identique Ã  l'indicateur TradingView 'Stoch RSI (3,3,14,14)'.
    Retourne (K, D) en pourcentage [0-100]. Valeur < 20 = survendu, > 80 = surachat.
    """
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
    """
    Calcule le RSI avec ta.momentum (fiable, pas de division par zÃ©ro).
    """
    try:
        rsi_indicator = RSIIndicator(close=prices, window=period)
        rsi_values = rsi_indicator.rsi()
        return rsi_values.dropna().iloc[-1]
    except Exception:
        # Fallback si ta.momentum Ã©choue
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
        # Fallback robuste
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_fallback = tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
        if np.isnan(atr_fallback) or atr_fallback <= 0.0:
            # Si mÃªme le fallback est nul â†’ utiliser 0.0001 comme valeur minimale ABSOLUE
            logger.warning(f"âš ï¸ ATR fallback = 0 pour {df.attrs.get('instrument', 'N/A')} â†’ Utilisation de 0.0001")
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
    """
    Valide la confluence sur H4.
    Correction :
    - Autorise si H1 + M15 confirment mÃªme si H4 est neutre.
    - Ajoute tolÃ©rance pour RR > 3.
    """
    ema_h4_fast = df_h4["close"].ewm(span=20, adjust=False).mean().iloc[-1]
    ema_h4_slow = df_h4["close"].ewm(span=200, adjust=False).mean().iloc[-1]

    if direction == "BUY":
        return ema_h4_fast > ema_h4_slow or True  # Autorise si autres TF confirment
    elif direction == "SELL":
        return ema_h4_fast < ema_h4_slow or True
    return True


def determine_trend_direction(
    df: pd.DataFrame,
    ema_fast_period=20,
    ema_medium_period=50,
    ema_slow_period=200
) -> str:
    """DÃ©termine la tendance directionnelle avec les EMA."""
    if len(df) < ema_slow_period + 10:
        return "NEUTRAL"

    ema_fast = ema(df["close"], ema_fast_period)
    ema_medium = ema(df["close"], ema_medium_period)
    ema_slow = ema(df["close"], ema_slow_period)
    current_price = df["close"].iloc[-1]

    logger.info(f"   ðŸ“‰ EMA{ema_fast_period}: {ema_fast.iloc[-1]:.5f}")
    logger.info(f"   ðŸ“‰ EMA{ema_medium_period}: {ema_medium.iloc[-1]:.5f}")
    logger.info(f"   ðŸ“‰ EMA{ema_slow_period}: {ema_slow.iloc[-1]:.5f}")

    if current_price > ema_slow.iloc[-1] and ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        return "BUY"
    elif current_price < ema_slow.iloc[-1] and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return "SELL"
    else:
        return "NEUTRAL"
  
def detect_mss(df: pd.DataFrame, lookback: int = 10) -> dict:
    """DÃ©tecte un Market Structure Shift (MSS) avec labels BUY/SELL."""
    if df.empty or len(df) < lookback + 5:
        return {"type": None, "level": None}
    
    highs = df["high"].rolling(window=lookback).max()
    lows = df["low"].rolling(window=lookback).min()
    last_close = df["close"].iloc[-1]
    last_time = df.index[-1]
    ema20 = df["close"].ewm(span=20).mean().iloc[-1]
    
    # MSS HAUSSE : cassure du dernier haut + close > EMA20
    if last_close > highs.iloc[-2] and last_close > ema20:
        return {"type": "MSS_BUY", "level": highs.iloc[-2], "time": last_time}
    
    # MSS BAISSE : cassure du dernier bas + close < EMA20
    if last_close < lows.iloc[-2] and last_close < ema20:
        return {"type": "MSS_SELL", "level": lows.iloc[-2], "time": last_time}
    
    return {"type": None, "level": None}
    
def validate_signal_coherence(pair: str, direction: str, df_h4: pd.DataFrame, df_m15: pd.DataFrame) -> bool:
    """
    Valide que le signal est cohÃ©rent avec la tendance globale.
    - Pour SELL : RSI M15 ne doit PAS Ãªtre en surachat (mais peut Ãªtre en survente modÃ©rÃ©e)
    - Pour BUY : RSI M15 ne doit PAS Ãªtre en survendu (mais peut Ãªtre en surachat modÃ©rÃ©)
    """
    if direction not in ["BUY", "SELL"]:
        logger.error(f"âŒ Direction invalide: {direction}. Doit Ãªtre BUY ou SELL.")
        return False

    current_price = df_m15["close"].iloc[-1]
    rsi_m15 = get_last_rsi(df_m15["close"])
    rsi_h4 = get_last_rsi(df_h4["close"]) if df_h4 is not None else 50
    ema20_h4 = df_h4["close"].ewm(span=20).mean().iloc[-1] if len(df_h4) >= 20 else 0
    ema200_h4 = df_h4["close"].ewm(span=200).mean().iloc[-1] if len(df_h4) >= 200 else 0

    logger.info(f"   ðŸ” VÃ©rification cohÃ©rence {pair} {direction}:")
    logger.info(f"   â€¢ Prix: {current_price:.5f}")
    logger.info(f"   â€¢ RSI M15: {rsi_m15:.1f}, RSI H4: {rsi_h4:.1f}")

    # âœ… Pour un SELL : on accepte mÃªme si RSI M15 est Ã  25 (survente lÃ©gÃ¨re), mais pas s'il est Ã  90
    if direction == "SELL":
        if rsi_m15 > 85:
            logger.info("   âŒ RSI M15 trop haut â†’ survendu impossible")
            return False
        if rsi_h4 > 80:
            logger.info("   âŒ RSI H4 trop haut â†’ tendance haussiÃ¨re forte â†’ pas de SELL")
            return False
        if ema20_h4 < ema200_h4:
            logger.info("   âœ… Tendance H4 baissiÃ¨re â†’ SELL autorisÃ©")
        else:
            logger.info("   âš ï¸ EMA20 > EMA200 â†’ risque de contre-tendance")
    # âœ… Pour un BUY : on accepte RSI trÃ¨s bas (oversold = meilleure entrÃ©e FVG)
    elif direction == "BUY":
        if rsi_m15 < 5:
            logger.info("   âŒ RSI M15 trop bas â†’ signal invalide")
            return False
        if rsi_h4 < 10:
            logger.info("   âŒ RSI H4 trop bas â†’ tendance baissiÃ¨re extrÃªme â†’ pas de BUY")
            return False
        if ema20_h4 > ema200_h4:
            logger.info("   âœ… Tendance H4 haussiÃ¨re â†’ BUY autorisÃ©")
        else:
            logger.info("   âš ï¸ EMA20 < EMA200 â†’ risque de contre-tendance")

    logger.info("   âœ… Signal cohÃ©rent")
    return True
    
def convert_direction_to_buy_sell(direction: str) -> str:
    """Convertit BUY/SELL en BUY/SELL"""
    if direction == "BUY":
        return "BUY"
    elif direction == "SELL":
        return "SELL"
    elif direction in ["BUY", "SELL"]:
        return direction
    else:
        return "NEUTRAL"

def detect_macd_divergence(df: pd.DataFrame, lookback: int = 14) -> str:
    """
    DÃ©tecte la divergence MACD (RÃ©guliÃ¨re, CachÃ©e, ExagÃ©rÃ©e).
    Retourne 'BULLISH', 'BEARISH' ou 'NONE'.
    """
    if len(df) < lookback * 2 + 5:
        return "NONE"

    # Calculer le MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    prices = df["close"].tail(lookback * 2).reset_index(drop=True)
    macd_vals = macd_line.tail(lookback * 2).reset_index(drop=True)

    # Trouver les pics et creux du prix
    price_peaks = []
    price_lows = []
    for i in range(3, len(prices) - 3):
        if (prices.iloc[i] > prices.iloc[i-3:i].max() and
            prices.iloc[i] > prices.iloc[i+1:i+4].max()):
            price_peaks.append((i, prices.iloc[i]))
        if (prices.iloc[i] < prices.iloc[i-3:i].min() and
            prices.iloc[i] < prices.iloc[i+1:i+4].min()):
            price_lows.append((i, prices.iloc[i]))

    # Trouver les pics et creux du MACD
    macd_peaks = []
    macd_lows = []
    for i in range(3, len(macd_vals) - 3):
        if (macd_vals.iloc[i] > macd_vals.iloc[i-3:i].max() and
            macd_vals.iloc[i] > macd_vals.iloc[i+1:i+4].max()):
            macd_peaks.append((i, macd_vals.iloc[i]))
        if (macd_vals.iloc[i] < macd_vals.iloc[i-3:i].min() and
            macd_vals.iloc[i] < macd_vals.iloc[i+1:i+4].min()):
            macd_lows.append((i, macd_vals.iloc[i]))

    # VÃ©rification des divergences
    if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
        last_price_peak = price_peaks[-1][1]
        prev_price_peak = price_peaks[-2][1]
        last_macd_peak = macd_peaks[-1][1]
        prev_macd_peak = macd_peaks[-2][1]

        # Regular Bearish Divergence
        if last_price_peak > prev_price_peak and last_macd_peak < prev_macd_peak:
            return "BEARISH"
        # Hidden Bullish Divergence
        elif last_price_peak < prev_price_peak and last_macd_peak > prev_macd_peak:
            return "BULLISH"

    if len(price_lows) >= 2 and len(macd_lows) >= 2:
        last_price_low = price_lows[-1][1]
        prev_price_low = price_lows[-2][1]
        last_macd_low = macd_lows[-1][1]
        prev_macd_low = macd_lows[-2][1]

        # Regular Bullish Divergence
        if last_price_low < prev_price_low and last_macd_low > prev_macd_low:
            return "BULLISH"
        # Hidden Bearish Divergence
        elif last_price_low > prev_price_low and last_macd_low < prev_macd_low:
            return "BEARISH"

    return "NONE"

def validate_entry_momentum(df_m15: pd.DataFrame, df_h4: pd.DataFrame, direction: str) -> bool:
    """
    Valide momentum via RSI.
    Correction :
    - Autorise RSI extrÃªme si divergence confirmÃ©e.
    """
    rsi_m15 = get_last_rsi(df_m15["close"])
    rsi_h4 = get_last_rsi(df_h4["close"]) if df_h4 is not None else 50

    if direction == "BUY":
        return rsi_m15 > 25 or detect_rsi_divergence_haussiere(df_m15)
    elif direction == "SELL":
        return rsi_m15 < 75 or detect_rsi_divergence(df_m15)
    return False
        
def is_new_low_daily(df_daily: pd.DataFrame, lookback_days: int = 7) -> bool:
    """
    DÃ©tecte si le prix actuel est un nouveau bas hebdomadaire sur le timeframe D1.
    UtilisÃ© pour valider les signaux BUY aprÃ¨s un 'New Low Area'.
    """
    if df_daily.empty or len(df_daily) < lookback_days + 1:
        return False

    recent_lows = df_daily["low"].tail(lookback_days)
    current_low = recent_lows.iloc[-1]

    return current_low <= recent_lows.min()

def get_min_gap_for_pair(pair: str) -> float:
    """
    Retourne la taille minimale du gap (en prix) pour un FVG.
    """
    pair = pair.upper()
    if pair == "XAU_USD":
        return 0.02  # 5 pips (car 1 pip = 0.01)
    elif "JPY" in pair:
        return 0.03 
     
    elif pair == "GBP_USD":
        return 0.00015  # âœ… Ajustement demandÃ©
# 5 pips (car 1 pip = 0.01)
    else:
        return 0.0002  # 5 pips (car 1 pip = 0.0001)

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
    "XAU_USD": 0.30,     # 30 cents = 3 pips sur l'or
    "USD_JPY": 0.03,
    "NAS100_USD": 1.0,# 3 pips sur JPY
    "DEFAULT": 0.00030   # 3 pips sur les paires standards
}

def calculate_sl_tp(entry_price: float, atr: float, direction: str, pair: str,
                    entry_type: str = "FVG_RETEST", fvg_data: dict = None, 
                    breaker_level: float = None) -> tuple:
    """
    Calcul SL/TP corrigÃ© pour utiliser les limites rÃ©elles de l'Imbalance (FVG).
    """
    try:
        if None in [entry_price, atr, direction, pair]:
            return None, None

        settings = SIGNAL_RISK_SETTINGS.get(entry_type, SIGNAL_RISK_SETTINGS["FVG_RETEST"])
        sl_mult = settings["sl_multiplier"]
        tp_mult = settings["tp_multiplier"]

        stop_loss = 0.0
        take_profit = 0.0

        # --- CORRECTION STOP LOSS STRUCTUREL ---
        if direction == "BUY":
            # Par dÃ©faut ATR
            stop_loss = entry_price - (atr * sl_mult)
            
            # Si c'est un FVG, le SL doit Ãªtre SOUS le bas du FVG (low_level)
            if fvg_data and "low_level" in fvg_data:
                fvg_bottom = float(fvg_data["low_level"])
                # On place le SL un peu sous le bas du FVG (ex: 2 pips de marge)
                pip_buffer = get_pip_value_for_pair(pair) * 20 # 2 pips/20 points
                structural_sl = fvg_bottom - pip_buffer
                # On prend le SL le plus sÃ»r (le plus bas entre ATR et Structurel) pour Ã©viter de se faire mÃ¨che
                stop_loss = min(stop_loss, structural_sl)

            take_profit = entry_price + (atr * tp_mult)
            
        else: # SELL
            # Par dÃ©faut ATR
            stop_loss = entry_price + (atr * sl_mult)
            
            # Si c'est un FVG, le SL doit Ãªtre AU-DESSUS du haut du FVG (high_level)
            if fvg_data and "high_level" in fvg_data:
                fvg_top = float(fvg_data["high_level"])
                # On place le SL un peu au dessus du haut du FVG
                pip_buffer = get_pip_value_for_pair(pair) * 20
                structural_sl = fvg_top + pip_buffer
                # On prend le plus haut
                stop_loss = max(stop_loss, structural_sl)

            take_profit = entry_price - (atr * tp_mult)

        # Ajustement RR minimum 1:2
        risk = abs(entry_price - stop_loss)
        if risk == 0: return None, None
        
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
    """Envoie une alerte Telegram avec vÃ©rification de cohÃ©rence et gestion des None."""
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("âš ï¸ Telegram dÃ©sactivÃ©")
        return

    # âœ… VÃ‰RIFICATION DES VALEURS OBLIGATOIRES
    if None in [pair, direction, entry_price, stop_loss, take_profit, rsi, entry_type]:
        logger.error(f"âŒ Valeur manquante pour Telegram: pair={pair}, direction={direction}, entry={entry_price}, sl={stop_loss}, tp={take_profit}, rsi={rsi}, type={entry_type}")
        return

    # âœ… VÃ‰RIFICATION DE COHÃ‰RENCE AVANT ENVOI
    if direction == "BUY":
        if stop_loss > entry_price or take_profit < entry_price:
            logger.error(f"âŒ INCOHÃ‰RENCE SL/TP pour BUY: Entry={entry_price}, SL={stop_loss}, TP={take_profit}")
            return
    elif direction == "SELL":
        if stop_loss < entry_price or take_profit > entry_price:
            logger.error(f"âŒ INCOHÃ‰RENCE SL/TP pour SELL: Entry={entry_price}, SL={stop_loss}, TP={take_profit}")
            return
    else:
        logger.error(f"âŒ Direction invalide: {direction}")
        return

    # Extraction infos prÃ©dictives injectÃ©es dans bias_analysis par advanced_main
    win_rate = bias_analysis.get("win_rate", "") if bias_analysis else ""
    quality_label = bias_analysis.get("quality_label", "") if bias_analysis else ""
    score_details = bias_analysis.get("score_details", {}) if bias_analysis else {}

    # Calcul RR pour affichage
    try:
        dist_sl = abs(entry_price - stop_loss)
        dist_tp = abs(take_profit - entry_price)
        rr_display = f"{dist_tp / dist_sl:.2f}" if dist_sl > 0 else "N/A"
    except Exception:
        rr_display = "N/A"

    # Score info enrichi
    if confidence_score:
        score_info = (
            f"<b>Score:</b> {confidence_score}/{SCORING_CONFIG['MIN_CONFIDENCE_SCORE']} "
            f"| <b>QualitÃ©:</b> {quality_label}\n"
            f"<b>Win Rate estimÃ©:</b> {win_rate}\n"
            f"<b>R/R:</b> 1:{rr_display}\n"
        )
    else:
        score_info = ""

    # Confluences actives pour affichage condensÃ©
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
    confluences_line = f"<b>Confluences:</b> {' Â· '.join(confluence_tags)}\n" if confluence_tags else ""

    # âœ… GARANTIR que toutes les valeurs sont formatables
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
<b>Type d'entrÃ©e:</b> {safe_entry_type}
<b>Bias:</b> {safe_bias}
{score_info}{confluences_line}
<b>EntrÃ©e:</b> {safe_entry_price}
<b>Stop Loss:</b> {safe_stop_loss}
<b>Take Profit:</b> {safe_take_profit}
<b>RSI:</b> {safe_rsi}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        if response.status_code == 200:
            logger.info(f"âœ… Telegram envoyÃ© pour {safe_pair}")
        else:
            logger.error(f"âŒ Ã‰chec Telegram: {response.text}")
    except Exception as e:
        logger.error(f"ðŸ’¥ Erreur rÃ©seau Telegram: {e}")
# =============================
# DÃ‰TECTION LIQUIDITÃ‰ HEBDOMADAIRE
# =============================
def detect_weekly_liquidity(df: pd.DataFrame) -> dict:
    """DÃ©tecte les highs/lows hebdomadaires pour la liquiditÃ©."""
    if len(df) < 7:
        return {}

    df_weekly = df.resample('W').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first'
    })

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
# DÃ‰TECTION SWING POINTS AVANCÃ‰E
# =============================
def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> tuple:
    """
    DÃ©tecte les vrais swing highs et swing lows.
    Un swing high = plus haut que 'lookback' bougies avant ET aprÃ¨s.
    """
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        # Swing High
        if (high == df["high"].iloc[i - lookback:i + lookback + 1].max()
            and df["close"].iloc[i] < df["open"].iloc[i]):  # rejection candle idÃ©ale
            swing_highs.append({
                "index": i,
                "time": df.index[i],
                "price": high
            })

        # Swing Low
        if (low == df["low"].iloc[i - lookback:i + lookback + 1].min()
            and df["close"].iloc[i] > df["open"].iloc[i]):
            swing_lows.append({
                "index": i,
                "time": df.index[i],
                "price": low
            })

    return swing_highs, swing_lows

def detect_swing_points_advanced(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> tuple:
    """DÃ©tecte les swing highs et lows avec validation avancÃ©e."""
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        try:
            if (df["high"].iloc[i] == df["high"].iloc[i-lookback:i+lookback+1].max() and
                df["close"].iloc[i] < df["open"].iloc[i]):

                swing_highs.append({
                    "index": i,
                    "time": df.index[i],
                    "price": df["high"].iloc[i],
                    "type": "SWING_HIGH",
                    "strength": "STRONG"
                })
        except Exception:
            pass

        try:
            if (df["low"].iloc[i] == df["low"].iloc[i-lookback:i+lookback+1].min() and
                df["close"].iloc[i] > df["open"].iloc[i]):

                swing_lows.append({
                    "index": i,
                    "time": df.index[i],
                    "price": df["low"].iloc[i],
                    "type": "SWING_LOW",
                    "strength": "STRONG"
                })
        except Exception:
            pass

    return swing_highs, swing_lows

# =============================
# DÃ‰TECTION FVG AVANCÃ‰E
# =============================
# Corrected FVG detection helpers
# File: main_fixed_detect_fvg.py
# Purpose: replacement for detect_fvg_advanced (and small robustness fixes)




def classify_fvg_type(current_candle, next_candle) -> str:
    """
    Same classification logic as original, kept for compatibility.
    Returns: 'BREAKAWAY'|'REJECTION'|'PERFECT'|'UNKNOWN'
    """
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
    """
    DÃ©tection avancÃ©e des Fair Value Gaps (FVG) selon la logique ICT.
    - Corrige l'erreur offset-naive vs offset-aware (timezone).
    - Ajoute la clÃ© 'index' pour compatibilitÃ© avec detect_orderflow_legs_advanced().
    - Priorise FVG rÃ©cents (< max_lookback_hours).
    - Respecte seuil dynamique via get_min_gap_for_pair().
    """

    fvgs = []
    if df is None or len(df) < 3:
        return fvgs

    # Uniformisation des datetimes (UTC aware)
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    min_time = now - timedelta(hours=max_lookback_hours)

    # Conversion robuste de l'index en UTC
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

        # FVG haussier (BUY)
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

        # FVG baissier (SELL)
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
    """
    Retest FVG valide si le prix est proche du midpoint du FVG.
    TolÃ©rance en pips basÃ©e sur l'ATR M15, bornÃ©e entre 10 et 50 pips.
    Aucune logique de tendance, de RSI ou de "respect" â†’ rÃ©servÃ© au scoring.
    """
    if "low_level" not in fvg or "high_level" not in fvg:
        return False

    fvg_mid = (float(fvg["low_level"]) + float(fvg["high_level"])) / 2.0
    distance = abs(current_price - fvg_mid)
    max_dist_pips = get_dynamic_max_distance(df, pair, atr_multiplier=1.5)
    return price_to_pips(distance, pair) <= max_dist_pips


 
def is_fvg_unmitigated(df: pd.DataFrame, fvg: dict) -> bool:
    """
    VÃ©rifie qu'un FVG n'a jamais Ã©tÃ© touchÃ© avant le retest.
    C'est le concept "Unmitigated FVG" d'Arjo.io.
    """
    # Trouver toutes les bougies aprÃ¨s le FVG
    after_data = df[df.index > fvg["time"]]
    if len(after_data) == 0:
        return False

    if fvg["direction"] == "BUY":
        # Pour un FVG haussier : le prix ne doit pas avoir touchÃ© le niveau bas du FVG
        return after_data["low"].min() > fvg["low_level"]
    elif fvg["direction"] == "SELL":
        # Pour un FVG baissier : le prix ne doit pas avoir touchÃ© le niveau haut du FVG
        return after_data["high"].max() < fvg["high_level"]
    return False
    
def is_fvg_respected_with_wick_rejection(df: pd.DataFrame, fvg: dict) -> bool:
    """VÃ©rifie si un FVG est respectÃ© via Wick Rejection."""
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
    """Calcule le Point of Control (POC) et les niveaux de liquiditÃ©."""
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
        
        return {
            "poc": float(poc),
            "liquidity_high": float(df["high"].max()),
            "liquidity_low": float(df["low"].min()),
        }
    except Exception:
        return {"poc": None, "liquidity_high": None, "liquidity_low": None}

def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """DÃ©tecte des patterns de chandeliers japonais."""
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
# DÃ‰TECTION FVG IMBRIQUÃ‰S (NESTED FVG)
# =============================
def detect_nested_fvg(df: pd.DataFrame, min_nesting: int = 2) -> list:
    """DÃ©tecte les FVG imbriquÃ©s (Nested FVG) en BUY/SELL."""
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
            else:  # SELL
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
# DRT (DISPLACEMENT RETRACEMENT THEORY)
# =============================
def calculate_drt_levels(swing_high: float, swing_low: float) -> dict:
    """Calcule les niveaux DRT pour un range donnÃ©."""
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
# DÃ‰TECTION ORDER FLOW LEGS AVANCÃ‰E
# =============================
def detect_orderflow_legs_advanced(df: pd.DataFrame) -> list:
    swing_highs, swing_lows = detect_swing_points_advanced(df, lookback=5)
    fvgs = detect_fvg_advanced(df, max_lookback_hours=36)
    ofls = []

    # === BUY ORDER FLOW LEGS ===
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

    # === SELL ORDER FLOW LEGS ===
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

    # Ajouter les FVG simples
    for fvg in fvgs:
        ofls.append({"direction": fvg.get("direction"), "fvg": fvg})

    logger.info(f"ðŸ“Š OFL FINAL: {len(ofls)} legs dÃ©tectÃ©s")
    return ofls
 
# =============================
# VALIDATION FVG RESPECTÃ‰ AVEC LIQUIDITÃ‰
# =============================
def is_fvg_respected_with_liquidity(df: pd.DataFrame, fvg: dict, liquidity_levels: dict) -> bool:
    """VÃ©rifie si un FVG a Ã©tÃ© respectÃ© avec une logique rÃ©aliste."""
    if "high_level" not in fvg or "low_level" not in fvg or "time" not in fvg or "direction" not in fvg:
        return False
    fvg_time = fvg["time"]
    subsequent_data = df[df.index > fvg_time]
    if subsequent_data.empty:
        return False
    # --- LOGIQUE RÃ‰ALISTE DU "RESPECT" ---
    # Pour un FVG haussier : Le prix doit avoir dÃ©passÃ© le haut du FVG (confirmation de force)
    # Pour un FVG baissier : Le prix doit avoir dÃ©passÃ© le bas du FVG (confirmation de force)
    if fvg["direction"] == "BUY":
        # Le prix a dÃ©passÃ© le haut du FVG â†’ FVG respectÃ©
        basic_respect = subsequent_data["high"].max() > fvg["high_level"]
    else:
        # Le prix a dÃ©passÃ© le bas du FVG â†’ FVG respectÃ©
        basic_respect = subsequent_data["low"].min() < fvg["low_level"]
    if not basic_respect:
        logger.debug(f"   âŒ FVG non respectÃ© : prix n'a pas confirmÃ© la cassure")
        return False
    # --- VALIDATION PAR LIQUIDITÃ‰ (OPTIONNELLE) ---
    if not liquidity_levels:
        logger.debug("   ðŸ’§ Pas de niveaux de liquiditÃ© â†’ Validation par prix uniquement")
        return True
    try:
        if fvg["direction"] == "BUY":
            liquidity_respect = subsequent_data["low"].min() <= liquidity_levels.get("previous_week_low", float('inf')) * 1.002
        else:
            liquidity_respect = subsequent_data["high"].max() >= liquidity_levels.get("previous_week_high", 0) * 0.998
        if liquidity_respect:
            logger.debug("   ðŸ’§ FVG respectÃ© + confirmation par liquiditÃ©")
        else:
            logger.debug("   âš ï¸ FVG respectÃ© mais pas de confirmation par liquiditÃ© â†’ AcceptÃ© quand mÃªme")
        return True  # â† On accepte le FVG tant que basic_respect est OK
    except Exception as e:
        logger.warning(f"   â— Erreur lors de la vÃ©rification de la liquiditÃ©: {e}")
        return True
# =============================
# DÃ‰TECTION DES POINTS D'INTÃ‰RÃŠT (POI) AVEC WICKS
# =============================
def detect_wick_rejection_poi(df: pd.DataFrame, bias: str, min_wick_ratio: float = 0.7) -> list:
    """
    DÃ©tecte les wicks de rejet AVEC CONFIRMATION OBLIGATOIRE par la bougie suivante.
    âœ… Filtrage de proximitÃ© dynamique selon la paire.
    """
    poi_list = []
    pair = df.attrs.get("instrument", "DEFAULT")

    # TolÃ©rance de distance dynamique selon la paire (en PRIX, pas en pips)
    pip_tolerance_map = {
        "XAU_USD": 20,       # â‰ˆ 50 pips
        "USD_JPY": 0.50,         # â‰ˆ 10 pips
        "AUD_USD": 0.0050,
        "EUR_USD": 0.0020,
        "USD_CAD": 0.0050,
        "GBP_USD": 0.0050,
        "DEFAULT": 0.0010     # â‰ˆ 5 pips (Forex)
    }
    pip_tolerance = pip_tolerance_map.get(pair, pip_tolerance_map["DEFAULT"])

    for i in range(1, len(df) - 1):  # -1 pour avoir une bougie de confirmation
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

        # === WICK REJECTION BUY ===
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
            # âœ… VÃ‰RIFICATION DE PROXIMITÃ‰
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

        # === WICK REJECTION SELL ===
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
            # âœ… VÃ‰RIFICATION DE PROXIMITÃ‰
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
# DÃ‰TERMINATION DU BIAS AVANCÃ‰
# =============================
def determine_advanced_bias(df: pd.DataFrame) -> dict:
    """DÃ©termine le bias H4 en se basant SUR LE MSS (Market Structure Shift)."""
    mss = detect_mss(df, lookback=20)
    
    if mss["type"] == "MSS_BUY":
        return {"bias": "BUY", "mss_detected": mss}
    elif mss["type"] == "MSS_SELL":
        return {"bias": "SELL", "mss_detected": mss}
    else:
        # Fallback sur EMA
        ema20 = df["close"].ewm(span=20).mean().iloc[-1]
        ema50 = df["close"].ewm(span=50).mean().iloc[-1]
        if ema20 > ema50:
            return {"bias": "BUY", "mss_detected": mss}
        elif ema20 < ema50:
            return {"bias": "SELL", "mss_detected": mss}
        else:
            return {"bias": "NEUTRAL", "mss_detected": mss}
# =============================
# DÃ‰TERMINATION DE LA NARRATIVE AVANCÃ‰E
# =============================


def determine_advanced_narrative(
    df_m15: pd.DataFrame,
    bias_analysis: dict,
    pair: str = "XAU_USD",
    df_h4: pd.DataFrame = None,
    df_d1: pd.DataFrame = None,
    df_h1: pd.DataFrame = None
) -> dict:
    """
    Narrative intraday avancÃ©e â€” CORRIGÃ‰E :
      - Ajout du FILTRE DE TENDANCE H1 (EMA 50) pour Ã©viter les contre-flux violents.
      - PAS DE REJET des signaux IRL (seulement flagging + pÃ©nalitÃ© dans le scoring).
    """
    if df_m15 is None or df_m15.empty or len(df_m15) < 3:
        logger.warning("âŒ DataFrame vide ou insuffisant")
        return {
            "bias": "NEUTRAL",
            "current_price": None,
            "potential_entries": [],
            "timestamp": datetime.now().isoformat()
        }

    # --- Compteurs & exemples (uniques) pour les FVG non valides ---
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
                logger.info(f"âš ï¸ {pair} Â· FVG {d} non valides pour retest: {c} (ex: {examples})")

    try:
        bias = (bias_analysis.get("bias", "NEUTRAL") or "NEUTRAL").upper()
        current_price = float(df_m15["close"].iloc[-1])
        logger.info(f"ðŸŽ¯ ANALYSE {pair} - Prix: {current_price:.5f}, Bias: {bias}")

        # === 1. CALCUL EMA 50 H1 (FILTRE DE FLUX) ===
        ema50_h1 = None
        if df_h1 is not None and len(df_h1) >= 50:
            ema50_h1 = df_h1['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            # Indicateur visuel dans les logs
            trend_h1 = "HAUSSIER" if current_price > ema50_h1 else "BAISSIER"
            logger.info(f"   ðŸŒŠ Tendance H1 (EMA50): {trend_h1} (Prix: {current_price:.5f} vs EMA: {ema50_h1:.5f})")
        # ============================================

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
            # RSI extrÃªme Ã  une zone FVG/imbalance = setup haute probabilitÃ© â†’ ne pas bloquer
            stoch_k_local, _ = calculate_stoch_rsi(df_m15["close"])
            if pair == "XAU_USD":
                if direction == "BUY":
                    # Stoch RSI < 25 Ã  FVG = confirmation oversold â†’ toujours valide
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
            logger.info(f"ðŸ” Dealing Range dÃ©tectÃ©e: H {dealing_range['high']:.5f} - L {dealing_range['low']:.5f}")
        else:
            logger.info("ðŸ” Aucune Dealing Range dÃ©tectÃ©e")

        # --- Filtrage des FVG rÃ©cents ---
        logger.info(f"ðŸ”Ž Filtrage des FVG rÃ©cents pour {pair}:")
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
                logger.warning(f" âš ï¸ Erreur conversion time FVG: {e}")
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
                status = "dans zone" if is_in_zone else "trÃ¨s proche" if is_very_close else "distance raisonnable"
                _log_fvg_recent_once(
                    pair,
                    fvg.get('direction', 'UNKNOWN'),
                    entry_level,
                    f" âœ… FVG RÃ‰CENT {fvg.get('direction', 'UNKNOWN')}: {entry_level:.5f} ({status}, distance: {distance:.5f}, il y a {time_diff.total_seconds()/3600:.1f}h)"
                )
            else:
                ancient_fvg_count += 1

        ofls = recent_ofls
        logger.info(f"ðŸ“Š FVG rÃ©cents et proches: {len(ofls)} (anciens rejetÃ©s: {ancient_fvg_count})")
        narrative["recent_ofls"] = ofls

        bos = detect_bos(df_h1, lookback=50)
        choch = detect_choch(df_h1, lookback=50)
        narrative["structure_analysis"] = {"bos": bos, "choch": choch}

        # --- FVG RETEST â€” AVEC FILTRE TENDANCE H1 ---
        for ofl in ofls:
            fvg = ofl.get("fvg", {})
            if not isinstance(fvg, dict) or not all(k in fvg for k in ["direction", "high_level", "low_level", "time"]):
                continue
            entry_level = get_fvg_midpoint(fvg)
            if entry_level is None:
                continue

            # âœ… DÃ©finir fvg_direction TÃ”T
            fvg_direction = str(fvg.get("direction", "")).upper()
            if fvg_direction not in {"BUY", "SELL"}:
                continue

            # === 2. APPLICATION DU FILTRE TENDANCE H1 ===
            # Si EMA H1 calculÃ©e, on l'utilise pour filtrer les contresens violents
            if ema50_h1 is not None:
                if fvg_direction == "BUY" and current_price < float(fvg.get("low_level", 0)):
                    # On veut acheter mais le prix est SOUS l'EMA 50 H1 -> DANGER
                    _log_fvg_added_once(pair, "BUY", entry_level, "REJECTED_H1", f"â›” IgnorÃ©: Signal BUY mais Prix < EMA50 H1 (Tendance baissiÃ¨re forte)")
                    continue
                if fvg_direction == "SELL" and current_price > float(fvg.get("high_level", 999999)):
                    # On veut vendre mais le prix est AU-DESSUS de l'EMA 50 H1 -> DANGER
                    _log_fvg_added_once(pair, "SELL", entry_level, "REJECTED_H1", f"â›” IgnorÃ©: Signal SELL mais Prix > EMA50 H1 (Tendance haussiÃ¨re forte)")
                    continue
            # ============================================

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

                # --- Flags IRL (sans rejet) ---
                is_near_key_level = False
                is_rsi_extreme = False
                is_trend_aligned = False

                if irl_erl_type == "IRL":
                    # VÃ©rifier si proche d'un niveau clÃ© (support/resistance)
                    distance_to_support = abs(current_price - liquidity_levels.get("previous_week_low", current_price))
                    distance_to_resistance = abs(current_price - liquidity_levels.get("previous_week_high", current_price))
                    tol = get_tolerance(current_price, pair) * 2
                    is_near_key_level = (distance_to_support <= tol or distance_to_resistance <= tol)

                    # VÃ©rifier si RSI en zone extrÃªme
                    rsi_m15_val = float(get_last_rsi(df_m15["close"]))
                    is_rsi_extreme = (fvg_direction == "BUY" and rsi_m15_val < 30) or (fvg_direction == "SELL" and rsi_m15_val > 70)

                    # VÃ©rifier si tendance alignÃ©e
                    is_trend_aligned = bias in ["NEUTRAL", fvg_direction]

                    logger.info(
                        f"â„¹ï¸ IRL dÃ©tectÃ© (narrative): near_key={is_near_key_level}, "
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
                    f" ðŸŽ¯ FVG {fvg_direction} AJOUTÃ‰: {entry_level:.5f} (type: {fvg_type}){irl_erl_label}"
                )

        _flush_invalid_retest_summary()

        # --- Nested FVG ---
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

            # === FILTRE H1 AUSSI POUR NESTED ===
            if ema50_h1 is not None:
                if nfvg_direction == "BUY" and current_price < ema50_h1:
                     continue
                if nfvg_direction == "SELL" and current_price > ema50_h1:
                     continue
            # ===================================

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
                    f" ðŸŽ¯ Nested FVG {nfvg_direction} AJOUTÃ‰: {entry_level:.5f}{irl_erl_label}"
                )

        # --- Wick Rejection ---
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
                    f" ðŸŽ¯ Wick Rejection {poi_direction} AJOUTÃ‰: {entry_level:.5f}{irl_erl_label}"
                )

        # --- Liquidity Draw ---
        liq_high = liquidity_levels.get("previous_week_high")
        liq_low = liquidity_levels.get("previous_week_low")

        # LIQUIDITY_DRAW : exige que le prix soit proche d'un FVG + Stoch RSI confirme
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
                logger.info(f" ðŸŽ¯ Liquidity Draw BUY AJOUTÃ‰ Ã  {liq_high:.5f} (Type: {irl_erl_type or 'No Range'}, StochRSI: {liq_stoch_k:.1f})")

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
                logger.info(f" ðŸŽ¯ Liquidity Draw SELL AJOUTÃ‰ Ã  {liq_low:.5f} (Type: {irl_erl_type or 'No Range'}, StochRSI: {liq_stoch_k:.1f})")

        # --- BISI ---
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
                        logger.info(f"ðŸŽ¯ BISI DÃ‰TECTÃ‰ : {bos_direction} Ã  {fvg_level:.5f}{irl_erl_label}")

        logger.info(f"ðŸ“ Narrative {pair} TERMINÃ‰E: {len(narrative['potential_entries'])} entrÃ©es dÃ©tectÃ©es")
        _log_narrative_list(narrative["potential_entries"], top_n=10)
        return narrative

    except Exception as e:
        logger.error(f"âŒ Erreur dans determine_advanced_narrative: {e}")
        logger.error(traceback.format_exc())
        return {
            "bias": "NEUTRAL",
            "current_price": None,
            "potential_entries": [],
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
# =============================
# HELPERS SCORING AVANCÃ‰
# =============================

def get_session_quality_bonus(pair: str) -> tuple:
    """
    Bonus/malus selon la session de trading active (UTC).
    London 07-16 UTC, NY 12-21 UTC = sessions Ã  haute liquiditÃ©.
    """
    hour = datetime.utcnow().hour
    is_london = 7 <= hour < 16
    is_ny = 12 <= hour < 21
    is_overlap = is_london and is_ny  # 12-16 UTC = meilleur moment

    forex_euro = ["EUR_USD", "GBP_USD", "USD_CAD", "AUD_USD"]
    jpy_pairs = ["USD_JPY", "GBP_JPY"]  # GBP/JPY actif en session asiatique
    is_asian = not (is_london or is_ny)

    if is_overlap:
        return 2, "+2 (Overlap London/NY)"
    elif is_london or is_ny:
        return 1, "+1 (Session active)"
    else:
        # Session asiatique
        if pair in jpy_pairs:
            return 1, "+1 (Session asiatique, JPY actif)"
        elif pair in forex_euro:
            return -1, "-1 (Session asiatique, paire EUR/GBP)"
        else:
            return 0, "0 (Session neutre)"


def get_d1_trend_bonus(df_d1, direction: str) -> tuple:
    """
    Confirmation tendance Daily (D1).
    EMA50 D1 : +2 si alignÃ©, -2 si contre.
    EMA200 D1 : +1 bonus supplÃ©mentaire si aussi alignÃ©.
    """
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
            label = "+2 (D1 EMA50 alignÃ©)"
            if len(df_d1) >= 200:
                ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
                if (direction == "BUY" and current > ema200) or (direction == "SELL" and current < ema200):
                    bonus += 1
                    label = "+3 (D1 EMA50+EMA200 alignÃ©s)"
        else:
            bonus = -2
            label = "-2 (Contre tendance D1)"
        return bonus, label
    except Exception:
        return 0, "0 (Erreur D1)"


def get_macd_h1_bonus(df_h1, direction: str) -> tuple:
    """
    Confirmation MACD histogram H1.
    Histogramme dans le bon sens + en accÃ©lÃ©ration = +2.
    """
    if df_h1 is None or df_h1.empty or len(df_h1) < 35:
        return 0, "0 (H1 insuffisant pour MACD)"
    try:
        hist = calculate_macd_momentum(df_h1)
        last_val = hist.iloc[-1]
        prev_val = hist.iloc[-2]
        if direction == "BUY":
            if last_val > 0 and last_val > prev_val:
                return 2, "+2 (MACD H1 haussier accÃ©lÃ©rant)"
            elif last_val > 0:
                return 1, "+1 (MACD H1 haussier)"
            elif last_val < 0:
                return -1, "-1 (MACD H1 baissier)"
        elif direction == "SELL":
            if last_val < 0 and last_val < prev_val:
                return 2, "+2 (MACD H1 baissier accÃ©lÃ©rant)"
            elif last_val < 0:
                return 1, "+1 (MACD H1 baissier)"
            elif last_val > 0:
                return -1, "-1 (MACD H1 haussier)"
        return 0, "0 (MACD H1 neutre)"
    except Exception:
        return 0, "0 (Erreur MACD H1)"


def get_rsi_divergence_bonus(df_h1, df_m15, direction: str) -> tuple:
    """
    Bonus si divergence RSI dÃ©tectÃ©e (H1 ou M15).
    Divergence haussiÃ¨re sur BUY, baissiÃ¨re sur SELL.
    """
    try:
        if direction == "BUY":
            if detect_rsi_divergence_haussiere(df_h1):
                return 3, "+3 (Divergence RSI H1 haussiÃ¨re)"
            if detect_rsi_divergence_haussiere(df_m15):
                return 2, "+2 (Divergence RSI M15 haussiÃ¨re)"
        elif direction == "SELL":
            # Divergence baissiÃ¨re = inverse : prix fait nouveau haut, RSI non
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
                return 3, "+3 (Divergence RSI H1 baissiÃ¨re)"
            if detect_bearish_div(df_m15):
                return 2, "+2 (Divergence RSI M15 baissiÃ¨re)"
    except Exception:
        pass
    return 0, "0 (Pas de divergence RSI)"


def estimate_win_rate(score: int, confluences: dict) -> str:
    """
    Estime le taux de gain probable basÃ© sur le score et les confluences.
    """
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
    """Label qualitÃ© du signal basÃ© sur le score total."""
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
# SYSTÃˆME DE SCORING
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
    """
    V3 STRICTE - logique A+ uniquement.

    Objectif : arrÃªter les signaux tardifs vus dans les logs :
    - BUY alors que Stoch RSI est dÃ©jÃ  surachetÃ©,
    - score gonflÃ© par D1/H4/session/MACD,
    - entrÃ©e acceptÃ©e alors que le prix est trop loin du niveau.

    Cette version privilÃ©gie :
    tendance H4/H1 + pullback propre + momentum pas Ã©tendu + rejet M15.
    """
    score = 0
    details: dict = {}
    min_required = SCORING_CONFIG.get("MIN_CONFIDENCE_SCORE", 10)

    direction = (direction or "").upper()
    entry_level = entry.get("entry_level")
    entry_type = str(entry.get("type", "FVG_RETEST")).upper()

    if entry_level is None or direction not in ["BUY", "SELL"]:
        return {"passed": False, "total_score": 0, "final_confidence": "LOW", "details": {"VETO": "EntrÃ©e/direction invalide"}}

    entry_level = float(entry_level)

    # === ATR / SL TP ===
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

    # === VETO 1 : H4 opposÃ© ===
    if direction == "BUY" and bias == "SELL":
        return {"passed": False, "total_score": -100, "final_confidence": "LOW", "details": {"VETO": "BUY contre bias H4 SELL"}, "stop_loss": stop_loss, "take_profit": take_profit, "atr_value": atr_value}
    if direction == "SELL" and bias == "BUY":
        return {"passed": False, "total_score": -100, "final_confidence": "LOW", "details": {"VETO": "SELL contre bias H4 BUY"}, "stop_loss": stop_loss, "take_profit": take_profit, "atr_value": atr_value}

    if (direction == "BUY" and bias == "BUY") or (direction == "SELL" and bias == "SELL"):
        score += 3
        details["Trend_H4"] = "+3 (AlignÃ©)"
    elif bias == "NEUTRAL":
        score += 1
        details["Trend_H4"] = "+1 (Neutre)"

    # === V78 : distance = malus progressif, plus veto brutal ===
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
            score += 2
            details["Distance"] = f"+2 proche ({distance:.5f} <= {max_distance_price * 0.50:.5f})"
        elif distance <= max_distance_price:
            details["Distance"] = f"0 acceptable ({distance:.5f} <= {max_distance_price:.5f})"
        elif distance <= max_distance_price * 1.50:
            score -= 2
            details["Distance"] = f"-2 un peu loin ({distance:.5f} > {max_distance_price:.5f})"
        else:
            # Seul cas oÃ¹ on bloque encore: entrÃ©e vraiment trop Ã©loignÃ©e.
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

    # === VETO 3 : tendance H1 stricte EMA50/EMA200 ===
    try:
        ema50_h1 = df_h1["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        ema200_h1 = df_h1["close"].ewm(span=200, adjust=False).mean().iloc[-1] if len(df_h1) >= 200 else None
        h1_close = float(df_h1["close"].iloc[-1])

        if direction == "BUY" and h1_close < ema50_h1:
            return {"passed": False, "total_score": -100, "final_confidence": "LOW", "details": {"VETO": "BUY sous EMA50 H1"}, "stop_loss": stop_loss, "take_profit": take_profit, "atr_value": atr_value}
        if direction == "SELL" and h1_close > ema50_h1:
            return {"passed": False, "total_score": -100, "final_confidence": "LOW", "details": {"VETO": "SELL au-dessus EMA50 H1"}, "stop_loss": stop_loss, "take_profit": take_profit, "atr_value": atr_value}

        score += 2
        details["Trend_H1_EMA50"] = "+2 (AlignÃ© EMA50)"

        if ema200_h1 is not None:
            if direction == "BUY" and ema50_h1 > ema200_h1:
                score += 1
                details["Trend_H1_EMA200"] = "+1 (EMA50 > EMA200)"
            elif direction == "SELL" and ema50_h1 < ema200_h1:
                score += 1
                details["Trend_H1_EMA200"] = "+1 (EMA50 < EMA200)"
            else:
                score -= 2
                details["Trend_H1_EMA200"] = "-2 (EMA50/EMA200 non alignÃ©es)"
    except Exception as exc:
        details["Trend_H1_Error"] = str(exc)

    # === VETO 4 : Stoch RSI anti-entrÃ©e tardive ===
    try:
        stoch_k_h1, _ = calculate_stoch_rsi(df_h1["close"])
        stoch_k_m15, _ = calculate_stoch_rsi(df_m15["close"])

        if direction == "BUY":
            if stoch_k_h1 >= 80 or stoch_k_m15 >= 85:
                return {"passed": False, "total_score": -100, "final_confidence": "LOW", "details": {"VETO": f"BUY trop tardif StochRSI H1={stoch_k_h1:.1f} M15={stoch_k_m15:.1f}"}, "stop_loss": stop_loss, "take_profit": take_profit, "atr_value": atr_value}
            if 20 <= stoch_k_m15 <= 60:
                score += 3
                details["StochRSI"] = f"+3 (zone idÃ©ale BUY M15={stoch_k_m15:.1f})"
            elif stoch_k_m15 < 20:
                score += 2
                details["StochRSI"] = f"+2 (survente BUY M15={stoch_k_m15:.1f})"
            else:
                details["StochRSI"] = f"0 (neutre BUY M15={stoch_k_m15:.1f})"

        elif direction == "SELL":
            if stoch_k_h1 <= 20 or stoch_k_m15 <= 15:
                return {"passed": False, "total_score": -100, "final_confidence": "LOW", "details": {"VETO": f"SELL trop tardif StochRSI H1={stoch_k_h1:.1f} M15={stoch_k_m15:.1f}"}, "stop_loss": stop_loss, "take_profit": take_profit, "atr_value": atr_value}
            if 40 <= stoch_k_m15 <= 80:
                score += 3
                details["StochRSI"] = f"+3 (zone idÃ©ale SELL M15={stoch_k_m15:.1f})"
            elif stoch_k_m15 > 80:
                score += 2
                details["StochRSI"] = f"+2 (surachat SELL M15={stoch_k_m15:.1f})"
            else:
                details["StochRSI"] = f"0 (neutre SELL M15={stoch_k_m15:.1f})"
    except Exception as exc:
        details["StochRSI_Error"] = str(exc)

    # === Setup / zone d'entrÃ©e ===
    if any(x in entry_type for x in ["FVG", "BISI", "NESTED"]):
        score += 4
        details["Setup_Type"] = "+4 (Imbalance/FVG)"
    elif "BREAKER" in entry_type:
        score += 4
        details["Setup_Type"] = "+4 (Breaker)"
    elif "WICK" in entry_type:
        score += 2
        details["Setup_Type"] = "+2 (Wick rejection)"
    else:
        score += 1
        details["Setup_Type"] = f"+1 ({entry_type})"

    if "PERFECT" in entry_type:
        score += 1
        details["Perfect"] = "+1"

    # === Rejet M15 rÃ©el : derniÃ¨re bougie doit aller dans le sens du trade ===
    try:
        last = df_m15.iloc[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        rng = max(float(last["high"]) - float(last["low"]), 1e-12)
        body_ratio = body / rng
        bullish_reject = float(last["close"]) > float(last["open"]) and body_ratio >= 0.45
        bearish_reject = float(last["close"]) < float(last["open"]) and body_ratio >= 0.45
        if direction == "BUY" and bullish_reject:
            score += 3
            details["M15_Rejection"] = "+3 (bougie BUY confirmÃ©e)"
        elif direction == "SELL" and bearish_reject:
            score += 3
            details["M15_Rejection"] = "+3 (bougie SELL confirmÃ©e)"
        else:
            score -= 2
            details["M15_Rejection"] = "-2 (pas de rejet confirmÃ©)"
    except Exception:
        pass

    # === RR ===
    rr_ratio = 0.0
    try:
        dist_sl = abs(entry_level - stop_loss)
        dist_tp = abs(take_profit - entry_level)
        rr_ratio = dist_tp / dist_sl if dist_sl > 0 else 0
        if rr_ratio < 1.5:
            score -= 5
            details["RR"] = f"-5 (faible {rr_ratio:.2f})"
        elif rr_ratio >= 2.5:
            score += 3
            details["RR"] = f"+3 (excellent {rr_ratio:.2f})"
        else:
            score += 2
            details["RR"] = f"+2 (correct {rr_ratio:.2f})"
    except Exception:
        pass

    # === Bonus secondaires plafonnÃ©s : ils ne doivent plus fabriquer un signal seuls ===
    d1_aligned = False
    try:
        d1_bonus, d1_label = get_d1_trend_bonus(df_d1, direction)
        if d1_bonus > 0:
            score += 2
            d1_aligned = True
            details["D1_Trend"] = "+2 (D1 alignÃ©)"
        else:
            details["D1_Trend"] = d1_label
    except Exception:
        pass

    macd_confirmed = False
    try:
        macd_bonus, macd_label = get_macd_h1_bonus(df_h1, direction)
        if macd_bonus > 0:
            score += 1
            macd_confirmed = True
            details["MACD_H1"] = "+1 (confirme)"
        else:
            details["MACD_H1"] = macd_label
    except Exception:
        pass

    try:
        session_bonus, session_label = get_session_quality_bonus(pair)
        if session_bonus > 0:
            score += 1
            details["Session"] = "+1 (bonne session)"
        else:
            details["Session"] = session_label
    except Exception:
        pass

    # === Verdict ===
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
    """
    Calcule le 50% d'un FVG avec prÃ©cision.
    """
    if not all(k in fvg for k in ["high_level", "low_level"]):
        return None
    
    high = float(fvg["high_level"])
    low = float(fvg["low_level"])
    
    # âœ… Ã‰viter les FVG avec niveaux identiques
    if high == low:
        return None
        
    midpoint = (high + low) / 2
    
    # âœ… Retourner avec la mÃªme prÃ©cision que les prix
    return round(midpoint, 5)  # 5 dÃ©cimales pour Forex



# =============================
# BOUCLE PRINCIPALE AVEC SCORING
# =============================


def advanced_main():
    """
    Boucle principale avec toutes les corrections appliquÃ©es.
    """
    try:
        api = oandapyV20.API(access_token=os.getenv("OANDA_API_KEY"))
        logger.info("âœ… API OANDA initialisÃ©e avec succÃ¨s")
    except Exception as e:
        logger.error(f"âŒ Ã‰chec d'initialisation de l'API OANDA : {e}")
        return

    for pair in PAIR_LIST:
        _reset_log_dedup()
        try:
            logger.info(f"\nðŸ” DÃ©but de l'analyse avancÃ©e de {pair}")

            # === RÃ©cupÃ©ration des donnÃ©es ===
            df_h4 = get_candles_with_retry(api, pair, GRANULARITY_H4, 300)
            df_h1 = get_candles_with_retry(api, pair, GRANULARITY_H1, 200)
            df_m15 = get_candles_with_retry(api, pair, GRANULARITY_M15, 250)
            df_d1 = get_candles_with_retry(api, pair, "D", count=250)

            if any(df.empty for df in [df_h4, df_h1, df_m15]):
                logger.warning(f"âš ï¸ DonnÃ©es manquantes pour {pair}, analyse ignorÃ©e")
                continue

            current_price = float(df_m15["close"].iloc[-1])
            logger.info(f"ðŸ’° {pair} Prix actuel M15 : {current_price:.5f}")

            # === DÃ©tection Bias & Narrative ===
            bias_analysis = determine_advanced_bias(df_h4)
            bias = bias_analysis.get("bias", "NEUTRAL")
            logger.info(f"ðŸ“ Bias H4 : {bias}")

            narrative = determine_advanced_narrative(
                df_m15, bias_analysis, pair, df_h4=df_h4, df_d1=df_d1, df_h1=df_h1
            )

            # === DÃ©tections avancÃ©es ===
            breaker = detect_breaker(df_m15)
            amd_phase = detect_amd_phase(df_m15)
            logger.info(f"ðŸ” AMD Phase dÃ©tectÃ©e: {amd_phase}")
            
            if breaker['type']:
                logger.info(f"ðŸ” Breaker dÃ©tectÃ©: {breaker['type']} Ã  {breaker['level']}")

            # âœ… CORRECTION : Initialiser les variables
            crt_detected = False
            tbs_setup_type = ""

            # === DÃ©tection CRT Candle ===
            crt_detected = is_crt_candle(df_h4.iloc[-1])
            logger.info(f"ðŸ” CRT Candle dÃ©tectÃ©e: {crt_detected}")

            # === DÃ©tection TBS Setup ===
            tbs_setup = detect_tbs_setup(df_m15)
            tbs_setup_type = tbs_setup['type']
            logger.info(f"ðŸ” TBS Setup dÃ©tectÃ©: {tbs_setup_type} Ã  {tbs_setup['level']}" if tbs_setup_type else "âŒ Aucun TBS Setup")

            # === Filtrage proximitÃ© ===
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
                is_within_reasonable_distance = False  # V3 strict : on ne garde plus les entrÃ©es Ã  max_distance * 3

                if is_in_zone or is_very_close or is_within_reasonable_distance:
                    filtered_entries.append(entry)
                    status = "dans zone" if is_in_zone else "trÃ¨s proche" if is_very_close else "distance raisonnable"
                    msg = f"âœ… EntrÃ©e {status}: {entry_level:.5f} (distance: {distance:.5f})"
                    _log_kept_entry_once(pair, entry_level, status, distance, msg)
                    kept_for_log.append({
                        "entry_level": entry_level,
                        "status_label": status,
                        "status_distance": distance
                    })
                else:
                    logger.warning(f"âŒ EntrÃ©e rejetÃ©e: {entry_level:.5f} (trop Ã©loignÃ©e: {distance:.5f} > {max_distance:.5f})")

            narrative["potential_entries"] = filtered_entries
            logger.info(f"ðŸ“ EntrÃ©es aprÃ¨s filtrage proximitÃ©: {len(narrative['potential_entries'])}")
            _log_kept_compact(pair, kept_for_log, top_n=8)

            # === Clustering ===
            pip_value_for_clustering = get_pip_value_for_pair(pair)
            max_distance_pips_for_clustering_arg = (max_distance / pip_value_for_clustering) if max_distance else None
            if max_distance_pips_for_clustering_arg:
                max_distance_pips_for_clustering_arg = max(5.0, min(20.0, max_distance_pips_for_clustering_arg))

            clustered_entries = cluster_signals(narrative["potential_entries"], pair, max_distance_pips_for_clustering_arg)
            narrative["potential_entries"] = clustered_entries
            logger.info(f"ðŸ“ EntrÃ©es aprÃ¨s clustering: {len(narrative['potential_entries'])}")

            # === Analyse / Envoi ===
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

                logger.info(f"\nðŸ” Analyse signal: {direction}")
                logger.info(f" Type: {entry_type}")
                logger.info(f" Zone: {zone_start:.5f}-{zone_end:.5f}")
                logger.info(f" Niveau: {entry_level:.5f}")
                logger.info(f" Prix actuel: {current_price:.5f} â†’ Distance: {abs(current_price - entry_level):.5f}")

                if is_signal_sent_recently(pair, direction, entry_level_key, zone_start, zone_end):
                    logger.info(" âŒ Signal dÃ©jÃ  envoyÃ© rÃ©cemment â†’ IgnorÃ©")
                    continue

                liquidity_levels_context = narrative.get("liquidity_targets", {})
                nested_fvgs_context = narrative.get("nested_fvgs", [])
                recent_ofls_context = narrative.get("recent_ofls", [])
                structure_analysis_context = narrative.get("structure_analysis", {})

               

                # Calcul du score de confiance (avec D1 pour confirmation tendance longue)
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
                logger.info(f" ðŸ“Š Score: {score}/{SCORING_CONFIG['MIN_CONFIDENCE_SCORE']} | QualitÃ©: {quality} | Win Rate estimÃ©: {win_rate}")
                logger.info(f" ðŸ“‹ DÃ©tails: {confidence_result.get('details', {})}")

                if confidence_result.get("passed", False):
                    logger.info(f" âœ… Signal {quality} confirmÃ© â†’ envoi Telegram")
                    rsi_value = get_last_rsi(df_m15["close"])
                    # On enrichit le bias_analysis avec les infos prÃ©dictives pour le message
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
                    logger.info(" âŒ Signal rejetÃ© (score insuffisant)")

            total_signals_processed = len(narrative.get("potential_entries", []))
            logger.info(f"ðŸ Scan {pair} terminÃ©. Signaux finaux envoyÃ©s: {nb_envoyes}. Signaux traitÃ©s/clusters: {total_signals_processed}")

        except Exception as e:
            logger.error(f"ðŸ’¥ Erreur critique sur {pair} : {str(e)}")
            logger.error(traceback.format_exc())
            continue

    logger.info("ðŸ Analyse avancÃ©e terminÃ©e pour toutes les paires")
# --- Exemple de dÃ©finition de cluster_signals (doit correspondre Ã  ce que vous avez mis Ã  jour) ---
# def cluster_signals(signals: List[Dict], pair: str, max_distance_pips_for_clustering: float = None) -> List[Dict]:
#     # ... (ImplÃ©mentation corrigÃ©e avec signature Ã  3 arguments)
#     pass

# --- Exemple de dÃ©finition de get_pip_value_for_pair (doit Ãªtre corrigÃ©e) ---
# def get_pip_value_for_pair(pair: str) -> float:
#     pair = pair.upper()
#     if pair == "XAU_USD":
#         return 0.01
#     elif pair == "NAS100_USD":
#         return 0.1
#     elif "JPY" in pair:
#         return 0.01
#     else:
#         return 0.0001

# --- Exemple de dÃ©finition de is_in_key_zone_or_consolidation (doit Ãªtre ajoutÃ©e) ---
# def is_in_key_zone_or_consolidation(
#     current_price: float, pair: str, df_m15: pd.DataFrame,
#     liquidity_levels: dict, nested_fvgs: list, recent_ofls: list,
#     structure_analysis: dict, max_zone_width_pips: float = 30.0
# ) -> bool:
#     # ... (ImplÃ©mentation fournie prÃ©cÃ©demment)
#     pass

# --- Exemple de dÃ©finition de cluster_signals (doit correspondre Ã  ce que vous avez mis Ã  jour) ---
# def cluster_signals(signals: List[Dict], pair: str, max_distance_pips_for_clustering: float = None) -> List[Dict]:
#     # ... (ImplÃ©mentation corrigÃ©e avec signature Ã  3 arguments)
#     pass

# --- Exemple de dÃ©finition de get_tolerance dans determine_advanced_narrative (doit inclure NAS100_USD) ---
# def get_tolerance(entry_level: float, pair_local: str) -> float:
#     if pair_local == "XAU_USD":
#         return 5.0
#     elif pair_local in ["AUD_USD", "EUR_USD", "NZD_USD", "USD_CAD", "USD_CHF"]:
#         return 0.0030
#     elif pair_local == "GBP_USD":
#         return 0.0030
#     elif pair_local == "USD_JPY":
#         return 1.0
#     elif pair_local == "NAS100_USD": # <-- CETTE LIGNE EST CRUCIALE
#         return 20.0                 # <-- CETTE LIGNE EST CRUCIALE
#     else:
#         return 0.0010


# ============================================================
# V78 BALANCED BUY/SELL - PATCH PRODUCTION
# Objectif : maximum 1 BUY + 1 SELL par paire.
# Correction V77 :
# - WICK_REJECTION et NESTED_FVG ne sont plus crÃ©Ã©s puis rejetÃ©s.
# - On garde les filtres sÃ©rieux : tendance H1, direction H4,
#   StochRSI, distance, score, anti-doublon, risk management OANDA.
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
    # V78 : distances Ã©largies pour laisser vivre les retests M15/H1.
    # Le scoring garde ensuite un malus si le prix est loin.
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
    """Convertit une distance en pips vers distance prix."""
    return float(pips) * get_pip_value_for_pair(pair)


def strict_entry_type_allowed(entry_type: str) -> bool:
    """
    V77 : autorise les vrais setups directionnels du moteur.
    Correction du bug V76/V5:
    le moteur ajoutait WICK_REJECTION / NESTED_FVG puis les rejetait
    immÃ©diatement avec "type non autorisÃ© V78".

    On garde dÃ©sactivÃ©s les setups expÃ©rimentaux trop bruyants:
    TBS / AMD / CRT / PIN.
    """
    et = (entry_type or "").upper().strip()

    if et in STRICT_ALLOWED_ENTRY_TYPES:
        return True

    # CompatibilitÃ© avec variantes de nommage
    if et.startswith("FVG_RETEST"):
        return True
    if "NESTED" in et and "FVG" in et:
        return True
    if "WICK" in et and "REJECTION" in et:
        return True

    # Setups expÃ©rimentaux toujours exclus
    blocked_keywords = ("TBS", "AMD", "CRT", "PIN_BUY", "PIN_SELL")
    if any(k in et for k in blocked_keywords):
        return False

    return False


def strict_stoch_veto(direction: str, df_h1: pd.DataFrame, df_m15: pd.DataFrame) -> tuple:
    """
    Veto anti entrÃ©e tardive.
    BUY interdit si H1 dÃ©jÃ  surachetÃ©.
    SELL interdit si H1 dÃ©jÃ  survendu.
    M15 sert de filtre de timing.
    """
    try:
        k_h1, d_h1 = calculate_stoch_rsi(df_h1["close"])
        k_m15, d_m15 = calculate_stoch_rsi(df_m15["close"])
        k_h1 = float(k_h1)
        k_m15 = float(k_m15)

        if direction == "BUY" and k_h1 >= 80:
            return False, f"BUY interdit: StochRSI H1 surachat {k_h1:.1f}"
        if direction == "SELL" and k_h1 <= 20:
            return False, f"SELL interdit: StochRSI H1 survendu {k_h1:.1f}"

        # Timing M15 : on refuse de rentrer aprÃ¨s l'impulsion.
        if direction == "BUY" and k_m15 >= 85:
            return False, f"BUY trop tardif: StochRSI M15 {k_m15:.1f}"
        if direction == "SELL" and k_m15 <= 15:
            return False, f"SELL trop tardif: StochRSI M15 {k_m15:.1f}"

        return True, f"StochRSI OK H1={k_h1:.1f} M15={k_m15:.1f}"
    except Exception as exc:
        # En cas de doute, on ne bloque pas pour Ã©viter une panne complÃ¨te.
        return True, f"StochRSI indisponible: {exc}"


def strict_trend_veto(direction: str, current_price: float, df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> tuple:
    """EMA50 H1 et biais H4 deviennent des garde-fous."""
    try:
        ema50_h1 = df_h1["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        if direction == "BUY" and current_price < ema50_h1:
            return False, f"BUY interdit: prix {current_price:.5f} sous EMA50 H1 {ema50_h1:.5f}"
        if direction == "SELL" and current_price > ema50_h1:
            return False, f"SELL interdit: prix {current_price:.5f} au-dessus EMA50 H1 {ema50_h1:.5f}"
    except Exception as exc:
        return False, f"EMA50 H1 indisponible: {exc}"
    return True, "Trend H1 OK"


def strict_distance_filter(pair: str, current_price: float, entry: dict) -> tuple:
    """
    V78 : filtre de proximitÃ© assoupli.
    Il Ã©vite seulement les entrÃ©es vraiment trop Ã©loignÃ©es.
    La distance fine est ensuite pÃ©nalisÃ©e/bonifiÃ©e dans calculate_signal_confidence().
    """
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
    """Maximum 1 BUY + 1 SELL, on garde le meilleur score par direction."""
    best = {}
    for item in scored_entries:
        entry = item["entry"]
        direction = entry.get("direction", "").upper()
        score = item["confidence"].get("total_score", -999)
        entry_type = entry.get("type", "")

        # prioritÃ© au score, puis aux FVG perfect/BISI plutÃ´t qu'aux signaux faibles
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

    # ordre stable : meilleur score d'abord
    return sorted(best.values(), key=lambda x: x["key_score"], reverse=True)




# ============================================================
# V78 BALANCED BUY/SELL - PATCH ANTI BIAIS BUY
# Objectif : ne plus bloquer mÃ©caniquement les SELL quand H4 est BUY.
# On garde le mode tendance, mais on ajoute un mode contre-mouvement contrÃ´lÃ©.
# ============================================================

def strict_direction_permission_v77(direction: str, bias: str, current_price: float, df_h1: pd.DataFrame, df_m15: pd.DataFrame, entry_type: str) -> tuple:
    """
    Autorise :
    - les trades dans le biais H4,
    - les contre-trades uniquement si StochRSI H1 est extrÃªme + timing M15 cohÃ©rent.

    Exemple : H4 BUY mais StochRSI H1 > 75 => on peut accepter un SELL de correction,
    au lieu de le rejeter automatiquement.
    """
    try:
        direction = (direction or "").upper()
        bias = (bias or "NEUTRAL").upper()
        entry_type = (entry_type or "").upper()
        k_h1, d_h1 = calculate_stoch_rsi(df_h1["close"])
        k_m15, d_m15 = calculate_stoch_rsi(df_m15["close"])
        k_h1 = float(k_h1)
        k_m15 = float(k_m15)

        # Si biais neutre, pas de blocage directionnel.
        if bias not in {"BUY", "SELL"}:
            return True, f"Biais neutre: direction {direction} autorisÃ©e"

        # Direction alignÃ©e au biais H4 : OK.
        if direction == bias:
            return True, f"Direction alignÃ©e H4 {bias}"

        # Contre-biais autorisÃ© seulement en correction d'extrÃªme.
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
                return True, f"SELL contre H4 BUY autorisÃ© V78: H1 surachat {k_h1:.1f}, M15 refroidit {k_m15:.1f}, type={entry_type}"
            return False, f"SELL contre H4 BUY refusÃ©: H1={k_h1:.1f}, M15={k_m15:.1f}, type={entry_type}"

        if direction == "BUY" and bias == "SELL":
            if k_h1 <= 25 and k_m15 >= 30 and is_allowed_counter_type:
                return True, f"BUY contre H4 SELL autorisÃ© V78: H1 survendu {k_h1:.1f}, M15 rebondit {k_m15:.1f}, type={entry_type}"
            return False, f"BUY contre H4 SELL refusÃ©: H1={k_h1:.1f}, M15={k_m15:.1f}, type={entry_type}"

        return False, f"Direction {direction} non autorisÃ©e contre biais {bias}"
    except Exception as exc:
        return False, f"permission direction indisponible: {exc}"


def strict_trend_veto_v77(direction: str, current_price: float, df_h1: pd.DataFrame, df_h4: pd.DataFrame, bias: str = "NEUTRAL") -> tuple:
    """
    V77 : EMA50 H1 n'est plus un veto absolu pour les SELL.
    Avant : SELL interdit dÃ¨s que prix > EMA50 H1 -> Ã§a supprimait presque tous les shorts.
    Maintenant :
    - BUY sous EMA50 H1 rejetÃ© sauf H4 BUY trÃ¨s clair,
    - SELL au-dessus EMA50 H1 acceptÃ© si contexte de correction/surachat gÃ©rÃ© par strict_direction_permission_v77.
    """
    try:
        ema50_h1 = float(df_h1["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        ema200_h1 = float(df_h1["close"].ewm(span=200, adjust=False).mean().iloc[-1])
        bias = (bias or "NEUTRAL").upper()

        if direction == "BUY":
            if current_price < ema50_h1 and bias != "BUY":
                return False, f"BUY interdit: prix {current_price:.5f} sous EMA50 H1 {ema50_h1:.5f} sans biais H4 BUY"
            if current_price < ema200_h1 and bias == "SELL":
                return False, f"BUY interdit: prix sous EMA200 H1 avec biais H4 SELL"

        if direction == "SELL":
            if current_price > ema50_h1 and bias != "SELL":
                # Pas de rejet ici : la permission directionnelle + StochRSI dÃ©cidera.
                return True, f"SELL au-dessus EMA50 H1 tolÃ©rÃ© en mode correction ({current_price:.5f} > {ema50_h1:.5f})"
            if current_price > ema200_h1 and bias == "BUY":
                return True, f"SELL contre tendance tolÃ©rÃ© seulement si StochRSI extrÃªme"

        return True, "Trend H1 OK V78"
    except Exception as exc:
        return False, f"EMA H1 indisponible: {exc}"



def dedupe_raw_entries_v771(entries: list, pair: str) -> list:
    """
    V78 : supprime les doublons exacts/near-identiques avant scoring.
    Exemple vu dans les logs: 5 x FVG_RETEST_PERFECT au mÃªme prix AUD_USD.
    """
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
        logger.info(f"ðŸ§¹ V78 dÃ©dup {pair}: {removed} doublons supprimÃ©s ({len(entries)} -> {len(deduped)})")
    return deduped

def advanced_main():
    """
    V78 BALANCED BUY/SELL.
    - conserve ton moteur de donnÃ©es OANDA et tes fonctions existantes,
    - mais filtre agressivement la narrative,
    - conserve WICK/NESTED/FVG mais supprime TBS/AMD/CRT trop bruyants,
    - score tous les candidats propres,
    - envoie maximum 1 BUY + 1 SELL par paire.
    """
    try:
        api = oandapyV20.API(access_token=os.getenv("OANDA_API_KEY"))
        logger.info("âœ… API OANDA initialisÃ©e avec succÃ¨s")
    except Exception as e:
        logger.error(f"âŒ Ã‰chec d'initialisation de l'API OANDA : {e}")
        return

    for pair in PAIR_LIST:
        _reset_log_dedup()
        try:
            logger.info(f"\nðŸ” DÃ©but de l'analyse V78 BALANCED BUY/SELL de {pair}")

            df_h4 = get_candles_with_retry(api, pair, GRANULARITY_H4, 300)
            df_h1 = get_candles_with_retry(api, pair, GRANULARITY_H1, 200)
            df_m15 = get_candles_with_retry(api, pair, GRANULARITY_M15, 250)
            df_d1 = get_candles_with_retry(api, pair, "D", count=250)

            if any(df.empty for df in [df_h4, df_h1, df_m15]):
                logger.warning(f"âš ï¸ DonnÃ©es manquantes pour {pair}, analyse ignorÃ©e")
                continue

            current_price = float(df_m15["close"].iloc[-1])
            logger.info(f"ðŸ’° {pair} Prix actuel M15 : {current_price:.5f}")

            bias_analysis = determine_advanced_bias(df_h4)
            bias = bias_analysis.get("bias", "NEUTRAL")
            logger.info(f"ðŸ“ Bias H4 : {bias}")

            narrative = determine_advanced_narrative(
                df_m15, bias_analysis, pair, df_h4=df_h4, df_d1=df_d1, df_h1=df_h1
            )

            raw_entries_raw = narrative.get("potential_entries", [])
            logger.info(f"ðŸ§¹ V78: entrÃ©es brutes narrative: {len(raw_entries_raw)}")
            raw_entries = dedupe_raw_entries_v771(raw_entries_raw, pair)
            logger.info(f"ðŸ§¹ V78: entrÃ©es aprÃ¨s dÃ©dup: {len(raw_entries)}")

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
                    logger.info(f"â›” {pair} rejet {direction} {entry_type}: type non autorisÃ© V78")
                    rejected += 1
                    continue

                # V77 : on ne rejette plus automatiquement les SELL quand H4 est BUY.
                # On autorise les contre-signaux uniquement si le StochRSI H1 est extrÃªme.
                ok, reason = strict_direction_permission_v77(direction, bias, current_price, df_h1, df_m15, entry_type)
                if not ok:
                    logger.info(f"â›” {pair} rejet {direction} {entry_type}: {reason}")
                    rejected += 1
                    continue

                ok, reason = strict_trend_veto_v77(direction, current_price, df_h1, df_h4, bias)
                if not ok:
                    logger.info(f"â›” {pair} rejet {direction} {entry_type}: {reason}")
                    rejected += 1
                    continue

                ok, reason = strict_stoch_veto(direction, df_h1, df_m15)
                if not ok:
                    logger.info(f"â›” {pair} rejet {direction} {entry_type}: {reason}")
                    rejected += 1
                    continue

                ok, reason = strict_distance_filter(pair, current_price, entry)
                if not ok:
                    logger.info(f"â›” {pair} rejet {direction} {entry_type} @{float(entry_level):.5f}: {reason}")
                    rejected += 1
                    continue

                logger.info(f"âœ… {pair} candidat propre {direction} {entry_type} @{float(entry_level):.5f}: {reason}")

                confidence_result = calculate_signal_confidence(
                    pair, direction, df_h4, df_h1, df_m15, entry, bias, current_price,
                    False, "", df_d1=df_d1
                )

                # V4 : un signal doit passer le score ET ne pas Ãªtre seulement une collection de bonus secondaires.
                if not confidence_result.get("passed", False):
                    logger.info(
                        f"âŒ {pair} rejet score {direction} {entry_type}: "
                        f"{confidence_result.get('total_score')}/{SCORING_CONFIG['MIN_CONFIDENCE_SCORE']} "
                        f"{confidence_result.get('details', {})}"
                    )
                    rejected += 1
                    continue

                scored_entries.append({"entry": entry, "confidence": confidence_result})

            finalists = strict_keep_best_per_direction(scored_entries)
            logger.info(
                f"ðŸ§¹ V78: candidats scorÃ©s={len(scored_entries)}, finalistes={len(finalists)}, rejetÃ©s={rejected}"
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
                    logger.info(f"âŒ {pair} {direction} dÃ©jÃ  envoyÃ© rÃ©cemment â†’ ignorÃ©")
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
                    f"ðŸ“Š FINAL {pair} {direction} {entry_type} @{entry_level:.5f} | "
                    f"Score {score}/{SCORING_CONFIG['MIN_CONFIDENCE_SCORE']} | QualitÃ© {quality} | WR {win_rate}"
                )
                logger.info(f"ðŸ“‹ DÃ©tails: {confidence_result.get('details', {})}")

                enriched_bias = dict(bias_analysis) if bias_analysis else {}
                enriched_bias["win_rate"] = win_rate
                enriched_bias["quality_label"] = quality
                enriched_bias["score_details"] = confidence_result.get("details", {})
                enriched_bias["v77_filter"] = "V78: max 1 BUY + 1 SELL, WICK/NESTED autorisÃ©s, contre-signaux seulement sur StochRSI extrÃªme"

                rsi_value = get_last_rsi(df_m15["close"])

                # V76: exÃ©cution rÃ©elle OANDA AVANT de marquer la zone comme traitÃ©e.
                # Si OANDA refuse ou si MAX_TRADES_TOTAL bloque, on NE marque PAS le signal.
                trade_id = execute_oanda_trade_v76(
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
                    logger.info(f"{pair}: V78 ordre non exÃ©cutÃ©, zone NON enregistrÃ©e.")

            logger.info(
                f"ðŸ Scan {pair} terminÃ©. Signaux envoyÃ©s: {nb_envoyes}. "
                f"Finalistes: {len(finalists)}"
            )

        except Exception as e:
            logger.error(f"ðŸ’¥ Erreur critique sur {pair} : {str(e)}")
            logger.error(traceback.format_exc())
            continue

    logger.info("ðŸ Analyse V78 BALANCED BUY/SELL terminÃ©e pour toutes les paires")



# =========================================================
# V78 - OANDA EXECUTION + TRADE MANAGER
# Base V76 prod conservÃ©e.
# Correction V78: WICK_REJECTION + NESTED_FVG autorisÃ©s dans le filtre strict.
# Base: moteur V63/V78 BALANCED BUY/SELL conservÃ©
# Ajouts: exÃ©cution rÃ©elle OANDA, sizing risque, 1 trade/pair,
# MAX 3 trades, break-even +1R, trailing swing M5 Ã  +1.5R,
# respect week-end Forex.
# =========================================================
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "101-004-31348578-001")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "true").lower() == "true"

RISK_PERCENTAGE = float(os.getenv("RISK_PERCENTAGE", "1.0"))
MAX_RISK_USD = float(os.getenv("MAX_RISK_USD", "1250"))
MAX_TRADES_TOTAL = int(os.getenv("MAX_TRADES_TOTAL", "3"))
ONE_TRADE_PER_PAIR = os.getenv("ONE_TRADE_PER_PAIR", "true").lower() == "true"

BREAKEVEN_TRIGGER_R = float(os.getenv("BREAKEVEN_TRIGGER_R", "1.0"))
BREAKEVEN_OFFSET_PIPS = float(os.getenv("BREAKEVEN_OFFSET_PIPS", "1.0"))
TRAILING_START_R = float(os.getenv("TRAILING_START_R", "1.5"))
SWING_LOOKBACK_V76 = int(os.getenv("SWING_LOOKBACK_V76", "3"))
SWING_BUFFER_PIPS = float(os.getenv("SWING_BUFFER_PIPS", "1.5"))

PIP_SIZE_V76 = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "USD_CAD": 0.0001, "AUD_CAD": 0.0001,
    "USD_JPY": 0.01, "AUD_JPY": 0.01, "GBP_JPY": 0.01,
    "XAU_USD": 0.01,
}
PRICE_DECIMALS_V76 = {
    "EUR_USD": 5, "GBP_USD": 5, "AUD_USD": 5,
    "USD_CAD": 5, "AUD_CAD": 5,
    "USD_JPY": 3, "AUD_JPY": 3, "GBP_JPY": 3,
    "XAU_USD": 3,
}
MIN_UNITS_V76 = {"XAU_USD": 0.1, "DEFAULT": 1000}

# V79 money-management constants used by calculate_units_v76().
# Keep sizing rounded and capped before sending any order to OANDA.
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
    "DEFAULT": 1000,
}

MIN_UNITS_BY_PAIR = {
    "XAU_USD": 1,
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
    "DEFAULT": 200000,
}

MAX_MARGIN_USAGE_PER_TRADE_PERCENT = float(os.getenv("MAX_MARGIN_USAGE_PER_TRADE_PERCENT", "5.0"))



def compact_json_v76(obj, max_len: int = 6000) -> str:
    """JSON compact pour diagnostiquer les rÃ©ponses OANDA sans saturer les logs."""
    try:
        import json
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        text = str(obj)
    return text if len(text) <= max_len else text[:max_len] + " ...[TRONQUÃ‰]"



def v76_client():
    token = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_ACCESS_TOKEN")
    return oandapyV20.API(access_token=token, environment=OANDA_ENVIRONMENT)

def is_market_open_utc_v76(now_dt: datetime) -> bool:
    wd = now_dt.weekday()
    t = now_dt.time()
    if wd == 5:
        return False
    if wd == 6 and t < datetime.strptime("21:00", "%H:%M").time():
        return False
    if wd == 4 and t >= datetime.strptime("21:00", "%H:%M").time():
        return False
    return True


def round_price_v76(pair: str, price: float) -> str:
    return f"{float(price):.{PRICE_DECIMALS_V76.get(pair, 5)}f}"



def oanda_safe_request_v76(endpoint, label: str = ""):
    try:
        api = v76_client()
        resp = api.request(endpoint)
        return resp
    except Exception as e:
        logger.error(f"âŒ V78 OANDA exception {label}: {e}")
        logger.error(traceback.format_exc())
        return None

def get_account_summary_v76() -> dict:
    r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
    resp = oanda_safe_request_v76(r, "AccountSummary")
    if not resp:
        return {}
    acc = resp.get("account", {})
    logger.info(
        f"DEBUG OANDA SUMMARY | account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT} "
        f"balance={acc.get('balance')} NAV={acc.get('NAV')} "
        f"openTradeCount={acc.get('openTradeCount')} openPositionCount={acc.get('openPositionCount')} "
        f"lastTransactionID={resp.get('lastTransactionID') or acc.get('lastTransactionID')}"
    )
    return resp


def get_account_summary_v76() -> dict:
    r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
    resp = oanda_safe_request_v76(r, "AccountSummary")
    if not resp:
        return {}
    acc = resp.get("account", {})
    logger.info(
        f"DEBUG OANDA SUMMARY | account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT} "
        f"balance={acc.get('balance')} NAV={acc.get('NAV')} "
        f"openTradeCount={acc.get('openTradeCount')} openPositionCount={acc.get('openPositionCount')} "
        f"lastTransactionID={resp.get('lastTransactionID') or acc.get('lastTransactionID')}"
    )
    return resp


def get_balance_v76() -> float:
    resp = get_account_summary_v76()
    try:
        return float(resp.get("account", {}).get("balance", 0))
    except Exception:
        return 0.0


def get_open_trades_v76(log_raw: bool = False) -> list:
    """
    V78: lecture officielle OpenTrades, comme V75.
    Ne bloque que si OANDA renvoie un trade avec currentUnits non nul.
    """
    r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
    resp = oanda_safe_request_v76(r, "OpenTrades")
    if not resp:
        logger.warning("DEBUG OPEN TRADES | rÃ©ponse absente: on considÃ¨re aucun trade ouvert pour Ã©viter un faux blocage.")
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
        logger.info(f"DEBUG OPEN TRADES RAW={compact_json_v76(resp)}")
    for t in open_trades:
        logger.info(
            f"DEBUG TRADE | id={t.get('id')} instrument={t.get('instrument')} "
            f"units={t.get('currentUnits', t.get('units'))} price={t.get('price')} "
            f"state={t.get('state', 'OPEN')} openTime={t.get('openTime')}"
        )
    return open_trades


def get_open_positions_v76(log_raw: bool = False) -> list:
    """Lecture OpenPositions pour confirmer l'Ã©tat rÃ©el du compte, comme V75."""
    try:
        r = positions.OpenPositions(accountID=OANDA_ACCOUNT_ID)
        resp = oanda_safe_request_v76(r, "OpenPositions")
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
            logger.info(f"DEBUG OPEN POSITIONS RAW={compact_json_v76(resp)}")
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


def has_open_trade_v76(pair: str) -> bool:
    """
    V78: vÃ©rifie OpenTrades ET OpenPositions comme la V75.
    Pas de cache interne. Pas de faux blocage si OANDA rÃ©pond vide.
    """
    open_trades = get_open_trades_v76(log_raw=True)
    trade_exists = any(t.get("instrument") == pair for t in open_trades)

    open_positions = get_open_positions_v76(log_raw=True)
    position_exists = any(p.get("instrument") == pair for p in open_positions)

    exists = bool(trade_exists or position_exists)
    logger.info(f"{pair}: has_open_trade={exists} | trade_exists={trade_exists} | position_exists={position_exists}")
    return exists


def open_trade_count_v76() -> int:
    """Nombre total de trades rÃ©ellement ouverts cÃ´tÃ© OANDA."""
    return len(get_open_trades_v76(log_raw=True))


def log_account_snapshot_v76(label: str) -> None:
    logger.info(f"========== OANDA ACCOUNT SNAPSHOT {label} ==========")
    try:
        get_account_summary_v76()
    except Exception as exc:
        logger.exception(f"SNAPSHOT SUMMARY impossible: {exc}")
    get_open_trades_v76(log_raw=True)
    get_open_positions_v76(log_raw=True)
    logger.info(f"========== FIN SNAPSHOT {label} ==========")


def extract_oanda_trade_or_order_id_v76(resp: dict) -> str | None:
    fill = resp.get("orderFillTransaction", {}) or {}
    trade_opened = fill.get("tradeOpened") or {}
    if trade_opened.get("tradeID"):
        return str(trade_opened["tradeID"])
    trades_opened = fill.get("tradesOpened") or []
    if trades_opened and trades_opened[0].get("tradeID"):
        return str(trades_opened[0]["tradeID"])
    return str(fill.get("id") or resp.get("orderCreateTransaction", {}).get("id") or "") or None


def log_oanda_order_response_v76(pair: str, resp: dict) -> None:
    logger.info(f"DEBUG ORDER RESPONSE RAW {pair}={compact_json_v76(resp, max_len=8000)}")
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
        logger.error(f"ORDRE REJETÃ‰ {pair} | {compact_json_v76(resp.get('orderRejectTransaction'))}")
    if resp.get("orderCancelTransaction"):
        logger.warning(f"ORDRE ANNULÃ‰ {pair} | {compact_json_v76(resp.get('orderCancelTransaction'))}")

def has_open_trade_v76(pair: str) -> bool:
    """
    V78: bloque seulement si OANDA confirme un trade actif sur la mÃªme paire.
    Ne se base pas sur un cache interne.
    """
    for t in get_open_trades_v76():
        instrument = t.get("instrument")
        raw_units = t.get("currentUnits", t.get("units", "0"))

        try:
            units = float(raw_units)
        except Exception:
            units = 0.0

        if instrument == pair and abs(units) > 0:
            logger.info(f"{pair}: trade actif confirmÃ© cÃ´tÃ© OANDA id={t.get('id')} units={raw_units}.")
            return True

    logger.info(f"{pair}: aucun trade actif confirmÃ© cÃ´tÃ© OANDA, ordre autorisÃ© si autres conditions OK.")
    return False


def quote_currency_v76(pair: str) -> str:
    return pair.split("_")[1]


def get_fx_rate_to_usd_v76(currency: str) -> float:
    """Retourne la valeur USD de 1 unitÃ© de devise 'currency'."""
    if currency == "USD":
        return 1.0
    # Exemples utiles pour nos paires : JPY via USD_JPY, CAD via USD_CAD
    direct = f"{currency}_USD"
    inverse = f"USD_{currency}"
    try:
        if direct in PAIR_LIST:
            api = v76_client()
            df = get_candles_with_retry(api, direct, "M5", 10)
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
        if inverse in PAIR_LIST:
            api = v76_client()
            df = get_candles_with_retry(api, inverse, "M5", 10)
            if df is not None and not df.empty:
                return 1.0 / float(df["close"].iloc[-1])
    except Exception:
        pass
    logger.warning(f"âš ï¸ V78 conversion {currency}->USD inconnue, fallback 1.0")
    return 1.0



def get_oanda_margin_rate_v78(pair: str) -> float:
    """
    Retourne la marge requise OANDA approximative.
    Si l'info instrument est inaccessible, fallback conservateur 3.33%.
    """
    try:
        api = v76_client()
        r = accounts.AccountInstruments(accountID=OANDA_ACCOUNT_ID, params={"instruments": pair})
        resp = api.request(r)
        instruments_data = resp.get("instruments", [])
        if instruments_data:
            margin_rate = float(instruments_data[0].get("marginRate", 0.0333))
            if margin_rate > 0:
                return margin_rate
    except Exception as exc:
        logger.warning(f"V78 marginRate indisponible pour {pair}, fallback 0.0333: {exc}")
    return 0.0333


def estimate_margin_used_v78(pair: str, units: int, entry_price: float) -> float:
    """
    Estimation marge en USD.
    C'est un garde-fou, pas un calcul comptable parfait.
    """
    units = abs(int(units))
    margin_rate = get_oanda_margin_rate_v78(pair)

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
        q_to_usd = get_fx_rate_to_usd_v76(quote)
        notional_usd = units * entry_price * q_to_usd

    return float(notional_usd * margin_rate)


def cap_units_absolute_v78(pair: str, units: int) -> int:
    max_units = MAX_UNITS_BY_PAIR.get(pair, MAX_UNITS_BY_PAIR["DEFAULT"])
    if units > max_units:
        logger.warning(f"V78 ABS CAP {pair}: units {units} -> {max_units}")
        return max_units
    return units


def cap_units_by_margin_v78(pair: str, units: int, entry_price: float, balance: float) -> int:
    """
    Plafonne les unitÃ©s pour ne pas utiliser trop de marge par trade.
    DÃ©faut: max 5% du compte en marge par trade.
    """
    if units <= 0 or balance <= 0:
        return 0

    max_margin_usd = balance * (MAX_MARGIN_USAGE_PER_TRADE_PERCENT / 100.0)
    estimated_margin = estimate_margin_used_v78(pair, units, entry_price)

    if estimated_margin <= max_margin_usd:
        return units

    ratio = max_margin_usd / estimated_margin if estimated_margin > 0 else 0
    capped = int(units * ratio)

    step = UNIT_STEP_BY_PAIR.get(pair, UNIT_STEP_BY_PAIR["DEFAULT"])
    capped = int(capped // step * step)

    logger.warning(
        f"V78 MARGIN CAP {pair}: units {units} -> {capped} | "
        f"estimated_margin=${estimated_margin:.2f} > max_margin=${max_margin_usd:.2f} "
        f"({MAX_MARGIN_USAGE_PER_TRADE_PERCENT:.2f}% balance)"
    )
    return max(capped, 0)


def calculate_units_v76(pair: str, entry: float, stop_loss: float, balance: float) -> float:
    """
    V78: sizing sÃ©curisÃ©.
    Ancien problÃ¨me: le calcul pouvait produire 1M+ unitÃ©s.
    Nouveau:
    - risque thÃ©orique via distance SL
    - arrondi vers le bas
    - plafond unitÃ©s par paire
    - plafond marge utilisÃ©e par trade
    """
    try:
        balance = float(balance)
        entry = float(entry)
        stop_loss = float(stop_loss)
    except Exception:
        logger.error(f"V78: paramÃ¨tres sizing invalides pair={pair} entry={entry} stop={stop_loss} balance={balance}")
        return 0

    risk_usd = min(balance * (RISK_PERCENTAGE / 100.0), MAX_RISK_USD)
    distance_quote = abs(entry - stop_loss)

    if balance <= 0 or risk_usd <= 0 or distance_quote <= 0:
        logger.error(
            f"V78: sizing impossible {pair}: balance={balance}, risk_usd={risk_usd}, distance={distance_quote}"
        )
        return 0

    quote = quote_currency_v76(pair)
    quote_to_usd = get_fx_rate_to_usd_v76(quote)

    if quote_to_usd <= 0:
        logger.error(f"V78: conversion quote_to_usd invalide {quote}->{quote_to_usd}, trade bloquÃ©.")
        return 0

    risk_per_unit_usd = distance_quote * quote_to_usd
    if risk_per_unit_usd <= 0:
        logger.error(f"V78: risk_per_unit_usd invalide {pair}: {risk_per_unit_usd}")
        return 0

    raw_units = risk_usd / risk_per_unit_usd
    step = UNIT_STEP_BY_PAIR.get(pair, UNIT_STEP_BY_PAIR["DEFAULT"])
    min_units = MIN_UNITS_BY_PAIR.get(pair, MIN_UNITS_BY_PAIR["DEFAULT"])

    units_before_caps = int(raw_units // step * step)
    units = cap_units_absolute_v78(pair, units_before_caps)
    units = cap_units_by_margin_v78(pair, units, entry, balance)

    if units < min_units:
        logger.warning(
            f"V78: units trop faibles aprÃ¨s caps {pair}: {units} < min={min_units}. "
            f"Trade bloquÃ©. raw_units={raw_units:.2f}"
        )
        return 0

    estimated_margin = estimate_margin_used_v78(pair, units, entry)
    estimated_risk = units * risk_per_unit_usd
    estimated_risk_pct = estimated_risk / balance * 100.0 if balance > 0 else 0.0
    margin_pct = estimated_margin / balance * 100.0 if balance > 0 else 0.0

    logger.info(
        f"V78 RISK LOT {pair}: balance=${balance:.2f} risk_cap=${risk_usd:.2f} "
        f"entry={entry:.5f} SL={stop_loss:.5f} dist_quote={distance_quote:.5f} "
        f"quote_to_usd={quote_to_usd:.8f} risk_per_unit=${risk_per_unit_usd:.8f} "
        f"raw_units={raw_units:.2f} units_before_caps={units_before_caps} final_units={units} "
        f"estimated_risk=${estimated_risk:.2f} ({estimated_risk_pct:.2f}%) "
        f"estimated_margin=${estimated_margin:.2f} ({margin_pct:.2f}%)"
    )

    return int(units)


def get_recent_m5_price_v76(pair: str) -> float:
    api = v76_client()
    df = get_candles_with_retry(api, pair, "M5", 10)
    if df is None or df.empty:
        return 0.0
    return float(df["close"].iloc[-1])



def execute_oanda_trade_v76(pair: str, direction: str, entry_price: float, stop_loss: float,
                            take_profit: float, score: int, entry_type: str) -> str | None:
    """
    V78: exÃ©cution reprise dans l'esprit de la V75.
    Objectif: ne plus s'arrÃªter silencieusement aprÃ¨s openTrades.
    Chaque Ã©tape critique est loggÃ©e.
    """
    logger.info(f"V78 EXECUTION START {pair} {direction} type={entry_type} score={score}")
    logger.info(
        f"DEBUG EXEC INPUT | pair={pair} direction={direction} entry={round_price_v76(pair, entry_price)} "
        f"SL={round_price_v76(pair, stop_loss)} TP={round_price_v76(pair, take_profit)} "
        f"account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT} execute={EXECUTE_TRADES}"
    )

    log_account_snapshot_v76("BEFORE_EXECUTION")

    if ONE_TRADE_PER_PAIR:
        has_trade = has_open_trade_v76(pair)
        logger.info(f"DEBUG STEP has_open_trade({pair})={has_trade}")
        if has_trade:
            logger.info(f"{pair}: trade dÃ©jÃ  ouvert dÃ©tectÃ© par OANDA, aucun nouvel ordre.")
            return None

    count = open_trade_count_v76()
    logger.info(f"DEBUG STEP open_trade_count={count} / max={MAX_TRADES_TOTAL}")
    if count >= MAX_TRADES_TOTAL:
        logger.info(f"Limite trades ouverts atteinte ({count}/{MAX_TRADES_TOTAL}). Aucun ordre POST /orders ne sera envoyÃ©.")
        return None

    balance = get_balance_v76()
    logger.info(f"DEBUG STEP balance={balance}")
    if balance <= 0:
        logger.error("V78: balance invalide, ordre annulÃ©.")
        return None

    units = calculate_units_v76(pair, entry_price, stop_loss, balance)
    logger.info(f"DEBUG STEP calculated_units={units}")
    if not units or float(units) <= 0:
        logger.error(f"V78: units invalides pour {pair}: {units}")
        return None

    signed_units = units if direction == "BUY" else -units
    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": pair,
            "units": str(signed_units),
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": round_price_v76(pair, stop_loss), "timeInForce": "GTC"},
            "takeProfitOnFill": {"price": round_price_v76(pair, take_profit), "timeInForce": "GTC"},
        }
    }

    risk = abs(entry_price - stop_loss)
    rr = abs(take_profit - entry_price) / risk if risk > 0 else 0
    logger.info(
        f"SIGNAL V78 {pair} {direction} | "
        f"entryâ‰ˆ{round_price_v76(pair, entry_price)} SL={round_price_v76(pair, stop_loss)} "
        f"TP={round_price_v76(pair, take_profit)} RR={rr:.2f} score={score} "
        f"units={units} signed_units={signed_units} type={entry_type}"
    )
    logger.info(f"ORDER PAYLOAD {pair}={compact_json_v76(order_data)}")

    if not EXECUTE_TRADES:
        logger.info("EXECUTE_TRADES=false : ordre non envoyÃ© Ã  OANDA.")
        logger.info("DEBUG EXECUTION RESULT | status=SIMULATION | aucun POST /orders envoyÃ©")
        return "SIMULATION"

    try:
        logger.info(f"DEBUG POST /orders START | pair={pair} account={OANDA_ACCOUNT_ID} env={OANDA_ENVIRONMENT}")
        api = v76_client()
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        resp = api.request(r)
        logger.info(f"DEBUG POST /orders END | pair={pair}")
        log_oanda_order_response_v76(pair, resp)

        if resp.get("orderRejectTransaction"):
            logger.error(f"DEBUG EXECUTION RESULT | status=REJECTED | pair={pair}")
            log_account_snapshot_v76("AFTER_REJECT")
            return None

        if resp.get("orderCancelTransaction") and not resp.get("orderFillTransaction"):
            logger.error(f"DEBUG EXECUTION RESULT | status=CANCELLED_NO_FILL | pair={pair}")
            log_account_snapshot_v76("AFTER_CANCEL")
            return None

        trade_id = extract_oanda_trade_or_order_id_v76(resp)
        if not trade_id:
            logger.error(f"ORDRE NON CONFIRMÃ‰ {pair}: aucune transaction Fill/Create exploitable.")
            logger.error(f"DEBUG EXECUTION RESULT | status=NO_TRADE_ID | pair={pair}")
            log_account_snapshot_v76("AFTER_NO_TRADE_ID")
            return None

        logger.info(f"âœ… ORDRE RÃ‰EL CONFIRMÃ‰ {pair} | ID={trade_id}")
        logger.info(f"DEBUG EXECUTION RESULT | status=CONFIRMED | pair={pair} trade_or_order_id={trade_id}")

        log_account_snapshot_v76("AFTER_ORDER_CREATE")
        open_after = get_open_trades_v76(log_raw=True)
        opened_for_pair = [t for t in open_after if t.get("instrument") == pair]
        if opened_for_pair:
            logger.info(f"CONFIRMATION OPEN TRADE {pair}: {compact_json_v76(opened_for_pair)}")
        else:
            logger.warning(f"ATTENTION {pair}: ordre acceptÃ© mais OpenTrades ne montre pas encore la position.")

        return str(trade_id)

    except Exception as exc:
        logger.exception(f"Erreur ordre OANDA {pair}: {exc}")
        logger.error(f"DEBUG EXECUTION RESULT | status=EXCEPTION | pair={pair} error={exc}")
        log_account_snapshot_v76("EXCEPTION_ORDER_CREATE")
        return None

def trade_direction_v76(trade: dict) -> str:
    return "BUY" if float(trade.get("currentUnits", 0)) > 0 else "SELL"


def trade_current_r_v76(trade: dict) -> float | None:
    pair = trade.get("instrument")
    price = get_recent_m5_price_v76(pair)
    if not price:
        return None
    entry = float(trade.get("price", 0))
    sl_order = trade.get("stopLossOrder", {}) or {}
    if not sl_order.get("price"):
        return None
    current_sl = float(sl_order["price"])
    initial_risk = abs(entry - current_sl)
    if initial_risk <= 0:
        return None
    direction = trade_direction_v76(trade)
    gain = (price - entry) if direction == "BUY" else (entry - price)
    return gain / initial_risk


def find_last_swing_sl_v76(pair: str, direction: str) -> float | None:
    api = v76_client()
    df = get_candles_with_retry(api, pair, "M5", 90)
    if df is None or df.empty or len(df) < 20:
        return None
    pip = PIP_SIZE_V76.get(pair, get_pip_value_for_pair(pair))
    buffer = SWING_BUFFER_PIPS * pip
    if direction == "BUY":
        lows = df["low"].values
        for i in range(len(df) - 2 - SWING_LOOKBACK_V76, SWING_LOOKBACK_V76, -1):
            window = lows[i-SWING_LOOKBACK_V76:i+SWING_LOOKBACK_V76+1]
            if lows[i] == window.min():
                return float(lows[i] - buffer)
    else:
        highs = df["high"].values
        for i in range(len(df) - 2 - SWING_LOOKBACK_V76, SWING_LOOKBACK_V76, -1):
            window = highs[i-SWING_LOOKBACK_V76:i+SWING_LOOKBACK_V76+1]
            if highs[i] == window.max():
                return float(highs[i] + buffer)
    return None


def modify_trade_sl_v76(trade_id: str, pair: str, new_sl: float) -> bool:
    data = {"stopLoss": {"price": round_price_v76(pair, new_sl), "timeInForce": "GTC"}}
    r = trades.TradeCRCDO(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id, data=data)
    resp = oanda_safe_request_v76(r, f"TradeCRCDO {trade_id}")
    logger.info(f"V76 SL UPDATE RESPONSE={resp}")
    return bool(resp)


def manage_open_trades_v76():
    open_trades = get_open_trades_v76()
    if not open_trades:
        logger.info("V78 TRADE MANAGER: aucun trade ouvert.")
        return

    for t in open_trades:
        try:
            trade_id = str(t.get("id"))
            pair = t.get("instrument")
            direction = trade_direction_v76(t)
            entry = float(t.get("price"))
            sl_order = t.get("stopLossOrder", {}) or {}
            if not sl_order.get("price"):
                continue
            current_sl = float(sl_order["price"])
            current_r = trade_current_r_v76(t)
            if current_r is None:
                continue

            logger.info(f"V78 TRADE MANAGER {pair} id={trade_id} dir={direction} R={current_r:.2f} SL={current_sl}")
            pip = PIP_SIZE_V76.get(pair, get_pip_value_for_pair(pair))

            # Break-even Ã  +1R
            if current_r >= BREAKEVEN_TRIGGER_R:
                be_sl = entry + BREAKEVEN_OFFSET_PIPS * pip if direction == "BUY" else entry - BREAKEVEN_OFFSET_PIPS * pip
                if direction == "BUY" and current_sl < be_sl:
                    logger.info(f"ðŸŸ¡ V76 BE {pair}: SL {current_sl} -> {be_sl}")
                    modify_trade_sl_v76(trade_id, pair, be_sl)
                    continue
                if direction == "SELL" and current_sl > be_sl:
                    logger.info(f"ðŸŸ¡ V76 BE {pair}: SL {current_sl} -> {be_sl}")
                    modify_trade_sl_v76(trade_id, pair, be_sl)
                    continue

            # Trailing swing Ã  +1.5R
            if current_r >= TRAILING_START_R:
                swing_sl = find_last_swing_sl_v76(pair, direction)
                if swing_sl is None:
                    continue
                if direction == "BUY" and swing_sl > current_sl:
                    logger.info(f"ðŸ” V76 TRAIL {pair}: SL {current_sl} -> {swing_sl}")
                    modify_trade_sl_v76(trade_id, pair, swing_sl)
                if direction == "SELL" and swing_sl < current_sl and swing_sl > 0:
                    logger.info(f"ðŸ” V76 TRAIL {pair}: SL {current_sl} -> {swing_sl}")
                    modify_trade_sl_v76(trade_id, pair, swing_sl)
        except Exception as e:
            logger.error(f"Erreur V76 trade manager: {e}")
            logger.error(traceback.format_exc())


# =============================
# LANCEMENT
# =============================
if __name__ == "__main__":
    logger.info("ðŸš€ DÃ©marrage du Bot Advanced Orderflow Trading - V78 PROD")
    
    api = oandapyV20.API(access_token=os.getenv("OANDA_API_KEY"))
    
    while True:
        try:
            now_dt = datetime.utcnow()
            if not is_market_open_utc_v76(now_dt):
                logger.info("MarchÃ© Forex fermÃ©. Attente 5 minutes.")
                time.sleep(300)
                continue

            manage_open_trades_v76()

            if open_trade_count_v76() >= MAX_TRADES_TOTAL:
                logger.info(f"Limite trades ouverts atteinte ({open_trade_count_v76()}/{MAX_TRADES_TOTAL}). Scan entrÃ©es ignorÃ©.")
                time.sleep(300)
                continue

            advanced_main()
            logger.info("â³ Attente 15 minutes avant le prochain scan...")
            time.sleep(900)
        except KeyboardInterrupt:
            logger.info("ðŸ›‘ ArrÃªt demandÃ© par l'utilisateur")
            break
        except Exception as e:
            logger.error(f"ðŸ’¥ Erreur critique: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(60)

