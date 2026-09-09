# ============================================================
# main.py - Version PROD "2R Strict" (v140)
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

# --- NOUVEAU : filtres qualité (win rate) ---
# Ces filtres utilisent des métriques déjà calculées (ADX, RSI) mais qui
# n'étaient jusqu'ici jamais utilisées pour rejeter un setup.
ENABLE_QUALITY_FILTERS = True   # coupe-circuit global, pour A/B tester facilement
MIN_ADX_TREND = 20.0            # ADX H1 minimum : sous ce seuil, marché sans tendance -> setups de continuation peu fiables
RSI_OVERBOUGHT = 78.0           # RSI M15 : au-dessus, on n'ouvre plus de BUY (mouvement déjà très étiré)
RSI_OVERSOLD = 22.0             # RSI M15 : en-dessous, on n'ouvre plus de SELL
STRICT_BIAS_ALIGNMENT = False   # True = exige HH+HL (ou LH+LL) complet en H4, rejette les biais "_WEAK" partiels

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
    """
    Retourne le taux de conversion de 1 unité de `currency`
    vers USD.

    Exemples :
        USD -> 1.0
        EUR -> EUR_USD
        GBP -> GBP_USD
        AUD -> AUD_USD
        NZD -> NZD_USD
        CAD -> 1 / USD_CAD
        CHF -> 1 / USD_CHF
        JPY -> 1 / USD_JPY

    Utilise directement OANDA.
    """
    try:
        currency = str(currency).upper().strip()

        if not currency:
            logger.error("[FX] Devise vide")
            return 0.0

        if currency == "USD":
            return 1.0

        # --------------------------------------------------------
        # Paires directes : XXX_USD
        # --------------------------------------------------------
        direct_pair = f"{currency}_USD"

        try:
            api = v88_client()

            r = pricing.PricingInfo(
                accountID=OANDA_ACCOUNT_ID,
                params={"instruments": direct_pair}
            )

            api.request(r)

            prices = r.response.get("prices", [])

            if prices:
                item = prices[0]

                bid = float(
                    item.get("bids", [{}])[0].get("price", 0)
                )

                ask = float(
                    item.get("asks", [{}])[0].get("price", 0)
                )

                if bid > 0 and ask > 0:
                    rate = (bid + ask) / 2.0

                    if rate > 0:
                        logger.debug(
                            f"[FX] {currency}->USD via {direct_pair} = {rate:.8f}"
                        )

                        return rate

        except Exception as e:
            logger.debug(
                f"[FX] Paire directe {direct_pair} indisponible: {e}"
            )

        # --------------------------------------------------------
        # Paires inverses : USD_XXX
        # --------------------------------------------------------
        inverse_pair = f"USD_{currency}"

        try:
            api = v88_client()

            r = pricing.PricingInfo(
                accountID=OANDA_ACCOUNT_ID,
                params={"instruments": inverse_pair}
            )

            api.request(r)

            prices = r.response.get("prices", [])

            if prices:
                item = prices[0]

                bid = float(
                    item.get("bids", [{}])[0].get("price", 0)
                )

                ask = float(
                    item.get("asks", [{}])[0].get("price", 0)
                )

                if bid > 0 and ask > 0:
                    rate_usd_currency = (bid + ask) / 2.0

                    if rate_usd_currency > 0:
                        rate = 1.0 / rate_usd_currency

                        logger.debug(
                            f"[FX] {currency}->USD via inverse "
                            f"{inverse_pair} = {rate:.8f}"
                        )

                        return rate

        except Exception as e:
            logger.debug(
                f"[FX] Paire inverse {inverse_pair} indisponible: {e}"
            )

        logger.error(
            f"[FX] Impossible de convertir {currency}->USD"
        )

        return 0.0

    except Exception as e:
        logger.error(
            f"[FX] Erreur conversion {currency}->USD: {e}"
        )
        return 0.0








def calculate_margin(pair: str, units: int, entry_price: float) -> dict:
    """
    Calcule la marge requise en USD.

    OANDA exprime les units dans la devise de base.
    Le notionnel doit donc être correctement converti en USD.
    """
    try:
        pair = str(pair).upper().strip()
        units = int(units)
        entry_price = float(entry_price)

        margin_available = float(get_available_margin())

        if units <= 0 or entry_price <= 0:
            return {
                "margin_required": 0.0,
                "margin_available": margin_available,
                "sufficient": False
            }

        parts = pair.split("_")

        if len(parts) != 2:
            logger.error(
                f"[MARGIN] Paire invalide: {pair}"
            )
            return {
                "margin_required": 0.0,
                "margin_available": margin_available,
                "sufficient": False
            }

        base = parts[0]
        quote = parts[1]

        margin_rate = float(get_oanda_margin_rate(pair))

        if margin_rate <= 0:
            margin_rate = 0.0333

        # ---------------------------------------------------------
        # NOTIONNEL EN USD
        # ---------------------------------------------------------

        # USD/XXX
        # Ex: USD_JPY -> 10000 units = 10000 USD
        if base == "USD":
            notional_usd = float(units)

        # XXX/USD
        # Ex: GBP_USD -> units * prix
        elif quote == "USD":
            notional_usd = float(units) * entry_price

        # XXX/YYY
        # Ex: AUD_JPY
        # units * prix = notionnel en JPY
        # puis JPY -> USD
        else:
            notional_quote = float(units) * entry_price

            quote_to_usd = float(
                get_fx_rate_to_usd(quote)
            )

            if quote_to_usd <= 0:
                logger.error(
                    f"[MARGIN] Conversion {quote}->USD invalide "
                    f"pour {pair}: {quote_to_usd}"
                )
                return {
                    "margin_required": 0.0,
                    "margin_available": margin_available,
                    "sufficient": False
                }

            notional_usd = (
                notional_quote * quote_to_usd
            )

        margin_required = (
            notional_usd * margin_rate
        )

        sufficient = (
            margin_available >= margin_required
        )

        logger.info(
            f"[MARGIN] {pair} | "
            f"units={units} | "
            f"notional_usd={notional_usd:.2f} | "
            f"margin_rate={margin_rate:.4f} | "
            f"margin_required={margin_required:.2f} | "
            f"available={margin_available:.2f} | "
            f"sufficient={sufficient}"
        )

        return {
            "margin_required": float(margin_required),
            "margin_available": float(margin_available),
            "sufficient": bool(sufficient)
        }

    except Exception as e:
        logger.error(
            f"[MARGIN] Erreur {pair}: {e}"
        )

        return {
            "margin_required": 0.0,
            "margin_available": 0.0,
            "sufficient": False
        }

def cap_units_by_margin(
    pair: str,
    units: int,
    entry_price: float,
    balance: float
) -> int:
    """
    Réduit les unités si la marge requise dépasse la marge autorisée.

    La limite retenue est la plus restrictive entre :
      - MAX_MARGIN_USAGE_PER_TRADE_PERCENT du solde
      - la marge réellement disponible OANDA
    """
    try:
        pair = str(pair).upper().strip()
        units = int(units)
        entry_price = float(entry_price)
        balance = float(balance)

        if units <= 0:
            return 0

        if balance <= 0 or entry_price <= 0:
            logger.error(
                f"[MARGIN_CAP] {pair} | "
                f"balance={balance} | entry={entry_price}"
            )
            return 0

        step = int(
            UNIT_STEP_BY_PAIR.get(
                pair,
                UNIT_STEP_BY_PAIR["DEFAULT"]
            )
        )

        if step <= 0:
            step = 1

        # Marge pour la taille demandée
        margin_info = calculate_margin(
            pair=pair,
            units=units,
            entry_price=entry_price
        )

        margin_required = float(
            margin_info.get("margin_required", 0.0)
        )

        margin_available = float(
            margin_info.get("margin_available", 0.0)
        )

        if margin_required <= 0:
            logger.error(
                f"[MARGIN_CAP] {pair} | "
                f"Marge invalide: {margin_required}"
            )
            return 0

        # Limite interne de marge par trade
        max_margin_by_balance = (
            balance
            * (
                float(MAX_MARGIN_USAGE_PER_TRADE_PERCENT)
                / 100.0
            )
        )

        # On prend la limite la plus restrictive
        allowed_margin = min(
            max_margin_by_balance,
            margin_available
        )

        if allowed_margin <= 0:
            logger.warning(
                f"[MARGIN_CAP] {pair} | "
                f"Aucune marge autorisée"
            )
            return 0

        # La taille demandée passe
        if margin_required <= allowed_margin:
            logger.info(
                f"[MARGIN_CAP] {pair} | "
                f"units={units} | "
                f"margin={margin_required:.2f} | "
                f"allowed={allowed_margin:.2f} | "
                f"OK"
            )
            return units

        # ---------------------------------------------------------
        # CALCUL DE LA NOUVELLE TAILLE
        # ---------------------------------------------------------

        ratio = (
            allowed_margin / margin_required
        )

        capped_raw = (
            float(units) * ratio
        )

        capped_units = (
            int(capped_raw // step) * step
        )

        logger.info(
            f"[MARGIN_CAP] {pair} | "
            f"units_initial={units} | "
            f"margin_required={margin_required:.2f} | "
            f"allowed_margin={allowed_margin:.2f} | "
            f"ratio={ratio:.4f} | "
            f"units_cap={capped_units}"
        )

        return max(0, int(capped_units))

    except Exception as e:
        logger.error(
            f"[MARGIN_CAP] Erreur {pair}: {e}"
        )
        return 0

def calculate_units(
    pair: str,
    entry: float,
    stop_loss: float,
    balance: float,
    risk_pct: float = None
) -> int:
    """Calcule la taille de position à partir du risque USD réel."""
    try:
        pair = str(pair).upper().strip()
        entry = float(entry)
        stop_loss = float(stop_loss)
        balance = float(balance)
        risk_pct = RISK_PERCENTAGE if risk_pct is None else float(risk_pct)

        if balance <= 0 or entry <= 0 or stop_loss <= 0:
            return 0

        risk_usd = balance * (risk_pct / 100.0)
        if MAX_RISK_USD is not None:
            risk_usd = min(risk_usd, float(MAX_RISK_USD))
        if risk_usd <= 0:
            return 0

        distance = abs(entry - stop_loss)
        if distance <= 0:
            return 0

        parts = pair.split("_")
        if len(parts) != 2:
            return 0
        quote = parts[1]

        quote_to_usd = float(get_fx_rate_to_usd(quote))
        if quote_to_usd <= 0:
            logger.error(f"[UNITS] {pair} | Conversion {quote}->USD invalide: {quote_to_usd}")
            return 0

        risk_per_unit = distance * quote_to_usd
        if risk_per_unit <= 0:
            return 0

        raw_units = risk_usd / risk_per_unit
        step = int(UNIT_STEP_BY_PAIR.get(pair, UNIT_STEP_BY_PAIR["DEFAULT"]))
        if step <= 0:
            step = 1

        units = int(raw_units // step) * step

        # Ces bornes étaient déclarées mais jamais appliquées.
        min_units = int(MIN_UNITS_BY_PAIR.get(pair, MIN_UNITS_BY_PAIR["DEFAULT"]))
        max_units = int(MAX_UNITS_BY_PAIR.get(pair, MAX_UNITS_BY_PAIR["DEFAULT"]))

        if max_units > 0 and units > max_units:
            units = (max_units // step) * step

        if units <= 0 or (min_units > 0 and units < min_units):
            logger.info(
                f"[UNITS] {pair} | taille insuffisante | raw={raw_units:.2f} | "
                f"min={min_units} | max={max_units}"
            )
            return 0

        logger.info(
            f"[UNITS] {pair} | risk_usd={risk_usd:.2f} | distance={distance:.6f} | "
            f"quote_to_usd={quote_to_usd:.10f} | risk/unit={risk_per_unit:.10f} | "
            f"raw_units={raw_units:.2f} | step={step} | units_avant_marge={units}"
        )

        units_before_margin = units
        units = cap_units_by_margin(
            pair=pair,
            units=units,
            entry_price=entry,
            balance=balance,
        )

        if 0 < units < min_units:
            logger.info(
                f"[UNITS] {pair} | marge trop restrictive | final={units} < min={min_units}"
            )
            return 0

        logger.info(
            f"[UNITS] {pair} | avant_marge={units_before_margin} | apres_marge={units}"
        )
        return max(0, int(units))

    except Exception as e:
        logger.error(f"[UNITS] Erreur calcul {pair}: {e}")
        return 0

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

def detect_fvg(df: pd.DataFrame, max_lookback_bars: int = 24) -> List[Dict]:
    """
    Détecte uniquement les FVG récents et réellement retestés.

    Logique :
    - FVG classique 3 bougies
    - uniquement les dernières `max_lookback_bars`
    - le prix actuel doit être dans ou très proche de la zone
    - la zone ne doit pas avoir été invalidée par une clôture complète
    - priorité aux FVG les plus récents
    """

    fvgs = []

    if df is None or len(df) < 5:
        return fvgs

    try:
        data = df.copy()

        # On travaille uniquement avec les bougies disponibles.
        # get_candles() fournit déjà normalement des bougies complètes.
        recent = data.iloc[-(max_lookback_bars + 3):].copy()

        pair = str(data.attrs.get("instrument", "")).upper()

        # Seuil minimal de taille du gap.
        # On évite de fabriquer des FVG minuscules dus au bruit.
        atr = calculate_atr(data)

        if atr is None or atr <= 0:
            atr = 0.0

        if "JPY" in pair:
            min_gap = max(0.015, atr * 0.05)
        elif "XAU" in pair:
            min_gap = max(0.20, atr * 0.05)
        else:
            min_gap = max(0.00015, atr * 0.05)

        # Prix actuel = dernière clôture disponible
        current_price = float(recent["close"].iloc[-1])

        # ---------------------------------------------------------
        # Détection FVG
        # ---------------------------------------------------------
        for i in range(1, len(recent) - 1):

            prev = recent.iloc[i - 1]
            middle = recent.iloc[i]
            nxt = recent.iloc[i + 1]

            # -----------------------------------------------------
            # FVG BUY
            # prev.high < nxt.low
            # -----------------------------------------------------
            if prev["high"] < nxt["low"]:

                low_level = float(prev["high"])
                high_level = float(nxt["low"])
                gap_size = high_level - low_level

                if gap_size < min_gap:
                    continue

                zone_mid = (low_level + high_level) / 2

                # -------------------------------------------------
                # Invalidation :
                # une clôture sous le bas de la zone invalide
                # le FVG haussier.
                # -------------------------------------------------
                invalidated = False

                for j in range(i + 2, len(recent)):
                    if float(recent.iloc[j]["close"]) < low_level:
                        invalidated = True
                        break

                if invalidated:
                    continue

                # -------------------------------------------------
                # Le prix doit être dans la zone ou suffisamment
                # proche pour constituer un vrai retest.
                # Tolérance = 0.35 ATR
                # -------------------------------------------------
                distance = 0.0

                if current_price > high_level:
                    distance = current_price - high_level
                elif current_price < low_level:
                    distance = low_level - current_price

                max_retest_distance = max(atr * 0.35, gap_size * 1.5)

                if distance > max_retest_distance:
                    continue

                # -------------------------------------------------
                # On garde uniquement les FVG récents.
                # -------------------------------------------------
                fvgs.append({
                    "direction": "BUY",
                    "high_level": high_level,
                    "low_level": low_level,
                    "midpoint": zone_mid,
                    "gap_size": gap_size,
                    "distance": distance,
                    "time": recent.index[i],
                })

            # -----------------------------------------------------
            # FVG SELL
            # prev.low > nxt.high
            # -----------------------------------------------------
            if prev["low"] > nxt["high"]:

                high_level = float(prev["low"])
                low_level = float(nxt["high"])
                gap_size = high_level - low_level

                if gap_size < min_gap:
                    continue

                zone_mid = (low_level + high_level) / 2

                # -------------------------------------------------
                # Invalidation :
                # une clôture au-dessus du haut de la zone invalide
                # le FVG baissier.
                # -------------------------------------------------
                invalidated = False

                for j in range(i + 2, len(recent)):
                    if float(recent.iloc[j]["close"]) > high_level:
                        invalidated = True
                        break

                if invalidated:
                    continue

                # -------------------------------------------------
                # Le prix doit être dans la zone ou suffisamment
                # proche pour constituer un vrai retest.
                # -------------------------------------------------
                distance = 0.0

                if current_price < low_level:
                    distance = low_level - current_price
                elif current_price > high_level:
                    distance = current_price - high_level

                max_retest_distance = max(atr * 0.35, gap_size * 1.5)

                if distance > max_retest_distance:
                    continue

                fvgs.append({
                    "direction": "SELL",
                    "high_level": high_level,
                    "low_level": low_level,
                    "midpoint": zone_mid,
                    "gap_size": gap_size,
                    "distance": distance,
                    "time": recent.index[i],
                })

        # ---------------------------------------------------------
        # Priorité :
        # 1. plus proche du prix
        # 2. plus récent
        # ---------------------------------------------------------
        fvgs.sort(
            key=lambda x: (
                float(x.get("distance", 999999)),
                -pd.Timestamp(x["time"]).timestamp()
            )
        )

        # Maximum 5 FVG réellement exploitables
        return fvgs[:5]

    except Exception as e:
        logger.warning(f"[FVG] Erreur détection : {e}")
        return []
        
def detect_wick_rejection(df: pd.DataFrame, bias: str) -> list:
    """
    Détecte uniquement les rejets de mèches récents et exploitables.

    Logique :
    - ne scanne que les dernières bougies M15
    - exige une vraie mèche dominante
    - exige une clôture du bon côté
    - conserve uniquement les niveaux proches du prix actuel
    - le niveau d'entrée est basé sur la clôture de la bougie de rejet,
      et non sur l'extrême de la mèche
    """

    poi = []

    if df is None or len(df) < 5:
        return poi

    try:
        data = df.copy()

        # ---------------------------------------------------------
        # Dernières bougies M15 uniquement
        # ---------------------------------------------------------
        recent = data.iloc[-8:].copy()

        atr = calculate_atr(data)

        if atr is None or atr <= 0:
            return poi

        atr = float(atr)

        # Prix actuel = dernière clôture disponible
        current_price = float(recent["close"].iloc[-1])

        # Tolérance maximale autour du niveau de rejet
        max_distance = atr * 0.75

        # ---------------------------------------------------------
        # On examine seulement les 4 dernières bougies fermées
        # ---------------------------------------------------------
        start_idx = max(1, len(recent) - 4)

        for i in range(start_idx, len(recent)):

            c = recent.iloc[i]

            open_price = float(c["open"])
            close_price = float(c["close"])
            high_price = float(c["high"])
            low_price = float(c["low"])

            total = high_price - low_price

            if total <= 0:
                continue

            # Corps de la bougie
            body = abs(close_price - open_price)

            # Évite qu'une micro-bougie soit interprétée
            # comme une énorme rejection.
            effective_body = max(
                body,
                total * 0.05
            )

            # -----------------------------------------------------
            # Taille des mèches
            # -----------------------------------------------------
            upper = high_price - max(
                open_price,
                close_price
            )

            lower = min(
                open_price,
                close_price
            ) - low_price

            # =====================================================
            # BUY
            # Forte mèche basse + clôture dans la partie haute
            # =====================================================
            if bias == "BUY":

                rejection_strength = lower / total

                close_position = (
                    close_price - low_price
                ) / total

                valid = (
                    rejection_strength >= 0.35
                    and lower >= effective_body * 0.9
                    and lower > upper * 1.25
                    and close_position >= 0.55
                )

                if not valid:
                    continue

                # IMPORTANT :
                # On ne prend PAS le bas de la mèche comme entrée.
                # L'entrée correspond au niveau confirmé par la
                # clôture de la bougie de rejet.
                level = close_price

                distance = abs(
                    current_price - level
                )

                if distance > max_distance:
                    continue

                poi.append({
                    "direction": "BUY",
                    "price_level": level,
                    "time": recent.index[i],
                    "rejection_strength": rejection_strength,
                    "distance": distance,
                })

            # =====================================================
            # SELL
            # Forte mèche haute + clôture dans la partie basse
            # =====================================================
            elif bias == "SELL":

                rejection_strength = upper / total

                close_position = (
                    high_price - close_price
                ) / total

                valid = (
                    rejection_strength >= 0.35
                    and upper >= effective_body * 0.9
                    and upper > lower * 1.25
                    and close_position >= 0.55
                )

                if not valid:
                    continue

                # IMPORTANT :
                # On ne prend PAS le haut de la mèche comme entrée.
                # L'entrée correspond au niveau confirmé par la
                # clôture de la bougie de rejet.
                level = close_price

                distance = abs(
                    current_price - level
                )

                if distance > max_distance:
                    continue

                poi.append({
                    "direction": "SELL",
                    "price_level": level,
                    "time": recent.index[i],
                    "rejection_strength": rejection_strength,
                    "distance": distance,
                })

        # ---------------------------------------------------------
        # Priorité :
        # 1. niveau le plus proche du prix
        # 2. niveau le plus récent
        # ---------------------------------------------------------
        poi.sort(
            key=lambda x: (
                float(x.get("distance", 999999)),
                -pd.Timestamp(x["time"]).timestamp()
            )
        )

        # Maximum 4 rejets exploitables
        return poi[:4]

    except Exception as e:
        logger.warning(
            f"[WICK] Erreur détection : {e}"
        )
        return []
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
    """Génère les setups récents : BOS_RETEST, FVG_RETEST, WICK_REJECTION."""
    setups = []

    if df_m15 is None or df_m15.empty or bias not in ("BUY", "SELL"):
        return setups

    try:
        current_price = float(df_m15["close"].iloc[-1])
        atr = calculate_atr(df_m15)
        if atr is None or atr <= 0:
            return setups
        atr = float(atr)

        # BOS_RETEST était codé mais jamais appelé par detect_setups().
        bos = detect_bos_retest(df_m15, bias)
        if bos and bos.get("direction") == bias:
            level = float(bos["entry_level"])
            distance_atr = abs(current_price - level) / atr
            if distance_atr <= 1.50:
                setups.append({**bos, "distance_atr": distance_atr})

        # FVG
        for f in detect_fvg(df_m15, max_lookback_bars=24):
            if f.get("direction") != bias:
                continue
            level = float(f["midpoint"])
            distance_atr = abs(current_price - level) / atr
            if distance_atr <= 1.50:
                setups.append({
                    "type": "FVG_RETEST",
                    "direction": bias,
                    "entry_level": level,
                    "fvg": f,
                    "distance_atr": distance_atr,
                    "rejection_strength": 0.0,
                    "time": f.get("time"),
                })

        # WICK
        for w in detect_wick_rejection(df_m15, bias):
            if w.get("direction") != bias:
                continue
            level = float(w["price_level"])
            distance_atr = abs(current_price - level) / atr
            if distance_atr <= 1.50:
                setups.append({
                    "type": "WICK_REJECTION",
                    "direction": bias,
                    "entry_level": level,
                    "distance_atr": distance_atr,
                    "rejection_strength": float(w.get("rejection_strength", 0.0)),
                    "time": w.get("time"),
                })

        # Dédoublonnage : BOS > FVG > WICK au même niveau.
        priority = {"BOS_RETEST": 1, "FVG_RETEST": 2, "WICK_REJECTION": 3}
        merge_distance = atr * 0.20
        setups.sort(key=lambda x: (
            float(x.get("distance_atr", 999999)),
            priority.get(x.get("type"), 99),
        ))

        deduped = []
        for setup in setups:
            duplicate = False
            for existing in deduped:
                if setup.get("direction") != existing.get("direction"):
                    continue
                if abs(float(setup["entry_level"]) - float(existing["entry_level"])) <= merge_distance:
                    if priority.get(setup.get("type"), 99) < priority.get(existing.get("type"), 99):
                        existing.clear()
                        existing.update(setup)
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(setup)

        deduped.sort(key=lambda x: (
            priority.get(x.get("type"), 99),
            float(x.get("distance_atr", 999999)),
        ))
        setups = deduped[:8]

        logger.info(
            f"[SETUPS_FILTER] {pair} | BIAS={bias} | retenus={len(setups)} | "
            f"prix={current_price:.5f} | ATR={atr:.5f}"
        )
        return setups

    except Exception as e:
        logger.warning(f"[SETUPS] {pair} erreur génération setups : {e}")
        return []

def get_directional_bias(
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame
) -> str:
    """
    Biais souverain H4.

    H4 décide uniquement de la direction.
    H1 sert à identifier la phase :
        - EXPANSION   : H1 aligné avec H4
        - RETRACEMENT : H1 opposé à H4
        - NEUTRAL_H1  : H1 indéterminé

    Le H1 ne peut pas annuler un biais H4 valide.
    """

    def structure_htf(df: pd.DataFrame) -> str:
        if df is None or len(df) < 20:
            return "NEUTRAL"

        try:
            highs, lows = detect_swing_points(df, 5)
        except Exception:
            return "NEUTRAL"

        if len(highs) < 2 or len(lows) < 2:
            return "NEUTRAL"

        last_high = float(highs[-1]["price"])
        prev_high = float(highs[-2]["price"])
        last_low = float(lows[-1]["price"])
        prev_low = float(lows[-2]["price"])

        hh = last_high > prev_high
        hl = last_low > prev_low

        lh = last_high < prev_high
        ll = last_low < prev_low

        # Structure clairement haussière
        if hh and hl:
            return "BUY"

        # Structure clairement baissière
        if lh and ll:
            return "SELL"

        # Structure partiellement haussière
        if hh or hl:
            return "BUY_WEAK"

        # Structure partiellement baissière
        if lh or ll:
            return "SELL_WEAK"

        # Structure contradictoire
        return "NEUTRAL"

    # =========================================================
    # STRUCTURE H4 / H1
    # =========================================================

    h4_struct = structure_htf(df_h4)
    h1_struct = structure_htf(df_h1)

    # =========================================================
    # MODE STRICT (optionnel) : on exige une structure H4
    # complète (HH+HL ou LH+LL), pas seulement partielle.
    # =========================================================
    if STRICT_BIAS_ALIGNMENT and h4_struct in ("BUY_WEAK", "SELL_WEAK"):
        logger.info(
            f"[BIAS_DIAG] H4={h4_struct} | H1={h1_struct} | "
            f"rejeté par STRICT_BIAS_ALIGNMENT -> NEUTRAL"
        )
        return "NEUTRAL"

    # =========================================================
    # H4 HAUSSIER = BIAIS BUY
    # =========================================================

    if h4_struct in ("BUY", "BUY_WEAK"):

        if h1_struct in ("BUY", "BUY_WEAK"):
            phase = "EXPANSION"

        elif h1_struct in ("SELL", "SELL_WEAK"):
            phase = "RETRACEMENT"

        else:
            phase = "NEUTRAL_H1"

        logger.info(
            f"[BIAS_DIAG] "
            f"H4={h4_struct} | "
            f"H1={h1_struct} | "
            f"Phase H1={phase} | "
            f"BIAIS=BUY"
        )

        return "BUY"

    # =========================================================
    # H4 BAISSIER = BIAIS SELL
    # =========================================================

    if h4_struct in ("SELL", "SELL_WEAK"):

        if h1_struct in ("SELL", "SELL_WEAK"):
            phase = "EXPANSION"

        elif h1_struct in ("BUY", "BUY_WEAK"):
            phase = "RETRACEMENT"

        else:
            phase = "NEUTRAL_H1"

        logger.info(
            f"[BIAS_DIAG] "
            f"H4={h4_struct} | "
            f"H1={h1_struct} | "
            f"Phase H1={phase} | "
            f"BIAIS=SELL"
        )

        return "SELL"

    # =========================================================
    # H4 INDETERMINE
    # =========================================================

    logger.info(
        f"[BIAS_DIAG] "
        f"H4={h4_struct} | "
        f"H1={h1_struct} | "
        f"Structure H4 incertaine -> BIAIS=NEUTRAL"
    )

    return "NEUTRAL"
    
def detect_bos_retest(
    df: pd.DataFrame,
    direction: str
) -> Optional[dict]:
    """
    Détecte un véritable BOS suivi d'un retest.

    BUY :
        1. cassure d'un swing high confirmé
        2. retour du prix vers le niveau cassé
        3. maintien au-dessus du niveau
        4. rejet OU micro-break haussier

    SELL :
        1. cassure d'un swing low confirmé
        2. retour du prix vers le niveau cassé
        3. maintien sous le niveau
        4. rejet OU micro-break baissier

    Le BOS et le retest peuvent être sur des bougies différentes.
    """

    # =========================================================
    # SÉCURITÉ
    # =========================================================

    if df is None or len(df) < 30:
        return None

    direction = direction.upper()

    if direction not in ("BUY", "SELL"):
        return None

    # =========================================================
    # BOUGIES CLÔTURÉES UNIQUEMENT
    # =========================================================

    data = df.copy()

    if len(data) < 25:
        return None

    atr = calculate_atr(data)

    if atr is None or atr <= 0:
        return None

    atr = float(atr)

    # =========================================================
    # PARAMÈTRES
    # =========================================================

    MAX_BOS_AGE = 8
    RETEST_TOLERANCE_ATR = 0.25
    MAX_RETEST_AGE = 2

    retest_tolerance = atr * RETEST_TOLERANCE_ATR

    # =========================================================
    # SWINGS CONFIRMÉS
    # =========================================================

    # On évite d'utiliser les structures les plus récentes
    # susceptibles d'être encore en formation.
    structure_cutoff = max(
        0,
        len(data) - MAX_BOS_AGE
    )

    structure_df = data.iloc[
        :structure_cutoff
    ]

    if len(structure_df) < 15:
        return None

    swing_highs, swing_lows = detect_swing_points(
        structure_df,
        5
    )

    # =========================================================
    # CONFIRMATION
    # =========================================================

    def confirmation_signal(
        candle,
        previous_candle,
        side: str
    ):
        high = float(candle["high"])
        low = float(candle["low"])
        open_ = float(candle["open"])
        close = float(candle["close"])

        candle_range = high - low

        if candle_range <= 0:
            return False, 0.0, False

        if side == "BUY":

            lower_wick = (
                min(open_, close) - low
            )

            rejection_ratio = (
                lower_wick / candle_range
            )

            micro_break = (
                close > float(previous_candle["high"])
            )

        else:

            upper_wick = (
                high - max(open_, close)
            )

            rejection_ratio = (
                upper_wick / candle_range
            )

            micro_break = (
                close < float(previous_candle["low"])
            )

        ok = (
            rejection_ratio >= 0.30
            or micro_break
        )

        return (
            ok,
            rejection_ratio,
            micro_break
        )

    # =========================================================
    # RECHERCHE DU BOS LE PLUS RÉCENT
    # =========================================================

    for bos_index in range(
        len(data) - 2,
        max(1, len(data) - MAX_BOS_AGE - 2),
        -1
    ):

        bos_candle = data.iloc[bos_index]
        previous_bos_candle = data.iloc[bos_index - 1]

        # =====================================================
        # BUY
        # =====================================================

        if direction == "BUY":

            valid_highs = [
                h for h in swing_highs
                if int(h.get("index", -1)) < bos_index
            ]

            if not valid_highs:
                continue

            swing_level = float(
                valid_highs[-1]["price"]
            )

            # -------------------------------------------------
            # BOS haussier
            # -------------------------------------------------

            bos_confirmed = (
                float(bos_candle["close"]) > swing_level
                and
                float(previous_bos_candle["close"]) <= swing_level
            )

            if not bos_confirmed:
                continue

            # -------------------------------------------------
            # Retest après le BOS
            # -------------------------------------------------

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

                # Le prix revient dans la zone du niveau cassé.
                touched = (
                    retest_low
                    <= swing_level + retest_tolerance
                )

                # Le niveau reste globalement défendu.
                held = (
                    retest_close
                    >= swing_level - retest_tolerance
                )

                if not touched or not held:
                    continue

                # Retest trop ancien = on ne prend pas.
                bars_since_retest = (
                    len(data) - 1 - retest_index
                )

                if bars_since_retest > MAX_RETEST_AGE:
                    continue

                previous_retest = data.iloc[
                    retest_index - 1
                ]

                (
                    confirmation_ok,
                    rejection_ratio,
                    micro_break
                ) = confirmation_signal(
                    retest_candle,
                    previous_retest,
                    "BUY"
                )

                if not confirmation_ok:
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
                            current_price - swing_level
                        ) / atr
                    ),
                }

        # =====================================================
        # SELL
        # =====================================================

        else:

            valid_lows = [
                l for l in swing_lows
                if int(l.get("index", -1)) < bos_index
            ]

            if not valid_lows:
                continue

            swing_level = float(
                valid_lows[-1]["price"]
            )

            # -------------------------------------------------
            # BOS baissier
            # -------------------------------------------------

            bos_confirmed = (
                float(bos_candle["close"]) < swing_level
                and
                float(previous_bos_candle["close"]) >= swing_level
            )

            if not bos_confirmed:
                continue

            # -------------------------------------------------
            # Retest après le BOS
            # -------------------------------------------------

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

                touched = (
                    retest_high
                    >= swing_level - retest_tolerance
                )

                held = (
                    retest_close
                    <= swing_level + retest_tolerance
                )

                if not touched or not held:
                    continue

                bars_since_retest = (
                    len(data) - 1 - retest_index
                )

                if bars_since_retest > MAX_RETEST_AGE:
                    continue

                previous_retest = data.iloc[
                    retest_index - 1
                ]

                (
                    confirmation_ok,
                    rejection_ratio,
                    micro_break
                ) = confirmation_signal(
                    retest_candle,
                    previous_retest,
                    "SELL"
                )

                if not confirmation_ok:
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
                            current_price - swing_level
                        ) / atr
                    ),
                }

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
    Confirmation M15 adaptative mais structurée.

    BUY :
        - rejet haussier + clôture acceptable
          OU
        - micro-break haussier
          OU
        - très fort rejet même sans bougie verte

    SELL :
        - rejet baissier + clôture acceptable
          OU
        - micro-break baissier
          OU
        - très fort rejet même sans bougie rouge

    La confirmation reste cohérente avec le biais
    et évite de valider une simple mèche isolée.
    """

    if df_m15 is None or len(df_m15) < 4:
        return False, "données insuffisantes"

    try:
        direction = str(direction).upper().strip()

        if direction not in ("BUY", "SELL"):
            return False, f"direction invalide: {direction}"

        last = df_m15.iloc[-1]
        prev = df_m15.iloc[-2]

        open_price = float(last["open"])
        close_price = float(last["close"])
        high_price = float(last["high"])
        low_price = float(last["low"])

        prev_high = float(prev["high"])
        prev_low = float(prev["low"])

        total = high_price - low_price

        if total <= 0:
            return False, "range nul"

        body = abs(
            close_price - open_price
        )

        upper_wick = (
            high_price
            - max(open_price, close_price)
        )

        lower_wick = (
            min(open_price, close_price)
            - low_price
        )

        close_position = (
            close_price - low_price
        ) / total

        lower_ratio = lower_wick / total
        upper_ratio = upper_wick / total

        # Évite qu'une bougie quasi sans corps
        # soit interprétée comme une énorme confirmation.
        effective_body = max(
            body,
            total * 0.05
        )

        # =========================================================
        # BUY
        # =========================================================

        if direction == "BUY":

            # Rejet haussier classique
            bullish_rejection = (
                lower_ratio >= 0.30
                and lower_wick >= effective_body * 0.90
                and lower_wick > upper_wick * 1.10
                and close_position >= 0.50
            )

            # Micro-break immédiat
            bullish_micro_break = (
                close_price > prev_high
            )

            # -----------------------------------------------------
            # Clôture favorable
            # -----------------------------------------------------
            #
            # On ne demande plus obligatoirement une bougie verte.
            # Une récupération nette de la partie basse suffit.
            #
            bullish_close = (
                close_price >= open_price
                or close_position >= 0.60
            )

            # -----------------------------------------------------
            # Confirmation forte
            # -----------------------------------------------------

            if (
                bullish_rejection
                and bullish_close
            ):
                return True, (
                    f"rejet haussier + clôture OK "
                    f"(rejet={lower_ratio:.2f}, "
                    f"close={close_position:.2f})"
                )

            # Le micro-break devient une confirmation autonome.
            # Il matérialise directement la reprise de structure.
            if bullish_micro_break:
                return True, (
                    f"micro-break haussier OK "
                    f"(rejet={lower_ratio:.2f})"
                )

            # Très gros rejet :
            # on autorise une clôture moins parfaite si la
            # pression acheteuse est clairement visible.
            if (
                lower_ratio >= 0.45
                and lower_wick > upper_wick * 1.40
                and close_position >= 0.55
            ):
                return True, (
                    f"fort rejet haussier OK "
                    f"(rejet={lower_ratio:.2f}, "
                    f"close={close_position:.2f})"
                )

            reasons = []

            if not bullish_rejection:
                reasons.append(
                    f"rejet insuffisant ({lower_ratio:.2f})"
                )

            if not bullish_micro_break:
                reasons.append(
                    "pas de micro-break"
                )

            if not bullish_close:
                reasons.append(
                    "clôture non favorable"
                )

            return False, ", ".join(reasons)

        # =========================================================
        # SELL
        # =========================================================

        if direction == "SELL":

            # Rejet baissier classique
            bearish_rejection = (
                upper_ratio >= 0.30
                and upper_wick >= effective_body * 0.90
                and upper_wick > lower_wick * 1.10
                and close_position <= 0.50
            )

            # Micro-break immédiat
            bearish_micro_break = (
                close_price < prev_low
            )

            # Clôture favorable
            bearish_close = (
                close_price <= open_price
                or close_position <= 0.40
            )

            # -----------------------------------------------------
            # Confirmation forte
            # -----------------------------------------------------

            if (
                bearish_rejection
                and bearish_close
            ):
                return True, (
                    f"rejet baissier + clôture OK "
                    f"(rejet={upper_ratio:.2f}, "
                    f"close={close_position:.2f})"
                )

            # Micro-break autonome
            if bearish_micro_break:
                return True, (
                    f"micro-break baissier OK "
                    f"(rejet={upper_ratio:.2f})"
                )

            # Très gros rejet
            if (
                upper_ratio >= 0.45
                and upper_wick > lower_wick * 1.40
                and close_position <= 0.45
            ):
                return True, (
                    f"fort rejet baissier OK "
                    f"(rejet={upper_ratio:.2f}, "
                    f"close={close_position:.2f})"
                )

            reasons = []

            if not bearish_rejection:
                reasons.append(
                    f"rejet insuffisant ({upper_ratio:.2f})"
                )

            if not bearish_micro_break:
                reasons.append(
                    "pas de micro-break"
                )

            if not bearish_close:
                reasons.append(
                    "clôture non favorable"
                )

            return False, ", ".join(reasons)

        return False, f"direction invalide: {direction}"

    except Exception as e:
        logger.warning(
            f"[CONFIRMATION] Erreur : {e}"
        )

        return False, (
            f"erreur confirmation: {e}"
        )

def calculate_sl_tp_structural(
    df_m15: pd.DataFrame,
    direction: str,
    entry: float,
    pair: str
) -> Tuple[float, float, float]:
    """
    Calcule un SL structurel M15 et un TP à 2R.

    Règles :
    - BUY  : sous un swing low M15 exploitable
    - SELL : au-dessus d'un swing high M15 exploitable
    - Buffer structurel : 5 pips
    - Recherche des swings sur les 64 dernières bougies
    - SL maximum : 2 ATR
    - Minimum SL : 10 pips si compatible avec 2 ATR
    - Si 10 pips > 2 ATR, le minimum effectif devient 2 ATR
    - USD_JPY / AUD_JPY : AUCUN fallback ATR
    - Autres paires : fallback ATR 1.5x si aucun swing exploitable
    - TP = 2R
    - Garantie finale RR >= 2.0 après arrondi
    """

    pair = str(pair).upper().strip()
    direction = str(direction).upper().strip()
    entry = float(entry)

    if direction not in ("BUY", "SELL"):
        raise ValueError(
            f"Direction inconnue: {direction}"
        )

    if df_m15 is None or len(df_m15) < 20:
        raise ValueError(
            f"Données M15 insuffisantes pour {pair}"
        )

    # ---------------------------------------------------------
    # SWINGS / ATR / PIP
    # ---------------------------------------------------------
    highs, lows = detect_swing_points(
        df_m15,
        5
    )

    pip = float(
        get_pip_value(pair)
    )

    atr = calculate_atr(
        df_m15
    )

    if atr is None or atr <= 0:
        atr = pip * 10

    atr = float(atr)

    if pip <= 0 or atr <= 0:
        raise ValueError(
            f"Paramètres SL invalides pour {pair}"
        )

    # ---------------------------------------------------------
    # PARAMÈTRES
    # ---------------------------------------------------------
    SL_BUFFER_PIPS = 5.0
    MIN_SL_PIPS = 10.0
    STRUCTURAL_SWING_LOOKBACK_BARS = 64
    MAX_SL_ATR = 2.0
    FALLBACK_SL_ATR = 1.5
    TARGET_RR = 2.0

    sl_buffer = (
        SL_BUFFER_PIPS * pip
    )

    max_sl_distance = (
        atr * MAX_SL_ATR
    )

    requested_min_sl_distance = (
        MIN_SL_PIPS * pip
    )

    # Le minimum 10 pips ne doit jamais
    # dépasser la limite absolue de 2 ATR.
    effective_min_sl_distance = min(
        requested_min_sl_distance,
        max_sl_distance
    )

    recent_cutoff = max(
        0,
        len(df_m15) - STRUCTURAL_SWING_LOOKBACK_BARS
    )

    sl = None
    sl_source = None

    # EPS uniquement pour éviter les rejets
    # artificiels liés aux arrondis flottants.
    EPS = max(
        pip * 0.05,
        atr * 0.001
    )

    # ---------------------------------------------------------
    # SL STRUCTUREL
    # ---------------------------------------------------------
    if direction == "BUY":

        candidates = []

        for low in lows:
            index = int(
                low.get("index", -1)
            )

            if index < recent_cutoff:
                continue

            level = float(
                low["price"]
            )

            if level >= entry:
                continue

            candidate_sl = (
                level - sl_buffer
            )

            candidate_risk = (
                entry - candidate_sl
            )

            if (
                0 < candidate_risk
                <= max_sl_distance + EPS
            ):
                candidates.append(
                    (
                        candidate_risk,
                        level,
                        candidate_sl
                    )
                )

        if candidates:

            # Plus petit risque structurel valide
            # = swing exploitable le plus proche.
            candidates.sort(
                key=lambda x: (
                    x[0],
                    -x[1]
                )
            )

            _, level, sl = candidates[0]

            sl_source = (
                f"SWING_LOW_NEAREST "
                f"{level:.5f}"
            )

        else:

            # IMPORTANT :
            # USD_JPY et AUD_JPY ne doivent plus
            # utiliser de fallback ATR.
            if pair in {
                "USD_JPY",
                "AUD_JPY"
            }:
                raise ValueError(
                    f"{pair} | aucun swing low M15 "
                    f"exploitable dans 2 ATR → rejet"
                )

            fallback_distance = min(
                atr * FALLBACK_SL_ATR,
                max_sl_distance
            )

            fallback_distance = max(
                fallback_distance,
                effective_min_sl_distance
            )

            fallback_distance = min(
                fallback_distance,
                max_sl_distance
            )

            sl = (
                entry - fallback_distance
            )

            sl_source = (
                "ATR_FALLBACK_NO_VALID_SWING"
            )

    else:  # SELL

        candidates = []

        for high in highs:
            index = int(
                high.get("index", -1)
            )

            if index < recent_cutoff:
                continue

            level = float(
                high["price"]
            )

            if level <= entry:
                continue

            candidate_sl = (
                level + sl_buffer
            )

            candidate_risk = (
                candidate_sl - entry
            )

            if (
                0 < candidate_risk
                <= max_sl_distance + EPS
            ):
                candidates.append(
                    (
                        candidate_risk,
                        level,
                        candidate_sl
                    )
                )

        if candidates:

            candidates.sort(
                key=lambda x: (
                    x[0],
                    x[1]
                )
            )

            _, level, sl = candidates[0]

            sl_source = (
                f"SWING_HIGH_NEAREST "
                f"{level:.5f}"
            )

        else:

            # IMPORTANT :
            # USD_JPY et AUD_JPY ne doivent plus
            # utiliser de fallback ATR.
            if pair in {
                "USD_JPY",
                "AUD_JPY"
            }:
                raise ValueError(
                    f"{pair} | aucun swing high M15 "
                    f"exploitable dans 2 ATR → rejet"
                )

            fallback_distance = min(
                atr * FALLBACK_SL_ATR,
                max_sl_distance
            )

            fallback_distance = max(
                fallback_distance,
                effective_min_sl_distance
            )

            fallback_distance = min(
                fallback_distance,
                max_sl_distance
            )

            sl = (
                entry + fallback_distance
            )

            sl_source = (
                "ATR_FALLBACK_NO_VALID_SWING"
            )

    # ---------------------------------------------------------
    # SÉCURITÉ DIRECTIONNELLE
    # ---------------------------------------------------------
    if direction == "BUY" and sl >= entry:

        if pair in {
            "USD_JPY",
            "AUD_JPY"
        }:
            raise ValueError(
                f"{pair} | SL structurel invalide "
                f"pour BUY → rejet"
            )

        sl = (
            entry - effective_min_sl_distance
        )

        sl_source = (
            "ATR_FALLBACK_INVALID_STRUCTURE"
        )

    elif direction == "SELL" and sl <= entry:

        if pair in {
            "USD_JPY",
            "AUD_JPY"
        }:
            raise ValueError(
                f"{pair} | SL structurel invalide "
                f"pour SELL → rejet"
            )

        sl = (
            entry + effective_min_sl_distance
        )

        sl_source = (
            "ATR_FALLBACK_INVALID_STRUCTURE"
        )

    # ---------------------------------------------------------
    # RISQUE AVANT ARRONDI
    # ---------------------------------------------------------
    risk_before_rounding = abs(
        entry - sl
    )

    if risk_before_rounding <= 0:
        raise ValueError(
            f"Risque nul {pair}"
        )

    if (
        risk_before_rounding
        > max_sl_distance + EPS
    ):
        raise ValueError(
            f"SL structurel > "
            f"{MAX_SL_ATR:.1f} ATR "
            f"(risk={risk_before_rounding:.5f}, "
            f"max={max_sl_distance:.5f})"
        )

    # ---------------------------------------------------------
    # MINIMUM SL
    # ---------------------------------------------------------
    if (
        risk_before_rounding
        < effective_min_sl_distance - EPS
    ):
        if direction == "BUY":
            sl = (
                entry - effective_min_sl_distance
            )
        else:
            sl = (
                entry + effective_min_sl_distance
            )

    # ---------------------------------------------------------
    # ARRONDI SL
    # ---------------------------------------------------------
    sl = float(
        round_price(
            pair,
            sl
        )
    )

    # ---------------------------------------------------------
    # SÉCURITÉ APRÈS ARRONDI
    # ---------------------------------------------------------
    if direction == "BUY" and sl >= entry:

        sl = float(
            round_price(
                pair,
                entry - effective_min_sl_distance
            )
        )

    elif direction == "SELL" and sl <= entry:

        sl = float(
            round_price(
                pair,
                entry + effective_min_sl_distance
            )
        )

    risk = abs(
        entry - sl
    )

    if risk <= 0:
        raise ValueError(
            f"Risk nul après arrondi {pair}"
        )

    if risk > max_sl_distance + EPS:
        raise ValueError(
            f"SL après arrondi > "
            f"{MAX_SL_ATR:.1f} ATR "
            f"(risk={risk:.5f}, "
            f"max={max_sl_distance:.5f})"
        )

    # ---------------------------------------------------------
    # TP = 2R
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # RR FINAL
    # ---------------------------------------------------------
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
        final_reward / final_risk
    )

    # Petite marge uniquement pour absorber
    # la granularité d'arrondi du prix.
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
            final_reward / final_risk
        )

    if rr < TARGET_RR:
        raise ValueError(
            f"RR final insuffisant après arrondi "
            f"(RR={rr:.3f})"
        )

    logger.info(
        f"[SLTP] {pair} | "
        f"{direction} | "
        f"ENTRY={entry:.5f} | "
        f"SL={sl:.5f} | "
        f"TP={tp:.5f} | "
        f"RISK={final_risk:.5f} | "
        f"ATR={atr:.5f} | "
        f"SL_ATR={final_risk / atr:.2f} | "
        f"RR={rr:.3f} | "
        f"MIN_SL={effective_min_sl_distance:.5f} | "
        f"SOURCE={sl_source}"
    )

    return (
        float(sl),
        float(tp),
        float(final_risk)
    )
def has_enough_room_to_tp(
    df_h1: pd.DataFrame,
    direction: str,
    entry: float,
    tp: float
) -> bool:
    """
    Vérifie que le TP à 2R dispose d'un espace structurel suffisant.

    Un swing H1 intermédiaire ne bloque pas le trade.

    Seul un swing situé dans les 15 derniers pourcents
    du trajet vers le TP est considéré comme obstacle.
    """

    if df_h1 is None or len(df_h1) < 20:
        return True

    try:
        highs, lows = detect_swing_points(
            df_h1,
            5
        )
    except Exception as e:
        logger.warning(
            f"[TP_SPACE] erreur swings H1: {e}"
        )
        return True

    total_distance = abs(
        tp - entry
    )

    if total_distance <= 0:
        return False

    critical_zone = (
        total_distance * 0.15
    )

    # =========================================================
    # BUY
    # =========================================================

    if direction == "BUY":

        for h in highs:

            level = float(
                h["price"]
            )

            if not (
                entry < level < tp
            ):
                continue

            distance_to_tp = (
                tp - level
            )

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

            level = float(
                l["price"]
            )

            if not (
                tp < level < entry
            ):
                continue

            distance_to_tp = (
                level - tp
            )

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
    pair,
    direction,
    entry,
    df_m15,
    df_h1,
    current_price
):
    # =========================================================
    # TYPES DE SETUPS AUTORISÉS
    # =========================================================
    setup_type = entry.get("type")

    if setup_type not in (
        "FVG_RETEST",
        "WICK_REJECTION",
        "BOS_RETEST"
    ):
        return {
            "passed": False,
            "reason": (
                f"type non autorisé: "
                f"{setup_type}"
            )
        }

    # =========================================================
    # SETUP LEVEL
    # =========================================================
    try:
        setup_level = float(
            entry["entry_level"]
        )
    except Exception:
        return {
            "passed": False,
            "reason": "entry_level invalide"
        }

    # =========================================================
    # ATR
    # =========================================================
    atr_price = calculate_atr(
        df_m15
    )

    if atr_price is None or atr_price <= 0:
        return {
            "passed": False,
            "reason": "ATR invalide"
        }

    atr_price = float(
        atr_price
    )

    # =========================================================
    # PIP
    # =========================================================
    pip = float(
        get_pip_value(pair)
    )

    if pip <= 0:
        return {
            "passed": False,
            "reason": "pip invalide"
        }

    atr_pips = (
        atr_price / pip
    )

    # =========================================================
    # DISTANCE MAXIMALE AU SETUP
    # =========================================================
    setup_distance = abs(
        float(current_price)
        - setup_level
    )

    distance_ratio = (
        setup_distance / atr_price
    )

    if distance_ratio > 2.0:
        return {
            "passed": False,
            "reason": (
                f"prix hors zone "
                f"(target={setup_level:.5f}, "
                f"price={current_price:.5f}, "
                f"dist={distance_ratio:.2f}ATR, "
                f"max=2.0ATR)"
            )
        }

    # =========================================================
    # CONFIRMATION
    # =========================================================

    if setup_type == "BOS_RETEST":

        confirmation_ok = True

        confirmation_msg = (
            f"BOS_RETEST "
            f"{entry.get('confirmation', 'OK')}"
        )

        confirmation = {
            "ok": True,
            "type": "BOS_RETEST",
            "message": confirmation_msg
        }

    elif setup_type == "WICK_REJECTION":

        try:
            rejection_strength = float(
                entry.get(
                    "rejection_strength",
                    0.0
                )
            )
        except Exception:
            rejection_strength = 0.0

        if rejection_strength < 0.35:
            return {
                "passed": False,
                "reason": (
                    f"rejet insuffisant "
                    f"({rejection_strength:.2f} < 0.35)"
                )
            }

        confirmation_ok = True

        confirmation_msg = (
            f"WICK confirmé "
            f"(rejet={rejection_strength:.2f})"
        )

        confirmation = {
            "ok": True,
            "type": "WICK_REJECTION",
            "rejection_strength": rejection_strength,
            "message": confirmation_msg
        }

    else:
        # =====================================================
        # FVG : confirmation locale
        # =====================================================
        try:
            confirmation_ok, confirmation_msg = (
                get_confirmation_signal(
                    df_m15,
                    direction
                )
            )
        except Exception as e:
            return {
                "passed": False,
                "reason": (
                    f"erreur confirmation FVG: {e}"
                )
            }

        confirmation = {
            "ok": bool(confirmation_ok),
            "type": "FVG_RETEST",
            "message": confirmation_msg
        }

        if not confirmation_ok:

            last = df_m15.iloc[-1]
            prev = df_m15.iloc[-2]

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
                )
            }

    # =========================================================
    # FILTRES QUALITÉ (WIN RATE)
    #
    # ADX H1 : un setup de continuation (retest dans le sens
    # du biais H4) a besoin d'une tendance H1 réelle. Sous le
    # seuil, le marché est en range et ces setups échouent
    # nettement plus souvent (faux retests).
    #
    # RSI M15 : on évite d'ouvrir un BUY quand le mouvement est
    # déjà extrêmement étiré à la hausse (et inversement pour
    # SELL), ce qui augmente le risque de retournement avant
    # d'atteindre le TP à 2R.
    # =========================================================
    if ENABLE_QUALITY_FILTERS:

        try:
            adx_h1 = float(calculate_adx(df_h1))
        except Exception:
            adx_h1 = 0.0

        if adx_h1 < MIN_ADX_TREND:
            return {
                "passed": False,
                "reason": (
                    f"ADX H1 trop faible "
                    f"({adx_h1:.1f} < {MIN_ADX_TREND}) "
                    f"-> marché sans tendance"
                )
            }

        try:
            rsi_m15 = float(get_last_rsi(df_m15["close"]))
        except Exception:
            rsi_m15 = 50.0

        if direction == "BUY" and rsi_m15 > RSI_OVERBOUGHT:
            return {
                "passed": False,
                "reason": (
                    f"RSI M15 suracheté "
                    f"({rsi_m15:.1f} > {RSI_OVERBOUGHT}) "
                    f"-> mouvement trop étiré pour un BUY"
                )
            }

        if direction == "SELL" and rsi_m15 < RSI_OVERSOLD:
            return {
                "passed": False,
                "reason": (
                    f"RSI M15 survendu "
                    f"({rsi_m15:.1f} < {RSI_OVERSOLD}) "
                    f"-> mouvement trop étiré pour un SELL"
                )
            }

    # =========================================================
    # EXECUTION ENTRY
    #
    # C'est le prix courant M15 utilisé comme base
    # pour calculer SL / TP.
    # =========================================================
    execution_entry = float(
        current_price
    )

    # =========================================================
    # SL / TP STRUCTURELS
    # =========================================================
    try:
        sl, tp, risk = (
            calculate_sl_tp_structural(
                df_m15,
                direction,
                execution_entry,
                pair
            )
        )
    except Exception as e:
        return {
            "passed": False,
            "reason": str(e)
        }

    # =========================================================
    # MIN SL ADAPTATIF
    # =========================================================
    max_sl_distance = (
        atr_price * 2.0
    )

    effective_min_sl_distance = min(
        10.0 * pip,
        max_sl_distance
    )

    # Tolérance uniquement pour les comparaisons
    # flottantes / arrondis.
    EPS = max(
        pip * 0.05,
        atr_price * 0.001
    )

    sl_distance = abs(
        execution_entry - sl
    )

    if (
        sl_distance
        < effective_min_sl_distance - EPS
    ):
        return {
            "passed": False,
            "reason": (
                f"SL trop proche "
                f"({sl_distance / pip:.2f} pips < "
                f"{effective_min_sl_distance / pip:.2f} pips)"
            )
        }

    # =========================================================
    # RR STRICT
    # =========================================================
    if risk <= 0:
        return {
            "passed": False,
            "reason": "risque nul"
        }

    reward = abs(
        tp - execution_entry
    )

    rr = (
        reward / risk
    )

    if rr < 2.0 - 0.001:
        return {
            "passed": False,
            "reason": (
                f"RR insuffisant "
                f"({rr:.3f} < 2.0)"
            )
        }

    # =========================================================
    # ROOM H1
    #
    # IMPORTANT :
    # has_enough_room_to_tp() retourne un BOOL,
    # pas (bool, message).
    # =========================================================
    try:

        room_ok = has_enough_room_to_tp(
            df_h1,
            direction,
            execution_entry,
            tp
        )

    except Exception as e:

        return {
            "passed": False,
            "reason": (
                f"contrôle room H1 erreur: {e}"
            )
        }

    if not room_ok:
        return {
            "passed": False,
            "reason": (
                "espace H1 insuffisant "
                "pour atteindre le TP"
            )
        }

    # =========================================================
    # MÉTRIQUES
    #
    # NOTE CORRECTIF : get_last_rsi() attend une pd.Series de
    # clôtures, pas un DataFrame. L'appel get_last_rsi(df_m15)
    # levait systématiquement une exception (dimensions
    # invalides pour talib), avalée par le except ci-dessous,
    # ce qui figeait "rsi" à 50.0 en permanence. On réutilise
    # ici les valeurs déjà calculées par le filtre qualité
    # quand celui-ci est actif, sinon on les recalcule.
    # =========================================================
    try:
        adx = float(adx_h1) if ENABLE_QUALITY_FILTERS else float(calculate_adx(df_h1))
    except Exception:
        adx = 0.0

    try:
        momentum = float(
            calculate_momentum(
                df_m15
            )
        )
    except Exception:
        momentum = 0.0

    try:
        rsi = float(rsi_m15) if ENABLE_QUALITY_FILTERS else float(get_last_rsi(df_m15["close"]))
    except Exception:
        rsi = 50.0

    metrics = {
        "atr": float(atr_pips),
        "atr_price": float(atr_price),
        "adx": float(adx),
        "momentum": float(momentum),
        "rsi": float(rsi),

        # Niveau structurel du setup
        "setup_level": float(
            setup_level
        ),

        # Prix utilisé pour la préparation de l'ordre
        "execution_entry": float(
            execution_entry
        ),

        "setup_distance": float(
            setup_distance
        ),

        "setup_distance_atr": float(
            distance_ratio
        ),

        "confirmation": confirmation,

        "confirmation_message": (
            confirmation_msg
        )
    }

    # =========================================================
    # RESULTAT FINAL
    # =========================================================
    return {
        "passed": True,

        "setup_level": float(
            setup_level
        ),

        "entry_level": float(
            execution_entry
        ),

        "execution_entry": float(
            execution_entry
        ),

        "sl": float(
            sl
        ),

        "tp": float(
            tp
        ),

        "risk": float(
            risk
        ),

        "rr": float(
            rr
        ),

        "confirmation": confirmation,

        "metrics": metrics
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
def execute_trade(
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

    pair = str(pair).upper().strip()
    direction = str(direction).upper().strip()

    # ========================================================
    # 1. COOLDOWN
    # ========================================================
    now = time.time()

    if (
        pair in last_execution_attempt
        and now - last_execution_attempt[pair]
        < EXECUTION_COOLDOWN_SECONDS
    ):
        logger.warning(
            f"[ORDER] Cooldown actif pour {pair}"
        )
        return None

    last_execution_attempt[pair] = now

    # ========================================================
    # 2. PARAMÈTRES INITIAUX
    # ========================================================
    expected_entry = float(entry_price)
    sl = float(stop_loss)
    tp = float(take_profit)

    metrics = dict(metrics or {})

    # Niveau structurel réel du setup.
    # Il sert uniquement au contrôle de distance d'entrée.
    setup_level = float(
        metrics.get(
            "setup_level",
            expected_entry
        )
    )

    # ========================================================
    # 3. CONTRÔLE INITIAL DU RISQUE / RR
    # ========================================================
    risk = abs(
        expected_entry - sl
    )

    if risk <= 0:
        logger.error(
            f"[ORDER] {pair} | Risque nul"
        )
        return None

    reward = abs(
        tp - expected_entry
    )

    rr = (
        reward / risk
        if risk > 0
        else 0.0
    )

    if rr < RR_MIN_EXECUTION - 0.001:
        logger.warning(
            f"[ORDER] {pair} | "
            f"RR={rr:.3f} < "
            f"{RR_MIN_EXECUTION} "
            f"avant exécution → rejet"
        )
        return None

    # ========================================================
    # 4. VÉRIFICATIONS GÉNÉRALES
    # ========================================================
    if (
        ONE_TRADE_PER_PAIR
        and has_open_trade(pair)
    ):
        logger.info(
            f"[ORDER] {pair}: trade déjà ouvert"
        )
        return None

    if (
        open_trade_count()
        >= MAX_TRADES_TOTAL
    ):
        logger.info(
            f"[ORDER] Limite trades atteinte "
            f"({MAX_TRADES_TOTAL})"
        )
        return None

    if is_maintenance_suspended():
        logger.warning(
            f"[ORDER] {pair} | "
            f"OANDA maintenance"
        )
        return None

    # ========================================================
    # 5. RÉCUPÉRATION DU PRIX MARCHÉ
    #
    # BUY  = ASK
    # SELL = BID
    # ========================================================
    pricing_data = get_price_spread(
        pair
    )

    bid = float(
        pricing_data.get(
            "bid",
            0
        ) or 0
    )

    ask = float(
        pricing_data.get(
            "ask",
            0
        ) or 0
    )

    if direction == "BUY":

        market_entry = (
            ask
            if ask > 0
            else float(
                pricing_data.get(
                    "mid",
                    0
                ) or 0
            )
        )

    elif direction == "SELL":

        market_entry = (
            bid
            if bid > 0
            else float(
                pricing_data.get(
                    "mid",
                    0
                ) or 0
            )
        )

    else:

        logger.error(
            f"[ORDER] {pair} | "
            f"Direction invalide: {direction}"
        )
        return None

    if market_entry <= 0:
        logger.warning(
            f"[ORDER] {pair} | "
            f"Prix marché indisponible → rejet"
        )
        return None

    pip = float(
        get_pip_value(pair)
    )

    if pip <= 0:
        logger.error(
            f"[ORDER] {pair} | "
            f"Valeur pip invalide"
        )
        return None

    # ========================================================
    # 6. CONTRÔLE DISTANCE SETUP -> MARCHÉ
    #
    # On compare le marché au setup structurel,
    # PAS à l'ancien prix d'évaluation.
    # ========================================================
    if pair == "XAU_USD":

        max_entry_deviation = max(
            pip * 20.0,
            setup_level * 0.00005
        )

    elif "JPY" in pair:

        max_entry_deviation = max(
            pip * 5.0,
            setup_level * 0.00003
        )

    else:

        max_entry_deviation = max(
            pip * 5.0,
            setup_level * 0.00003
        )

    setup_deviation = abs(
        market_entry - setup_level
    )

    logger.info(
        f"[ENTRY_CHECK] {pair} | "
        f"{direction} | "
        f"SETUP={setup_level:.5f} | "
        f"EXEC_EXPECTED={expected_entry:.5f} | "
        f"MARKET={market_entry:.5f} | "
        f"DEV_SETUP={setup_deviation:.5f} | "
        f"MAX={max_entry_deviation:.5f}"
    )

    # Très petite tolérance technique pour éviter
    # qu'un flottant provoque un rejet à la frontière.
    deviation_eps = (
        pip * 0.05
    )

    if (
        setup_deviation
        > max_entry_deviation + deviation_eps
    ):
        logger.warning(
            f"[ENTRY_REJECT] {pair} | "
            f"marché trop éloigné du setup | "
            f"setup={setup_level:.5f} | "
            f"market={market_entry:.5f} | "
            f"écart={setup_deviation:.5f} > "
            f"max={max_entry_deviation:.5f}"
        )
        return None

    # ========================================================
    # 7. PRIX D'ORDRE = PRIX MARCHÉ CAPTURÉ
    # ========================================================
    expected_entry = float(
        market_entry
    )

    # ========================================================
    # 8. RECALCUL DU RISQUE LIVE
    #
    # Le SL reste structurel.
    # Le prix d'entrée devient le prix ASK/BID réel.
    # ========================================================
    risk = abs(
        expected_entry - sl
    )

    if risk <= 0:
        logger.error(
            f"[ORDER] {pair} | "
            f"Risque nul après refresh prix"
        )
        return None

    # ========================================================
    # 9. TP LIVE = EXACTEMENT 2R
    #
    # IMPORTANT :
    # On ne conserve PAS l'ancien TP calculé avant
    # le refresh du prix.
    # ========================================================
    if direction == "BUY":

        tp = (
            expected_entry
            + risk * RR_MIN_EXECUTION
        )

    else:

        tp = (
            expected_entry
            - risk * RR_MIN_EXECUTION
        )

    tp = float(
        round_price(
            pair,
            tp
        )
    )

    # ========================================================
    # 10. RR APRÈS ARRONDI
    # ========================================================
    reward = abs(
        tp - expected_entry
    )

    rr = (
        reward / risk
        if risk > 0
        else 0.0
    )

    # L'arrondi peut exceptionnellement faire passer
    # le RR légèrement sous 2.0.
    # On pousse alors très légèrement le TP.
    if rr < RR_MIN_EXECUTION:

        if direction == "BUY":

            tp = float(
                round_price(
                    pair,
                    expected_entry
                    + risk * 2.01
                )
            )

        else:

            tp = float(
                round_price(
                    pair,
                    expected_entry
                    - risk * 2.01
                )
            )

        reward = abs(
            tp - expected_entry
        )

        rr = (
            reward / risk
            if risk > 0
            else 0.0
        )

    if rr < RR_MIN_EXECUTION - 0.001:

        logger.warning(
            f"[ORDER] {pair} | "
            f"RR={rr:.3f} < "
            f"{RR_MIN_EXECUTION} "
            f"après recalcul live → rejet"
        )
        return None

    # ========================================================
    # 11. MINIMUM SL ADAPTATIF
    # ========================================================
    atr_pips = float(
        metrics.get(
            "atr",
            0.0
        ) or 0.0
    )

    effective_min_sl_pips = min(
        10.0,
        atr_pips * 2.0
        if atr_pips > 0
        else 10.0
    )

    risk_pips = (
        risk / pip
    )

    eps_pips = max(
        0.05,
        atr_pips * 0.001
        if atr_pips > 0
        else 0.05
    )

    if (
        risk_pips
        < effective_min_sl_pips - eps_pips
    ):
        logger.warning(
            f"[ORDER] {pair} | "
            f"SL trop proche "
            f"({risk_pips:.2f} pips < "
            f"{effective_min_sl_pips:.2f} pips) "
            f"→ rejet"
        )
        return None

    # ========================================================
    # 12. BALANCE
    # ========================================================
    balance = get_balance()

    if balance <= 0:
        logger.error(
            f"[ORDER] {pair} | "
            f"Balance invalide"
        )
        return None

    # ========================================================
    # 13. RISK PERCENTAGE
    # ========================================================
    hour = datetime.utcnow().hour

    is_asia = (
        21 <= hour
        or hour < 7
    )

    risk_pct = (
        0.5
        if is_asia
        else RISK_PERCENTAGE
    )

    # ========================================================
    # 14. CALCUL UNITÉS
    # ========================================================
    units = calculate_units(
        pair,
        expected_entry,
        sl,
        balance,
        risk_pct
    )

    if units <= 0:
        logger.error(
            f"[ORDER] {pair} | "
            f"Units invalides: {units}"
        )
        return None

    # ========================================================
    # 15. VÉRIFICATION MARGE
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
                f"[RISK] {pair} | "
                f"Marge insuffisante"
            )
            return None

    # ========================================================
    # 16. RECHECK FINAL DES PARAMÈTRES
    #
    # Sécurité avant envoi OANDA.
    # ========================================================
    final_risk = abs(
        expected_entry - sl
    )

    final_reward = abs(
        tp - expected_entry
    )

    final_rr = (
        final_reward / final_risk
        if final_risk > 0
        else 0.0
    )

    if (
        final_risk <= 0
        or final_rr < RR_MIN_EXECUTION - 0.001
    ):
        logger.warning(
            f"[ORDER] {pair} | "
            f"Validation finale échouée | "
            f"risk={final_risk:.5f} | "
            f"RR={final_rr:.3f}"
        )
        return None

    # ========================================================
    # 17. MARKET ORDER
    # ========================================================
    signed_units = (
        units
        if direction == "BUY"
        else -units
    )

    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": pair,
            "units": str(
                int(signed_units)
            ),
            "positionFill": "DEFAULT",

            "stopLossOnFill": {
                "price": round_price(
                    pair,
                    sl
                ),
                "timeInForce": "GTC"
            },

            "takeProfitOnFill": {
                "price": round_price(
                    pair,
                    tp
                ),
                "timeInForce": "GTC"
            }
        }
    }

    logger.info(
        f"[ORDER_EXPECTED] {pair} | "
        f"{direction} | "
        f"SETUP={setup_level:.5f} | "
        f"ENTRY={expected_entry:.5f} | "
        f"SL={sl:.5f} | "
        f"TP={tp:.5f} | "
        f"RISK={final_risk:.5f} | "
        f"RR={final_rr:.3f} | "
        f"UNITS={units}"
    )

    if not EXECUTE_TRADES:
        logger.info(
            "[ORDER] EXECUTE_TRADES=false"
        )
        return "SIMULATION"

    # ========================================================
    # 18. ENVOI OANDA
    # ========================================================
    try:

        api = v88_client()

        r = orders.OrderCreate(
            accountID=OANDA_ACCOUNT_ID,
            data=order_data
        )

        api.request(r)

        resp = r.response

        if resp.get(
            "orderRejectTransaction"
        ):

            reject = resp[
                "orderRejectTransaction"
            ]

            logger.error(
                f"[ORDER] REJECT {pair}: "
                f"{reject.get('rejectReason')}"
            )

            return None

        # ====================================================
        # 19. RÉCUPÉRATION DU FILL
        # ====================================================
        fill = resp.get(
            "orderFillTransaction",
            {}
        )

        trade_id = None
        actual_entry = None

        if fill.get(
            "tradeOpened"
        ):

            trade_id = (
                fill[
                    "tradeOpened"
                ].get(
                    "tradeID"
                )
            )

        if fill.get(
            "price"
        ) is not None:

            try:

                actual_entry = float(
                    fill["price"]
                )

            except Exception:

                actual_entry = None

        # ====================================================
        # 20. FALLBACK TRADE OUVERT
        # ====================================================
        if not trade_id:

            time.sleep(1)

            open_trades = get_open_trades(
                force_refresh=True
            )

            candidates = []

            for t in open_trades:

                if (
                    t.get("instrument")
                    != pair
                ):
                    continue

                current_units = float(
                    t.get(
                        "currentUnits",
                        0
                    )
                )

                if current_units == 0:
                    continue

                t_direction = (
                    "BUY"
                    if current_units > 0
                    else "SELL"
                )

                if (
                    t_direction
                    != direction
                ):
                    continue

                t_entry = float(
                    t.get(
                        "price",
                        0
                    )
                )

                if t_entry <= 0:
                    continue

                candidates.append(
                    (
                        abs(
                            t_entry
                            - expected_entry
                        ),
                        t
                    )
                )

            if candidates:

                candidates.sort(
                    key=lambda x: x[0]
                )

                _, best_trade = (
                    candidates[0]
                )

                trade_id = (
                    best_trade.get(
                        "id"
                    )
                )

                try:
                    actual_entry = float(
                        best_trade.get(
                            "price"
                        )
                    )
                except Exception:
                    actual_entry = None

        if not trade_id:

            logger.error(
                f"[ORDER] {pair} | "
                f"Trade non confirmé"
            )

            return None

        # ====================================================
        # 21. FALLBACK PRIX FILL
        # ====================================================
        if actual_entry is None:

            try:

                trade_details = (
                    get_trade_details(
                        str(trade_id)
                    )
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
                    f"Impossible récupérer "
                    f"fill réel: {e}"
                )

        if actual_entry is None:
            actual_entry = expected_entry

        actual_entry = float(
            actual_entry
        )

        # ====================================================
        # 22. RR RÉEL APRÈS FILL
        # ====================================================
        fill_deviation = abs(
            actual_entry
            - expected_entry
        )

        risk_real = abs(
            actual_entry
            - sl
        )

        reward_real = abs(
            tp
            - actual_entry
        )

        rr_real = (
            reward_real / risk_real
            if risk_real > 0
            else 0.0
        )

        slippage_pips = (
            (
                actual_entry
                - expected_entry
            ) / pip
            if pip > 0
            else 0.0
        )

        logger.info(
            f"[FILL_REAL] {pair} | "
            f"ID={trade_id} | "
            f"ENTRY_EXPECTED={expected_entry:.5f} | "
            f"ENTRY_FILLED={actual_entry:.5f} | "
            f"SLIPPAGE={slippage_pips:+.2f} pips | "
            f"RR_REAL={rr_real:.3f}"
        )

        # ====================================================
        # 23. PROTECTION RR APRÈS FILL
        # ====================================================
        if (
            rr_real
            < RR_MIN_EXECUTION - 0.001
        ):

            logger.error(
                f"[ORDER_ABORT] {pair} | "
                f"RR réel {rr_real:.3f} < "
                f"{RR_MIN_EXECUTION}"
            )

            try:

                close_data = {
                    "units": "ALL"
                }

                close_request = (
                    trades.TradeClose(
                        accountID=OANDA_ACCOUNT_ID,
                        tradeID=str(
                            trade_id
                        ),
                        data=close_data
                    )
                )

                api.request(
                    close_request
                )

                logger.error(
                    f"[ORDER_ABORT] {pair} | "
                    f"Trade {trade_id} "
                    f"fermé immédiatement"
                )

            except Exception as close_error:

                logger.critical(
                    f"[ORDER_ABORT] {pair} | "
                    f"IMPOSSIBLE DE FERMER "
                    f"LE TRADE {trade_id}: "
                    f"{close_error}"
                )

            return None

        # ====================================================
        # 24. LOG SLIPPAGE
        # ====================================================
        if (
            fill_deviation
            > max_entry_deviation
        ):

            logger.warning(
                f"[SLIPPAGE] {pair} | "
                f"Fill hors tolérance | "
                f"écart={fill_deviation:.5f} | "
                f"max={max_entry_deviation:.5f}"
            )

        # ====================================================
        # 25. ENREGISTREMENT TRADE
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

        open_trade_details[
            str(trade_id)
        ] = {
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
            f"RR={rr_real:.3f}"
        )

        return str(
            trade_id
        )

    except Exception as e:

        logger.error(
            f"[ORDER] Erreur {pair}: {e}",
            exc_info=True
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

    # Diagnostic compact de la structure HTF
    def _struct_brief(df, label):
        highs, lows = detect_swing_points(df, 5)

        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1]["price"] > highs[-2]["price"]
            hl = lows[-1]["price"] > lows[-2]["price"]
            lh = highs[-1]["price"] < highs[-2]["price"]
            ll = lows[-1]["price"] < lows[-2]["price"]

            if hh and hl:
                return f"{label} BULLISH"

            if lh and ll:
                return f"{label} BEARISH"

            return (
                f"{label} MIXED "
                f"(HH={hh},HL={hl},LH={lh},LL={ll})"
            )

        return f"{label} INDETERMINE"

    for pair in PAIR_LIST:

        # ========================================================
        # 1. UN SEUL TRADE PAR PAIRE
        # ========================================================
        if has_open_trade(pair):
            logger.info(f"[INFO] {pair}: trade déjà ouvert")
            continue

        try:
            # ====================================================
            # 2. CHARGEMENT DES DONNÉES
            # ====================================================
            df_h4 = get_candles(
                api,
                pair,
                GRANULARITY_H4,
                300
            )

            df_h1 = get_candles(
                api,
                pair,
                GRANULARITY_H1,
                200
            )

            df_m15 = get_candles(
                api,
                pair,
                GRANULARITY_M15,
                250
            )

            if any(
                df.empty
                for df in [df_h4, df_h1, df_m15]
            ):
                logger.info(
                    f"{pair} | Données insuffisantes"
                )
                continue

            # Prix M15 utilisé pour le contexte / qualification
            current_price = float(
                df_m15["close"].iloc[-1]
            )

            # ====================================================
            # 3. BIAIS H4 SOUVERAIN
            # ====================================================
            bias = get_directional_bias(
                df_h4,
                df_h1
            )

            # ====================================================
            # 4. DIAGNOSTIC SI NEUTRE
            # ====================================================
            if bias == "NEUTRAL":
                logger.info(
                    f"{pair} | BIAS_DIAG: "
                    f"H4={_struct_brief(df_h4, 'H4')} | "
                    f"H1={_struct_brief(df_h1, 'H1')} | "
                    f"-> NEUTRAL"
                )
                continue

            # ====================================================
            # 5. DÉTECTION DES SETUPS M15
            # ====================================================
            setups = detect_setups(
                pair,
                df_m15,
                df_h1,
                bias
            )

            if not setups:
                logger.info(
                    f"{pair} | Aucun setup {bias} détecté"
                )
                continue

            setup_summary = ", ".join(
                [
                    (
                        f"{s.get('type')}"
                        f"@"
                        f"{round(float(s.get('entry_level', 0)), 5)}"
                    )
                    for s in setups[:3]
                ]
            )

            logger.info(
                f"{pair} | SETUPS_FOUND: {len(setups)} | "
                f"Types: {setup_summary}"
                f"{' ...' if len(setups) > 3 else ''}"
            )

            # ====================================================
            # 6. ÉVALUATION DES SETUPS
            # ====================================================
            valid_trades = []

            for entry in setups:

                result = evaluate_setup(
                    pair=pair,
                    direction=bias,
                    entry=entry,
                    df_m15=df_m15,
                    df_h1=df_h1,
                    current_price=current_price
                )

                if not result.get("passed"):
                    logger.info(
                        f"[REJECT] {pair} {bias} "
                        f"{entry.get('type')} : "
                        f"{result.get('reason', 'raison inconnue')}"
                    )
                    continue

                # -----------------------------------------------
                # setup_level = niveau structurel du setup
                # execution_entry = prix utilisable pour l'ordre
                # -----------------------------------------------
                setup_level = float(
                    result.get(
                        "setup_level",
                        entry.get("entry_level")
                    )
                )

                execution_entry = float(
                    result["execution_entry"]
                )

                valid_trades.append(
                    {
                        "entry": entry,
                        "setup_level": setup_level,
                        "execution_entry": execution_entry,
                        "sl": float(result["sl"]),
                        "tp": float(result["tp"]),
                        "risk": float(result["risk"]),
                        "rr": float(result["rr"]),
                        "confirmation": result.get(
                            "confirmation",
                            {}
                        ),
                        "metrics": result.get(
                            "metrics",
                            {}
                        ),
                    }
                )

            # ====================================================
            # 7. AUCUN TRADE VALIDE
            # ====================================================
            if not valid_trades:
                continue

            # ====================================================
            # 8. TRI :
            #    - RR prioritaire
            #    - puis proximité du setup
            # ====================================================
            valid_trades.sort(
                key=lambda x: (
                    x["rr"],
                    -abs(
                        x["execution_entry"]
                        - x["setup_level"]
                    )
                ),
                reverse=True
            )

            best = valid_trades[0]

            # ====================================================
            # 9. RÉCUPÉRATION DES VALEURS
            # ====================================================
            setup_level = float(
                best["setup_level"]
            )

            execution_entry = float(
                best["execution_entry"]
            )

            stop_loss = float(
                best["sl"]
            )

            take_profit = float(
                best["tp"]
            )

            rr = float(
                best["rr"]
            )

            metrics = dict(
                best.get("metrics") or {}
            )

            setup_type = best["entry"].get(
                "type",
                "UNKNOWN"
            )

            # On conserve explicitement le niveau structurel
            metrics["setup_level"] = setup_level
            metrics["execution_entry"] = execution_entry

            # ====================================================
            # 10. LOG AVANT EXÉCUTION
            # ====================================================
            logger.info(
                f"[TRADE] {pair} {bias} "
                f"| SETUP={setup_type} "
                f"| SETUP_LEVEL={setup_level:.5f} "
                f"| EXEC_ENTRY={execution_entry:.5f} "
                f"| SL={stop_loss:.5f} "
                f"| TP={take_profit:.5f} "
                f"| RR={rr:.2f}"
            )

            # ====================================================
            # 11. EXÉCUTION
            #
            # IMPORTANT :
            # On ne transmet plus le setup_level comme
            # prix d'entrée de l'ordre.
            # ====================================================
            trade_id = execute_trade(
                pair=pair,
                direction=bias,
                entry_price=execution_entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                score=0,
                entry_type=setup_type,
                eqs=0,
                setup_type=setup_type,
                metrics=metrics
            )

            # ====================================================
            # 12. RÉSULTAT
            # ====================================================
            if trade_id:
                logger.info(
                    f"✅ {pair} trade exécuté "
                    f"(ID {trade_id})"
                )

                send_telegram(
                    pair,
                    bias,
                    execution_entry,
                    stop_loss,
                    take_profit,
                    rr,
                    setup_type
                )
            else:
                logger.error(
                    f"❌ {pair} échec exécution"
                )

        except Exception as e:
            logger.error(
                f"💥 Erreur sur {pair}: {e}",
                exc_info=True
            )

    # ============================================================
    # 13. SUMMARY
    # ============================================================
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
    logger.info("🚀 Démarrage du Bot 2R Strict - Version Optimisée v140")
    logger.info("✅ SL structurel | TP = 2R (immuable) | RR ≥ 2.0 avant ordre")
    logger.info("✅ Structure H1 assouplie (2/3) + tolérance retracement H4 fort")
    logger.info("✅ Distance max 2.0 ATR | Confirmation rejet OU micro-break")
    logger.info("✅ SL structurel exploitable ≤ 2×ATR | fallback ATR 1.5× | TP 2R")
    logger.info(f"✅ MAX TRADES: {MAX_TRADES_TOTAL}")
    if DEMO_MODE:
        logger.info("🔬 MODE DEMO ACTIVÉ")
    if DEBUG_MODE:
        logger.info("🔍 MODE DEBUG ACTIVÉ")

    SIGNAL_SCAN_INTERVAL = 900  # 15 min
    last_signal_scan = time.time() - SIGNAL_SCAN_INTERVAL
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
