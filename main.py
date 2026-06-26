import os
import time
import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime, time as dtime
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
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")  # practice ou live

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

# Sécurité : pour envoyer de vrais ordres, mets EXECUTE_TRADES=true dans Render/.env
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

client = oandapyV20.API(access_token=OANDA_API_KEY, environment=OANDA_ENVIRONMENT)

# =========================================================
# STRATEGIE V66 - SEQUENCE SNIPER
# Impulsion -> zone unique -> retracement -> rejet -> cassure locale
# =========================================================
PAIR_LIST = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD",
    "AUD_USD", "AUD_CAD", "AUD_JPY", "XAU_USD", "GBP_JPY"
]

SESSION_START = dtime(7, 0)     # Londres UTC
SESSION_END = dtime(21, 0)      # Fin NY UTC
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

# Plus strict = moins de trades, meilleur winrate attendu
MIN_SEQUENCE_SCORE = 8
MIN_FVG_ATR_RATIO = 0.25          # FVG trop petit = rejet
MAX_ZONE_AGE_H4_CANDLES = 8       # FVG récent uniquement
RETRACE_TOLERANCE_ATR = 0.15      # tolérance autour de la zone
MIN_REJECTION_BODY_RATIO = 0.45
MIN_IMPULSE_ATR_MULTIPLIER = 1.20
MAX_SPREAD_PIPS = {
    "XAU_USD": 35.0,
    "GBP_JPY": 4.0,
    "AUD_JPY": 3.5,
    "USD_JPY": 2.5,
    "DEFAULT": 2.0,
}

PAIR_DECIMALS = {
    "XAU_USD": 2,
    "USD_JPY": 3,
    "GBP_JPY": 3,
    "AUD_JPY": 3,
    "DEFAULT": 5,
}

PIP_VALUE = {
    "XAU_USD": 0.01,
    "USD_JPY": 0.01,
    "GBP_JPY": 0.01,
    "AUD_JPY": 0.01,
    "DEFAULT": 0.0001,
}

MIN_UNITS = {
    "XAU_USD": 1,
    "DEFAULT": 1000,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("oanda_v66_sequence_sniper.log")],
)
logger = logging.getLogger("OANDA-V66-Sequence-Sniper")

last_signal_key = set()


@dataclass
class FVGZone:
    direction: str          # BUY ou SELL
    low: float
    high: float
    midpoint: float
    age: int
    size: float
    impulse_strength: float


# =========================================================
# OUTILS
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
        logger.info("Email non configuré, notification ignorée.")
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
        logger.info(f"📧 Email envoyé: {subject}")
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
    # fallback simple via dernière M1 bid/ask indisponible ici: utiliser mid et spread filtré par historique
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
    df = df.copy()
    df["ema50"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)
    return df


def market_bias(df_h4: pd.DataFrame, df_h1: pd.DataFrame) -> Optional[str]:
    h4 = add_indicators(df_h4)
    h1 = add_indicators(df_h1)
    if len(h4) < 210 or len(h1) < 210:
        return None

    h4_last = h4.iloc[-1]
    h1_last = h1.iloc[-1]
    h4_slope = h4["ema50"].iloc[-1] - h4["ema50"].iloc[-6]
    h1_slope = h1["ema50"].iloc[-1] - h1["ema50"].iloc[-6]

    buy = (
        h4_last.close > h4_last.ema50 > h4_last.ema200 and
        h1_last.close > h1_last.ema50 and
        h4_slope > 0 and h1_slope > 0
    )
    sell = (
        h4_last.close < h4_last.ema50 < h4_last.ema200 and
        h1_last.close < h1_last.ema50 and
        h4_slope < 0 and h1_slope < 0
    )
    if buy:
        return "BUY"
    if sell:
        return "SELL"
    return None


def detect_impulse_and_fvg(df_h4: pd.DataFrame, bias: str) -> Optional[FVGZone]:
    """Garde une seule zone : le FVG récent le plus propre dans le sens du bias."""
    df = add_indicators(df_h4)
    candidates: List[FVGZone] = []
    if len(df) < 30:
        return None

    start = max(2, len(df) - MAX_ZONE_AGE_H4_CANDLES - 2)
    for i in range(start, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        nxt = df.iloc[i + 1]
        atr = float(df["atr"].iloc[i]) if not np.isnan(df["atr"].iloc[i]) else 0
        if atr <= 0:
            continue

        body = abs(curr.close - curr.open)
        impulse_strength = body / atr
        if impulse_strength < MIN_IMPULSE_ATR_MULTIPLIER:
            continue

        # Bullish FVG: low de la bougie suivante > high précédente
        if bias == "BUY" and nxt.low > prev.high:
            low, high = prev.high, nxt.low
            size = high - low
            if size / atr < MIN_FVG_ATR_RATIO:
                continue
            candidates.append(FVGZone("BUY", low, high, (low + high) / 2, len(df) - 1 - i, size, impulse_strength))

        # Bearish FVG: high de la bougie suivante < low précédente
        if bias == "SELL" and nxt.high < prev.low:
            low, high = nxt.high, prev.low
            size = high - low
            if size / atr < MIN_FVG_ATR_RATIO:
                continue
            candidates.append(FVGZone("SELL", low, high, (low + high) / 2, len(df) - 1 - i, size, impulse_strength))

    if not candidates:
        return None

    # Meilleur = récent + impulsion forte + zone large
    candidates.sort(key=lambda z: (z.age, -z.impulse_strength, -z.size))
    return candidates[0]


def price_retraced_to_zone(df_m15: pd.DataFrame, zone: FVGZone) -> bool:
    df = add_indicators(df_m15)
    atr = float(df["atr"].iloc[-1]) if not np.isnan(df["atr"].iloc[-1]) else 0
    tolerance = max(atr * RETRACE_TOLERANCE_ATR, pip_value("EUR_USD"))
    last = df.iloc[-1]
    touched = last.low <= zone.high + tolerance and last.high >= zone.low - tolerance
    return bool(touched)


def rejection_confirmed(pair: str, df_m15: pd.DataFrame, zone: FVGZone) -> Tuple[bool, str]:
    df = add_indicators(df_m15)
    if len(df) < 10:
        return False, "données M15 insuffisantes"

    last = df.iloc[-1]
    prev = df.iloc[-2]
    total = max(last.high - last.low, 1e-9)
    body = abs(last.close - last.open)
    body_ratio = body / total
    upper_wick = last.high - max(last.close, last.open)
    lower_wick = min(last.close, last.open) - last.low

    if zone.direction == "BUY":
        if last.close <= last.open:
            return False, "bougie M15 non haussière"
        if body_ratio < MIN_REJECTION_BODY_RATIO:
            return False, f"corps trop faible {body_ratio:.2f}"
        if lower_wick < upper_wick:
            return False, "pas de rejet bas clair"
        if last.close <= prev.high:
            return False, "pas de cassure locale M15"
        if df["rsi"].iloc[-1] < 48:
            return False, f"RSI M15 faible {df['rsi'].iloc[-1]:.1f}"
        return True, "rejet BUY + cassure locale confirmés"

    if zone.direction == "SELL":
        if last.close >= last.open:
            return False, "bougie M15 non baissière"
        if body_ratio < MIN_REJECTION_BODY_RATIO:
            return False, f"corps trop faible {body_ratio:.2f}"
        if upper_wick < lower_wick:
            return False, "pas de rejet haut clair"
        if last.close >= prev.low:
            return False, "pas de cassure locale M15"
        if df["rsi"].iloc[-1] > 52:
            return False, f"RSI M15 trop haut {df['rsi'].iloc[-1]:.1f}"
        return True, "rejet SELL + cassure locale confirmés"

    return False, "direction inconnue"


def spread_ok(pair: str) -> Tuple[bool, str, Optional[float]]:
    prices = fetch_pricing_spread(pair)
    if not prices:
        return True, "spread indisponible, filtre ignoré", None
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

    # Approximation OANDA : pour compte USD et paires non USD en devise de contrepartie,
    # c'est volontairement conservateur. Pour du démo/live, vérifier marge OANDA.
    units_float = risk_usd / distance
    min_units = MIN_UNITS.get(pair, MIN_UNITS["DEFAULT"])

    if pair == "XAU_USD":
        units = int(max(round(units_float, 0), min_units))
    else:
        units = int(max(round(units_float / min_units) * min_units, min_units))

    return units


def build_trade_levels(pair: str, zone: FVGZone, entry: float, df_m15: pd.DataFrame) -> Tuple[float, float]:
    df = add_indicators(df_m15)
    atr = float(df["atr"].iloc[-1]) if not np.isnan(df["atr"].iloc[-1]) else abs(zone.high - zone.low)
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
        logger.info(f"🚫 {pair}: trade déjà ouvert, aucun nouvel ordre.")
        return None
    if open_trade_count() >= MAX_TRADES_TOTAL:
        logger.info(f"🚫 Limite trades ouverts atteinte ({MAX_TRADES_TOTAL}).")
        return None

    balance = get_account_balance()
    units = calculate_units(pair, entry, stop, balance)
    if units <= 0:
        logger.warning(f"❌ {pair}: taille position invalide.")
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

    logger.info(
        f"💖 SIGNAL V66 {pair} {direction} | entry≈{entry:.5f} SL={stop:.5f} TP={tp:.5f} "
        f"RR={RISK_REWARD:.2f} score={score}/10 units={units} | {reason}"
    )

    if not EXECUTE_TRADES:
        logger.info("🧪 EXECUTE_TRADES=false : ordre non envoyé à OANDA.")
        return "SIMULATION"

    try:
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        resp = client.request(r)
        trade_id = resp.get("orderFillTransaction", {}).get("id") or resp.get("orderCreateTransaction", {}).get("id")
        logger.info(f"✅ ORDRE RÉEL ENVOYÉ OANDA {pair} | ID={trade_id}")
        send_email(
            f"💖 V66 OANDA {pair} {direction}",
            f"Trade réel envoyé\nPaire: {pair}\nDirection: {direction}\nEntrée: {entry}\nSL: {stop}\nTP: {tp}\nScore: {score}/10\nRaison: {reason}",
        )
        return trade_id
    except Exception as exc:
        logger.error(f"❌ Erreur ordre OANDA {pair}: {exc}")
        return None


def analyze_pair(pair: str) -> None:
    logger.info(f"\n🔎 V66 analyse {pair}")
    try:
        spread_valid, spread_msg, _ = spread_ok(pair)
        logger.info(f"{pair} · {spread_msg}")
        if not spread_valid:
            return

        df_h4 = fetch_candles(pair, "H4", 260)
        df_h1 = fetch_candles(pair, "H1", 260)
        df_m15 = fetch_candles(pair, "M15", 120)
        if df_h4.empty or df_h1.empty or df_m15.empty:
            logger.warning(f"{pair}: données insuffisantes.")
            return

        bias = market_bias(df_h4, df_h1)
        if not bias:
            logger.info(f"⛔ {pair}: pas d'alignement H4/H1 propre.")
            return
        logger.info(f"📍 {pair}: bias HTF = {bias}")

        zone = detect_impulse_and_fvg(df_h4, bias)
        if not zone:
            logger.info(f"⛔ {pair}: aucune impulsion + FVG HTF propre.")
            return
        logger.info(
            f"📦 {pair}: zone unique {zone.direction} [{zone.low:.5f}-{zone.high:.5f}] "
            f"mid={zone.midpoint:.5f}, âge={zone.age} H4, impulse={zone.impulse_strength:.2f} ATR"
        )

        if not price_retraced_to_zone(df_m15, zone):
            logger.info(f"⏳ {pair}: prix pas encore revenu dans la zone.")
            return

        ok_reject, reject_reason = rejection_confirmed(pair, df_m15, zone)
        if not ok_reject:
            logger.info(f"⛔ {pair}: pas de confirmation M15: {reject_reason}")
            return

        current_price = df_m15["close"].iloc[-1]
        stop, tp = build_trade_levels(pair, zone, current_price, df_m15)
        risk = abs(current_price - stop)
        reward = abs(tp - current_price)
        rr = reward / risk if risk > 0 else 0
        if rr < 1.8:
            logger.info(f"⛔ {pair}: RR insuffisant {rr:.2f}")
            return

        score = 0
        score += 2  # alignement HTF
        score += 2  # impulsion
        score += 2  # FVG unique
        score += 2  # retracement
        score += 2  # rejet + cassure M15
        if zone.impulse_strength > 1.7:
            score += 1
        if zone.age <= 3:
            score += 1

        if score < MIN_SEQUENCE_SCORE:
            logger.info(f"⛔ {pair}: score insuffisant {score}/{MIN_SEQUENCE_SCORE}")
            return

        key = (pair, zone.direction, round(zone.midpoint, decimals(pair)))
        if key in last_signal_key:
            logger.info(f"🔁 {pair}: signal déjà traité sur cette zone.")
            return
        last_signal_key.add(key)

        place_trade(pair, zone.direction, current_price, stop, tp, score, reject_reason)

    except Exception as exc:
        logger.exception(f"Erreur analyse {pair}: {exc}")


def manage_notifications_on_closed_trades() -> None:
    # V66 simple: OANDA gère SL/TP. Ajout possible ensuite pour email TP/SL via transaction history.
    return


def main() -> None:
    logger.info("🚀 Démarrage OANDA V66 Sequence Sniper")
    logger.info(f"Compte: {OANDA_ACCOUNT_ID} | env={OANDA_ENVIRONMENT} | EXECUTE_TRADES={EXECUTE_TRADES}")
    balance = get_account_balance()
    logger.info(f"💰 Solde compte: {balance:.2f}")
    logger.info(f"Paires: {', '.join(PAIR_LIST)}")

    while True:
        now = datetime.utcnow().time()
        logger.info(f"\n===== Nouvelle boucle {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC =====")
        try:
            if SESSION_START <= now <= SESSION_END:
                logger.info("⏱ Session Londres/NY active")
                for pair in PAIR_LIST:
                    analyze_pair(pair)
                    time.sleep(0.2)
                manage_notifications_on_closed_trades()
                time.sleep(LOOP_SECONDS)
            else:
                logger.info("🛑 Hors session trading. Attente 5 min.")
                time.sleep(300)
        except Exception as exc:
            logger.exception(f"Erreur boucle principale: {exc}")
            time.sleep(60)


if __name__ == "__main__":
    main()
