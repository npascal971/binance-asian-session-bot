#!/usr/bin/env python3
"""
==============================================================================
 BOT VWAP OR — VWAP mensuel + 1 écart-type, or H1, acheteur seul, TP 2R
==============================================================================

 LA RÈGLE (ton script 04_VWAP_OR_SOLO, à l'identique)
   - VWAP pondéré par le volume, remis à zéro le 1er de chaque mois (heure UTC)
   - bande haute = VWAP + 1 écart-type
   - SIGNAL : la bougie H1 précédente clôture sous (ou sur) la bande, la bougie
     qui vient de se terminer clôture AU-DESSUS ; au moins 3 bougies dans le mois
   - ACHAT à l'ouverture suivante (le bot agit dès la clôture de la bougie signal)
   - STOP = plus bas des 10 dernières bougies (bougie signal comprise)
   - OBJECTIF = 2 fois le risque (2R)
   - une seule position à la fois

 CE QUE DISENT LES TESTS (vwap_robust.py, or H1 2010 -> 2026)
   - avant 2021 (jamais servi à régler) : 371 trades, +0,18 R/trade, t = +2,39
   - 2021-2026 : 227 trades, +0,39 R/trade, t = +3,92
   - 98 % des 45 réglages voisins gagnent avant 2021 ; 9 années sur 11 gagnantes
   - tient avec frais doublés et entrée à l'ouverture suivante
   - la VENTE au même signal perd (t = -2,3) : le signal a un sens

 CE QUI RESTE FRAGILE — honnêtement
   - t = 2,39 hors réglage : en dessous de notre barre t > 3
   - ne marche QUE sur l'or (argent, Nasdaq, S&P, EUR/USD, USD/JPY : ~0)
   - dépend du volume et de la remise à zéro MENSUELLE (hebdo : rien)
   - avec frais triplés, l'avantage avant 2021 n'est plus mesurable

 ENVIRON 30 À 45 TRADES PAR AN. COMPTE DÉMO UNIQUEMENT.

 Ce bot ne touche QUE ses propres trades (commentaire "vwap|..."). Le bot
 Donchian (commentaire "trend|...") peut tourner sur le même compte.
==============================================================================
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import oandapyV20
from oandapyV20.endpoints import accounts, instruments, orders, pricing, trades

load_dotenv()


def _env_float(n, d):
    try:
        return float(os.getenv(n, d))
    except (TypeError, ValueError):
        return d


def _env_bool(n, d):
    v = os.getenv(n)
    return d if v is None else v.strip().lower() in ("1", "true", "yes", "oui", "on")


def utcnow():
    return datetime.now(timezone.utc)


# ===========================================================================
# CONFIGURATION — la règle est FIGÉE : c'est ce qui a été testé
# ===========================================================================
PAIR = "XAU_USD"
SIGMA = 1.0
SL_BARS = 10
RR = 2.0
MIN_AGE = 3
CANDLES = 1000                 # > 1 mois de bougies H1 : le mois en cours est complet

OANDA_ACCOUNT_ID = (os.getenv("VWAP_ACCOUNT_ID") or os.getenv("OANDA_ACCOUNT_ID", "")).strip()
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")
EXECUTE_TRADES = _env_bool("EXECUTE_TRADES", True)
RISK_PCT = _env_float("VWAP_RISK_PCT", 0.5)
MAX_SPREAD = _env_float("VWAP_MAX_SPREAD", 1.0)           # en dollars ; au-delà on attend
MAX_MARGIN_USAGE_PCT = _env_float("MAX_MARGIN_USAGE_PCT", 20.0)
MAX_UNITS = 500
LATE_MINUTES = 20              # signal vu trop tard (bot redémarré) -> on ne court pas après
CHECK_SECONDS = 30
STATE_FILE = os.getenv("VWAP_STATE_FILE", "/data/vwap_state.json")
TAG = "vwap"

logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
logger = logging.getLogger("VwapGold")
for noisy in ("urllib3", "requests", "oandapyV20"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


def short_error(e, limit=200):
    s = str(e)
    return s if len(s) <= limit else f"{s[:limit]}... [tronqué {len(s)}]"


# ===========================================================================
# OANDA
# ===========================================================================
_API = None


def api():
    global _API
    if _API is None:
        token = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_ACCESS_TOKEN")
        if not token:
            raise RuntimeError("OANDA_API_KEY manquant")
        _API = oandapyV20.API(access_token=token, environment=OANDA_ENVIRONMENT,
                              request_params={"timeout": 15})
    return _API


def read(req, retries=3):
    """Lecture avec nouvel essai. JAMAIS utilisée pour un ordre (risque de doublon)."""
    for k in range(retries + 1):
        try:
            api().request(req)
            return req.response
        except Exception:
            if k == retries:
                raise
            time.sleep(1.5 * (k + 1))


def get_candles():
    r = instruments.InstrumentsCandles(instrument=PAIR, params={
        "granularity": "H1", "price": "M", "count": CANDLES})
    rows, idx = [], []
    for c in read(r).get("candles", []):
        if not c.get("complete"):
            continue                              # jamais la bougie en formation
        m = c["mid"]
        idx.append(pd.Timestamp(c["time"]).tz_convert("UTC") if pd.Timestamp(c["time"]).tzinfo
                   else pd.Timestamp(c["time"], tz="UTC"))
        rows.append({"open": float(m["o"]), "high": float(m["h"]), "low": float(m["l"]),
                     "close": float(m["c"]), "volume": float(c.get("volume", 0))})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def get_price():
    r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params={"instruments": PAIR})
    p = (read(r).get("prices") or [{}])[0]
    bid = float((p.get("bids") or [{}])[0].get("price", 0) or 0)
    ask = float((p.get("asks") or [{}])[0].get("price", 0) or 0)
    return bid, ask, bool(p.get("tradeable", False))


def get_account():
    return read(accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)).get("account", {})


def get_open_trades():
    return read(trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)).get("trades", [])


def px(p):
    return f"{float(p):.3f}"


# ===========================================================================
# LA RÈGLE — même calcul que vwap_robust.py / pine_backtest.py
# ===========================================================================
def upper_band(df):
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    v = df["volume"].fillna(0)
    g = pd.Series(df.index.year * 100 + df.index.month, index=df.index)
    sv = v.groupby(g).cumsum()
    sp = (v * hlc3).groupby(g).cumsum()
    sp2 = (v * hlc3 * hlc3).groupby(g).cumsum()
    age = pd.Series(1, index=df.index).groupby(g).cumsum()
    vw = sp / sv.replace(0, np.nan)
    sd = np.sqrt((sp2 / sv.replace(0, np.nan) - vw * vw).clip(lower=0))
    return (vw + SIGMA * sd).where(sd > 0).values, age.values, v.values


def evaluate(df):
    """Sur la dernière bougie TERMINÉE : (signal, stop, bande, clôture)."""
    up, age, v = upper_band(df)
    c, i = df["close"].values, len(df) - 1
    sig = bool(c[i - 1] <= up[i - 1] and c[i] > up[i] and age[i] >= MIN_AGE and v[i] > 0)
    sl = float(df["low"].values[-SL_BARS:].min())
    return sig, sl, float(up[i]), float(c[i])


# ===========================================================================
# ÉTAT ET JOURNAL
# ===========================================================================
def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    try:
        os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        logger.warning(f"[STATE] non enregistré : {short_error(e, 100)}")


def log_event(env_name, tag, rec):
    line = json.dumps(rec, default=str, ensure_ascii=False)
    logger.info(f"[{tag}] {line}")
    path = os.getenv(env_name)
    if path:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.warning(f"[{tag}] fichier {path} : {short_error(e, 80)}")


def our_trades(all_trades):
    out = []
    for t in all_trades:
        com = str((t.get("clientExtensions") or {}).get("comment", ""))
        parts = com.split("|")
        if len(parts) >= 3 and parts[0] == TAG and parts[2].startswith("sl0="):
            out.append({"id": str(t["id"]), "entry": float(t.get("price", 0) or 0),
                        "sl0": float(parts[2][4:]), "opened_at": t.get("openTime"),
                        "has_tp": bool(t.get("takeProfitOrder")),
                        "has_sl": bool(t.get("stopLossOrder"))})
    return out


_KNOWN = {}


def track_closures(mine):
    ids = {t["id"] for t in mine}
    for t in mine:
        _KNOWN.setdefault(t["id"], t)
    for tid in list(_KNOWN):
        if tid in ids:
            continue
        info = _KNOWN[tid]
        try:
            td = read(trades.TradeDetails(accountID=OANDA_ACCOUNT_ID, tradeID=tid)).get("trade", {})
            if td.get("state") != "CLOSED":
                continue
        except Exception:
            continue
        close = float(td.get("averageClosePrice", 0) or 0)
        risk = info["entry"] - info["sl0"]
        log_event("JOURNAL_FILE", "JOURNAL", {
            "id": tid, "pair": PAIR, "dir": "BUY", "entry": info["entry"], "exit": close,
            "sl0": info["sl0"], "pl": round(float(td.get("realizedPL", 0) or 0), 2),
            "r": round((close - info["entry"]) / risk, 2) if risk > 0 and close else None,
            "opened_at": info.get("opened_at"), "closed_at": td.get("closeTime")})
        del _KNOWN[tid]


# ===========================================================================
# ACTIONS
# ===========================================================================
def units_for(entry, sl, acct):
    balance = float(acct.get("balance", 0) or 0)
    dist = entry - sl
    if balance <= 0 or dist <= 0:
        return 0
    u = min(int(balance * RISK_PCT / 100.0 / dist), MAX_UNITS)
    est_margin = u * entry * 0.05
    cap = min(balance * MAX_MARGIN_USAGE_PCT / 100.0,
              float(acct.get("marginAvailable", 0) or 0) or balance)
    if est_margin > cap > 0:
        u = int(u * cap / est_margin)
    return max(0, u)


def ensure_protection(t):
    """Un trade du bot sans objectif ou sans stop est réparé (après un ordre partiel)."""
    if t["has_tp"] and t["has_sl"]:
        return
    tp = t["entry"] + RR * (t["entry"] - t["sl0"])
    data = {}
    if not t["has_tp"]:
        data["takeProfit"] = {"price": px(tp), "timeInForce": "GTC"}
    if not t["has_sl"]:
        data["stopLoss"] = {"price": px(t["sl0"]), "timeInForce": "GTC"}
    if not EXECUTE_TRADES:
        return
    try:
        api().request(trades.TradeCRCDO(accountID=OANDA_ACCOUNT_ID, tradeID=t["id"], data=data))
        logger.warning(f"[PROTECT] #{t['id']} protection remise : {list(data)}")
    except Exception as e:
        logger.error(f"[PROTECT] #{t['id']} échec, nouvel essai : {short_error(e, 120)}")


def fix_tp_on_fill(tid, fill, sl0):
    """L'objectif envoyé avec l'ordre vient du prix demandé ; on le recale sur le
    prix réellement obtenu, comme dans le test (objectif = entrée + 2 x risque)."""
    tp = fill + RR * (fill - sl0)
    try:
        api().request(trades.TradeCRCDO(accountID=OANDA_ACCOUNT_ID, tradeID=tid,
                                        data={"takeProfit": {"price": px(tp), "timeInForce": "GTC"}}))
    except Exception as e:
        logger.warning(f"[ORDER] #{tid} objectif non recalé (reste celui de l'ordre) : {short_error(e, 100)}")


def open_long(sl, ask, candle):
    """Renvoie 'OK', 'UNCERTAIN' (ne pas réessayer), 'RETRY' (marché fermé) ou 'SKIP'."""
    acct = get_account()
    u = units_for(ask, sl, acct)
    if u < 1:
        logger.info(f"[ORDER] taille < 1 once -> pas de trade")
        return "SKIP"
    tp = ask + RR * (ask - sl)
    log_event("SNAPSHOT_FILE", "SIGNAL", {"ts": utcnow().isoformat(), "candle": candle,
              "pair": PAIR, "dir": "BUY", "ask": ask, "sl0": round(sl, 3), "tp": round(tp, 3),
              "units": u, "risk_pct": RISK_PCT})
    if not EXECUTE_TRADES:
        logger.info("[ORDER] EXECUTE_TRADES=false -> simulation")
        return "OK"
    order = {"order": {
        "type": "MARKET", "instrument": PAIR, "units": str(u), "positionFill": "DEFAULT",
        "stopLossOnFill": {"price": px(sl), "timeInForce": "GTC"},
        "takeProfitOnFill": {"price": px(tp), "timeInForce": "GTC"},
        "tradeClientExtensions": {"tag": TAG, "comment": f"{TAG}|{PAIR}|sl0={sl:.5f}"}}}
    try:
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order)
        api().request(r)                       # pas de nouvel essai : risque de doublon
        resp = r.response
    except Exception as e:
        msg = str(e)
        if "MARKET_HALTED" in msg or "MARKET_CLOSED" in msg:
            return "RETRY"
        logger.error(f"[ORDER] erreur : {short_error(e)} -> pas de nouvel essai sur cette bougie")
        return "UNCERTAIN"
    rej = resp.get("orderRejectTransaction") or resp.get("orderCancelTransaction")
    if rej:
        reason = rej.get("rejectReason") or rej.get("reason")
        if reason in ("MARKET_HALTED", "MARKET_CLOSED"):
            return "RETRY"
        logger.error(f"[ORDER] refusé : {reason}")
        return "SKIP"
    fill = resp.get("orderFillTransaction") or {}
    tid = (fill.get("tradeOpened") or {}).get("tradeID")
    if not tid:
        logger.warning("[ORDER] état incertain -> pas de nouvel essai sur cette bougie")
        return "UNCERTAIN"
    fpx = float(fill.get("price", ask) or ask)
    if abs(fpx - ask) <= 5.0:                  # prix aberrant : on garde l'objectif de l'ordre
        fix_tp_on_fill(str(tid), fpx, sl)
    logger.info(f"[ORDER_OK] BUY {u} oz #{tid} à {fpx:.2f} | stop {sl:.2f} | "
                f"objectif {fpx + RR * (fpx - sl):.2f} | risque {RISK_PCT:.2f}%")
    return "OK"


# ===========================================================================
# CYCLE
# ===========================================================================
STATE = {}
_LAST_WAIT_LOG = {"t": 0.0}


def cycle():
    try:
        mine = [t for t in our_trades(get_open_trades())]
    except Exception as e:
        logger.warning(f"[CYCLE] positions illisibles -> cycle sauté : {short_error(e, 120)}")
        return
    track_closures(mine)
    for t in mine:
        ensure_protection(t)

    df = get_candles()
    if len(df) < 200:
        logger.warning(f"[CYCLE] seulement {len(df)} bougies -> cycle sauté")
        return
    candle = df.index[-1].isoformat()
    if STATE.get("done_for") == candle:
        return

    sig, sl, band, close = evaluate(df)
    end = df.index[-1] + timedelta(hours=1)
    if not sig:
        logger.info(f"[ATTENTE] bougie {df.index[-1]:%d/%m %H:%M} UTC clôture {close:.2f} | "
                    f"bande VWAP+1σ {band:.2f} ({close - band:+.2f})")
        STATE["done_for"] = candle
        save_state(STATE)
        return
    if mine:
        logger.info(f"[SIGNAL] {candle} ignoré : position #{mine[0]['id']} déjà ouverte (1 max)")
        STATE["done_for"] = candle
        save_state(STATE)
        return

    bid, ask, tradeable = get_price()
    first_seen = STATE.setdefault("seen", {}).setdefault(candle, utcnow().isoformat())
    if not tradeable:
        if time.time() - _LAST_WAIT_LOG["t"] > 1800:
            logger.info(f"[SIGNAL] {candle} : marché fermé, entrée à la réouverture")
            _LAST_WAIT_LOG["t"] = time.time()
        save_state(STATE)
        return
    late = (pd.Timestamp(first_seen) - end) > pd.Timedelta(minutes=LATE_MINUTES)
    if late:
        logger.warning(f"[SIGNAL] {candle} découvert {pd.Timestamp(first_seen) - end} après la "
                       f"clôture (bot arrêté ?) -> pas d'entrée en retard")
        STATE["done_for"] = candle
        STATE["seen"] = {}
        save_state(STATE)
        return
    if ask - bid > MAX_SPREAD:
        if time.time() - _LAST_WAIT_LOG["t"] > 300:
            logger.info(f"[SIGNAL] spread {ask - bid:.2f} $ > {MAX_SPREAD} $ -> on attend")
            _LAST_WAIT_LOG["t"] = time.time()
        return
    if ask <= sl:
        logger.info(f"[SIGNAL] prix {ask:.2f} déjà sous le stop {sl:.2f} -> pas de trade")
        res = "SKIP"
    else:
        res = open_long(sl, ask, candle)
    if res == "RETRY":
        return
    STATE["done_for"] = candle
    STATE["seen"] = {}
    save_state(STATE)


def validate_config():
    ok = True
    if not OANDA_ACCOUNT_ID:
        logger.critical("OANDA_ACCOUNT_ID manquant"); ok = False
    if not (os.getenv("OANDA_API_KEY") or os.getenv("OANDA_ACCESS_TOKEN")):
        logger.critical("OANDA_API_KEY manquant"); ok = False
    if OANDA_ENVIRONMENT == "live" and not _env_bool("VWAP_ALLOW_LIVE", False):
        logger.critical("⛔ ENVIRONNEMENT LIVE refusé : t = 2,39 hors réglage, sous la barre t > 3.")
        ok = False
    elif OANDA_ENVIRONMENT not in ("practice", "live"):
        logger.critical(f"OANDA_ENVIRONMENT invalide : {OANDA_ENVIRONMENT!r}"); ok = False
    if not 0 < RISK_PCT <= 2.0:
        logger.critical(f"VWAP_RISK_PCT={RISK_PCT} hors bornes (0 ; 2]"); ok = False
    for name in ("JOURNAL_FILE", "SNAPSHOT_FILE"):
        logger.info(f"[OUTPUT] {name}={os.getenv(name) or '(logs seulement)'}")
    return ok


def main():
    logger.info("🚀 Bot VWAP Or — VWAP mensuel +1σ, H1, acheteur seul, stop plus bas 10, objectif 2R")
    logger.info(f"✅ Risque {RISK_PCT:.2f}% par trade | 1 position max | spread max {MAX_SPREAD} $")
    logger.info("ℹ️  ~30-45 trades/an. t = 2,39 hors réglage : prometteur, pas prouvé. Démo seulement.")
    if not validate_config():
        sys.exit(1)
    STATE.update(load_state())
    last_hb = 0.0
    while True:
        try:
            if time.time() - last_hb >= 3600:
                a = get_account()
                logger.info(f"💓 solde={float(a.get('balance', 0) or 0):.2f} | "
                            f"P/L latent={float(a.get('unrealizedPL', 0) or 0):+.2f}")
                last_hb = time.time()
            cycle()
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt demandé")
            break
        except Exception as e:
            logger.error(f"💥 Erreur boucle : {short_error(e)}")
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
