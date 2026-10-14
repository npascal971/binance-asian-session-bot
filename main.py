#!/usr/bin/env python3
"""
==============================================================================
 BOT TENDANCE OR — Donchian journalier, acheteur seul, stop suiveur
==============================================================================

 POURQUOI CE BOT, ET CE QU'IL VAUT VRAIMENT (à lire avant de le lancer)

 C'est la SEULE configuration qui a donné un résultat positif dans nos tests,
 reproduite à l'identique :
   - or (XAU_USD) uniquement, bougies JOURNALIÈRES
   - entrée : clôture au-dessus du plus haut des 20 jours précédents
   - ACHAT SEULEMENT (les ventes perdaient : PF 0,26 sur les métaux en D1)
   - stop initial 1,5 ATR, puis stop SUIVEUR à 2,5 ATR qui ne fait que monter
   - AUCUN objectif fixe : on laisse courir les tendances

 Ce que disent les tests (octobre 2020 -> octobre 2026, banc classic_strategies) :
   - or seul, achat + vente : 62 trades, PF 1,92, +31,5 R
   - côté acheteur (or + argent) : PF 2,21, +49,9 R
   - balayage du canal 10 -> 50 : 7/7 valeurs positives hors échantillon

 Ce qu'il N'A PAS passé — honnêtement :
   - retirer les 3 meilleurs trades rendait l'ensemble perdant (PF 0,82)
   - seules 2 années sur 6 étaient positives : il gagne par à-coups
   - garder simplement de l'or a rapporté PLUS (+59 R) sur la même période
   - sur 2013-2019 (or plat), l'avantage était quasi nul (PF ~1,1)

 Autrement dit : ce bot est une façon d'être acheteur d'or AVEC un stop, qui
 sort quand la tendance casse et revient quand elle repart. Il gagne quand l'or
 monte fort, il perd par petites touches quand l'or stagne. Ce n'est PAS un edge
 démontré. Il prend peu de trades (de l'ordre de 5 à 10 par an) et les garde des
 semaines. De longues périodes sans gain sont normales.

 COMPTE DÉMO. Ne pas passer en réel sur la foi de ces chiffres.
==============================================================================
"""
import os
import sys
import time
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import oandapyV20
from oandapyV20.endpoints import instruments, pricing, orders, accounts, trades

load_dotenv()


# ===========================================================================
# CONFIGURATION
# Noms de variables DISTINCTS de l'ancien bot (préfixe TREND_) : si ton Railway
# contient encore GRANULARITY=H1 ou DONCHIAN_CHANNELS=20,30,40, ils n'auront
# AUCUN effet ici. L'échelle de temps est figée en journalier : c'est la stratégie.
# ===========================================================================
def _env_float(n, d):
    try:
        return float(os.getenv(n, d))
    except (TypeError, ValueError):
        return float(d)


def _env_int(n, d):
    try:
        return int(os.getenv(n, d))
    except (TypeError, ValueError):
        return int(d)


def _env_bool(n, d):
    return str(os.getenv(n, str(d))).strip().lower() in ("1", "true", "yes", "on")


def utcnow():
    return datetime.now(timezone.utc)


OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "").strip()
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")
OANDA_HTTP_TIMEOUT = _env_float("OANDA_HTTP_TIMEOUT", 15.0)
EXECUTE_TRADES = _env_bool("EXECUTE_TRADES", True)

ASSETS = [x.strip().upper() for x in os.getenv("TREND_ASSETS", "XAU_USD").split(",") if x.strip()]
GRANULARITY = "D"                               # figé : c'est ce qui a été testé
CHANNEL = _env_int("TREND_CHANNEL", 20)
ATR_PERIOD = 14
SL_ATR = _env_float("TREND_SL_ATR", 1.5)        # stop initial
TRAIL_ATR = _env_float("TREND_TRAIL_ATR", 2.5)  # stop suiveur
RISK_PCT = _env_float("TREND_RISK_PCT", 0.5)    # % du solde risqué par trade
MAX_MARGIN_USAGE_PCT = _env_float("MAX_MARGIN_USAGE_PCT", 20.0)
CHECK_INTERVAL_SECONDS = _env_int("TREND_CHECK_SECONDS", 900)   # vérifie toutes les 15 min

TAG = "trend"   # préfixe du commentaire OANDA : seuls ces trades sont gérés par ce bot

PRICE_DECIMALS = {"XAU_USD": 3, "XAG_USD": 4}
MAX_UNITS = {"XAU_USD": 500, "XAG_USD": 5000}

RAILWAY_ENV = bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID")
                   or os.getenv("RAILWAY_SERVICE_ID"))

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("TrendGold")
for noisy in ("urllib3", "requests", "oandapyV20"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


def short_error(e, limit=200):
    s = str(e)
    return s if len(s) <= limit else f"{s[:limit]}... [tronqué {len(s)}]"


# ===========================================================================
# OANDA (repris de la version précédente : retries 401/5xx déjà éprouvés)
# ===========================================================================
_API = None


def api_client():
    global _API
    if _API is None:
        token = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_ACCESS_TOKEN")
        if not token:
            raise RuntimeError("OANDA_API_KEY manquant")
        _API = oandapyV20.API(access_token=token, environment=OANDA_ENVIRONMENT,
                              request_params={"timeout": OANDA_HTTP_TIMEOUT})
    return _API


def oanda_read(request, retries=3, backoff=1.0):
    """Lecture avec retries sur erreurs transitoires (401 de débit inclus). JAMAIS pour un ordre."""
    api = api_client()
    for attempt in range(retries + 1):
        try:
            api.request(request)
            return request.response
        except Exception as e:
            code = getattr(e, "code", None)
            transient = code is None or code >= 500 or code == 429 or code == 401
            if not transient or attempt >= retries:
                raise
            time.sleep(backoff * (attempt + 1))
    return {}


def get_daily_candles(pair, count=120):
    """Bougies journalières FERMÉES uniquement. Lève en cas d'échec (fail-closed)."""
    params = {"granularity": GRANULARITY, "count": min(count + 1, 500), "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    resp = oanda_read(r)
    rows = []
    for c in resp.get("candles", []):
        if not c.get("complete", False):
            continue
        m = c.get("mid")
        if m:
            rows.append({"time": c["time"], "open": float(m["o"]), "high": float(m["h"]),
                         "low": float(m["l"]), "close": float(m["c"])})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    return df


def get_price(pair):
    try:
        r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params={"instruments": pair})
        pr = oanda_read(r).get("prices", [])
        if pr:
            bid = float(pr[0].get("bids", [{}])[0].get("price", 0) or 0)
            ask = float(pr[0].get("asks", [{}])[0].get("price", 0) or 0)
            return {"bid": bid, "ask": ask}
    except Exception as e:
        logger.debug(f"[PRICE] {pair} : {short_error(e, 100)}")
    return {"bid": 0.0, "ask": 0.0}


def get_open_trades():
    """Lève en cas d'échec : l'appelant DOIT savoir que l'état est inconnu."""
    r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
    return oanda_read(r).get("trades", [])


def get_account():
    try:
        r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
        return oanda_read(r).get("account", {})
    except Exception as e:
        logger.warning(f"[ACCOUNT] indisponible : {short_error(e, 120)}")
        return {}


def round_price(pair, price):
    return f"{float(price):.{PRICE_DECIMALS.get(pair, 3)}f}"


# ===========================================================================
# INDICATEURS — identiques au banc de test
# ===========================================================================
def atr_series(df, n=ATR_PERIOD):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def breakout_up(df, chan=CHANNEL):
    """Vrai si la DERNIÈRE clôture dépasse le plus haut des `chan` jours PRÉCÉDENTS.
    C'est un état (comme dans le banc), pas un croisement : après une sortie au stop,
    on re-rentre si la cassure tient toujours, exactement comme le backtest."""
    if len(df) < chan + 2:
        return False
    hi = df["high"].rolling(chan).max().shift(1)
    return bool(df["close"].iloc[-1] > hi.iloc[-1])


# ===========================================================================
# ÉTAT DES POSITIONS
# Le stop INITIAL est stocké dans le commentaire du trade. Indispensable : le stop
# suiveur se déplace, et mesurer le résultat en R depuis le stop COURANT fausse
# tout (bug déjà rencontré sur une version précédente).
# ===========================================================================
def make_comment(pair, sl0):
    return f"{TAG}|{pair}|sl0={sl0:.5f}"


def parse_comment(comment):
    try:
        parts = str(comment).split("|")
        if len(parts) >= 3 and parts[0] == TAG and parts[2].startswith("sl0="):
            return parts[1], float(parts[2][4:])
    except Exception:
        pass
    return None, None


def our_trades(all_trades):
    """Trades ouverts gérés par CE bot, indexés par paire."""
    out = {}
    for t in all_trades:
        ext = t.get("clientExtensions") or {}
        pair, sl0 = parse_comment(ext.get("comment", ""))
        if pair:
            out.setdefault(pair, []).append({
                "id": str(t.get("id")),
                "units": float(t.get("currentUnits", 0) or 0),
                "entry": float(t.get("price", 0) or 0),
                "sl": float((t.get("stopLossOrder") or {}).get("price", 0) or 0),
                "sl0": sl0,
                "opened_at": t.get("openTime"),
            })
    return out


# ===========================================================================
# JOURNAL
# ===========================================================================
def _append(path, line):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_event(env_name, tag, rec):
    try:
        line = json.dumps(rec, default=str, ensure_ascii=False)
        logger.info(f"[{tag}] {line}")
        path = os.getenv(env_name)
        if path:
            _append(path, line)
    except Exception as e:
        logger.debug(f"[{tag}] {short_error(e, 80)}")


_KNOWN = {}   # trade_id -> infos, pour journaliser les clôtures


def track_closures(mine):
    """Journalise les trades qui ont disparu. Si le résultat est illisible, on
    RÉESSAIE au cycle suivant au lieu de journaliser un résultat vide (défaut de la
    version précédente : 9 trades journalisés plusieurs fois sans résultat)."""
    open_ids = {t["id"] for lst in mine.values() for t in lst}
    for lst in mine.values():
        for t in lst:
            _KNOWN.setdefault(t["id"], t)
    for tid in list(_KNOWN.keys()):
        if tid in open_ids:
            continue
        info = _KNOWN[tid]
        try:
            r = trades.TradeDetails(accountID=OANDA_ACCOUNT_ID, tradeID=tid)
            td = oanda_read(r).get("trade", {})
            if td.get("state") != "CLOSED":
                continue                       # pas encore réglé côté OANDA : on attend
            pl = float(td.get("realizedPL", 0) or 0)
            close_price = float(td.get("averageClosePrice", 0) or 0)
        except Exception as e:
            logger.debug(f"[CLOSE] {tid} illisible, nouvel essai plus tard : {short_error(e, 80)}")
            continue
        risk = abs(info["entry"] - info["sl0"]) if info.get("sl0") else 0
        r_mult = ((close_price - info["entry"]) / risk) if (risk > 0 and close_price) else None
        log_event("JOURNAL_FILE", "JOURNAL", {
            "id": tid, "pair": td.get("instrument"), "dir": "BUY",
            "entry": info["entry"], "exit": close_price, "sl0": info.get("sl0"),
            "pl": round(pl, 2), "r": round(r_mult, 2) if r_mult is not None else None,
            "opened_at": info.get("opened_at"), "closed_at": td.get("closeTime"),
        })
        del _KNOWN[tid]


# ===========================================================================
# ACTIONS
# ===========================================================================
def calculate_units(pair, entry, sl, balance, margin_avail):
    risk_usd = balance * RISK_PCT / 100.0
    dist = entry - sl
    if dist <= 0 or risk_usd <= 0:
        return 0
    units = int(risk_usd / dist)
    units = min(units, MAX_UNITS.get(pair, 100000))
    est_margin = units * entry * 0.05            # ~5 % sur les métaux, prudent
    cap = min(balance * MAX_MARGIN_USAGE_PCT / 100.0, margin_avail or balance)
    if est_margin > cap > 0:
        units = int(units * cap / est_margin)
    return max(0, units)


def open_long(pair, atr_now):
    acct = get_account()
    balance = float(acct.get("balance", 0) or 0)
    if balance <= 0:
        logger.error(f"[ORDER] {pair} solde illisible -> pas d'ordre")
        return "UNCERTAIN"
    entry = get_price(pair)["ask"]
    if entry <= 0:
        logger.warning(f"[ORDER] {pair} prix indisponible -> pas d'ordre")
        return None
    sl0 = entry - SL_ATR * atr_now
    units = calculate_units(pair, entry, sl0, balance,
                            float(acct.get("marginAvailable", 0) or 0))
    if units < 1:
        logger.info(f"[ORDER] {pair} taille < 1 unité -> pas de trade")
        return None

    log_event("SNAPSHOT_FILE", "SIGNAL", {
        "ts": utcnow().isoformat(), "pair": pair, "dir": "BUY", "entry": round(entry, 3),
        "sl0": round(sl0, 3), "units": units, "atr": round(atr_now, 3),
        "risk_pct": RISK_PCT})

    if not EXECUTE_TRADES:
        logger.info("[ORDER] EXECUTE_TRADES=false -> simulation, aucun ordre envoyé")
        return "SIMULATION"

    order = {"order": {
        "type": "MARKET", "instrument": pair, "units": str(int(units)),
        "positionFill": "DEFAULT",
        "stopLossOnFill": {"price": round_price(pair, sl0), "timeInForce": "GTC"},
        # PAS de takeProfitOnFill : c'est le stop suiveur qui sort. Un objectif fixe
        # décapitait les gros gagnants, c'est la leçon principale des tests.
        "tradeClientExtensions": {"tag": TAG, "comment": make_comment(pair, sl0)},
    }}
    try:
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order)
        api_client().request(r)            # PAS de retry sur un ordre : risque de doublon
        resp = r.response
        if resp.get("orderRejectTransaction"):
            logger.error(f"[ORDER] REJET {pair} : {resp['orderRejectTransaction'].get('rejectReason')}")
            return None
        tid = ((resp.get("orderFillTransaction") or {}).get("tradeOpened") or {}).get("tradeID")
        if tid:
            logger.info(f"[ORDER_OK] {pair} BUY {units} u. ID={tid} | entrée~{entry:.2f} "
                        f"stop={sl0:.2f} | risque {RISK_PCT:.2f}%")
            return str(tid)
        if resp.get("orderCancelTransaction"):
            logger.warning(f"[ORDER] {pair} annulé ({resp['orderCancelTransaction'].get('reason')})")
            return None
        logger.warning(f"[ORDER] {pair} état incertain -> pas de nouvel essai sur cette bougie")
        return "UNCERTAIN"
    except Exception as e:
        logger.error(f"[ORDER] {pair} erreur : {short_error(e)} -> pas de nouvel essai sur cette bougie")
        return "UNCERTAIN"


def raise_stop(pair, trade, new_sl):
    """Remonte le stop. Ne le baisse JAMAIS (garanti ici et par l'appelant)."""
    if new_sl <= trade["sl"] + 1e-9:
        return
    if not EXECUTE_TRADES:
        logger.info(f"[TRAIL] (simulation) {pair} #{trade['id']} stop {trade['sl']:.2f} -> {new_sl:.2f}")
        return
    try:
        data = {"stopLoss": {"price": round_price(pair, new_sl), "timeInForce": "GTC"}}
        r = trades.TradeCRCDO(accountID=OANDA_ACCOUNT_ID, tradeID=trade["id"], data=data)
        api_client().request(r)
        logger.info(f"[TRAIL] {pair} #{trade['id']} stop {trade['sl']:.2f} -> {new_sl:.2f}")
    except Exception as e:
        # Le stop actuel reste en place : on réessaiera au prochain cycle (idempotent).
        logger.warning(f"[TRAIL] {pair} #{trade['id']} échec, nouvel essai plus tard : {short_error(e, 120)}")


# ===========================================================================
# CYCLE
# ===========================================================================
_ENTRY_DONE_FOR = {}   # pair -> horodatage de la bougie déjà traitée pour l'ENTRÉE


def cycle():
    # FAIL-CLOSED : sans état fiable des positions, on ne fait rien.
    try:
        mine = our_trades(get_open_trades())
    except Exception as e:
        logger.warning(f"[CYCLE] positions illisibles -> cycle sauté : {short_error(e, 120)}")
        return
    track_closures(mine)

    for i, pair in enumerate(ASSETS):
        if i:
            time.sleep(1.0)            # pas de rafale de requêtes (401 de débit)
        try:
            df = get_daily_candles(pair)
        except Exception as e:
            logger.warning(f"[CYCLE] {pair} bougies illisibles -> rien : {short_error(e, 100)}")
            continue
        if len(df) < CHANNEL + ATR_PERIOD + 2:
            logger.info(f"[CYCLE] {pair} historique insuffisant")
            continue
        a = atr_series(df)
        atr_now = float(a.iloc[-1])
        if not np.isfinite(atr_now) or atr_now <= 0:
            continue
        last_close = float(df["close"].iloc[-1])
        candle_ts = str(df.index[-1])

        positions = mine.get(pair, [])

        # 1) STOP SUIVEUR : à chaque clôture journalière, stop = clôture - 2,5 ATR,
        #    seulement s'il monte. Idempotent : relancer le calcul sur la même
        #    bougie redonne la même valeur, donc aucun risque de double action.
        for t in positions:
            raise_stop(pair, t, last_close - TRAIL_ATR * atr_now)

        # 2) ENTRÉE : une seule position par actif, comme dans le banc.
        if positions:
            continue
        if _ENTRY_DONE_FOR.get(pair) == candle_ts:
            continue
        if breakout_up(df):
            hi = float(df["high"].rolling(CHANNEL).max().shift(1).iloc[-1])
            logger.info(f"[SIGNAL] {pair} clôture {last_close:.2f} > plus haut {CHANNEL}j "
                        f"{hi:.2f} -> ACHAT")
            res = open_long(pair, atr_now)
            if res:                    # succès, simulation OU incertain : pas de 2e essai
                _ENTRY_DONE_FOR[pair] = candle_ts
        else:
            _ENTRY_DONE_FOR[pair] = candle_ts


def check_output_file(env_name):
    path = os.getenv(env_name)
    if not path:
        logger.info(f"[OUTPUT] {env_name} non défini (lignes seulement dans les logs)")
        return
    try:
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"_startup_check": env_name, "at": utcnow().isoformat()}) + "\n")
        logger.info(f"[OUTPUT] {env_name}={path} : écriture OK")
        if RAILWAY_ENV and not os.getenv("RAILWAY_VOLUME_MOUNT_PATH"):
            logger.warning(f"[OUTPUT] ⚠️ {env_name} hors volume Railway : effacé au redéploiement")
    except Exception as e:
        logger.error(f"[OUTPUT] ❌ {env_name}={path} non inscriptible : {short_error(e)}")


def validate_config():
    ok = True
    if not OANDA_ACCOUNT_ID:
        logger.critical("OANDA_ACCOUNT_ID manquant"); ok = False
    if not (os.getenv("OANDA_API_KEY") or os.getenv("OANDA_ACCESS_TOKEN")):
        logger.critical("OANDA_API_KEY manquant"); ok = False
    if OANDA_ENVIRONMENT not in ("practice", "live"):
        logger.critical(f"OANDA_ENVIRONMENT invalide : {OANDA_ENVIRONMENT!r}"); ok = False
    elif OANDA_ENVIRONMENT == "live":
        logger.critical("⛔ ENVIRONNEMENT LIVE refusé : ce bot n'a pas d'edge démontré. "
                        "Pour passer outre, définir TREND_ALLOW_LIVE=true en connaissance de cause.")
        if not _env_bool("TREND_ALLOW_LIVE", False):
            ok = False
    if not 0 < RISK_PCT <= 2.0:
        logger.critical(f"TREND_RISK_PCT={RISK_PCT} hors bornes (0 ; 2]"); ok = False
    for old in ("GRANULARITY", "DONCHIAN_CHANNELS", "ASSETS", "TOTAL_RISK_PCT"):
        if os.getenv(old):
            logger.warning(f"[CONFIG] variable de l'ancien bot {old}={os.getenv(old)!r} "
                           f"ignorée (ce bot lit TREND_*)")
    check_output_file("JOURNAL_FILE")
    check_output_file("SNAPSHOT_FILE")
    return ok


def main_loop():
    logger.info("🚀 Bot Tendance Or — Donchian journalier, acheteur seul, stop suiveur")
    logger.info(f"✅ Actifs: {', '.join(ASSETS)} | canal {CHANNEL} jours | bougies D")
    logger.info(f"✅ Stop initial {SL_ATR} ATR | stop suiveur {TRAIL_ATR} ATR | aucun objectif")
    logger.info(f"✅ Risque {RISK_PCT:.2f}% du solde par trade | 1 position max par actif")
    logger.info("ℹ️  Peu de trades (≈5-10/an), tenus des semaines. Pas d'edge démontré : voir l'en-tête.")
    if not EXECUTE_TRADES:
        logger.warning("EXECUTE_TRADES=false : SIMULATION, aucun ordre envoyé")
    if not validate_config():
        sys.exit(1)

    last_hb = 0.0
    while True:
        try:
            now = time.time()
            if now - last_hb >= 3600:
                acct = get_account()
                logger.info(f"💓 solde={float(acct.get('balance', 0) or 0):.2f} | "
                            f"P/L latent={float(acct.get('unrealizedPL', 0) or 0):+.2f}")
                last_hb = now
            cycle()
            time.sleep(CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt demandé")
            break
        except Exception as e:
            logger.error(f"💥 Erreur boucle : {short_error(e)}")
            logger.debug("trace", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main_loop()
