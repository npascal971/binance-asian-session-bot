#!/usr/bin/env python3
"""VWAP mensuel or H1 — v6 candidate DEMO. Voir README.md avant déploiement.
Signal identique à v4 ; aucune promesse de rendement. Compte practice uniquement.
"""
import fcntl
import hashlib
import json
import logging
import math
import os
import signal
import urllib.request
import urllib.parse
from dataclasses import dataclass
from types import SimpleNamespace
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name, default):
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in ('1', 'true', 'yes', 'oui', 'on')


VERSION = 'vwap-v6-demo-candidate'
PAIR, SIGMA, SL_BARS, RR, MIN_AGE, CANDLES = 'XAU_USD', 1.0, 10, 2.0, 3, 1000
CHECK_SECONDS, ENTRY_WINDOW_MIN = 30, 20
ACCOUNT = (os.getenv('VWAP_ACCOUNT_ID') or os.getenv('OANDA_ACCOUNT_ID', '')).strip()
ENVIRONMENT = os.getenv('OANDA_ENVIRONMENT', 'practice').strip()
EXECUTE_TRADES = _env_bool("EXECUTE_TRADES", True)
RISK_PCT = float(os.getenv('VWAP_RISK_PCT', '0.5'))
MAX_SPREAD = float(os.getenv('VWAP_MAX_SPREAD', '1.0'))
MAX_SLIPPAGE = float(os.getenv('VWAP_MAX_SLIPPAGE', '1.0'))
MAX_MARGIN_USAGE_PCT = float(os.getenv('MAX_MARGIN_USAGE_PCT', '20'))
MAX_UNITS, TOL = 500, 0.0011
STATE_FILE = Path(os.getenv('VWAP_STATE_FILE', '/data/vwap_state.json'))
JOURNAL_FILE = Path(os.getenv('JOURNAL_FILE', '/data/vwap_journal_v6.jsonl'))
SNAPSHOT_FILE = Path(os.getenv('SNAPSHOT_FILE', '/data/vwap_signals_v6.jsonl'))
STATE, _API, _LOCK = {}, None, None
_STOP = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', stream=sys.stdout)
logger = logging.getLogger('VwapGold')
for noisy in ('oandapyV20', 'urllib3', 'requests'):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)


class StateError(RuntimeError):
    pass


def utcnow():
    return datetime.now(timezone.utc)


def short_error(exc):
    # Ne journalise jamais un corps HTTP entier ni des credentials.
    return type(exc).__name__


def number(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError('Valeur numérique non finie')
    return x


def px(value):
    return f'{number(value):.3f}'


@dataclass
class Endpoint:
    method: str
    path: str
    params: dict | None = None
    data: dict | None = None


def endpoint_factory(method, template):
    def make(**kwargs):
        params, data = kwargs.pop('params', None), kwargs.pop('data', None)
        parts = {k: urllib.parse.quote(str(v), safe='') for k, v in kwargs.items()}
        return Endpoint(method, template.format(**parts), params, data)
    return make


# Petit adaptateur REST explicite ; aucune dépendance au SDK OANDA / à six.
accounts = SimpleNamespace(
    AccountSummary=endpoint_factory('GET', '/accounts/{accountID}/summary'),
    AccountInstruments=endpoint_factory('GET', '/accounts/{accountID}/instruments'))
instruments = SimpleNamespace(InstrumentsCandles=endpoint_factory('GET', '/instruments/{instrument}/candles'))
pricing = SimpleNamespace(PricingInfo=endpoint_factory('GET', '/accounts/{accountID}/pricing'))
trades = SimpleNamespace(
    OpenTrades=endpoint_factory('GET', '/accounts/{accountID}/openTrades'),
    TradeDetails=endpoint_factory('GET', '/accounts/{accountID}/trades/{tradeID}'),
    TradeCRCDO=endpoint_factory('PUT', '/accounts/{accountID}/trades/{tradeID}/orders'))
orders = SimpleNamespace(OrderCreate=endpoint_factory('POST', '/accounts/{accountID}/orders'))
transactions = SimpleNamespace(
    TransactionsSinceID=endpoint_factory('GET', '/accounts/{accountID}/transactions/sinceid'))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class RestClient:
    def __init__(self, token):
        self.token = token
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, req):
        if ENVIRONMENT != 'practice':
            raise RuntimeError('Compte practice uniquement')
        url = 'https://api-fxpractice.oanda.com/v3' + req.path
        if req.params:
            url += '?' + urllib.parse.urlencode(req.params)
        data = json.dumps(req.data, allow_nan=False).encode() if req.data is not None else None
        request = urllib.request.Request(url, data=data, method=req.method, headers={
            'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339'})
        with self.opener.open(request, timeout=15) as response:
            return json.load(response)


def api():
    global _API
    if _API is None:
        token = os.getenv('OANDA_API_KEY') or os.getenv('OANDA_ACCESS_TOKEN')
        if not token:
            raise RuntimeError('Clé OANDA manquante')
        _API = RestClient(token)
    return _API


def read(req, retries=2):
    if req.method != 'GET':
        raise ValueError('read est réservé aux lectures')
    for k in range(retries + 1):
        try:
            return api().request(req)
        except Exception:
            if k == retries:
                raise
            time.sleep(1 + k)


def save_state():
    """Toute panne d'écriture arrête le processus ; aucun ordre sans état durable."""
    tmp = STATE_FILE.with_name(STATE_FILE.name + '.tmp')
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(STATE, f, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE_FILE)
        fd = os.open(str(STATE_FILE.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception as exc:
        raise StateError('État non sauvegardé : processus arrêté') from exc


def load_state():
    if not STATE_FILE.exists():
        return {}
    try:
        value = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('Objet JSON requis')
        for name in ('known', 'fills', 'windows'):
            if name in value and not isinstance(value[name], dict):
                raise ValueError('Structure état invalide')
        return value
    except Exception as exc:
        raise StateError('État illisible : ne pas effacer ; restaurer une sauvegarde') from exc


def acquire_lock():
    """Verrou Linux limité au même fichier/volume. Pas un verrou distribué."""
    global _LOCK
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOCK = STATE_FILE.with_name(STATE_FILE.name + '.lock').open('a')
    try:
        fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise StateError('Une instance utilise déjà cet état sur ce volume') from exc


def append_json(path, rec):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
    except Exception as exc:
        raise StateError('Journal/signaux non sauvegardés : processus arrêté') from exc


def log_event(path, label, rec):
    # La ligne JSON complète reste dans le fichier ; stdout reste lisible.
    append_json(path, rec)
    logger.info('[%s] id=%s candle=%s', label, rec.get('id'), rec.get('candle'))


def get_account_response():
    response = read(accounts.AccountSummary(accountID=ACCOUNT))
    if not response.get('account') or response['account'].get('id') != ACCOUNT:
        raise RuntimeError('Compte absent de la réponse')
    return response


def get_open_trades():
    response = read(trades.OpenTrades(accountID=ACCOUNT))
    if not isinstance(response.get('trades'), list):
        raise RuntimeError('Liste des trades absente')
    return response['trades']


def get_price():
    p = read(pricing.PricingInfo(accountID=ACCOUNT, params={'instruments': PAIR}))['prices'][0]
    bid, ask = number(p['bids'][0]['price']), number(p['asks'][0]['price'])
    stamp = pd.Timestamp(p['time'])
    if stamp.tzinfo is None or not (0 < bid <= ask):
        raise ValueError('Cotation invalide')
    if p.get('tradeable'):
        age = (pd.Timestamp(utcnow()) - stamp).total_seconds()
        if not -5 <= age <= 60:
            raise ValueError('Cotation périmée')
    return bid, ask, p.get('tradeable') is True


def get_candles():
    result = read(instruments.InstrumentsCandles(instrument=PAIR, params={
        'granularity': 'H1', 'price': 'M', 'count': CANDLES}))
    rows, idx = [], []
    for c in result['candles']:
        if not c.get('complete'):
            continue
        m = c['mid']
        idx.append(pd.Timestamp(c['time']))
        row = {k: number(m[v]) for k, v in [('open','o'),('high','h'),('low','l'),('close','c')]}
        row['volume'] = number(c['volume'])
        if not (0 < row['low'] <= min(row['open'], row['close']) <=
                max(row['open'], row['close']) <= row['high']) or row['volume'] < 0:
            raise ValueError('Bougie invalide')
        rows.append(row)
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    if not df.empty and (not df.index.is_monotonic_increasing or not df.index.is_unique):
        raise ValueError('Historique non ordonné ou dupliqué')
    return df


# Fonctions de signal reprises sans changement de la v4.
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


def conversion_to_usd(ccy):
    if ccy == 'USD':
        return 1.0
    for ins, inverse in ((f'{ccy}_USD', False), (f'USD_{ccy}', True)):
        try:
            p = read(pricing.PricingInfo(accountID=ACCOUNT, params={'instruments': ins}), retries=0)['prices'][0]
            bid, ask = number(p['bids'][0]['price']), number(p['asks'][0]['price'])
            age = (pd.Timestamp(utcnow()) - pd.Timestamp(p['time'])).total_seconds()
            if 0 < bid <= ask and p.get('tradeable') and -5 <= age <= 60:
                mid = (bid + ask) / 2
                return 1 / mid if inverse else mid
        except Exception:
            continue
    raise RuntimeError('Conversion devise indisponible : entrée suspendue')


def size_order(ask, sl, acct):
    """Dimensionne au pire prix accepté ; pas seulement à la cotation avant envoi."""
    ccy = acct['currency']
    k = conversion_to_usd(ccy)
    balance, nav = number(acct['balance']) * k, number(acct['NAV']) * k
    available, used = number(acct['marginAvailable']) * k, number(acct['marginUsed']) * k
    item = read(accounts.AccountInstruments(accountID=ACCOUNT, params={'instruments': PAIR}))['instruments'][0]
    if item['name'] != PAIR or int(item['displayPrecision']) != 3:
        raise RuntimeError('Instrument/précision inattendu')
    rate = max(number(item['marginRate']), number(acct['marginRate']))
    if rate <= 0 or rate > 1:
        raise ValueError('Taux de marge invalide')
    worst = number(px(ask + MAX_SLIPPAGE))
    dist = worst - sl
    if min(balance, nav, available, dist) <= 0:
        return {'units': 0}
    cap = max(0, min(available * 0.95, nav * MAX_MARGIN_USAGE_PCT / 100 - used))
    max_allowed = min(MAX_UNITS, int(number(item['maximumOrderUnits'])))
    u = max(0, min(int(balance * RISK_PCT / 100 / dist), int(cap / (worst * rate)), max_allowed))
    if u < max(1, math.ceil(number(item['minimumTradeSize']))):
        u = 0
    return {'units': u, 'usd_per_ccy': k, 'currency': ccy, 'price_bound': worst,
            'risk_home_max': u * dist / k, 'margin_rate': rate}


def our_trades(all_trades):
    found = []
    for t in all_trades:
        ext = t.get('clientExtensions') or {}
        parts = str(ext.get('comment', '')).split('|')
        # Un marqueur VWAP corrompu doit bloquer les entrées, pas rendre le trade invisible.
        if parts[0] != 'vwap':
            continue
        if len(parts) != 3 or parts[1] != PAIR or not parts[2].startswith('sl0='):
            raise RuntimeError('Marqueur VWAP invalide : intervention nécessaire')
        initial_sl, entry, units = number(parts[2][4:]), number(t['price']), number(t['currentUnits'])
        if t['instrument'] != PAIR or units <= 0 or not 0 < initial_sl < entry:
            raise RuntimeError('Trade VWAP incohérent : entrée suspendue')
        found.append({'id': str(t['id']), 'entry': entry, 'sl0': initial_sl,
                      'units': abs(number(t.get('initialUnits', units))), 'opened_at': t['openTime'],
                      'tp': (t.get('takeProfitOrder') or {}).get('price'),
                      'sl': (t.get('stopLossOrder') or {}).get('price'),
                      'trailing': bool(t.get('trailingStopLossOrder'))})
    return found


def register_fill(fill, pending):
    opened = fill.get('tradeOpened') or {}
    if not opened.get('tradeID'):
        raise RuntimeError('Exécution sans tradeOpened : réconciliation bloquée')
    tid = str(opened['tradeID'])
    entry = number(opened.get('price', fill['price']))
    u = abs(number(opened['units']))
    if u <= 0 or entry <= pending['sl0']:
        raise RuntimeError('Exécution incohérente : vérifier le compte')
    info = {'entry': entry, 'sl0': pending['sl0'], 'units': u,
            'opened_at': fill['time'], 'currency': pending['currency'],
            'usd_per_ccy': pending['usd_per_ccy'],
            'risk_home': u * (entry - pending['sl0']) / pending['usd_per_ccy'],
            'slippage': entry - pending['ask'], 'tp_sent': pending['tp_sent'],
            'tp_rebased': False, 'client_id': pending['client_id']}
    STATE.setdefault('known', {})[tid] = info
    STATE['done_for'] = pending['candle']
    STATE['pending'] = None
    save_state()
    logger.info('[ORDER_OK] trade=%s unités=%s prix=%.3f risque_max_prévu=%.2f',
                tid, u, entry, pending['risk_home_max'])


def resolve_response(response):
    pending = STATE.get('pending')
    if not pending:
        return
    fill = response.get('orderFillTransaction')
    if fill:
        register_fill(fill, pending)
        return
    rejection = response.get('orderRejectTransaction') or response.get('orderCancelTransaction')
    if rejection:
        # Un refus confirmé consomme le signal. Aucune boucle d'ouverture/fermeture.
        STATE['done_for'] = pending['candle']
        STATE['pending'] = None
        save_state()
        logger.warning('[ORDER_REFUSED] %s', rejection.get('rejectReason', rejection.get('reason')))


def reconcile_pending():
    p = STATE.get('pending')
    if not p:
        return True
    result = read(transactions.TransactionsSinceID(accountID=ACCOUNT, params={'id': p['since_id']}))
    txs = result['transactions']
    ids = {str(t['id']) for t in txs if (t.get('clientExtensions') or {}).get('id') == p['client_id']}
    matched = [t for t in txs if t.get('clientOrderID') == p['client_id'] or
               str(t.get('orderID', '')) in ids or
               (t.get('clientExtensions') or {}).get('id') == p['client_id']]
    for t in matched:
        if t.get('type') == 'ORDER_FILL':
            register_fill(t, p)
            return True
    for t in matched:
        if t.get('type') in ('ORDER_CANCEL', 'MARKET_ORDER_REJECT'):
            resolve_response({'orderCancelTransaction': t})
            return True
    # Même si rien n'est trouvé : ne pas supposer que l'ordre n'a pas été exécuté.
    logger.error('[PENDING] ordre %s non résolu ; nouvelles entrées bloquées', p['client_id'])
    return False


def ensure_protection(t):
    info = STATE['known'][t['id']]
    target = t['entry'] + RR * (t['entry'] - t['sl0'])
    data = {}
    if not t['trailing'] and (t['sl'] is None or number(t['sl']) < t['sl0'] - TOL):
        data['stopLoss'] = {'price': px(t['sl0']), 'timeInForce': 'GTC'}
    # Un TP existant des anciennes versions est respecté. Pour les nouveaux, recaler
    # une seule fois, sauf si l'utilisateur a déjà déplacé le TP envoyé initialement.
    rebase = not info.get('tp_rebased', True)
    manual_tp = rebase and t['tp'] is not None and abs(number(t['tp']) - info['tp_sent']) > TOL
    if t['tp'] is None or (rebase and not manual_tp and abs(number(t['tp']) - target) > TOL):
        data['takeProfit'] = {'price': px(target), 'timeInForce': 'GTC'}
    if data and not EXECUTE_TRADES:
        logger.info('[DRY_RUN] protection à vérifier trade=%s champs=%s', t['id'], list(data))
        return
    if data:
        api().request(trades.TradeCRCDO(accountID=ACCOUNT, tradeID=t['id'], data=data))
        logger.warning('[PROTECT] trade=%s champs=%s', t['id'], list(data))
    if rebase:
        info['tp_rebased'] = True
        save_state()


_JOURNALED = set()


def load_journal_index():
    _JOURNALED.clear()
    if JOURNAL_FILE.exists():
        with JOURNAL_FILE.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get('id'):
                        _JOURNALED.add(str(rec['id']))


def track_closures(mine):
    known = STATE.setdefault('known', {})
    for t in mine:
        if t['id'] not in known:
            # Reprise d'une position v4 ou après migration ; taux initial inconnu.
            known[t['id']] = {k: t[k] for k in ('entry', 'sl0', 'units', 'opened_at')}
            known[t['id']]['restored'] = True
            known[t['id']]['tp_rebased'] = True
            save_state()
    ids = {t['id'] for t in mine}
    for tid in list(known):
        if tid in ids:
            continue
        info = known[tid]
        if tid in _JOURNALED:
            del known[tid]
            save_state()
            continue
        td = read(trades.TradeDetails(accountID=ACCOUNT, tradeID=tid))['trade']
        if td.get('state') != 'CLOSED':
            continue
        entry, sl, close = info['entry'], info['sl0'], number(td['averageClosePrice'])
        pl = number(td['realizedPL'])
        fin = number(td['financing']) if 'financing' in td else None
        risk_home = info.get('risk_home')
        rec = {'version': VERSION, 'id': tid, 'pair': PAIR, 'dir': 'BUY',
               'entry': entry, 'sl0': sl, 'exit': close, 'units': info['units'],
               'r_prix': (close - entry) / (entry - sl), 'pl': pl, 'financement': fin,
               'r_apres_financement': (pl + fin) / risk_home if fin is not None and risk_home else None,
               'r_net': None, 'commissions_non_incluses': True,
               'devise_compte': info.get('currency'), 'risque_initial_devise_compte': risk_home,
               'glissement': info.get('slippage'), 'restored': info.get('restored', False),
               'opened_at': info.get('opened_at'), 'closed_at': td['closeTime']}
        # P/L contient déjà l'effet des prix bid/ask : ne pas déduire le spread à nouveau.
        log_event(JOURNAL_FILE, 'JOURNAL', rec)
        _JOURNALED.add(tid)
        del known[tid]
        STATE.get('fills', {}).pop(tid, None)
        save_state()


def open_long(sl, ask, candle):
    if STATE.get('pending'):
        logger.error('[PENDING] nouvelle tentative interdite')
        return 'UNCERTAIN'
    response = get_account_response()
    acct = response['account']
    plan = size_order(ask, sl, acct)
    if plan['units'] < 1:
        logger.info('[SKIP] taille/marge insuffisante')
        return 'SKIP'
    tp = number(px(ask + RR * (ask - sl)))
    rec = {'version': VERSION, 'candle': candle, 'ts': utcnow().isoformat(),
           'pair': PAIR, 'ask': ask, 'sl0': sl, 'tp': tp, **plan}
    log_event(SNAPSHOT_FILE, 'SIGNAL', rec)
    if not EXECUTE_TRADES:
        logger.info('[DRY_RUN] signal calculé ; aucun ordre ni modification')
        return 'OK'
    # Ne plus utiliser une cotation si les lectures de compte/marge ont pris trop longtemps.
    bid_now, ask_now, tradeable = get_price()
    if not tradeable or ask_now > plan['price_bound'] or ask_now - bid_now > MAX_SPREAD or bid_now <= sl:
        return 'RETRY'
    if pd.Timestamp(utcnow()) >= pd.Timestamp(candle) + pd.Timedelta(hours=1, minutes=ENTRY_WINDOW_MIN):
        return 'SKIP'
    seq = int(STATE.get('order_seq', 0)) + 1
    STATE['order_seq'] = seq
    cid = 'vw6-' + hashlib.sha256(f'{ACCOUNT}|{candle}|{seq}'.encode()).hexdigest()[:24]
    pending = {**plan, 'client_id': cid, 'since_id': str(response['lastTransactionID']),
               'candle': candle, 'ask': ask, 'sl0': sl, 'tp_sent': tp}
    STATE['pending'] = pending
    save_state()  # impérativement AVANT la requête d'ordre
    data = {'order': {'type': 'MARKET', 'timeInForce': 'FOK', 'instrument': PAIR,
            'units': str(plan['units']), 'positionFill': 'OPEN_ONLY', 'priceBound': px(plan['price_bound']),
            'clientExtensions': {'id': cid, 'tag': 'vwap'},
            'tradeClientExtensions': {'id': cid, 'tag': 'vwap', 'comment': f'vwap|{PAIR}|sl0={sl:.5f}'},
            'stopLossOnFill': {'price': px(sl), 'timeInForce': 'GTC'},
            'takeProfitOnFill': {'price': px(tp), 'timeInForce': 'GTC'}}}
    try:
        result = api().request(orders.OrderCreate(accountID=ACCOUNT, data=data))
    except Exception as exc:
        logger.error('[ORDER_UNCERTAIN] %s ; réconciliation requise', short_error(exc))
        return 'UNCERTAIN'
    resolve_response(result)
    fill = result.get('orderFillTransaction') or {}
    tid = str((fill.get('tradeOpened') or {}).get('tradeID', ''))
    if tid and tid in STATE.get('known', {}):
        # Réduire le délai avant recalage ; en cas d'échec, le cycle suivant réessaie.
        info = STATE['known'][tid]
        try:
            ensure_protection({'id': tid, 'entry': info['entry'], 'sl0': info['sl0'],
                               'sl': sl, 'tp': tp, 'trailing': False})
        except StateError:
            raise
        except Exception as exc:
            logger.warning('[TP_A_RECALER] trade=%s %s', tid, short_error(exc))
    return 'UNCERTAIN' if STATE.get('pending') else 'OK'


def mark_done(candle, reason):
    STATE['done_for'] = candle
    STATE['windows'] = {}
    save_state()
    logger.info('[%s] bougie=%s', reason, candle)


def window_valid(candle, now):
    # Échéance absolue : jamais prolongée par attente, week-end ou redémarrage.
    deadline = (pd.Timestamp(candle) + pd.Timedelta(hours=1, minutes=ENTRY_WINDOW_MIN)).isoformat()
    windows = STATE.setdefault('windows', {})
    if candle not in windows:
        STATE['windows'] = {candle: deadline}
        save_state()
    return pd.Timestamp(candle) + pd.Timedelta(hours=1) <= pd.Timestamp(now) < pd.Timestamp(deadline)


def cycle():
    resolved = True
    if STATE.get('pending'):
        try:
            resolved = reconcile_pending()
        except StateError:
            raise
        except Exception as exc:
            logger.error('[PENDING] lecture impossible %s ; entrées bloquées', short_error(exc))
            resolved = False
    mine = our_trades(get_open_trades())  # échec de lecture = échec du cycle
    track_closures(mine)
    for t in mine:
        ensure_protection(t)
    if not resolved or STATE.get('pending'):
        return
    df = get_candles()
    if len(df) < 200:
        return
    candle = df.index[-1].isoformat()
    if STATE.get('done_for') and pd.Timestamp(candle) <= pd.Timestamp(STATE['done_for']):
        return
    sig, sl, band, close = evaluate(df)
    if not sig:
        mark_done(candle, f'ATTENTE close={close:.2f} bande={band:.2f}')
        return
    if mine:
        mark_done(candle, 'POSITION_EXISTANTE')
        return
    if not window_valid(candle, utcnow()):
        mark_done(candle, 'EXPIRE')
        return
    bid, ask, tradeable = get_price()
    if not tradeable or ask - bid > MAX_SPREAD:
        return
    if not 0 < sl < bid:
        mark_done(candle, 'STOP_DEJA_FRANCHI')
        return
    result = open_long(number(px(sl)), ask, candle)
    if result in ('OK', 'SKIP'):
        mark_done(candle, result)


def validate_config():
    if ENVIRONMENT != 'practice':
        raise ValueError('Cette candidate v6 est réservée à OANDA practice')
    if not ACCOUNT or not (os.getenv('OANDA_API_KEY') or os.getenv('OANDA_ACCESS_TOKEN')):
        raise ValueError('Compte ou clé OANDA manquant')
    for value in (RISK_PCT, MAX_SPREAD, MAX_SLIPPAGE, MAX_MARGIN_USAGE_PCT):
        number(value)
    if not (0 < RISK_PCT <= 2 and MAX_SPREAD > 0 and MAX_SLIPPAGE >= 0 and 0 < MAX_MARGIN_USAGE_PCT <= 100):
        raise ValueError('Paramètres hors bornes')
    if len({str(p.resolve()) for p in (STATE_FILE, JOURNAL_FILE, SNAPSHOT_FILE)}) != 3:
        raise ValueError('État, journal et signaux doivent avoir des chemins distincts')


def request_stop(*_):
    global _STOP
    _STOP = True


def main():
    validate_config()
    acquire_lock()
    STATE.update(load_state())
    identity = {'account': ACCOUNT, 'environment': ENVIRONMENT, 'pair': PAIR}
    if STATE.get('identity', identity) != identity:
        raise StateError('État lié à un autre compte/environnement')
    if STATE and 'identity' not in STATE and os.getenv('VWAP_ACCEPT_LEGACY_STATE', 'false').lower() != 'true':
        raise StateError('État v4/v5 : vérifier le compte puis autoriser VWAP_ACCEPT_LEGACY_STATE=true')
    STATE['identity'] = identity
    STATE['schema_version'] = 6
    save_state()
    load_journal_index()
    for path in (JOURNAL_FILE, SNAPSHOT_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.flush()
            os.fsync(f.fileno())
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    logger.info('Démarrage %s | practice | execute=%s risque=%.2f%% | échéance clôture+20min',
                VERSION, EXECUTE_TRADES, RISK_PCT)
    last_hb = 0
    while not _STOP:
        try:
            cycle()
            if time.monotonic() - last_hb >= 3600:
                acct = get_account_response()['account']
                logger.info('[HEARTBEAT] %s solde=%s devise=%s pending=%s', VERSION,
                            acct['balance'], acct['currency'], bool(STATE.get('pending')))
                last_hb = time.monotonic()
        except StateError:
            raise
        except Exception as exc:
            logger.error('[CYCLE_SUSPENDU] %s', short_error(exc))
        for _ in range(CHECK_SECONDS):
            if _STOP:
                break
            time.sleep(1)
    save_state()
    logger.info('Arrêt propre %s', VERSION)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        logger.critical('[ARRET] %s', str(exc)[:180])
        sys.exit(1)
