#!/usr/bin/env python3
"""VWAP mensuel or H1 — v6.4 autonome DEMO, exécution H4/H1 sur quatre marchés.
Signal identique à v4 ; aucune promesse de rendement. Compte practice uniquement.
"""
import fcntl
import hashlib
import json
import logging
import math
import os
import signal
import re
import urllib.error
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


VERSION = 'vwap-v6.4-multipair-demo'
PATTERN_ENABLED = _env_bool('PATTERN_ENABLED', True)
# PATTERN_SHADOW est obsolète : EXECUTE_TRADES commande les deux stratégies.
PATTERN_SHADOW = False
MAX_TOTAL_RISK_PCT = float(os.getenv('MAX_TOTAL_RISK_PCT', '2.0'))
MAX_OPEN_TRADES = 4
PAIR_PRECISION = {'XAU_USD': 3, 'EUR_USD': 5, 'GBP_USD': 5, 'USD_JPY': 3}
PAIR_SPREAD = {'XAU_USD': 1.0, 'EUR_USD': .00020, 'GBP_USD': .00025, 'USD_JPY': .025}
PAIR_SLIPPAGE = {'XAU_USD': 1.0, 'EUR_USD': .00005, 'GBP_USD': .00005, 'USD_JPY': .005}
PATTERN_PAIRS = ('XAU_USD', 'EUR_USD', 'GBP_USD', 'USD_JPY')
_PATTERN_LAST = {}
_PATTERN_NEXT = {}
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
    if isinstance(exc, ModuleNotFoundError):
        name = re.sub(r"[^A-Za-z0-9_.-]", "?", exc.name or "inconnu")[:100]
        return f"ModuleNotFoundError module={name}"
    if isinstance(exc, urllib.error.HTTPError):
        # Aucune URL, query, clé ni corps HTML ; chemin contrôlé ajouté au transport.
        return (f'HTTPError status={exc.code} method={getattr(exc, "bot_method", "?")} '
                f'endpoint={getattr(exc, "bot_endpoint", "?")}')
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
        try:
            with self.opener.open(request, timeout=15) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            exc.bot_method = req.method
            exc.bot_endpoint = re.sub(r'/accounts/[^/]+', '/accounts/<compte>', req.path)
            raise


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
        for name in ('known', 'fills', 'windows', 'pattern_done'):
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


def get_candles(pair=PAIR, retries=2):
    result = read(instruments.InstrumentsCandles(instrument=pair, params={
        'granularity': 'H1', 'price': 'M', 'count': CANDLES}), retries=retries)
    if result.get('instrument') != pair:
        raise RuntimeError('Historique reçu pour un autre instrument')
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


def price_text(pair, value):
    return f'{number(value):.{PAIR_PRECISION[pair]}f}'


def quote_home(pair, acct):
    """Cotation fraîche et facteurs OANDA quote -> devise du compte."""
    response = read(pricing.PricingInfo(accountID=ACCOUNT, params={
        'instruments': pair, 'includeHomeConversions': 'true'}))
    matches = [p for p in response['prices'] if p.get('instrument') == pair]
    if len(matches) != 1:
        raise ValueError('Cotation instrument absente')
    p = matches[0]
    bid, ask = number(p['bids'][0]['price']), number(p['asks'][0]['price'])
    stamp = pd.Timestamp(p['time'])
    age = (pd.Timestamp(utcnow()) - stamp).total_seconds()
    if stamp.tzinfo is None or not (0 < bid <= ask and -5 <= age <= 60):
        raise ValueError('Cotation invalide ou périmée')
    if p.get('tradeable') is not True:
        raise ValueError('Marché fermé')
    quote = pair.split('_')[1]
    if quote == acct['currency']:
        loss, value = 1., 1.
    else:
        conversion = next(c for c in response['homeConversions'] if c['currency'] == quote)
        loss = max(number(conversion['accountLoss']), number(conversion['accountGain']))
        value = number(conversion['positionValue'])
    if min(loss, value) <= 0:
        raise ValueError('Conversion invalide')
    return {'bid': bid, 'ask': ask, 'quote_to_home': loss, 'value_to_home': value}


def available_risk(pair, acct, all_trades):
    """Une position par instrument ; budget cumulé initial et stops effectifs.
    Les positions manuelles consomment aussi le budget. Un stop inconnu bloque.
    Ce plafond théorique exclut gaps, financement et évolution des conversions.
    """
    if len(all_trades) >= MAX_OPEN_TRADES or any(t['instrument'] == pair for t in all_trades):
        return 0.
    budget = min(number(acct['balance']), number(acct['NAV'])) * MAX_TOTAL_RISK_PCT / 100
    for t in all_trades:
        units = number(t['currentUnits'])
        if not units:
            continue
        stop = (t.get('stopLossOrder') or {}).get('price')
        if stop is None:
            return 0.
        q = quote_home(t['instrument'], acct)
        effective = max(0., (number(t['price']) - number(stop)) * units) * q['quote_to_home']
        info = STATE.get('known', {}).get(str(t['id']), {})
        initial = 0.
        if info.get('sl0') is not None:
            initial = abs(number(t['price']) - number(info['sl0'])) * abs(units) * q['quote_to_home']
        budget -= max(effective, initial)
    return max(0., budget)


def pattern_plan(pair, side, sl, acct, q, budget):
    item = read(accounts.AccountInstruments(accountID=ACCOUNT, params={'instruments': pair}))['instruments'][0]
    precision = PAIR_PRECISION[pair]
    if item['name'] != pair or int(item['displayPrecision']) != precision or side not in (-1, 1):
        raise ValueError('Instrument ou direction invalide')
    sl = number(price_text(pair, sl))
    entry = q['ask'] if side == 1 else q['bid']
    bound = number(price_text(pair, entry + side * PAIR_SLIPPAGE[pair]))
    dist = side * (bound - sl)
    if sl <= 0 or side * ((q['bid'] if side == 1 else q['ask']) - sl) <= 0 or dist <= 0:
        return {'units': 0}
    if q['ask'] - q['bid'] > PAIR_SPREAD[pair] or (q['ask'] - q['bid']) / (side * (entry - sl)) > RULE.max_entry_spread_over_risk:
        return {'units': 0}
    rate = max(number(item['marginRate']), number(acct['marginRate']))
    if not 0 < rate <= 1:
        raise ValueError('Marge invalide')
    nav, available, used = number(acct['NAV']), number(acct['marginAvailable']), number(acct['marginUsed'])
    cap = max(0., min(available * .95, nav * MAX_MARGIN_USAGE_PCT / 100 - used))
    risk = max(0., min(min(number(acct['balance']), nav) * RISK_PCT / 100, budget))
    # Majoration de conversion de 1 % pour la cotation finale avant envoi.
    conversion = q['quote_to_home'] * 1.01
    per_unit_margin = max(q['ask'], bound) * q['value_to_home'] * 1.01 * rate
    u = max(0, min(math.floor(risk / (dist * conversion)), math.floor(cap / per_unit_margin),
                   int(number(item['maximumOrderUnits'])), MAX_UNITS if pair == PAIR else 200000))
    if u < max(1, math.ceil(number(item['minimumTradeSize']))):
        u = 0
    return {'units': u, 'currency': acct['currency'], 'quote_to_home': q['quote_to_home'],
            'quote_to_home_limit': conversion,
            'price_bound': bound, 'risk_home_max': u * dist * conversion,
            'margin_rate': rate, 'pair': pair, 'side': side, 'strategy': 'h4h1',
            'sl0': sl, 'ask': entry, 'precision': precision}


def our_trades(all_trades):
    found = []
    for t in all_trades:
        parts = str((t.get('clientExtensions') or {}).get('comment', '')).split('|')
        if parts[0] not in ('vwap', 'h4h1'):
            continue
        if len(parts) != 3 or parts[1] not in PATTERN_PAIRS or not parts[2].startswith('sl0='):
            raise RuntimeError('Marqueur bot invalide : entrée suspendue')
        strategy, pair = parts[:2]
        sl, entry, units = number(parts[2][4:]), number(t['price']), number(t['currentUnits'])
        side = 1 if units > 0 else -1
        if t['instrument'] != pair or not units or sl <= 0 or side * (entry-sl) <= 0:
            raise RuntimeError('Trade bot incohérent : entrée suspendue')
        if strategy == 'vwap' and (pair != PAIR or side != 1):
            raise RuntimeError('Trade VWAP incohérent')
        found.append({'id': str(t['id']), 'pair': pair, 'side': side, 'strategy': strategy,
                      'entry': entry, 'sl0': sl, 'units': abs(number(t.get('initialUnits', units))),
                      'opened_at': t['openTime'], 'tp': (t.get('takeProfitOrder') or {}).get('price'),
                      'sl': (t.get('stopLossOrder') or {}).get('price'),
                      'trailing': bool(t.get('trailingStopLossOrder'))})
    return found


def consume_pending(p):
    if p.get('strategy') == 'h4h1':
        STATE.setdefault('pattern_done', {})[p['pair']] = p['candle']
    else:
        STATE['done_for'] = p['candle']


def register_fill(fill, pending):
    opened = fill.get('tradeOpened') or {}
    if not opened.get('tradeID'):
        raise RuntimeError('Exécution sans tradeOpened : réconciliation bloquée')
    tid = str(opened['tradeID'])
    entry = number(opened.get('price', fill['price']))
    signed_units = number(opened['units'])
    u, side = abs(signed_units), pending.get('side', 1)
    if u <= 0 or signed_units * side <= 0 or side * (entry - pending['sl0']) <= 0:
        raise RuntimeError('Exécution incohérente : vérifier le compte')
    k = pending.get('quote_to_home')
    if k is None:
        k = 1 / pending['usd_per_ccy']
    info = {'entry': entry, 'sl0': pending['sl0'], 'units': u,
            'pair': pending.get('pair', PAIR), 'side': side, 'strategy': pending.get('strategy', 'vwap'),
            'opened_at': fill['time'], 'currency': pending['currency'],
            'usd_per_ccy': pending.get('usd_per_ccy'), 'quote_to_home': k,
            'risk_home': u * side * (entry - pending['sl0']) * k,
            'slippage': side * (entry - pending['ask']), 'tp_sent': pending['tp_sent'],
            'tp_rebased': False, 'client_id': pending['client_id']}
    STATE.setdefault('known', {})[tid] = info
    consume_pending(pending)
    STATE['pending'] = None
    save_state()
    logger.info('[ORDER_OK] stratégie=%s pair=%s direction=%s trade=%s unités=%s prix=%s risque_max_prévu=%.2f',
                info['strategy'], info['pair'], 'BUY' if side == 1 else 'SELL',
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
        consume_pending(pending)
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
    pair, side = info.get('pair', PAIR), info.get('side', 1)
    tol = 1.1 * 10 ** -PAIR_PRECISION[pair]
    target = t['entry'] + RR * (t['entry'] - t['sl0'])
    data = {}
    if not t['trailing'] and (t['sl'] is None or side * (number(t['sl']) - t['sl0']) < -tol):
        data['stopLoss'] = {'price': price_text(pair, t['sl0']), 'timeInForce': 'GTC'}
    rebase = not info.get('tp_rebased', True)
    manual_tp = rebase and t['tp'] is not None and abs(number(t['tp']) - info['tp_sent']) > tol
    if t['tp'] is None or (rebase and not manual_tp and abs(number(t['tp']) - target) > tol):
        data['takeProfit'] = {'price': price_text(pair, target), 'timeInForce': 'GTC'}
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
            known[t['id']] = {k: t[k] for k in ('entry', 'sl0', 'units', 'opened_at', 'pair', 'side', 'strategy') if k in t}
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
        rec = {'version': VERSION, 'id': tid, 'pair': info.get('pair', PAIR), 'dir': 'BUY' if info.get('side', 1) == 1 else 'SELL',
               'strategy': info.get('strategy', 'vwap'),
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
    if available_risk(PAIR, acct, get_open_trades()) + 1e-8 < plan['risk_home_max']:
        logger.info('[SKIP] exposition totale ou position existante')
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


# Calcul H4/H1 intégré : aucun fichier recherche_pattern.py requis.
@dataclass(frozen=True)
class Rule:
    trend_ema_h4: int = 50
    trend_slope_h4: int = 3
    pullback_ema_h1: int = 20
    pullback_window: int = 3
    trigger_window: int = 2
    atr_h1: int = 14
    stop_buffer_atr: float = 0.1
    rr: float = 2.0
    max_entry_spread_over_risk: float = 0.1

RULE = Rule()
OHLC = ('open', 'high', 'low', 'close')


def features(df):
    """Index = ouverture H1 ; toutes les colonnes décrivent la décision à sa clôture."""
    h1=df.rename(columns={f'mid_{k}':k for k in OHLC})[list(OHLC)].copy()
    groups=h1.resample('4h',origin='start_day',closed='left',label='left')
    h4=groups.agg({'open':'first','high':'max','low':'min','close':'last'})
    # Quatre heures réellement présentes, aucune H4 partielle à travers une fermeture.
    h4=h4[groups['close'].count()==4].dropna()
    h4['ema']=h4.close.ewm(span=RULE.trend_ema_h4,adjust=False,min_periods=RULE.trend_ema_h4).mean()
    h4['slope']=h4.ema-h4.ema.shift(RULE.trend_slope_h4)
    h4['context']=np.select([(h4.close>h4.ema)&(h4.slope>0),
                              (h4.close<h4.ema)&(h4.slope<0)],[1,-1],default=0)
    # Une H4 ouverte à 00h n'est disponible qu'à 04h. Jointure à la clôture H1.
    h4.index=h4.index+pd.Timedelta(hours=4)
    times=h1.index+pd.Timedelta(hours=1)
    context=h4[['context']].reindex(times,method='ffill',tolerance=pd.Timedelta(hours=4))
    out=pd.DataFrame(index=df.index)
    out['context']=context['context'].fillna(0).to_numpy(dtype=int)
    out['ema20']=h1.close.ewm(span=RULE.pullback_ema_h1,adjust=False,min_periods=RULE.pullback_ema_h1).mean()
    prev=h1.close.shift(1)
    tr=pd.concat([h1.high-h1.low,(h1.high-prev).abs(),(h1.low-prev).abs()],axis=1).max(axis=1)
    out['atr']=tr.rolling(RULE.atr_h1,min_periods=RULE.atr_h1).mean()
    # Repli : une des 3 bougies PRÉCÉDENTES touche EMA20 et une clôture va à contresens.
    touched=(h1.low<=out.ema20)&(h1.high>=out.ema20)
    touch=touched.shift(1).rolling(RULE.pullback_window).max().fillna(0).astype(bool)
    down=(h1.close<h1.open).shift(1).rolling(RULE.pullback_window).max().fillna(0).astype(bool)
    up=(h1.close>h1.open).shift(1).rolling(RULE.pullback_window).max().fillna(0).astype(bool)
    prev_high=h1.high.shift(1).rolling(RULE.trigger_window).max()
    prev_low=h1.low.shift(1).rolling(RULE.trigger_window).min()
    buy=(out.context==1)&touch&down&(h1.close>prev_high)&(h1.close>out.ema20)
    sell=(out.context==-1)&touch&up&(h1.close<prev_low)&(h1.close<out.ema20)
    out['side']=np.select([buy,sell],[1,-1],default=0)
    # Un même élan consécutif ne génère qu'un déclenchement.
    repeat=out.side.eq(out.side.shift(1))&out.side.ne(0)
    out.loc[repeat,'side']=0
    out['sl_buy']=h1.low.rolling(RULE.pullback_window+1).min()-RULE.stop_buffer_atr*out.atr
    out['sl_sell']=h1.high.rolling(RULE.pullback_window+1).max()+RULE.stop_buffer_atr*out.atr
    return out


def open_pattern(pair, side, sl, candle):
    if ENVIRONMENT != 'practice':
        raise ValueError('Practice uniquement')
    if STATE.get('pending'):
        return 'RETRY'
    response = get_account_response()
    acct = response['account']
    budget = available_risk(pair, acct, get_open_trades())
    if budget <= 0:
        logger.info('[PATTERN_BLOQUE] pair=%s position existante ou budget atteint', pair)
        return 'SKIP'
    q = quote_home(pair, acct)
    plan = pattern_plan(pair, side, sl, acct, q, budget)
    if plan['units'] < 1:
        logger.info('[PATTERN_BLOQUE] pair=%s taille, spread ou stop invalide', pair)
        return 'SKIP'
    tp = number(price_text(pair, plan['ask'] + RR * (plan['ask'] - plan['sl0'])))
    if min(tp, plan['price_bound']) <= 0:
        return 'SKIP'
    if not EXECUTE_TRADES:
        logger.info('[DRY_RUN] H4/H1 pair=%s direction=%s unités=%s', pair, side, plan['units'])
        return 'OK'
    # Prix et conversion relus juste avant l'envoi ; aucune répétition d'un POST.
    fresh = quote_home(pair, acct)
    entry = fresh['ask'] if side == 1 else fresh['bid']
    close_side = fresh['bid'] if side == 1 else fresh['ask']
    if (side * (entry - plan['price_bound']) > 0 or side * (close_side - plan['sl0']) <= 0 or
            fresh['ask'] - fresh['bid'] > PAIR_SPREAD[pair] or
            fresh['quote_to_home'] > plan['quote_to_home_limit'] or
            fresh['value_to_home'] > q['value_to_home'] * 1.01):
        return 'RETRY'
    if not (pd.Timestamp(candle) + pd.Timedelta(hours=1) <= pd.Timestamp(utcnow()) <
            pd.Timestamp(candle) + pd.Timedelta(hours=1, minutes=ENTRY_WINDOW_MIN)):
        return 'SKIP'
    seq = int(STATE.get('order_seq', 0)) + 1
    STATE['order_seq'] = seq
    cid = 'h41-' + hashlib.sha256(f'{ACCOUNT}|{pair}|{candle}|{seq}'.encode()).hexdigest()[:24]
    pending = {**plan, 'client_id': cid, 'since_id': str(response['lastTransactionID']),
               'candle': candle, 'tp_sent': tp}
    log_event(SNAPSHOT_FILE, 'PATTERN_SIGNAL', {'version': VERSION, **pending, 'ts': utcnow().isoformat()})
    STATE['pending'] = pending
    save_state()
    order = {'type': 'MARKET', 'timeInForce': 'FOK', 'instrument': pair,
             'units': str(side * plan['units']), 'positionFill': 'OPEN_ONLY',
             'priceBound': price_text(pair, plan['price_bound']),
             'clientExtensions': {'id': cid, 'tag': 'h4h1'},
             'tradeClientExtensions': {'id': cid, 'tag': 'h4h1',
                 'comment': f"h4h1|{pair}|sl0={price_text(pair, plan['sl0'])}"},
             'stopLossOnFill': {'price': price_text(pair, plan['sl0']), 'timeInForce': 'GTC'},
             'takeProfitOnFill': {'price': price_text(pair, tp), 'timeInForce': 'GTC'}}
    try:
        result = api().request(orders.OrderCreate(accountID=ACCOUNT, data={'order': order}))
    except Exception as exc:
        logger.error('[ORDER_UNCERTAIN] pair=%s %s ; réconciliation requise', pair, short_error(exc))
        return 'UNCERTAIN'
    resolve_response(result)
    return 'UNCERTAIN' if STATE.get('pending') else 'OK'


def scan_pattern_pairs():
    """Décision sur H1 clôturée ; exécution démo autorisée par EXECUTE_TRADES.
    Les bougies consommées persistent. Panne réseau : réessai sous une minute.
    """
    if not PATTERN_ENABLED:
        return
    labels = {1: 'BUY', -1: 'SELL', 0: 'NONE'}
    for pair in PATTERN_PAIRS:
        if _STOP:
            return
        tick = time.monotonic()
        if tick < _PATTERN_NEXT.get(pair, 0):
            continue
        _PATTERN_NEXT[pair] = tick + 60
        try:
            df = get_candles(pair, retries=0)
            if len(df) < 240:
                logger.warning('[PATTERN_INDISPONIBLE] pair=%s historique=%s bougies', pair, len(df))
                continue
            candle = df.index[-1]
            now = pd.Timestamp(utcnow())
            if candle + pd.Timedelta(hours=1) > now:
                raise ValueError('Bougie H1 non clôturée')
            done = STATE.get('pattern_done', {}).get(pair)
            if done and candle <= pd.Timestamp(done):
                continue
            candidate = features(df.rename(columns={k: 'mid_' + k for k in OHLC})).iloc[-1]
            side = int(candidate['side'])
            logger.info('[PATTERN_SCAN] pair=%s bougie=%s contexte_H4=%s candidat=%s mode=%s',
                        pair, candle.isoformat(), labels[int(candidate['context'])], labels[side],
                        'execution_demo' if EXECUTE_TRADES else 'simulation')
            result = 'SKIP'
            if side and candle + pd.Timedelta(hours=1) <= now < candle + pd.Timedelta(hours=1, minutes=ENTRY_WINDOW_MIN):
                sl = number(candidate['sl_buy' if side == 1 else 'sl_sell'])
                result = open_pattern(pair, side, sl, candle.isoformat())
            elif side:
                logger.info('[PATTERN_EXPIRE] pair=%s bougie=%s', pair, candle.isoformat())
            if result in ('OK', 'SKIP'):
                STATE.setdefault('pattern_done', {})[pair] = candle.isoformat()
                save_state()
                seconds = max(30, (now.floor('h') + pd.Timedelta(hours=1, seconds=2) - now).total_seconds())
                _PATTERN_NEXT[pair] = time.monotonic() + seconds
        except StateError:
            raise
        except Exception as exc:
            logger.warning('[PATTERN_INDISPONIBLE] pair=%s %s ; nouvel essai dans 1 min',
                           pair, short_error(exc))


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
    if any(t.get('pair', PAIR) == PAIR for t in mine):
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
    for value in (RISK_PCT, MAX_SPREAD, MAX_SLIPPAGE, MAX_MARGIN_USAGE_PCT, MAX_TOTAL_RISK_PCT):
        number(value)
    if not (0 < RISK_PCT <= 2 and MAX_SPREAD > 0 and MAX_SLIPPAGE >= 0 and 0 < MAX_MARGIN_USAGE_PCT <= 100 and RISK_PCT <= MAX_TOTAL_RISK_PCT <= 4):
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
    logger.info('[PATTERN_CONFIG] enabled=%s mode=%s pairs=%s risque_total_max=%.2f%% max_positions=%s',
                PATTERN_ENABLED, 'execution_demo' if EXECUTE_TRADES else 'simulation',
                ','.join(PATTERN_PAIRS), MAX_TOTAL_RISK_PCT, MAX_OPEN_TRADES)
    last_hb = 0
    while not _STOP:
        management_ok = False
        try:
            cycle()
            management_ok = True
            if time.monotonic() - last_hb >= 3600:
                acct = get_account_response()['account']
                logger.info('[HEARTBEAT] %s solde=%s devise=%s pending=%s', VERSION,
                            acct['balance'], acct['currency'], bool(STATE.get('pending')))
                last_hb = time.monotonic()
        except StateError:
            raise
        except Exception as exc:
            logger.error('[CYCLE_SUSPENDU] %s', short_error(exc))
        if management_ok:
            scan_pattern_pairs()
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
