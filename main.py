import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.types import StopLossDetails, TakeProfitDetails
from oandapyV20.endpoints.accounts import AccountDetails
from oandapyV20.endpoints.positions import OpenPositions
from datetime import datetime, time
import pandas as pd
import time as t
import smtplib
from email.message import EmailMessage
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
INSTRUMENT = "EUR_USD"
RISK_PERCENTAGE = 0.01
TRAILING_ACTIVATION_THRESHOLD_PIPS = 20
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 3.0
SESSION_START = time(7, 0)

client = oandapyV20.API(access_token=ACCESS_TOKEN)

def send_email(subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_ADDRESS
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

def get_account_balance():
    r = AccountDetails(accountID=ACCOUNT_ID)
    client.request(r)
    return float(r.response["account"]["balance"])

def get_atr():
    params = {"granularity": "H1", "count": 20}
    r = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
    client.request(r)
    candles = r.response["candles"]
    highs = [float(c["mid"]["h"]) for c in candles if c["complete"]]
    lows = [float(c["mid"]["l"]) for c in candles if c["complete"]]
    closes = [float(c["mid"]["c"]) for c in candles if c["complete"]]
    tr_list = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        for i in range(1, len(candles))
    ]
    atr = sum(tr_list) / len(tr_list)
    return atr

def calculate_position_size(entry_price, stop_loss_price):
    risk_amount = get_account_balance() * RISK_PERCENTAGE
    pip_value = abs(entry_price - stop_loss_price)
    units = risk_amount / pip_value if pip_value != 0 else 0
    return int(units)

def get_current_price(direction):
    r = instruments.InstrumentsPrice(instrument=INSTRUMENT)
    client.request(r)
    price = float(r.response['prices'][0]['bids'][0]['price']) if direction == "sell" else float(r.response['prices'][0]['asks'][0]['price'])
    return price

def generate_trade_signal():
    params = {"granularity": "H1", "count": 50}
    r = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
    client.request(r)
    candles = r.response["candles"]
    closes = [float(c["mid"]["c"]) for c in candles if c["complete"]]
    if len(closes) < 30:
        return None
    df = pd.DataFrame(closes, columns=['close'])
    df['sma_fast'] = df['close'].rolling(window=10).mean()
    df['sma_slow'] = df['close'].rolling(window=30).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['macd'], df['macd_signal'] = compute_macd(df['close'])

    signal = None
    if df['sma_fast'].iloc[-2] < df['sma_slow'].iloc[-2] and df['sma_fast'].iloc[-1] > df['sma_slow'].iloc[-1] and df['rsi'].iloc[-1] > 50 and df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
        signal = "buy"
    elif df['sma_fast'].iloc[-2] > df['sma_slow'].iloc[-2] and df['sma_fast'].iloc[-1] < df['sma_slow'].iloc[-1] and df['rsi'].iloc[-1] < 50 and df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]:
        signal = "sell"
    return signal

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, short=12, long=26, signal=9):
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Les autres fonctions place_trade(), update_trailing_stop_tp(), modify_trade(), monitor_trade(), main() restent inchangées.
