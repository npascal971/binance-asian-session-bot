import ccxt
import time
import pandas as pd
import pandas_ta as ta
import csv
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_API_SECRET'),
    'urls': {
        'api': {
            'public': 'https://testnet.binance.vision/api',
            'private': 'https://testnet.binance.vision/api',
        },
    },
    'options': {
        'adjustForTimeDifference': True,
        'enableRateLimit': True,
    },
})

exchange.set_sandbox_mode(True)

try:
    balance = exchange.fetch_balance()
    print(f"Solde disponible : {balance['total']['USDT']} USDT")
except Exception as e:
    print(f"Erreur : {e}")

# (Le reste du code est inchangé, à compléter ici si besoin)