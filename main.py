import os
import time
import threading
import logging
from datetime import datetime, timedelta
import schedule
import pandas as pd
import pandas_ta as ta
import ccxt
from flask import Flask
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AsianSessionTrader:
    def __init__(self):
        self.exchange = self.configure_exchange()
        self.symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        self.risk_per_trade = 0.01
        self.tp_percent = 2.0
        self.sl_percent = 1.0
        self.trailing_stop_percent = 0.5
        self.break_even_trigger = 1.0
        self.session_data = {}
        self.active_trades = {}
        self.asian_session = {"start": {"hour": 23, "minute": 0}, "end": {"hour": 10, "minute": 0}}

        if not os.path.exists("reports"):
            os.makedirs("reports")

        self.update_balance()

    def configure_exchange(self):
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'urls': {
                'api': {
                    'public': 'https://testnet.binance.vision/api',
                    'private': 'https://testnet.binance.vision/api',
                },
            },
            'options': {'adjustForTimeDifference': True, 'enableRateLimit': True},
        })
        exchange.set_sandbox_mode(True)
        return exchange

    def update_balance(self):
        balance = self.exchange.fetch_balance()
        usdt_balance = balance.get("total", {}).get("USDT", 0)
        logging.info(f"Solde actuel : {usdt_balance} USDT")

    def is_within_session(self, current_time):
        start = self.asian_session['start']
        end = self.asian_session['end']
        start_time = timedelta(hours=start['hour'], minutes=start['minute'])
        end_time = timedelta(hours=end['hour'], minutes=end['minute'])
        now_time = timedelta(hours=current_time.hour, minutes=current_time.minute)
        if start_time < end_time:
            return start_time <= now_time <= end_time
        else:
            return now_time >= start_time or now_time <= end_time

    def analyze_session(self):
        try:
            for symbol in self.symbols:
                ohlcv = self.exchange.fetch_ohlcv(symbol, "1h", limit=200)
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("datetime", inplace=True)

                df["ema200"] = ta.ema(df["close"], length=200)
                df["rsi"] = ta.rsi(df["close"], length=14)
                macd = ta.macd(df["close"])
                if macd is None or macd.empty:
                    continue
                macd = macd.dropna()

                if df["close"].iloc[-1] > df["ema200"].iloc[-1] and df["rsi"].iloc[-1] > 50 and macd.iloc[-1]["MACD_12_26_9"] > 0:
                    self.session_data[symbol] = {
                        "ema200": df["ema200"].iloc[-1],
                        "rsi": df["rsi"].iloc[-1],
                        "macd": macd.iloc[-1]["MACD_12_26_9"],
                        "trend_ok": True
                    }
                else:
                    self.session_data[symbol] = {"trend_ok": False}
        except Exception as e:
            logging.error(f"Erreur analyse session : {str(e)}")

    def execute_post_session_trades(self):
        for symbol in self.symbols:
            data = self.session_data.get(symbol, {})
            if not data.get("trend_ok"):
                logging.info(f"Pas d'entrée pour {symbol} - tendance non confirmée.")
                continue

            price = self.exchange.fetch_ticker(symbol)['last']
            usdt_balance = self.exchange.fetch_balance()['total']['USDT']
            trade_amount = self.risk_per_trade * usdt_balance / price
            sl = price * (1 - self.sl_percent / 100)
            tp = price * (1 + self.tp_percent / 100)

            try:
                order = self.exchange.create_market_buy_order(symbol, trade_amount)
                logging.info(f"ORDRE exécuté {symbol}: {order}")
                self.active_trades[symbol] = {
                    "entry": price,
                    "sl": sl,
                    "tp": tp,
                    "amount": trade_amount,
                    "order_id": order['id'],
                    "open": True,
                    "trailing_stop": sl
                }
            except Exception as e:
                logging.error(f"Erreur exécution ordre : {e}")

    def manage_take_profit_stop_loss(self, symbol, trade):
        try:
            price = self.exchange.fetch_ticker(symbol)['last']

            if price >= trade['tp']:
                trade['open'] = False
                self.exchange.create_market_sell_order(symbol, trade['amount'])
                logging.info(f"TP atteint {symbol} à {price:.2f}")
                return

            if price <= trade['sl']:
                trade['open'] = False
                self.exchange.create_market_sell_order(symbol, trade['amount'])
                logging.info(f"SL touché {symbol} à {price:.2f}")
                return

            new_sl = price * (1 - self.trailing_stop_percent / 100)
            if new_sl > trade["sl"]:
                logging.info(f"🔁 Trailing SL mis à jour pour {symbol} : {trade['sl']:.2f} → {new_sl:.2f}")
                trade["sl"] = new_sl

            if price >= trade["entry"] * (1 + self.break_even_trigger / 100) and trade["sl"] < trade["entry"]:
                logging.info(f"🔐 Break-even activé pour {symbol} → SL remonté à l'entrée : {trade['entry']:.2f}")
                trade["sl"] = trade["entry"]

        except Exception as e:
            logging.error(f"Erreur SL/TP dynamique : {e}")

    def monitor_trades(self):
        for symbol, trade in list(self.active_trades.items()):
            if trade.get("open"):
                self.manage_take_profit_stop_loss(symbol, trade)

def scheduled_task():
    logging.info("===== Tâche programmée =====")
    trader.analyze_session()
    trader.execute_post_session_trades()
    trader.monitor_trades()

@app.route("/")
def home():
    return "Asian Session Bot is running 🚀", 200

def monitor_trades_runner(trader):
    while True:
        open_trades = len([t for t in trader.active_trades.values() if t.get("open")])
        logging.info(f"💓 Monitor tick | Trades actifs : {open_trades}")
        trader.monitor_trades()
        time.sleep(60)


def run_scheduler():
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logging.error(f"Erreur scheduler : {e}")
        time.sleep(10)

if __name__ == "__main__":
    trader = AsianSessionTrader()
    scheduled_task()
    schedule.every().day.at("02:10").do(scheduled_task)

    threading.Thread(target=run_scheduler, daemon=True).start()
    threading.Thread(target=monitor_trades_runner, args=(trader,), daemon=True).start()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)
