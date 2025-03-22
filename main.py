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
        self.session_data = {}
        self.active_trades = {}
        self.asian_session = {"start": {"hour": 23, "minute": 0}, "end": {"hour": 10, "minute": 0}}

        if not os.path.exists("reports"):
            os.makedirs("reports")

        self.update_balance()
        logging.info(f"Configuration session : {self.asian_session}")
        logging.info(f"UTC maintenant : {datetime.utcnow()}")

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
            'options': {
                'adjustForTimeDifference': True,
                'enableRateLimit': True,
            },
        })
        exchange.set_sandbox_mode(True)
        return exchange

    def update_balance(self):
        balance = self.exchange.fetch_balance()
        usdt_balance = balance.get("total", {}).get("USDT", 0)
        logging.info(f"Solde actuel : {usdt_balance} USDT")

    def get_session_start(self):
        now = datetime.utcnow()
        start_time = now.replace(hour=self.asian_session['start']['hour'], minute=self.asian_session['start']['minute'], second=0, microsecond=0)
        if now < start_time:
            start_time -= timedelta(days=1)
        return int(start_time.timestamp() * 1000)

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
                ohlcv = self.exchange.fetch_ohlcv(symbol, "1h", limit=100)
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("datetime")

                session_start = datetime.utcnow().replace(hour=self.asian_session['start']['hour'], 
                                                          minute=self.asian_session['start']['minute'], 
                                                          second=0, microsecond=0)
                if datetime.utcnow() < session_start:
                    session_start -= timedelta(days=1)
                session_end = session_start + timedelta(hours=11)

                df_session = df[(df.index >= session_start) & (df.index <= session_end)]

                if len(df["close"]) < 30:
                    logging.warning(f"Trop peu de données pour MACD sur {symbol}")
                    continue

                df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
                df["ema200"] = ta.ema(df["close"], length=200)
                df["rsi"] = ta.rsi(df["close"], length=14)
                macd = ta.macd(df["close"])
                if macd is None or macd.empty:
                    continue

                macd = macd.dropna()
                if macd.empty:
                    continue

                df_ltf = pd.DataFrame(self.exchange.fetch_ohlcv(symbol, timeframe="5m"),
                                      columns=["timestamp", "open", "high", "low", "close", "volume"])

                ob = self.detect_order_block(df)
                fvg = self.detect_fvg(df)
                structure = self.detect_structure_break(df_ltf)

                self.session_data[symbol] = {
                    "high": df["high"].max(),
                    "low": df["low"].min(),
                    "vwap": df["vwap"].mean(),
                    "ema200": df["ema200"].iloc[-1],
                    "rsi": df["rsi"].iloc[-1],
                    "macd": macd.iloc[-1]["MACD_12_26_9"],
                    "order_block": ob,
                    "fvg": fvg,
                    "structure_break": structure,
                }

            logging.info("Analyse de session terminée")
            self.save_report_to_file(self.generate_report())

        except Exception as e:
            logging.error(f"Erreur analyse session : {str(e)}")

    def detect_order_block(self, df):
        return "Order Block Exemple"

    def detect_fvg(self, df):
        return "FVG Exemple"

    def detect_structure_break(self, df):
        return "Structure Break Exemple"

    def safe_fmt(self, value, fmt=":.2f", default="N/A"):
        try:
            if pd.isna(value) or value is None:
                return default
            return format(value, fmt)
        except Exception:
            return default

    def generate_report(self):
        report = "Rapport de Session\n\n"
        for symbol, data in self.session_data.items():
            report += f"{symbol}\n"
            report += f"- HIGH: {self.safe_fmt(data.get('high'))}\n"
            report += f"- LOW: {self.safe_fmt(data.get('low'))}\n"
            report += f"- VWAP: {self.safe_fmt(data.get('vwap'))}\n"
            report += f"- EMA200: {self.safe_fmt(data.get('ema200'))}\n"
            report += f"- RSI: {self.safe_fmt(data.get('rsi'))}\n"
            report += f"- MACD: {self.safe_fmt(data.get('macd'), ':.4f')}\n"
            report += f"- OB: {data.get('order_block', 'N/A')}\n"
            report += f"- FVG: {data.get('fvg', 'N/A')}\n"
            report += f"- Structure: {data.get('structure_break', 'N/A')}\n\n"
        return report

    def save_report_to_file(self, text_report):
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
            filename_txt = f"reports/report_{timestamp}.txt"
            with open(filename_txt, "w") as file:
                file.write(text_report)

            filename_csv = f"reports/report_{timestamp}.csv"
            df = pd.DataFrame(self.session_data).T
            df.to_csv(filename_csv)

            logging.info(f"Rapport sauvegardé : {filename_txt} / {filename_csv}")
        except Exception as e:
            logging.error(f"Erreur sauvegarde rapport : {str(e)}")

    def send_email(self, subject, body):
        sender_email = os.getenv('EMAIL_ADDRESS')
        receiver_email = sender_email
        password = os.getenv('EMAIL_PASSWORD')

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender_email, password)
            text = message.as_string()
            server.sendmail(sender_email, receiver_email, text)
            server.quit()
            logging.info(f"E-mail envoyé avec succès - Sujet : {subject}")
        except Exception as e:
            logging.error(f"Erreur lors de l'envoi de l'e-mail : {e}")

    def execute_post_session_trades(self):
        for symbol in self.symbols:
            if symbol not in self.session_data:
                continue
            price = self.exchange.fetch_ticker(symbol)['last']
            usdt_balance = self.exchange.fetch_balance()['total']['USDT']
            trade_amount = self.risk_per_trade * usdt_balance / price
            sl = price * (1 - self.sl_percent / 100)
            tp = price * (1 + self.tp_percent / 100)

            try:
                order = self.exchange.create_market_buy_order(symbol, trade_amount)
                logging.info(f"✅ ORDRE exécuté pour {symbol} : {order}")

                self.active_trades[symbol] = {
                    "entry": price,
                    "sl": sl,
                    "tp": tp,
                    "amount": trade_amount,
                    "order_id": order['id'],
                    "open": True
                }

            except Exception as e:
                logging.error(f"❌ Échec de l'ordre sur {symbol} : {e}")

    def manage_take_profit_stop_loss(self, symbol, trade):
        try:
            price = self.exchange.fetch_ticker(symbol)['last']
            if price >= trade['tp']:
                trade['open'] = False
                sell_order = self.exchange.create_market_sell_order(symbol, trade['amount'])
                logging.info(f"✅ TP atteint pour {symbol} à {price:.2f} - Vente exécutée : {sell_order}")
                self.send_email(f"TP atteint - {symbol}", f"Le TP a été atteint pour {symbol} à {price:.2f}\nPosition clôturée.")
            elif price <= trade['sl']:
                trade['open'] = False
                sell_order = self.exchange.create_market_sell_order(symbol, trade['amount'])
                logging.info(f"❌ SL touché pour {symbol} à {price:.2f} - Vente exécutée : {sell_order}")
                self.send_email(f"SL touché - {symbol}", f"Le SL a été touché pour {symbol} à {price:.2f}\nPosition clôturée.")
        except Exception as e:
            logging.error(f"Erreur gestion TP/SL pour {symbol} : {e}")

    def monitor_trades(self):
        for symbol, trade in list(self.active_trades.items()):
            if trade.get("open"):
                self.manage_take_profit_stop_loss(symbol, trade)

    def has_open_trade(self):
        return any(trade['open'] for trade in self.active_trades.values())

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
        trader.monitor_trades()
        time.sleep(60)

def continuous_market_monitor(trader, interval_minutes=5):
    while True:
        now = datetime.utcnow().time()
        if trader.is_within_session(now):
            logging.info("🔁 Analyse pendant session asiatique...")
        else:
            logging.info("🌐 Analyse hors session asiatique...")
        if not trader.has_open_trade():
            trader.analyze_session()
        time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    trader = AsianSessionTrader()
    scheduled_task()
    schedule.every().day.at("02:10").do(scheduled_task)

    threading.Thread(target=lambda: schedule.run_pending(), daemon=True).start()
    threading.Thread(target=monitor_trades_runner, args=(trader,), daemon=True).start()
    threading.Thread(target=continuous_market_monitor, args=(trader,), daemon=True).start()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)
