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

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AsianSessionTrader:
    def __init__(self):
        self.exchange = self.configure_exchange()
        self.symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        self.risk_per_trade = 0.01
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
    'apiKey': 'LisbDeNvATic1cNKkxMCt5lupsA6Ly8TiNhSjEC1GQYXqUn7YTSQbhAC1h0J41yX',
    'secret': 'xIlgTJaLNAfXZNfXEN6sTWJ8fLwp2H95kcXN5L8qpX1MBukUAwm1TvOO8a0jEXyc',
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

        # Activation du mode sandbox (Testnet)
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

    def analyze_session(self):
        try:
            for symbol in self.symbols:
                ohlcv = self.exchange.fetch_ohlcv(symbol, "1h", since=self.get_session_start())
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

                if len(df["close"]) < 30:
                    logging.warning(f"Trop peu de données pour MACD sur {symbol} ({len(df)} lignes)")
                    continue


                df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
                df["ema200"] = ta.ema(df["close"], length=200)
                df["rsi"] = ta.rsi(df["close"], length=14)
                macd = ta.macd(df["close"])
                if macd is None or macd.empty or not all(col in macd.columns for col in ["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]):
                    logging.warning(f"Erreur MACD pour {symbol} (colonnes manquantes ou données vides)")
                    continue

# Nettoyage des NaN et utilisation du dernier MACD valide
                macd = macd.dropna()
                if macd.empty:
                    logging.warning(f"MACD vide après nettoyage pour {symbol}")
                    continue

                macd_value = macd["MACD_12_26_9"].iloc[-1]


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

    def generate_report(self):
        report = "Rapport de Session\n\n"
        for symbol, data in self.session_data.items():
            report += f"{symbol}\n"
            report += f"- HIGH: {data['high']:.2f}\n- LOW: {data['low']:.2f}\n- VWAP: {data['vwap']:.2f}\n"
            report += f"- EMA200: {data['ema200']:.2f}\n- RSI: {data['rsi']:.2f}\n- MACD: {data['macd']:.4f}\n"
            report += f"- OB: {data['order_block']}\n- FVG: {data['fvg']}\n- Structure: {data['structure_break']}\n\n"
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
        logging.info(f"Email envoyé - Sujet: {subject}")

    def execute_post_session_trades(self):
        for symbol in self.symbols:
            if symbol not in self.session_data:
                continue

            price = self.exchange.fetch_ticker(symbol)['last']
            trade_amount = self.risk_per_trade * self.exchange.fetch_balance()['total']['USDT'] / price

            sl = price * 0.99  # SL à -1%
            tp = price * 1.02  # TP à +2%

            self.active_trades[symbol] = {
                "entry": price,
                "sl": sl,
                "tp": tp,
                "amount": trade_amount,
                "open": True
            }

            logging.info(f"SIMULATION : Achat {trade_amount:.4f} {symbol} à {price:.2f} (SL: {sl:.2f}, TP: {tp:.2f})")

    def monitor_trades(self):
        for symbol, trade in self.active_trades.items():
            if not trade['open']:
                continue

            price = self.exchange.fetch_ticker(symbol)['last']

            if price <= trade['sl']:
                trade['open'] = False
                logging.info(f"SL touché pour {symbol} à {price:.2f} ❌")
                self.send_email(f"SL touché - {symbol}", f"Le SL a été touché pour {symbol} à {price:.2f}")
            elif price >= trade['tp']:
                trade['open'] = False
                logging.info(f"TP touché pour {symbol} à {price:.2f} ✅")
                self.send_email(f"TP atteint - {symbol}", f"Le TP a été atteint pour {symbol} à {price:.2f}")

# Tâche planifiée

def scheduled_task():
    logging.info("\n===== Tâche quotidienne programmée lancée =====")
    trader = AsianSessionTrader()
    trader.analyze_session()
    trader.execute_post_session_trades()
    trader.monitor_trades()

@app.route("/")
def home():
    return "Asian Session Bot is running 🚀", 200

if __name__ == "__main__":
    scheduled_task()
    schedule.every().day.at("02:10").do(scheduled_task)

    def schedule_runner():
        while True:
            schedule.run_pending()
            time.sleep(30)

    bot_thread = threading.Thread(target=schedule_runner, daemon=True)
    bot_thread.start()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)
