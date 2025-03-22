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
        self.asian_session = {"start": {"hour": 23, "minute": 0}, "end": {"hour": 10, "minute": 0}}

        if not os.path.exists("reports"):
            os.makedirs("reports")

        self.update_balance()
        logging.info(f"Configuration session : {self.asian_session}")
        logging.info(f"UTC maintenant : {datetime.utcnow()}")

    def configure_exchange(self):
        return ccxt.binance({"enableRateLimit": True})

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

                if len(df) < 3:
                    logging.warning(f"Pas assez de données pour {symbol} (seulement {len(df)} bougies)")
                    continue

                df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
                df["ema200"] = ta.ema(df["close"], length=200)
                df["rsi"] = ta.rsi(df["close"], length=14)
                macd = ta.macd(df["close"])

                if macd is None or "MACD_12_26_9" not in macd.columns:
                    logging.warning(f"Erreur MACD pour {symbol}")
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

            if self.session_data:
                report_txt = self.generate_report()
                self.save_report_to_file(report_txt)
                self.send_email("Rapport de Session", report_txt)
            else:
                logging.info("Aucun signal exploitable trouvé. Aucun email envoyé.")

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
        logging.info("Simulation d'exécution de trades")

# Fonction planifiée

def scheduled_task():
    logging.info("\n===== Tâche quotidienne programmée lancée =====")
    trader = AsianSessionTrader()
    trader.analyze_session()
    trader.execute_post_session_trades()

if __name__ == "__main__":
    schedule.every().day.at("10:30").do(scheduled_task)  # ⏰ Planifié après la session asiatique

    def schedule_runner():
        while True:
            schedule.run_pending()
            time.sleep(30)

    bot_thread = threading.Thread(target=schedule_runner, daemon=True)
    bot_thread.start()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)
