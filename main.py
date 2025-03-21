import ccxt
import time
import pandas as pd
import pandas_ta as ta
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Chargement des variables d'environnement
load_dotenv()

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)  # 🔥 Ajout explicite de sys.stdout
    ],
)


class AsianSessionTrader:
    def __init__(self):
        self.exchange = self.configure_exchange()
        self.symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        self.risk_per_trade = 0.02
        self.session_data = {}
        self.asian_session = {"start": {"hour": 19, "minute": 0}, "end": {"hour": 19, "minute": 55}}
        self.update_balance()
        logging.info(f"Configuration session : {self.asian_session}")
        logging.info(f"UTC maintenant : {datetime.utcnow()}")

    def configure_exchange(self):
        exchange = ccxt.binance({
            "apiKey": os.getenv("BINANCE_API_KEY"),
            "secret": os.getenv("BINANCE_API_SECRET"),
            "enableRateLimit": True,
        })
        exchange.set_sandbox_mode(True)
        return exchange

    def update_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            self.balance = balance["total"].get("USDT", 0)
            logging.info(f"Solde actuel : {self.balance:.2f} USDT")
        except Exception as e:
            logging.error(f"Erreur de solde : {str(e)}")

    def analyze_session(self):
        try:
            for symbol in self.symbols:
                ohlcv = self.exchange.fetch_ohlcv(symbol, "1h", since=self.get_session_start())
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

                if df.empty or len(df) < 26:
                    logging.warning(f"Pas assez de données pour {symbol} (seulement {len(df)} bougies)")
                    continue

                df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
                macd = ta.macd(df["close"])

                if macd is None or macd.empty or "MACD_12_26_9" not in macd.columns:
                    logging.warning(f"MACD non calculable pour {symbol}")
                    continue

                last_macd_value = macd["MACD_12_26_9"].dropna().iloc[-1] if not macd["MACD_12_26_9"].dropna().empty else None
                if last_macd_value is None:
                    logging.warning(f"MACD vide pour {symbol}")
                    continue

                self.session_data[symbol] = {
                    "high": df["high"].max(),
                    "low": df["low"].min(),
                    "vwap": df["vwap"].mean(),
                    "macd": last_macd_value,
                }

            logging.info("Analyse de session terminée")
            if self.session_data:
                self.send_email("\ud83d\udcc8 Rapport de Session", self.generate_report())
            else:
                logging.info("Aucun signal exploitable trouvé. Aucun email envoyé.")

        except Exception as e:
            logging.error(f"\u274c \u00c9chec critique de l'analyse : {str(e)}")

    def execute_post_session_trades(self):
        try:
            for symbol, data in self.session_data.items():
                current_price = self.get_current_price(symbol)
                if current_price is None:
                    continue
                if current_price > data["vwap"] and data["macd"] > 0:
                    self.place_order(symbol, "buy", current_price)
                elif current_price < data["vwap"] and data["macd"] < 0:
                    self.place_order(symbol, "sell", current_price)
            self.session_data = {}
            self.update_balance()
        except Exception as e:
            logging.error(f"Erreur d'exécution des trades : {str(e)}")

    def run_cycle(self):
        while True:
            now = datetime.utcnow()
            start_time = datetime(now.year, now.month, now.day, self.asian_session["start"]["hour"], self.asian_session["start"]["minute"])
            end_time = datetime(now.year, now.month, now.day, self.asian_session["end"]["hour"], self.asian_session["end"]["minute"])

            if start_time <= now < end_time:
                if not self.session_data:
                    logging.info("\ud83d\ude80 D\u00e9but de l'analyse...")
                    self.analyze_session()
            elif now >= end_time:
                if self.session_data:
                    logging.info("\ud83d\udca1 Ex\u00e9cution des trades...")
                    self.execute_post_session_trades()
            time.sleep(60)

    def place_order(self, symbol, side, price):
        try:
            position_size = (self.balance * self.risk_per_trade) / price
            if os.getenv("ENVIRONMENT") == "LIVE":
                self.exchange.create_market_order(symbol, side, position_size)
                logging.info(f"Ordre ex\u00e9cut\u00e9 : {side} {position_size:.4f} {symbol}")
            else:
                logging.info(f"SIMULATION : {side} {position_size:.4f} {symbol}")
            self.send_email("\ud83c\udfaf Nouveau Trade", f"**{symbol}** | {side.upper()}\nMontant: {position_size:.4f}\nPrix: {price:.2f}")
        except Exception as e:
            logging.error(f"Erreur d'ordre : {str(e)}")

    def get_current_price(self, symbol):
        try:
            return self.exchange.fetch_ticker(symbol)["last"]
        except Exception as e:
            logging.error(f"Erreur r\u00e9cup\u00e9ration prix {symbol} : {str(e)}")
            return None

    def get_session_start(self):
        return int(datetime(datetime.utcnow().year, datetime.utcnow().month, datetime.utcnow().day,
                            self.asian_session["start"]["hour"], self.asian_session["start"]["minute"]).timestamp() * 1000)

    def generate_report(self):
        report = "\ud83d\udcc8 **Rapport de Session**\n\n"
        for symbol, data in self.session_data.items():
            report += f"**{symbol}**\n- HIGH: {data['high']:.2f}\n- LOW: {data['low']:.2f}\n- VWAP: {data['vwap']:.2f}\n- MACD: {data['macd']:.4f}\n\n"
        return report

    def send_email(self, subject, body):
        try:
            msg = MIMEMultipart()
            msg["From"] = os.getenv("EMAIL_ADDRESS")
            msg["To"] = os.getenv("EMAIL_ADDRESS")
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(os.getenv("EMAIL_ADDRESS"), os.getenv("EMAIL_PASSWORD"))
                server.send_message(msg)
        except Exception as e:
            logging.error(f"Erreur envoi email : {str(e)}")


app = Flask(__name__)


@app.route("/")
def status():
    return "<h1>Trading Bot Actif</h1><p>Strat\u00e9gie : Post-Session Asiatique</p><p>\ud83d\udd52 Prochaine analyse : 17h00 UTC</p>"

def run_bot():
    trader = AsianSessionTrader()
    trader.run_cycle()

if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)
