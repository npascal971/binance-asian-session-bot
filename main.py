import ccxt
import time
import pandas as pd
import pandas_ta as ta
import os
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, Response
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
        logging.StreamHandler(sys.stdout)
    ],
)

class AsianSessionTrader:
    def __init__(self):
        self.exchange = self.configure_exchange()
        self.symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        self.risk_per_trade = 0.01  # 1% de risque
        self.session_data = {}
        self.positions_taken = set()
        self.asian_session = {"start": {"hour": 23, "minute": 0}, "end": {"hour": 10, "minute": 0}}
        self.update_balance()
        logging.info(f"Configuration session : {self.asian_session}")
        logging.info(f"UTC maintenant : {datetime.utcnow()}")

    def configure_exchange(self):
        exchange = ccxt.binance({
            "apiKey": os.getenv("BINANCE_API_KEY"),
            "secret": os.getenv("BINANCE_API_SECRET"),
            "enableRateLimit": True,
        })
        exchange.set_sandbox_mode(True)  # Met à False pour environnement réel
        return exchange

    def update_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            self.balance = balance["total"].get("USDT", 0)
            logging.info(f"Solde actuel : {self.balance:.2f} USDT")
        except Exception as e:
            logging.error(f"Erreur de solde : {str(e)}")

    def already_in_position(self, symbol):
        return symbol in self.positions_taken

    def analyze_session(self):
        try:
            for symbol in self.symbols:
                ohlcv = self.exchange.fetch_ohlcv(symbol, "1h", since=self.get_session_start())
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

                if len(df) < 2:
                    logging.warning(f"Pas assez de données pour {symbol} (seulement {len(df)} bougies)")
                    continue

                df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
                macd = ta.macd(df["close"])
                if macd is None or "MACD_12_26_9" not in macd.columns:
                    logging.warning(f"Erreur calcul MACD pour {symbol}")
                    continue

                vwap = df["vwap"].mean()
                macd_value = macd.iloc[-1]["MACD_12_26_9"]
                current_price = self.get_current_price(symbol)

                self.session_data[symbol] = {
                    "high": df["high"].max(),
                    "low": df["low"].min(),
                    "vwap": vwap,
                    "macd": macd_value,
                }

                # 📈 Prise de trade immédiate si conditions remplies
                if current_price and macd_value > 0 and current_price > vwap and not self.already_in_position(symbol):
                    self.update_balance()
                    position_size = round((self.balance * self.risk_per_trade) / current_price, 4)
                    self.place_order(symbol, "buy", current_price, position_size)
                    self.place_tp_sl_order(symbol, "buy", current_price, position_size)
                    self.positions_taken.add(symbol)

            logging.info("Analyse de session terminée")

            if self.session_data:
                self.send_email("Rapport de Session", self.generate_report())
            else:
                logging.info("Aucun signal exploitable trouvé. Aucun email envoyé.")

        except Exception as e:
            logging.error(f"Erreur d'analyse : {str(e)}")

    def place_order(self, symbol, side, price, quantity):
        try:
            if os.getenv("ENVIRONMENT") == "LIVE":
                self.exchange.create_market_order(symbol, side.upper(), quantity)
                logging.info(f"Ordre exécuté : {side.upper()} {quantity:.4f} {symbol}")
            else:
                logging.info(f"SIMULATION : {side.upper()} {quantity:.4f} {symbol}")

            self.send_email(
                "Nouveau Trade",
                f"{symbol} | {side.upper()}\nMontant: {quantity:.4f}\nPrix: {price:.2f}",
            )

        except Exception as e:
            logging.error(f"Erreur d'ordre : {str(e)}")

    def place_tp_sl_order(self, symbol, side, entry_price, quantity):
        try:
            tp_price = entry_price * 1.02
            sl_price = entry_price * 0.98
            logging.info(f"TP/SL configurés pour {symbol} | TP: {tp_price:.2f} | SL: {sl_price:.2f}")
            # Ajout d'ordre réel TP/SL possible ici si souhaité (create_order avec stop/limit)

        except Exception as e:
            logging.error(f"Erreur TP/SL pour {symbol} : {str(e)}")

    def execute_post_session_trades(self):
        try:
            for symbol, data in self.session_data.items():
                current_price = self.get_current_price(symbol)
                if current_price is None:
                    continue

                if current_price > data["vwap"] and data["macd"] > 0 and not self.already_in_position(symbol):
                    self.update_balance()
                    position_size = round((self.balance * self.risk_per_trade) / current_price, 4)
                    self.place_order(symbol, "buy", current_price, position_size)
                    self.place_tp_sl_order(symbol, "buy", current_price, position_size)
                    self.positions_taken.add(symbol)

            self.session_data = {}
            self.update_balance()

        except Exception as e:
            logging.error(f"Erreur exécution des trades : {str(e)}")

    def run_cycle(self):
        while True:
            now = datetime.utcnow()
            start_time = datetime(now.year, now.month, now.day, self.asian_session["start"]["hour"], self.asian_session["start"]["minute"])
            end_time = datetime(now.year, now.month, now.day, self.asian_session["end"]["hour"], self.asian_session["end"]["minute"])

            if start_time <= now < end_time:
                if not self.session_data:
                    logging.info("Debut de l'analyse...")
                    self.positions_taken.clear()
                    self.analyze_session()
            elif now >= end_time:
                if self.session_data:
                    logging.info("Execution des trades...")
                    self.execute_post_session_trades()

            time.sleep(60)

    def get_current_price(self, symbol):
        try:
            return self.exchange.fetch_ticker(symbol)["last"]
        except Exception as e:
            logging.error(f"Erreur récupération prix {symbol} : {str(e)}")
            return None

    def get_session_start(self):
        return int(
            datetime(datetime.utcnow().year, datetime.utcnow().month, datetime.utcnow().day,
                     self.asian_session["start"]["hour"], self.asian_session["start"]["minute"]).timestamp() * 1000
        )

    def generate_report(self):
        report = "Rapport de Session\n\n"
        for symbol, data in self.session_data.items():
            report += f"{symbol}\n- HIGH: {data['high']:.2f}\n- LOW: {data['low']:.2f}\n- VWAP: {data['vwap']:.2f}\n- MACD: {data['macd']:.4f}\n\n"
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
                server.login(os.getenv("EMAIL_ADDRESS"], os.getenv("EMAIL_PASSWORD"))
                server.send_message(msg)

        except Exception as e:
            logging.error(f"Erreur envoi email : {str(e)}")

# API Flask simple
app = Flask(__name__)

@app.route("/")
def status():
    html = "<h1>Trading Bot Actif</h1><p>Stratégie : Post-Session Asiatique</p>"
    return Response(html, content_type="text/html; charset=utf-8")

def run_bot():
    trader = AsianSessionTrader()
    trader.run_cycle()

if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)
