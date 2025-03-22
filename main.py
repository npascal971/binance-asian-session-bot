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
        self.risk_per_trade = 0.01
        self.session_data = {}
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
        exchange.set_sandbox_mode(True)
        return exchange

    def update_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            self.balance = balance["total"].get("USDT", 0)
            logging.info(f"Solde actuel : {self.balance:.2f} USDT")
        except Exception as e:
            logging.error(f"Erreur de solde : {str(e)}")

    def detect_order_block(self, df):
        try:
            last_bullish = df[(df['close'] > df['open'])].iloc[-2]
            last_bearish = df[(df['close'] < df['open'])].iloc[-2]
            if df.iloc[-1]['close'] < last_bullish['low']:
                return {"type": "bearish", "price": last_bullish['high']}
            elif df.iloc[-1]['close'] > last_bearish['high']:
                return {"type": "bullish", "price": last_bearish['low']}
        except:
            return None

    def detect_fvg(self, df):
        try:
            if df.iloc[-3]['high'] < df.iloc[-1]['low']:
                return {"type": "bullish", "zone": (df.iloc[-3]['high'], df.iloc[-1]['low'])}
            elif df.iloc[-3]['low'] > df.iloc[-1]['high']:
                return {"type": "bearish", "zone": (df.iloc[-1]['high'], df.iloc[-3]['low'])}
        except:
            return None

    def detect_structure_break(self, df):
        try:
            if df.iloc[-3]['high'] < df.iloc[-2]['high'] and df.iloc[-2]['high'] > df.iloc[-1]['high']:
                return "CHoCH possible (bearish)"
            if df.iloc[-3]['low'] > df.iloc[-2]['low'] and df.iloc[-2]['low'] < df.iloc[-1]['low']:
                return "CHoCH possible (bullish)"
        except:
            return None

    def analyze_session(self):
        try:
            for symbol in self.symbols:
                ohlcv = self.exchange.fetch_ohlcv(symbol, "1h", since=self.get_session_start())
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

                if len(df) < 3:
                    logging.warning(f"Pas assez de données pour {symbol} (seulement {len(df)} bougies)")
                    continue

                df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
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
                    "macd": macd.iloc[-1]["MACD_12_26_9"],
                    "order_block": ob,
                    "fvg": fvg,
                    "structure_break": structure,
                }

            logging.info("Analyse de session terminée")

            if self.session_data:
                self.send_email("Rapport de Session", self.generate_report())
            else:
                logging.info("Aucun signal exploitable trouvé. Aucun email envoyé.")

        except Exception as e:
            logging.error(f"Erreur analyse session : {str(e)}")

    def execute_post_session_trades(self):
        try:
            for symbol, data in self.session_data.items():
                current_price = self.get_current_price(symbol)
                if current_price is None:
                    continue

                if data["structure_break"] and data["order_block"]:
                    if data["order_block"]["type"] == "bullish" and current_price >= data["order_block"]["price"]:
                        self.place_order(symbol, "buy", current_price)
                    elif data["order_block"]["type"] == "bearish" and current_price <= data["order_block"]["price"]:
                        self.place_order(symbol, "sell", current_price)

            self.session_data = {}
            self.update_balance()
        except Exception as e:
            logging.error(f"Erreur exécution trade : {str(e)}")

    def place_order(self, symbol, side, price):
        try:
            position_size = (self.balance * self.risk_per_trade) / price
            if os.getenv("ENVIRONMENT") == "LIVE":
                self.exchange.create_market_order(symbol, side, position_size)
                logging.info(f"Ordre exécuté : {side} {position_size:.4f} {symbol}")
            else:
                logging.info(f"SIMULATION : {side} {position_size:.4f} {symbol}")

            self.send_email(
                "Nouveau Trade",
                f"{symbol} | {side.upper()}\nMontant: {position_size:.4f}\nPrix: {price:.2f}",
            )

        except Exception as e:
            logging.error(f"Erreur d'ordre : {str(e)}")

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
            report += f"{symbol}\n- HIGH: {data['high']:.2f}\n- LOW: {data['low']:.2f}\n- VWAP: {data['vwap']:.2f}\n- MACD: {data['macd']:.4f}\n"
            report += f"- OB: {data['order_block']}\n- FVG: {data['fvg']}\n- Structure: {data['structure_break']}\n\n"
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
            logging.error(f"Erreur email : {str(e)}")

# Flask API
app = Flask(__name__)

@app.route("/")
def status():
    html = "<h1>Trading Bot Actif</h1><p>Stratégie : Post-Session Asiatique avec SMC</p>"
    return Response(html, content_type="text/html; charset=utf-8")

def run_bot():
    trader = AsianSessionTrader()
    trader.run_cycle()

if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)
