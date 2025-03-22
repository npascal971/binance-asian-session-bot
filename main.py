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
        self.tp_percent = 0.1
        self.sl_percent = 1.0
        self.trailing_stop_percent = 0.5
        self.break_even_trigger = 1.0
        self.session_data = {}
        self.active_trades = {}
        self.asian_session = {"start": {"hour": 23, "minute": 0}, "end": {"hour": 10, "minute": 0}}
        self.us_session = {"start": {"hour": 14, "minute": 0}, "end": {"hour": 21, "minute": 0}}
        if not os.path.exists("reports"):
            os.makedirs("reports")

        self.update_balance()
        self.last_ob = {} 

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
        # Vérifie si l'heure actuelle est dans la session asiatique ou US
        asian_start = timedelta(hours=self.asian_session['start']['hour'], minutes=self.asian_session['start']['minute'])
        asian_end = timedelta(hours=self.asian_session['end']['hour'], minutes=self.asian_session['end']['minute'])
        us_start = timedelta(hours=self.us_session['start']['hour'], minutes=self.us_session['start']['minute'])
        us_end = timedelta(hours=self.us_session['end']['hour'], minutes=self.us_session['end']['minute'])
        now_time = timedelta(hours=current_time.hour, minutes=current_time.minute)

        in_asian_session = asian_start <= now_time <= asian_end if asian_start < asian_end else now_time >= asian_start or now_time <= asian_end
        in_us_session = us_start <= now_time <= us_end if us_start < us_end else now_time >= us_start or now_time <= us_end

        return in_asian_session or in_us_session
        

    def detect_order_blocks(self, df, bullish=True):
        try:
            df['body'] = abs(df['close'] - df['open'])
            df['prev_close'] = df['close'].shift(1)
            df['prev_open'] = df['open'].shift(1)

            # Initialiser ob_candidates à un DataFrame vide
            ob_candidates = pd.DataFrame()

            if bullish:
                ob_candidates = df[(df['open'] < df['close']) &
                                   (df['high'].shift(-1) > df['high']) &
                                   (df['low'].shift(-1) > df['low'])]
            else:
                ob_candidates = df[(df['open'] > df['close']) &
                                   (df['low'].shift(-1) < df['low']) &
                                   (df['high'].shift(-1) < df['high'])]

            if not ob_candidates.empty:
                last_ob = ob_candidates.iloc[-1]
                ob_zone = {
                    "open": last_ob['open'],
                    "close": last_ob['close'],
                    "high": last_ob['high'],
                    "low": last_ob['low'],
                    "timestamp": last_ob.name
                }
                logging.info(f"📦 OB détecté ({'Bullish' if bullish else 'Bearish'}) : {ob_zone}")
                return ob_zone
            else:
                logging.info(f"📦 Aucun OB détecté ({'Bullish' if bullish else 'Bearish'})")
                return None
        except Exception as e:
            logging.error(f"Erreur détection OB : {e}")
            return None
        
    def detect_ltf_structure_shift(self, symbol, timeframe="3m"):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=50)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("datetime", inplace=True)

            df['higher_high'] = df['high'] > df['high'].shift(1)
            df['lower_low'] = df['low'] < df['low'].shift(1)

            hh_detected = df['higher_high'].iloc[-2] and not df['higher_high'].iloc[-1]
            ll_detected = df['lower_low'].iloc[-2] and not df['lower_low'].iloc[-1]

            if hh_detected and ll_detected:
                logging.info(f"🌀 {symbol} → Possible CHoCH détecté (retournement)")
                return True
            return False
        except Exception as e:
            logging.error(f"Erreur détection structure LTF {symbol}: {e}")
            return False

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

                trend_ok = df["close"].iloc[-1] > df["ema200"].iloc[-1] and df["rsi"].iloc[-1] > 50 and macd.iloc[-1]["MACD_12_26_9"] > 0

                ob = self.detect_order_blocks(df, bullish=trend_ok)

                self.session_data[symbol] = {
                    "ema200": df["ema200"].iloc[-1],
                    "rsi": df["rsi"].iloc[-1],
                    "macd": macd.iloc[-1]["MACD_12_26_9"],
                    "trend_ok": trend_ok,
                    "order_block": ob
                }
        except Exception as e:
            logging.error(f"Erreur analyse session : {str(e)}")

    def execute_post_session_trades(self):
        for symbol in self.symbols:
            data = self.session_data.get(symbol, {})
            if not data.get("trend_ok"):
                logging.info(f"Pas d'entrée pour {symbol} - tendance non confirmée.")
                continue

            if not self.detect_ltf_structure_shift(symbol, timeframe="3m"):
                logging.info(f"⚠️ {symbol} - Structure LTF pas encore confirmée, on attend")
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
                    "trailing_stop": sl,
                    "entry_time": datetime.now()
                }
                logging.info(f"🎯 Nouveau trade {symbol} | Entrée: {price:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")
            except Exception as e:
                logging.error(f"Erreur exécution ordre : {e}")
                
    def send_trade_notification(self, subject, body, trade):
        sender_email = os.getenv('EMAIL_ADDRESS')
        receiver_email = os.getenv('EMAIL_ADDRESS')
        password = os.getenv('EMAIL_PASSWORD')

        # Calculer le profit ou le drawdown en USD
        entry_price = trade['entry']
        exit_price = trade['exit_price']  # Prix de sortie (TP ou SL)
        amount = trade['amount']
        profit_usd = (exit_price - entry_price) * amount
        drawdown_usd = (entry_price - exit_price) * amount

        # Ajouter les informations au corps de l'e-mail
        if profit_usd > 0:
            body += f"\nProfit réalisé : {profit_usd:.2f} USD"
        else:
            body += f"\nDrawdown subi : {drawdown_usd:.2f} USD"

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, msg.as_string())
                logging.info("📩 Notification email envoyée.")
        except Exception as e:
            logging.error(f"Erreur envoi email : {e}")

    def manage_take_profit_stop_loss(self, symbol, trade):
        try:
            price = self.exchange.fetch_ticker(symbol)['last']
            if price >= trade['tp']:
                duration = datetime.now() - trade["entry_time"]
                minutes = int(duration.total_seconds() // 60)
                logging.info(f"✅ TP atteint {symbol} à {price:.2f} | Durée : {minutes} min")
                trade['open'] = False
                trade['exit_price'] = price  # Ajouter le prix de sortie
                self.exchange.create_market_sell_order(symbol, trade['amount'])

                subject = f"[TP ATTEINT] {symbol}"
                body = f"✅ Take Profit atteint sur {symbol}\n\nPrix d'entrée : {trade['entry']:.2f} USDT\nPrix de sortie : {price:.2f} USDT\nDurée : {minutes} minutes"
                self.send_trade_notification(subject, body, trade)  # Passer le trade
                return

            if price <= trade['sl']:
                duration = datetime.now() - trade["entry_time"]
                minutes = int(duration.total_seconds() // 60)
                logging.info(f"🛑 SL touché {symbol} à {price:.2f} | Durée : {minutes} min")
                trade['open'] = False
                trade['exit_price'] = price  # Ajouter le prix de sortie
                self.exchange.create_market_sell_order(symbol, trade['amount'])

                subject = f"[SL TOUCHÉ] {symbol}"
                body = f"🛑 Stop Loss touché sur {symbol}\n\nPrix d'entrée : {trade['entry']:.2f} USDT\nPrix de sortie : {price:.2f} USDT\nDurée : {minutes} minutes"
                self.send_trade_notification(subject, body, trade)  # Passer le trade
                return

        # Trailing SL
            new_sl = price * (1 - self.trailing_stop_percent / 100)
            if new_sl > trade["sl"]:
                logging.info(f"🔁 Trailing SL mis à jour pour {symbol} : {trade['sl']:.2f} → {new_sl:.2f}")
                trade["sl"] = new_sl

        # Break-even
            if price >= trade["entry"] * (1 + self.break_even_trigger / 100) and trade["sl"] < trade["entry"]:
                logging.info(f"🔐 Break-even activé pour {symbol} → SL remonté à l'entrée : {trade['entry']:.2f}")
                trade["sl"] = trade["entry"]

        except Exception as e:
            logging.error(f"Erreur SL/TP dynamique : {e}")


    def monitor_trades(self):
        for symbol, trade in list(self.active_trades.items()):
            if trade.get("open"):
                self.manage_take_profit_stop_loss(symbol, trade)

# Les autres fonctions restent inchangées...
# (scheduled_task, monitor_trades_runner, run_scheduler, Flask app etc.)


def scheduled_task():
    logging.info("===== Début de la tâche programmée =====")
    trader.analyze_session()
    trader.execute_post_session_trades()
    trader.monitor_trades()
    logging.info("===== Fin de la tâche programmée =====")

@app.route("/")
def home():
    return "Asian Session Bot is running 🚀", 200

def monitor_trades_runner(trader):
    while True:
        open_trades = [t for t in trader.active_trades.values() if t.get("open")]
        if open_trades:
            for trade in open_trades:
                duration = datetime.now() - trade["entry_time"]
                minutes = int(duration.total_seconds() // 60)
                logging.info(f"💓 Monitor tick | Trades actifs : {len(open_trades)} | Durée trade actif : {minutes} min")
        else:
            logging.info("💓 Monitor tick | Trades actifs : 0")
        trader.monitor_trades()
        time.sleep(60)

def run_scheduler():
    while True:
        current_time = datetime.now()
        if trader.is_within_session(current_time):  # Vérifie si l'heure est entre 10h00 et 17h00
            logging.info("===== Début de la tâche programmée =====")
            trader.analyze_session()
            trader.execute_post_session_trades()
            trader.monitor_trades()
            logging.info("===== Fin de la tâche programmée =====")
        else:
            logging.info("En dehors de la plage horaire de trading (10h00-17h00). Attente...")
        
        time.sleep(300)  # Attendre 5 minute avant de vérifier à nouveau

if __name__ == "__main__":
    trader = AsianSessionTrader()
    scheduled_task()
    schedule.every().day.at("02:10").do(scheduled_task)

    threading.Thread(target=run_scheduler, daemon=True).start()
    threading.Thread(target=monitor_trades_runner, args=(trader,), daemon=True).start()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)
