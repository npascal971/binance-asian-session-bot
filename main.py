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
        self.us_session = {"start": {"hour": 10, "minute": 0}, "end": {"hour": 17, "minute": 0}}
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
        # Vérifie uniquement si l'heure actuelle est dans la session asiatique
        asian_start = timedelta(hours=self.asian_session['start']['hour'], minutes=self.asian_session['start']['minute'])
        asian_end = timedelta(hours=self.asian_session['end']['hour'], minutes=self.asian_session['end']['minute'])
        now_time = timedelta(hours=current_time.hour, minutes=current_time.minute)

        # Gestion du cas où la session asiatique traverse minuit
        if asian_start < asian_end:
            in_asian_session = asian_start <= now_time <= asian_end
        else:
            in_asian_session = now_time >= asian_start or now_time <= asian_end

        return in_asian_session

    def get_asian_session_range(self, symbol):
        try:
            # Récupérer les données OHLCV pour la session asiatique
            ohlcv = self.exchange.fetch_ohlcv(symbol, "1h", limit=12)  # 12 heures pour couvrir la session asiatique
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("datetime", inplace=True)

            # Filtrer pour ne garder que les heures de la session asiatique
            asian_start_hour = self.asian_session['start']['hour']
            asian_end_hour = self.asian_session['end']['hour']
            df_asian = df.between_time(f"{asian_start_hour}:00", f"{asian_end_hour}:00")

            # Repérer le high et le low de la session asiatique
            asian_high = df_asian['high'].max()
            asian_low = df_asian['low'].min()

            logging.info(f"📊 Asian Range pour {symbol} : High = {asian_high}, Low = {asian_low}")
            return asian_high, asian_low
        except Exception as e:
            logging.error(f"Erreur calcul Asian Range pour {symbol} : {e}")
            return None, None

    def is_price_near_liquidity_zone(self, symbol, price, asian_high, asian_low, threshold=0.005):
        try:
            # Calculer la distance en pourcentage par rapport au high et low de la session asiatique
            distance_to_high = abs(price - asian_high) / asian_high
            distance_to_low = abs(price - asian_low) / asian_low

            # Vérifier si le prix est proche d'une zone de liquidité
            if distance_to_high <= threshold:
                logging.info(f"🎯 Prix proche du high de la session asiatique pour {symbol} : {asian_high}")
                return True
            elif distance_to_low <= threshold:
                logging.info(f"🎯 Prix proche du low de la session asiatique pour {symbol} : {asian_low}")
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Erreur vérification zones de liquidité pour {symbol} : {e}")
            return False

    def detect_order_blocks(self, df, symbol, bullish=True):
        try:
            df['body'] = abs(df['close'] - df['open'])
            df['prev_close'] = df['close'].shift(1)
            df['prev_open'] = df['open'].shift(1)

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
                ob_timestamp = last_ob.name

                if ob_timestamp != self.last_ob.get(symbol, {}).get("timestamp"):
                    ob_zone = {
                        "open": last_ob['open'],
                        "close": last_ob['close'],
                        "high": last_ob['high'],
                        "low": last_ob['low'],
                        "timestamp": ob_timestamp
                    }
                    logging.info(f"📦 OB détecté ({'Bullish' if bullish else 'Bearish'}) pour {symbol} : {ob_zone}")
                    self.last_ob[symbol] = ob_zone
                    return ob_zone
                else:
                    logging.info(f"📦 OB déjà traité pour {symbol} (timestamp: {ob_timestamp})")
                    return None
            else:
                logging.info(f"📦 Aucun OB détecté pour {symbol} ({'Bullish' if bullish else 'Bearish'})")
                return None
        except Exception as e:
            logging.error(f"Erreur détection OB pour {symbol} : {e}")
            return None

    def detect_ltf_structure_shift(self, symbol, timeframe="5m"):
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
            else:
                logging.info(f"⚠️ Retournement de structure non confirmé pour {symbol} : HH = {hh_detected}, LL = {ll_detected}")
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

                trend_ok = (df["close"].iloc[-1] > df["ema200"].iloc[-1]) and \
                           (df["rsi"].iloc[-1] > 45) and \
                           (macd.iloc[-1]["MACD_12_26_9"] > -10)

                logging.info(f"EMA200 pour {symbol} : {df['ema200'].iloc[-1]}")
                logging.info(f"RSI pour {symbol} : {df['rsi'].iloc[-1]}")
                logging.info(f"MACD pour {symbol} : {macd.iloc[-1]['MACD_12_26_9']}")
                logging.info(f"Tendance pour {symbol} : {'OK' if trend_ok else 'Non confirmée'}")

                ob = self.detect_order_blocks(df, symbol, bullish=trend_ok)

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

            # Récupérer le high et le low de la session asiatique
            asian_high, asian_low = self.get_asian_session_range(symbol)
            if asian_high is None or asian_low is None:
                continue

            # Vérifier si le prix est proche d'une zone de liquidité
            price = self.exchange.fetch_ticker(symbol)['last']
            if not self.is_price_near_liquidity_zone(symbol, price, asian_high, asian_low):
                logging.info(f"⚠️ {symbol} - Prix pas proche des zones de liquidité, on attend")
                continue

            # Vérifier un retournement de structure en LTF
            if not self.detect_ltf_structure_shift(symbol, timeframe="5m"):
                logging.info(f"⚠️ {symbol} - Structure LTF pas encore confirmée, on attend")
                continue

            # Passer l'ordre
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

        entry_price = trade['entry']
        exit_price = trade.get('exit_price', entry_price)  # Utiliser entry_price par défaut si exit_price n'est pas défini
        amount = trade['amount']
        profit_usd = (exit_price - entry_price) * amount
        drawdown_usd = (entry_price - exit_price) * amount

        # Ajouter le profit ou le drawdown au corps de l'e-mail
        if profit_usd > 0:
            body += f"\nProfit réalisé : {profit_usd:.2f} USD"
            logging.info(f"💰 Profit réalisé pour le trade : {profit_usd:.2f} USD")
        else:
            body += f"\nDrawdown subi : {drawdown_usd:.2f} USD"
            logging.info(f"💸 Drawdown subi pour le trade : {drawdown_usd:.2f} USD")

        # Créer et envoyer l'e-mail
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

            # TP atteint
            if price >= trade['tp']:
                trade['exit_price'] = price  # Définir exit_price pour TP
                duration = datetime.now() - trade["entry_time"]
                minutes = int(duration.total_seconds() // 60)
                logging.info(f"✅ TP atteint {symbol} à {price:.2f} | Durée : {minutes} min")
                logging.info(f"📊 Prix d'entrée : {trade['entry']:.2f} | Prix de sortie : {trade['exit_price']:.2f}")
                self.send_trade_notification(f"[TP ATTEINT] {symbol}", f"✅ Take Profit atteint sur {symbol}\n\nPrix d'entrée : {trade['entry']:.2f} USDT\nPrix de sortie : {price:.2f} USDT\nDurée : {minutes} minutes", trade)
                trade['open'] = False
                self.exchange.create_market_sell_order(symbol, trade['amount'])
                return

            # SL touché
            if price <= trade['sl']:
                trade['exit_price'] = price  # Définir exit_price pour SL
                duration = datetime.now() - trade["entry_time"]
                minutes = int(duration.total_seconds() // 60)
                logging.info(f"🛑 SL touché {symbol} à {price:.2f} | Durée : {minutes} min")
                logging.info(f"📊 Prix d'entrée : {trade['entry']:.2f} | Prix de sortie : {trade['exit_price']:.2f}")
                self.send_trade_notification(f"[SL TOUCHÉ] {symbol}", f"🛑 Stop Loss touché sur {symbol}\n\nPrix d'entrée : {trade['entry']:.2f} USDT\nPrix de sortie : {price:.2f} USDT\nDurée : {minutes} minutes", trade)
                trade['open'] = False
                self.exchange.create_market_sell_order(symbol, trade['amount'])
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
        if trader.is_within_session(current_time):
            logging.info("===== Début de la tâche programmée =====")
            trader.analyze_session()
            trader.execute_post_session_trades()
            trader.monitor_trades()
            logging.info("===== Fin de la tâche programmée =====")
        else:
            logging.info("En dehors de la plage horaire de trading (10h00-17h00). Attente...")
        
        time.sleep(60)

if __name__ == "__main__":
    trader = AsianSessionTrader()
    scheduled_task()
    schedule.every().day.at("02:10").do(scheduled_task)

    threading.Thread(target=run_scheduler, daemon=True).start()
    threading.Thread(target=monitor_trades_runner, args=(trader,), daemon=True).start()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)
