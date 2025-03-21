# Dans la fonction main_loop
def main_loop():
    bot = TradingBot()
    while True:
        try:
            # Nouvelle logique d'exécution
            if bot.in_trading_session():
                bot.analyze_markets()
                bot.execute_strategy()
            
            bot.manage_risk()
            time.sleep(60)  # Vérification toutes les minutes

        except Exception as e:
            logging.critical(f"Crash du bot : {str(e)}")
            bot.send_email("CRASH DU BOT", str(e))
            time.sleep(300)
