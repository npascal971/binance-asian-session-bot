def main():
    try:
        now = pd.Timestamp.now(tz='UTC')
        print(f"Heure actuelle : {now}")

        # Récupérer et afficher le solde du compte test
        balance = exchange.fetch_balance()['total']['USDT']
        print(f"💰 Solde du compte test : {balance:.2f} USDT")

        # Si session asiatique en cours
        if asian_session_start <= now.hour or now.hour < asian_session_end:
            print("🌏 Session asiatique en cours - Enregistrement des données...")
            for symbol in symbols:
                htf_df = fetch_ohlcv(symbol, timeframe)
                if htf_df is not None:
                    record_asian_session_data(symbol, htf_df)

        # Après la session asiatique, analyse des signaux
        elif now.hour >= asian_session_end:
            print("✅ Session asiatique terminée - Analyse des données...")
            for symbol in symbols:
                asian_high = asian_session_data[symbol]['high']
                asian_low = asian_session_data[symbol]['low']
                print(f"{symbol} - 📊 High: {asian_high}, Low: {asian_low}")

                if asian_high is not None and asian_low is not None:
                    ltf_df = fetch_ohlcv(symbol, ltf_timeframe, limit=50)
                    if ltf_df is not None:
                        action = check_reversal_setup(ltf_df, asian_high, asian_low)
                        print(f"{symbol} - ⚡ Signal détecté : {action}")
                        
                        if action in ['buy', 'sell']:
                            execute_trade(symbol, action, balance)

        # Gestion des trades ouverts
        manage_active_trades()

    except Exception as e:
        print(f"❌ Erreur dans main: {e}")

    return "Script exécuté avec succès"

