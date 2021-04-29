from portfolio import Portfolio, telegramBot
etoro_p = Portfolio('etoro', 71676.96)
bh_p = Portfolio('bh', 70000)

tele_token = '1685123002:AAHby83yoK0JNrpp2-g11pphnvWs5gVNo1g'
tbot = telegramBot(tele_token, portfolios=[etoro_p, bh_p])
tbot.loop_for_telegram()