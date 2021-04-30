from portfolio import Portfolio, telegramBot
from token_ import tele_token

while(True):
    etoro_p = Portfolio('etoro', 71676.96) #, portfolio_csv='csvs/etoro_portfolio_2021-04-29.csv', pnl_csv='csvs/etoro_pnl_2021-04-29.csv')
    bh_p = Portfolio('bh', 70000) #, portfolio_csv='csvs/bh_portfolio_2021-04-29.csv', pnl_csv='csvs/bh_pnl_2021-04-29.csv')
    tbot = telegramBot(tele_token, portfolios=[etoro_p, bh_p])
    tbot.loop_for_telegram()
