import os
import json
import requests
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import seaborn as sns
import numpy as np
from forex_python.converter import CurrencyRates
from yahoo_fin import stock_info as si
import yfinance as yf
pd.set_option("precision", 2)

def trading_day(open_market='16:30'):
    def hour_to_float(hours=16, minutes=30):
        return int(hours)+int(minutes)/60

    def str_hour_to_float(hour='16:30'):
        return int(hour.split(':')[0])+int(hour.split(':')[1])/60
    
    trade_day = datetime.datetime.now()
    day_off = -1
    if hour_to_float(trade_day.hour, trade_day.minute) > str_hour_to_float(open_market):
        day_off = 0
    td = (datetime.datetime.now() + datetime.timedelta(days=day_off))
    return str(datetime.datetime(year=td.year, day=td.day, month=td.month)).split()[0]

class Portfolio():
    def __init__(self, broker, start_balance=10000, portfolio_csv='', pnl_csv=''):
        self.hist_data = {}
        self.broker = broker
        self.start_balance = start_balance
        self.set_trade_day()
        self.converter = CurrencyRates()
        self.USDILS = self.converter.get_rate('USD', 'ILS')
        self.USDHKD = self.converter.get_rate('USD', 'HKD')
        self.otc_dict = {'XIACY': 14.70, 'XIACF': 3.3, 'INVZ': 10} 
        self.load_portfolio_from_csv(portfolio_csv)
        self.load_pnl_csv(pnl_csv)
        self.update_pnl()
        
    def set_trade_day(self):
        self.trade_day = self.get_trading_day()
        
    def print_status(self):
        status = []
        status.append(f'account: {self.broker}')
        status.append(f'equity: ${self.equity:.2f}')
        status.append(f'allocated: ${self.allocated:.2f}')
        status.append(f'cash: ${self.cash:.2f}')
        status.append(f'current pnl: ${self.current_pnl:.2f}')
        if len(self.pnl.pnl.values) > 1:
            status.append(f'dif from previous day: ${round(self.pnl.pnl.values[-1]-self.pnl.pnl.values[-2], 2):.2f}')
        print('\n'.join(status))
        return status
        
    def save_csvs(self):
        self.set_trade_day()
        pos_path = f'csvs/{self.broker}_pnl_{self.trade_day}.csv'
        pnl_path = f'csvs/{self.broker}_portfolio_{self.trade_day}.csv'
        self.pnl.to_csv(pos_path)
        self.positions.to_csv(pnl_path)
        print('positions saved to path:', pos_path)
        print('pnl saved to path:', pnl_path)

    def load_portfolio_from_csv(self, portfolio_csv=''):
        if portfolio_csv == '':
            portfolio_csv = f'csvs/{self.broker}_portfolio_{self.trade_day}.csv'
        if os.path.exists(portfolio_csv):
            self.positions = pd.read_csv(portfolio_csv, index_col=0)
            self.positions['symbol'] = self.positions['symbol'].apply(lambda x: x.upper())
            print('loaded positions from path:', portfolio_csv)
        else:
            self.positions = pd.DataFrame()
            
    def get_symbol_data(self, symbol):
        try:
            data = si.get_quote_table(symbol)
            # print(symbol, data)
            if data != None and not np.isnan(data['Quote Price']):
                return data
        except:
            pass
        if symbol in self.otc_dict:
            print(f'price collected from otc dict: {symbol}')
            return {'Quote Price': self.otc_dict[symbol], 'Previous Close': self.otc_dict[symbol]}
        print(f'failed to get data: {symbol}')
    
    def get_precision(self, price):
        if price < 1:
            return 5
        return 2
    
    def get_daily_dif(self, row):
        return round((row['price']-row['prev'])/row['prev']*100, 2)
    
    def get_price_text_out(self, row):
        data = row['data']
        symbol = row['symbol']
        prec = row['prec']
        price = row['price']
        curr = 'USD'
        if row['symbol'] == 'TCEHY':
            curr = 'HKD'
        dif_prec = row['daily_dif']
        dif = round(row['daily_dif']*row['prev']/100, prec)
        
        sign = '+' if dif >= 0 else ''
        space1 = ' '*(6-len(symbol))
        space2 = ' '*(8-len(f'{price}'))
        space3 = ' '*(8-len(f'{sign}{dif}'))
        return f'{symbol}: {space1}{curr}{price},{space2}dif:{curr}{sign}{dif},{space3}{sign}{dif_prec:.2f}%'
    
    def get_position_text_out(self, row):
        symbol = row['symbol']
        dif_prec = row['daily_dif']
        price = row['price']
        quant = row['quantity']
        pnl = round(row['p/l'], row['prec'])
        curr = 'USD'
        if row['symbol'] == 'TCEHY':
            curr = 'HKD'
        dif_prec = row['daily_dif']
        sign = '+' if dif_prec >= 0 else ''
        space1 = ' '*(6-len(symbol))
        space2 = ' '*(8-len(f'{price}'))
        space3 = ' '*(8-len(f'{sign}{dif_prec}'))
        return f'{symbol}:{space1}{curr}{price},amount:{quant}{space2}dif:{space3}{sign}{dif_prec:.2f}%, pnl:{pnl}'
            
    def get_price(self, symbol):
        data = si.get_quote_table(symbol)
        if 'USD' in symbol or 'usd' in symbol:
            prec = 5
        else:
            prec = 2
        price = round(data['Quote Price'], prec)
        prev = data['Previous Close']
        dif = round(price - prev, prec)
        dif_prec = round(dif/prev*100, 2)
        sign = '+' if dif >= 0 else ''
        space1 = ' '*(6-len(symbol))
        space2 = ' '*(8-len(f'{price}'))
        space3 = ' '*(8-len(f'{sign}{dif}'))
        text_out = f'{symbol}: {space1} ${price},{space2}dif: ${sign}{dif},{space3}{sign}{dif_prec:.2f}%'
        return price, prev, text_out
            
    def current_price(self, row, feat='Quote Price'):
        symbol = row['symbol']
        price = None
        price = row['data'][feat]
        if row['symbol'] == 'TCEHY':
            price = price*self.USDHKD
        return price
        
    def add_current_price(self, row, feat='Quote Price'):
        return self.current_price(row, feat=feat)
    
    def set_current_value(self):
        self.positions['data'] = self.positions['symbol'].apply(lambda symbol: self.get_symbol_data(symbol))
        self.positions['price'] = self.positions.apply(lambda row: self.add_current_price(row, feat='Quote Price'), axis=1)
        self.positions['prec'] = self.positions['price'].apply(lambda x: self.get_precision(x))
        self.positions['prev'] = self.positions.apply(lambda row: self.add_current_price(row, feat='Previous Close'), axis=1)
        self.positions['daily_dif'] = self.positions.apply(lambda row: self.get_daily_dif(row), axis=1)
        self.positions['text_out'] = self.positions.apply(lambda row: self.get_price_text_out(row), axis=1)
        self.set_usd_value('price')
        self.allocated = round(self.positions.value_usd.sum(), 2)
    
    def get_usd_value(self, row, feat='price'):
        if row['currency'] == 'USD':
            return row[feat]*row['quantity']
        else:
            return row[feat]*row['quantity']/self.converter.get_rate('USD', row['currency'])
    
    def set_usd_value(self, feat='price'):
        trg_feat = {'open_price': 'open_value_usd', 'price': 'value_usd'}[feat]
        self.positions[trg_feat] = self.positions.apply(lambda row: self.get_usd_value(row, feat), axis=1)
    
    def add_position(self, symbol, price, quantity, currency='USD', date=datetime.datetime.today()):
        line = {'symbol': symbol.upper(), 'open_price': price, 'quantity': quantity, 'currency': currency, 'open_date': date,
                'broker': self.broker}
        line['open_value_usd'] = self.get_usd_value(line, 'open_price')
        self.positions = self.positions.append(line, ignore_index=True)
        
    def set_pnl(self):
        self.positions['p/l'] = self.positions.apply(lambda row: row['value_usd']-row['open_value_usd'], axis=1)
        self.positions = self.positions.sort_values(by='p/l')
        self.current_pnl = round(self.positions['p/l'].sum(), 2)
        self.positions['p/l_ils'] = self.positions['p/l'].apply(lambda x: x*self.USDILS)
        if len(self.positions) > 0:
            self.invested = round(self.positions.open_value_usd.sum(), 2)
            self.cash = round(self.start_balance - self.invested, 2)
            self.equity = round(self.cash + self.positions.value_usd.sum(), 2)
            self.positions['pos_out'] = self.positions.apply(lambda row: self.get_position_text_out(row), axis=1)
        
    def get_hwm(self, row):
        if row.name == 0:
            return row['equity']
        return max(self.pnl[:row.name].equity.max(), row.equity)
    
    def hour_to_float(self, hours=16, minutes=30):
        return hours+minutes/60

    def str_hour_to_float(self, hour='16:30'):
        return int(hour.split(':')[0])+int(hour.split(':')[1])/60

    def get_trading_day(self, open_market='00:30'):
        trade_day = datetime.datetime.now()
        day_off = -1
        if self.hour_to_float(trade_day.hour, trade_day.minute) > self.str_hour_to_float(open_market):
            day_off = 0
        td = (datetime.datetime.now() + datetime.timedelta(days=day_off))
        return str(datetime.datetime(year=td.year, day=td.day, month=td.month)).split()[0]
    
    def load_pnl_csv(self, csv_path=''):
        if csv_path == '':
            csv_path = f'csvs/{self.broker}_pnl_{self.trade_day}.csv'
        self.pnl = pd.read_csv(csv_path, index_col=0)
        self.pnl = self.pnl.sort_values(by='date')
        print('loaded pnl log from:', csv_path)
    
    def update_pnl_log(self):
        self.pnl = self.pnl[self.pnl.date < self.trade_day]
        self.pnl = self.pnl.append({'date': self.trade_day, 'equity': self.equity}, ignore_index=True)
        self.pnl['water_mark'] = self.pnl.apply(lambda row: self.get_hwm(row),axis=1)
        self.pnl['pnl'] = self.pnl['equity'].apply(lambda x: x - self.pnl['equity'].values[0])
        self.pnl['max_gain'] = self.pnl.apply(lambda row: row['water_mark'] - self.pnl['equity'].values[0], axis=1)
        
    def update_pnl(self):
        self.set_current_value()
        self.set_pnl()
        print(f'current pnl: {self.current_pnl}')
        self.update_pnl_log()
        status = self.print_status()
        self.save_csvs()
        return status
    
    def plot_chart(self, name='equity', figsize=(16,9), major_loc=20, minor_loc=4, to_file=False, broker='', days=None):
        if name not in ['equity', 'pnl']:
            print('supported plot types: equity/pnl only')
            return
        fig, ax = plt.subplots(facecolor='white', figsize=figsize)
        ax.xaxis.set_major_locator(MultipleLocator(major_loc))
        ax.xaxis.set_minor_locator(AutoMinorLocator(minor_loc))
        if not days:
            days = len(self.pnl['date'].values)
        x = self.pnl['date'].values[-days:]
        y = self.pnl[name].values[-days:]
        scale = {'equity': 0.0025, 'pnl': 0.025}[name]
        top = {'equity':'water_mark', 'pnl': 'max_gain'}[name]
        color = {'equity':'blue', 'pnl': 'green'}[name]
        y_max = np.max(y)*(1+scale)
        y_min = np.min(y)*(1-scale) 
        plt.fill_between(x, self.pnl[top].values[-days:], label=top, color=color, alpha=0.3)
        plt.plot(x, y, label=name, alpha=1)
        plt.fill_between(x, y, self.pnl[top].values[-days:], label=f'under {top}', alpha=0.3, color='red')
        plt.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.ylabel('$')
        plt.ylim(top=y_max, bottom=y_min)
        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        if to_file:
            path = f'pngs/{broker}_{name}_{self.trade_day}.png'
            self.image_to_file(plt, path)
            self.equity_image_path = path
            return path
        else:
            plt.show()
    
    def image_to_file(self, plt, path):
        plt.savefig(path)
        print(f'image saved to path: {path}')
    
    def get_historic_data_for_symbol(self, symbol, date):
        if symbol not in self.hist_data:
            yf_data = yf.Ticker(symbol)
            data = yf_data.history(start=date)
            data['date'] = [str(d).split('T')[0] for d in data.index.values]
            data['date'] = data['date'].apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d'))
            self.hist_data[symbol] = data
            print(f'loaded historic data for {symbol}')

    def get_price_for_date_for_symbol(self, dt, symbol, td=datetime.timedelta(days=1)):
        slice_df = self.hist_data[symbol][(self.hist_data[symbol].date>=dt) & (self.hist_data[symbol].date<dt+td)]
        if len(slice_df) >= 1:
            self.last[symbol] = slice_df.Close.values[-1]
            return self.last[symbol]
        elif symbol in self.last:
            return self.last[symbol]
        elif symbol in self.otc_dict:
            return self.otc_dict[symbol]
        else:
            print(f'failed to get historic price for symbol: {symbol}, date: {dt}')

    def create_historic_equity(self):
        self.last = {}
        self.pnl = pd.DataFrame()
        log_cols = ['currency', 'open_date', 'open_price', 'open_value_usd', 'quantity',
           'symbol']
        dts = sorted(set(bh_p.positions.open_date.unique()))
        str_dates = [str(d).split('T')[0] for d in dts]
        s_date = datetime.datetime.strptime(str(str_dates[0]).split('T')[0], '%Y-%m-%d')-datetime.timedelta(days=3)
        self.pnl = self.pnl.append({'date': s_date, 'equity': self.start_balance}, ignore_index=True)
        for i, date in enumerate(str_dates):
            next_dt_str = str_dates[i+1] if i < len(str_dates)-1 else '2025-12-31'
            next_dt = datetime.datetime.strptime(next_dt_str, '%Y-%m-%d')
            prev_dt = datetime.datetime.strptime(date, '%Y-%m-%d')
            slice_pos = bh_p.positions[bh_p.positions.open_date <= date][log_cols]
            print(date, dt, slice_pos.symbol.values)
            for symbol in slice_pos.symbol.values:
                get_historic_data_for_symbol(self, symbol, date)
            for s_date in bh_p.hist_data[symbol].index.values:
                s_date = datetime.datetime.strptime(str(s_date).split('T')[0], '%Y-%m-%d')
                if prev_dt < s_date <= next_dt:
                    slice_pos['price'] = slice_pos['symbol'].apply(lambda symbol: get_price_for_date_for_symbol(self, s_date, symbol))
                    slice_pos['value_usd'] = slice_pos.apply(lambda row: row['price']*row['quantity'], axis=1)
        #             self.set_pnl()
                    invested = round(slice_pos.open_value_usd.sum(), 2)
                    cash = round(self.start_balance - invested, 2)
                    equity = cash + slice_pos.value_usd.sum()
                    print(s_date, invested, cash, slice_pos.value_usd.sum(), equity)
                    self.pnl = self.pnl.append({'date': s_date, 'equity': equity}, ignore_index=True)
                    

class telegramBot():
    def __init__(self, token, portfolios=[], timeout=100):
        self.token = token
        self.portfolios = {p.broker: p for p in portfolios}
        self.names = list(self.portfolios.keys())
        self.timeout = timeout
        self.url = f"https://api.telegram.org/bot{self.token}"
        self.from_ = None
        
    def get_updates(self,offset=None):
            # In 100 seconds if user input query then process that, use it as the read timeout from the server
            url = self.url+f"/getUpdates?timeout={self.timeout}"    
            if offset:
                url = url+f"&offset={offset+1}"
            url_info = requests.get(url)
            return json.loads(url_info.content)
        
    def send_message(self, msg, chat_id):
            url = self.url + f"/sendMessage?chat_id={chat_id}&text={msg}"
            if msg is not None:
                requests.get(url)
                
    def grab_token(self):
            return tokens
        
    def loop_for_telegram(self, expire_in=60):
        update_id = None
        t0 = datetime.datetime.now()
        td = datetime.datetime.now() - t0
        self.tbot_out('tracking your portfolio..')
        while (td.seconds < expire_in*60):
            updates = self.get_updates(offset=update_id)
            
            if updates:
                updates = updates['result']
                for item in updates:
                    update_id = item["update_id"]
                    print(update_id)
                    try:
                        message = item["message"]["text"]
                        from_ = item["message"]["from"]["id"]
                    except:
                        message, from_ = None, None
                        
                    print('user:', from_, 'message:', message)
                    response = self.add_user_input(message, from_=from_)
                    print("standing by ...")
            td = datetime.datetime.now() - t0
        print('telegram loop exit')
        
    def tbot_out(self, message):
        if isinstance(message, list):
            message = '\n'.join(message)
        if self.from_:
            print('sending tbot message ..')
            self.send_message(message, self.from_)
            
    def send_photo(self, file_path):
        method = "sendPhoto"
        if self.from_:
            print('sending tbot image ..')
        params = {'chat_id': self.from_}
        files = {'photo': open(file_path, 'rb')}
        api_url = self.url + f"/{method}?"
        resp = requests.post(api_url, params, files=files)
        return resp
    
    def match_words_in_user_input(self, user_input, words, ret_ind=False):
        for w in words:
            if w in user_input:
                if ret_ind:
                    return True, user_input.index(w)
                return True
        if ret_ind:
            return False, None
        return False
    
    def get_matches_in_user_input(self, user_input, words, ret_org_if_no_match=True):
        matches = []
        for w in words:
            if w in user_input:
                matches.append(w)
        if ret_org_if_no_match and len(matches) == 0:
            return words
        return matches
    
    def get_text_features(self, user_input):
        user_input_split = user_input.split()
        plot = self.match_words_in_user_input(user_input_split, ['plot', 'chart'])
        names = self.get_matches_in_user_input(user_input_split, self.names)
        charts = self.get_matches_in_user_input(user_input_split, ['equity', 'pnl'])
        prices = self.match_words_in_user_input(user_input_split, ['price', 'prices'])
        symbols, s_ind = self.match_words_in_user_input(
            user_input_split, ['symbol', 'symbols', 'stock', 'stocks', 'symbols:', 'symbol:'], ret_ind=True)
        status = self.match_words_in_user_input(user_input_split, ['profit', 'status'])
        open_pos = self.get_matches_in_user_input(
            user_input_split, ['open', 'symbol', 'quantity', 'price', 'broker'], ret_org_if_no_match=False)
        return user_input_split, plot, names, charts, prices, symbols, s_ind, status, open_pos
    
    def print_out_positions_by_symbol(self, user_input_split):
        responded = False
        req_pos, pos_ind = self.match_words_in_user_input(
            user_input_split, ['position', 'positions', 'position:', 'positions:'], ret_ind=True)
        if req_pos:
            for broker in self.names:
                symbols = [s.upper() for s in user_input_split[pos_ind+1:]]
                positions = self.portfolios[broker].positions.symbol.unique()
                if len(symbols) == 0:
                    symbols = positions
                for s in symbols:
                    if s in positions:
                        for p in self.portfolios[broker].positions[self.portfolios[broker].positions.symbol.isin([s])].pos_out.values:
                            self.tbot_out(p)
                            responded = True
            if not responded:
                self.tbot_out(f'could not find symbol in portfolio.\nto view positions write back positions..')
                responded = True
        return responded
    
    def get_totals(self, ret_text=True, feats=['value_usd', 'pnl', 'max_gain']):
        totals = dict.fromkeys(feats,0)
        for broker in self.names:
            totals['value_usd'] += self.portfolios[broker].positions.value_usd.sum()
            totals['pnl'] += self.portfolios[broker].pnl.pnl.values[-1]
            totals['max_gain'] += self.portfolios[broker].pnl.max_gain.values[-1]
        if ret_text:
            totals_out = []
            for f in feats:
                totals_out.append(f'total {f}: ${totals[f]:.2f}')
            return ['\n']+totals_out+['\n']
        return totals
                              
    
    def print_out_status(self, status, names, prices):
        if status:
            for broker in names:
                self.tbot_out(f'collecting prices for stocks under: {broker}..')
                response = []
                response.extend(self.portfolios[broker].update_pnl())
                if prices:
                    for t in self.portfolios[broker].positions.text_out.values:
                        self.tbot_out(t)
                self.tbot_out(response)
            self.tbot_out(self.get_totals())
            return True
        return False
    
    def print_out_symbols(self, symbols, user_input_split, s_ind):
        if symbols and 'open' not in user_input_split:
            for s in user_input_split[s_ind+1:]:
                try:
                    p, prev, t = self.portfolios[self.names[0]].get_price(s)
                    self.tbot_out(t)
                except Exception as e:
                    print(e)
            return True
        return False
    
    def plot_charts(self, plot, user_input_split, names, charts):
        if plot:
            days = None
            if 'days' in user_input_split:
                try:
                    days = int(user_input_split[user_input_split.index('days')+1])
                except:
                    pass
            for broker in names:
                major_loc = {'bh':4, 'etoro':1}[broker]
                for c in charts:
                    self.tbot_out(f'fetching {c} chart for: {broker}..')
                    path = self.portfolios[broker].plot_chart(c, to_file=True, major_loc=major_loc, broker=broker, days=days)
                    resp = self.send_photo(path)
            return True
        return False
    
    def send_open_pos(self, open_pos, user_input_split):
        if 0 < len(open_pos) < 5:
            self.tbot_out('to open position use this format:\nopen symbol AAPL price 120.25 quatity 12 broker etoro')
            responded = True
        elif len(open_pos) == 5:
            broker = user_input_split[user_input_split.index('broker')+1]
            if broker in self.names:
                symbol = user_input_split[user_input_split.index('symbol')+1].upper()
                quantity = float(user_input_split[user_input_split.index('quantity')+1])
                price = float(user_input_split[user_input_split.index('price')+1])
                self.tbot_out(f'opening position. broker: {broker}, symbol: {symbol}, quantity: {quantity}, price: {price}')
                self.portfolios[broker].add_position(symbol, price, quantity)
                self.tbot_out(self.portfolios[broker].update_pnl())
            else:
                self.tbot_out('broker could only be etoro/bh')
            return True
        return False
    
    def add_user_input(self, user_input, from_=None):
        print(f'from:{from_}, text:{user_input}')
        responded = []

        if from_:
            self.from_ = from_
        user_input_split, plot, names, charts, prices, symbols, s_ind, status, open_pos = self.get_text_features(user_input)
                
        responded.append(self.print_out_status(status, names, prices))
        responded.append(self.print_out_symbols(symbols, user_input_split, s_ind))
        responded.append(self.plot_charts(plot, user_input_split, names, charts))
        responded.append(self.send_open_pos(open_pos, user_input_split))
        responded.append(self.print_out_positions_by_symbol(user_input_split))
                    
        if not np.any(responded):
            self.tbot_out('try one of these:\nplot equity or pnl\nget status\pnl\nprice status\nprice for symbols: APPL, NVDA, etc..')