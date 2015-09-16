# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import tushare as ts
import datetime as dt
import dt_tool
from model_tool import *
from stock_tools import trade
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import sys
from tabulate import tabulate
matplotlib.style.use('ggplot')

# <codecell>

# get all time ticks
def get_ticks(id, end, delta):
    date_list = dt_tool.dt_range(end, delta)
    ticks = None
    for date_str in date_list:
        try:
            tick = ts.get_tick_data(id, date_str)
        except:
            continue
        if tick is None:
            continue
        tick['date'] = date_str
        if ticks is None:
            ticks = tick
        else:
            ticks = ticks.append(tick)
        ticks['dt'] = ticks['date'] + ' '+ ticks['time']
        ticks['id'] = id
    return ticks

def cmp_top(ticks):
    if ticks is None:
        print 'no'
        return None
    try:
        today = ts.get_today_ticks(id).sort('amount', ascending=False)
    except:
        return None
    sticks = ticks.sort('amount', ascending=False).head(20)
    sticks = sticks.reset_index()
    today = today.reset_index()
    top = sticks.ix[0]
    top_today = today.ix[0]

    return [(round((top.price - top_today.price)/top.price, 3)), round(top.price, 3), round(top_today.price, 3), top.amount, top_today.amount]


def tick_plot(ticks): 
    # init
    ticks = ticks.sort('dt', ascending=True)
    fig = ticks.plot(x='dt', y=['price', 'amount'],  secondary_y='amount',  legend=True, title=id, rot=10, figsize=(18,8)) 
    return fig.get_figure()

def hist_plot(id, end, delta):
    start_date = dt_tool.add(end, delta)
    df = ts.get_h_data(id, autype='qfq', start=start_date)
    df['r13'] = pd.rolling_mean(df['close'], window=13, min_periods=1, center=True)
    df['r34'] = pd.rolling_mean(df['close'], window=34, min_periods=1, center=True)
    df['r55'] = pd.rolling_mean(df['close'], window=55, min_periods=1, center=True)
    fig = df.plot(y=['close', 'r13',  'r34', 'r55', 'volume'], title=id,  secondary_y='volume', grid=True, legend=True, figsize=(18, 8))
    return fig.get_figure()
    
id = sys.argv[1]
delta = int(sys.argv[2])
print id
base = '/Users/haoyuanhu/Write/notes'
today = dt_tool.get_today()
ticks = get_ticks(id, today, -delta)
name, now, begin, rat = trade.get_realtime(id)
fig = tick_plot(ticks)
fig.savefig('%s/image/%s_tick_%s.png'%(base, id, today), dpi=400)
fig = hist_plot(id, today, -delta)
fig.savefig('%s/image/%s_hist_%s.png'%(base, id, today), dpi=400)
sticks = ticks.sort('amount', ascending=False).head(5)
sticks.type = sticks.type.apply(lambda x : type2id(x))
sticks.to_csv('%s/data/%s_%s.csv'%(base, id, today),  index=None)

# tick today
yestoday = dt_tool.add(today, -3)
#ticks_today = ts.get_today_ticks(id).sort('amount', ascending=False).head(10)
#ticks_today = ts.get_tick_data(id, today).sort('amount', ascending=False).head(5)
ticks_today = ts.get_tick_data(id, yestoday).sort('amount', ascending=False).head(5)
ticks_today.type = ticks_today.type.apply(lambda x : type2id(x))
txt = ticks_today.reset_index().drop(['index'], 1)
tab = tabulate(txt, headers='keys', tablefmt='pipe')
tick_today = tab.encode('utf-8')

# hist tick
txt = sticks.drop(['dt', 'id'], 1)
txt = txt.reset_index().drop(['index'], 1)
tab = tabulate(txt, headers='keys', tablefmt='pipe')
tick_hist = tab.encode('utf-8')
top_tick_day = txt.ix[0].date
top_tick_type = txt.ix[0].type

# write
params = {"rate":'%s%%'%(100 *rat), "id":id, "name":name, "today":today, "tick_hist":tick_hist, "tick_today":tick_today, 'begin':begin, 'now':now, 'top_tick_day':top_tick_day, "top_tick_type":top_tick_type}

fr = open('tp.md', 'r').read()%params
f = open('%s/stock/%s_%s.md'%(base,  id, today), 'w')
f.write(fr)
f.close()


