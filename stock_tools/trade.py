import tushare as ts
from pandas import DataFrame
import numpy as np
from datetime import datetime
import  urllib2


def get_new_history():
    news = ts.get_gem_classified()
    return get_data_by_column(news)

def get_all_history():
    all = ts.get_stock_basics()
    all = all.reset_index()
    return get_data_by_column(all)


def get_sz50s_history():
    szs = ts.get_sz50s()
    return get_data_by_column(szs)

def get_zz500s_history():
    zz500s = ts.get_zz500s()
    return get_data_by_column(zz500s)

def get_hs300s_history():
    hs300s = ts.get_hs300s()
    return get_data_by_column(hs300s)

def get_data_by_column(stock_pd):
    ids = stock_pd['code']
    rs = get_data(ids[0])
    for id in ids[1:]:
        try:
            code = get_data(id)
        except:
            continue
        rs = rs.append(code)
    return rs

def get_data(id):
    if id is None:
        return None
    now = datetime.now()
    now_str = now.strftime('20%y-%m-%d')
    df = ts.get_h_data(id, autype='hfq', start='2013-01-01', end=now_str)
    if df is None:
        return None
    df['code'] = id
    delta = df['open'] - df['close']
    rate = delta/df['open']
    df['delta'] = np.round(delta, 3)
    df['rate'] = np.round(rate, 3)
    df = df.reset_index()
    return df

def get_realtime(id):
    if id[0] == '6':
            cmd = "http://hq.sinajs.cn/list=sh%s"
    else:
        cmd = "http://hq.sinajs.cn/list=sz%s"
    if len(id) < 1:
        return 
    id = id.zfill(6)
    cmd = cmd%id
    rs = urllib2.urlopen(cmd)
    rs = rs.read()
    items = rs.split(',')
    if len(items) < 3:
        return ''
    name = items[0].strip().split('"')[1]
    now = float(items[3])
    begin = float(items[2])
    rat = round((now - begin)/begin, 3)
    name = name.decode('gbk').encode('utf-8')
    return  name, now, begin, rat

