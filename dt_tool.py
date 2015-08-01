import datetime
import tushare as ts
import pandas as pd
def is_weekend(date):
    dt = str2dt(date)
    if dt.weekday() == 5 or dt.weekday()==6:
        return True
    return False

def add(start_str, delta):
    start_dt = datetime.datetime.strptime(start_str, '%Y-%m-%d')
    end_dt = start_dt + datetime.timedelta(delta)
    return dt2str(end_dt)

def dt_range(start_str, delta=90):
    if delta > 0:
        date_list = [add(start_str, x) for x in range(0, delta)]
    else:
        date_list = [add(start_str, x) for x in range(-1, -delta)]
    return date_list

def format(dt_str):
    dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d')
    return dt2str(dt)

def dt2str(dt):
    return dt.strftime('%Y-%m-%d')

def str2dt(dt_str):
    return datetime.datetime.strptime(dt_str, '%Y-%m-%d')

def delta_day(start, end):
    start_dt = str2dt(start)
    end_dt = str2dt(end)
    return (end_dt - start_dt).days

def delta_id(start, end):
    return '%d_%s'%(delta_day(start, end)/7, str2dt(start).weekday())

today = datetime.datetime.today()
today_str = dt2str(today)
delta = delta_id('2015-06-30', '2015-07-01')
print delta
delta = delta_id('2015-06-28', '2015-07-01')
print delta
a = pd.DataFrame(['2015-06-02', '2015-07-01'])
a.columns = ['dt']
print  delta_id('2015-04-24', '2015-07-30')
print  delta_id('2015-04-27', '2015-07-30')
