# -*- coding: utf-8 -*-

import pandas
import math
import numpy  as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Ridge 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import tushare as ts
import datetime as dt
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cross_validation import train_test_split,cross_val_score

pd.options.mode.chained_assignment = None  # default='warn'


ids = pd.read_csv('./id', sep='|', names=('id', 'name'),dtype={'time':'string', 'id':'string'})
ids.index = ids.id
ss = StandardScaler()

## 获取分笔数据
# get all time ticks
def get_tick_feature(id, end_str, delta):
    if isinstance(end_str, str):
        end = dt.datetime.strptime(end_str, '%Y-%m-%d')
    date_list = [end - dt.timedelta(days=x) for x in range(0, 90)]
    ticks = None
    for date in date_list:
        date_str = dt.datetime.strftime(date, '%Y-%m-%d')
        try:
            tick = ts.get_tick_data(id, date_str)
        except:
            continue
        if tick is None:
            continue
        ft = tick.sort('amount', ascending=False).head(10).reset_index().drop(['index', 'time'], 1).stack().reset_index()
        ft['fea'] = ft.level_0.map(str) + '_' + ft.level_1
        fea_pd = ft.drop(['level_0', 'level_1'], 1)
        fea_pd.index = fea_pd.fea
        fT = fea_pd.T
        fT['date'] = date_str

        if ticks is None:
            ticks = fT
        else:
            ticks = ticks.append(fT)
    return ticks

# ## 将数据stack为series, 方便处理
def get_sd(id):
    try:
        hist = ts.get_h_data(id, autype='qfq', start='2005-06-10')
    except:
        return None
    fea_tick = get_tick_feature(id, '2015-06-10', 1500)
    print 'fea', fea_tick.columns
    print 'hist', hist.columns
    hist.to_csv('hist.csv')
    fea_tick.to_csv('tick.csv')
    h = hist.merge(fea_tick, how='left', left_index=True, right_on='date')
    s = h.stack()
    spd = pd.DataFrame(s)
    sd = spd.reset_index()
    sd.columns = ('date', 'type', 'value')
    return sd


# ## 根据给定的label时间点， 返回fx, y
def get_trainset(id, sd, label_dt, delta = 7):
    label_date = dt.datetime.strftime(label_dt, '%Y-%m-%d')
    end_date =  dt.datetime.strftime(label_dt + dt.timedelta(delta), '%Y-%m-%d')
    x = get_feature(id, sd, label_date)
    if x is None:
        return None
    if label_dt.weekday() == 5 or label_dt.weekday() == 6:
            return None
    ssd = sd.set_index('date')
    print end_date
    print label_date
    label_close = sd.query('date == "%s" and type == "close"'%(label_date)).value
    if label_close is None or len(label_close) == 0:
        return None
    label_close = label_close.values[0]
    print label_close
    print type(label_close)
    end_close = sd.query('date == "%s" and type == "close"'%(end_date)).value
    if end_close is None or len(end_close) == 0:
        return None
    end_close = end_close.values[0]
    y = (end_close - label_close)/label_close
    print y
    return x, y


def get_feature(id, sd, label_date): 
    label_dt = dt.datetime.strptime(label_date, '%Y-%m-%d')
    if label_dt.weekday() == 5 or label_dt.weekday()==6:
        return None
    start_dt = label_dt - dt.timedelta(90)
    start_date = dt.datetime.strftime(start_dt, '%Y-%m-%d')
    trainset = sd.query('date <= "%s" and date >= "%s"'%(label_date, start_date))
    trainset['delta'] = label_dt - trainset.date
    trainset['delta_str'] = trainset.delta.dt.days.map(str)
    trainset['feature'] = trainset.type  + '_' + trainset.delta_str
    fea = trainset.loc[:, ['feature', 'value']]
    ifea = fea.set_index('feature')
    x = ifea.T
    
    ft = get_tick_feature(id, label_date, 90)
    return x.merge(ft, on='date')


def train(id, sd, end='2015-07-15'):
    end_dt = dt.datetime.strptime(end, '%Y-%m-%d')
    date_list = [end_dt - dt.timedelta(days=x) for x in range(0, 2000)]
    rs = get_trainset(id, sd, date_list[0])
    if rs is None:
        return None
    rx, y = rs
    ry = [y]
    for d in date_list:
        rs = get_trainset(id, sd, d)
        if rs is None:
            continue    
        x, y = rs
        rx = rx.append(x)
        ry.append(y)
    rx.fillna(0, inplace=True)
    train_x = np.asarray(rx)
    train_y = np.asarray(ry).ravel()
    train_scale_x = ss.fit_transform(train_x)
    clf = GradientBoostingRegressor()
    clf.fit(train_scale_x, train_y)
    scores = cross_val_score(clf, train_scale_x, train_y, scoring='mean_squared_error', cv=5)
    return clf, scores.mean(), rx


def predict(sd, rx, clf, date='2015-07-30'):
    fx = get_feature(sd, date)
    tmpx = rx.append(fx)
    cfx = tmpx.tail(1)
    cfx.fillna(0, inplace=True)
    pred_x = np.asarray(cfx)
    pred_scale_x = ss.fit_transform(pred_x)
    pred = clf.predict(pred_scale_x)
    return pred


f = open('pred.csv', 'w')
for id in ids.id:
    print id
    try:
        sd = get_sd(id)
        if sd is None:
            continue
        rs = train(id, sd)
        if rs is  None:
            continue
        clf, score, rx = rs 
        rmse = math.sqrt(-score)
        pred = predict(sd, rx, clf)[0]
        f.write('%s, %s, %s, %s\n'%(id, ids.ix[id]['name'], pred, rmse))
        f.flush()
    except     Exception,e:
        print e
        print '=== STEP ERROR INFO START'
        import traceback
        traceback.print_exc()
        print '=== STEP ERROR INFO END'
        continue
f.close()



