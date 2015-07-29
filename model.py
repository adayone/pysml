# -*- coding: utf-8 -*-

import pandas
import numpy  as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Ridge 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import tushare as ts
import datetime as dt
import pandas as pd
from stock_tools import trade 
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cross_validation import train_test_split,cross_val_score

pd.options.mode.chained_assignment = None  # default='warn'


ids = pd.read_csv('./id', sep='|', names=('id', 'name'),dtype={'time':'string', 'id':'string'})
ss = StandardScaler()


# ## 将数据stack为series, 方便处理


def get_sd(id):
    try:
        hist = ts.get_h_data(id, autype='qfq', start='2013-06-10')
    except:
        return None
    s = hist.stack()
    spd = pd.DataFrame(s)
    sd = spd.reset_index()
    sd.columns = ('date', 'type', 'value')
    return sd


# ## 根据给定的label时间点， 返回fx, y


def get_trainset(sd, label_dt):
    label_date = dt.datetime.strftime(label_dt, '%Y-%m-%d')
    x = get_feature(sd, label_date)
    if x is None:
        return None
    if label_dt.weekday() == 0:
        if 'close_0' not in x or 'close_3' not in x:
            return None
        y = (x.close_0 - x.close_3)/x.close_3
        fx = x.ix[:, 'open_3':]
    else:
        if 'close_0' not in x or 'close_1' not in x:
            return None
        y = (x.close_0 - x.close_1)/x.close_1
        fx = x.ix[:, 'open_1':]
    return fx, y


def get_feature(sd, label_date): 
    label_dt = dt.datetime.strptime(label_date, '%Y-%m-%d')
    if label_dt.weekday() == 5 or label_dt.weekday()==6:
        return None
    start_dt = label_dt - dt.timedelta(180)
    start_date = dt.datetime.strftime(start_dt, '%Y-%m-%d')
    trainset = sd.query('date <= "%s" and date >= "%s"'%(label_date, start_date))
    trainset['delta'] = label_dt - trainset.date
    trainset['delta_str'] = trainset.delta.dt.days.map(str)
    trainset['feature'] = trainset.type  + '_' + trainset.delta_str
    fea = trainset.loc[:, ['feature', 'value']]
    ifea = fea.set_index('feature')
    x = ifea.T
    return x


def train(sd, end='2015-07-28'):
    end_dt = dt.datetime.strptime(end, '%Y-%m-%d')
    date_list = [end_dt - dt.timedelta(days=x) for x in range(0, 600)]
    rs = get_trainset(sd, date_list[0])
    if rs is None:
        return None
    rx, y = rs

    ry = [y]
    for d in date_list:
        rs = get_trainset(sd, d)
        if rs is None:
            continue    
        x, y = rs
        rx = rx.append(x)
        ry.append(y)
    rx.fillna(0, inplace=True)
    train_x = np.asarray(rx)
    train_y = np.asarray(ry).ravel()
    train_scale_x = ss.fit_transform(train_x)
    clf = Ridge(0.5)
    clf.fit(train_scale_x, train_y)
    scores = cross_val_score(clf, train_scale_x, train_y, scoring='mean_squared_error', cv=5)
    return clf, scores.mean(), rx


def predict(sd, rx, clf, date='2015-07-29'):
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
    
    sd = get_sd(id)
    if sd is None:
        continue
    rs = train(sd)
    if rs is  None:
        continue
    clf, score, rx = rs 
    try:
        pred = predict(sd, rx, clf)
    except:
        continue
    f.write('pred:%s score:%s'%(pred, score))
    f.flush()
f.close()



