# -*- coding: utf-8 -*-

import pandas
import sys
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
import dt_tool

pd.options.mode.chained_assignment = None  # default='warn'


ss = StandardScaler()
end = '2015-07-31'
delta = 15
id = '000002'

## 获取分笔数据
# get all time ticks
# 创建一个日期列表
ticks = None
date_list = dt_tool.dt_range(end, -delta)
for date in date_list:
    try:
        tick = ts.get_tick_data(id, date)
    except Exception, e:
        print e
        continue
    if tick is None:
        continue
    ft = tick.sort('amount', ascending=False).head(10).reset_index().drop(['index', 'time', 'change'], 1).stack().reset_index()
    ft.index = ft.level_0.map(str) + '_' + ft.level_1
    fT = ft.drop(['level_0', 'level_1'], 1).T
    fT['date'] = dt_tool.format(date)
    if ticks is None:
        ticks = fT
    else:
        ticks = ticks.append(fT)
    ticks.to_csv('data/ticks.csv', index=None)

# 获取历史数据矩阵
# 将数据stack为series, 方便处理
try:
    #hist = ts.get_h_data(id, autype='qfq', start='2005-06-10')
    hist = ts.get_hist_data(id)
except Exception, e:
    print e
ls = hist.index.format()
hist['date'] = [dt_tool.format(x) for x in ls]
hist.to_csv('data/hist.csv', index=None)

# 聚合两份数据
h = hist.merge(ticks, how='left',  on='date')
h.to_csv('data/merged.csv', index=None)
h.index = h.date
h = h.drop('date', 1)
s = h.stack()
spd = pd.DataFrame(s)
sd = spd.reset_index()
sd.columns = ('date', 'type', 'value')
sd.to_csv('data/sd.csv', index=None)

# 开始构建训练数据
# 根据给定的label时间点， 返回fx, y
label_start_date = '2015-07-23'
delta = 7
label_end_date = dt_tool.add(label_start_date, delta)

# 将日期更换为delta
if dt_tool.is_weekend(label_start_date):
    sys/exit(-1)
fea_start_date = dt_tool.add(label_start_date, -90)
trainset = sd.query('date >= "%s" and date <= "%s"'%(fea_start_date, label_start_date))
trainset['delta'] = trainset.date.apply(lambda x : dt_tool.delta_id(x, label_start_date))
trainset['feature'] = trainset.type  + '_' + trainset.delta
trainset.to_csv('data/trainset.csv', index=None)
fea = trainset.loc[:, ['feature', 'value']]
fea.to_csv('data/fea.csv', index=None)
ifea = fea.set_index('feature')
x = ifea.T
x.to_csv('data/x.csv', index=None)




#x = get_feature(id, sd, label_date)
#if x is None:
#    return None
#if label_dt.weekday() == 5 or label_dt.weekday() == 6:
#        return None
#ssd = sd.set_index('date')
#label_close = sd.query('date == "%s" and type == "close"'%(label_date)).value
#if label_close is None or len(label_close) == 0:
#    return None
#label_close = label_close.values[0]
#end_close = sd.query('date == "%s" and type == "close"'%(end_date)).value
#if end_close is None or len(end_close) == 0:
#    return None
#end_close = end_close.values[0]
#y = (end_close - label_close)/label_close
#print y



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


#f = open('pred.csv', 'w')
#for id in ids.id:
#    print id
#    try:
#        sd = get_sd(id)
#        if sd is None:
#            continue
#        rs = train(id, sd)
#        if rs is  None:
#            continue
#        clf, score, rx = rs 
#        rmse = math.sqrt(-score)
#        pred = predict(sd, rx, clf)[0]
#        f.write('%s, %s, %s, %s\n'%(id, ids.ix[id]['name'], pred, rmse))
#        f.flush()
#    except     Exception,e:
#        print e
#        print '=== STEP ERROR INFO START'
#        import traceback
#        traceback.print_exc()
#        print '=== STEP ERROR INFO END'
#        continue
#f.close()



