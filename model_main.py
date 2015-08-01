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


def type2id(x):
    if x == '买盘':
        return 1
    elif x == '中性盘':
        return 0
    return -1

def get_sd(id, label_start_date, fea_delta = 120):
    # delta 指的是 确定label；的时候 是一个星期的涨幅还是什么
    # fea delta 指的是， 我要准备多久的数据
    ## 获取分笔数据
    # get all time ticks
    # 创建一个日期列表
    ticks = None
    date_list = dt_tool.dt_range(label_start_date, -fea_delta)
    for date in date_list:
        try:
            tick = ts.get_tick_data(id, date)
        except Exception, e:
            print e
            continue
        if tick is None:
            continue
        tick.type = tick.type.apply(lambda x : type2id(x))
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

    if hist is None or ticks is None:
        return None
   
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
    # 看了下 之前写的效率太低了 数据获取应该只有一遍的
    return sd
    
def get_trainset(sd, label_start_date, label_delta=7, is_predict=False):
    # 开始构建训练数据
    # 根据给定的label时间点， 返回fx, y
    label_end_date = dt_tool.add(label_start_date, label_delta)
    
    # 将日期更换为delta
    if dt_tool.is_weekend(label_start_date):
        return None
    # 这个决定了获取多久的数据作为特征
    fea_start_date = dt_tool.add(label_start_date, -900)
    trainset = sd.query('date >= "%s" and date <= "%s"'%(fea_start_date, label_start_date))
    trainset['delta'] = trainset.date.apply(lambda x : dt_tool.delta_id(x, label_start_date))
    trainset['feature'] = trainset.type  + '_' + trainset.delta
    trainset.to_csv('data/trainset.csv', index=None)
    fea = trainset.loc[:, ['feature', 'value']]
    fea.to_csv('data/fea.csv', index=None)
    ifea = fea.set_index('feature')
    x = ifea.T
    x.to_csv('data/x.csv', index=None)
   
    if is_predict:
        return x
    
    # 确定label数据， 以七天为一个周期
    
    label_close = sd.query('date == "%s" and type == "close"'%(label_start_date)).value
    if label_close is None or len(label_close) == 0:
        return None
    label_close = label_close.values[0]
    end_close = sd.query('date == "%s" and type == "close"'%(label_end_date)).value
    if end_close is None or len(end_close) == 0:
        return None
    end_close = end_close.values[0]
    y = (end_close - label_close)/label_close
    print y
    return x, y

# 到此为止， 获取一条训练数据的操作已经结束了
# 在整个过程中， 随着目标时间的推移， 训练数据逐步增多
# 那么， 让我们把上面的过程，包装成一个函数吧

ss = StandardScaler()
label_start_date = '2015-07-15'
train_delta = -900
id = '600588'
sd = get_sd(id, label_start_date)
date_list = dt_tool.dt_range(label_start_date, train_delta)
rs = get_trainset(sd, date_list[0])
rx, y = rs
rx.to_csv('data/rx_1.csv', index=None)
ry = [y]
for d in date_list:
    rs = get_trainset(sd, d)
    if rs is None:
        continue    
    x, y = rs
    rx = rx.append(x)
    ry.append(y)
rx.fillna(0, inplace=True)
rx.to_csv('data/rx.csv', index=None)

train_x = np.asarray(rx)
train_y = np.asarray(ry).ravel()
np.savetxt('data/ry.csv', train_y)
train_scale_x = ss.fit_transform(train_x)
clf = GradientBoostingRegressor()
clf.fit(train_scale_x, train_y)
scores = cross_val_score(clf, train_scale_x, train_y, scoring='mean_squared_error', cv=10)
print clf, math.sqrt(-scores.mean())

## 做了这么多， 我们终于拿到了一个
## 骑士特征还算丰富的模型
## 欧巴， 我们开始预测吧， 希望不会又是然并卵
## 可不可以， rmse降到0.03以下
fx = get_trainset(sd, '2015-07-31', is_predict=True)
tmpx = rx.append(fx)
cfx = tmpx.tail(1)
cfx.fillna(0, inplace=True)
pred_x = np.asarray(cfx)
pred_scale_x = ss.fit_transform(pred_x)
pred = clf.predict(pred_scale_x)
print pred


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



