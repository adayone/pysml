# -*- coding: utf-8 -*-
from sklearn.svm import SVR
import pandas
import sys
import math
import numpy  as np
import tushare as ts
import datetime as dt
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Ridge 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cross_validation import train_test_split,cross_val_score
import dt_tool
from model_tool import *

pd.options.mode.chained_assignment = None  # default='warn'
pred_date = '2015-08-10'

rxy = pd.read_csv('data/rxy.csv')
sd = pd.read_csv('data/sd.csv')
ss = StandardScaler()
train_x = np.asarray(rxy.drop('label', 1))
train_y = np.asarray(rxy.label).ravel()
train_scale_x = ss.fit_transform(train_x)
#clf = GradientBoostingRegressor()
#clf = SVR(kernel='poly', C=1e3, degree=2)
#clf = LR(C=10.0, penalty='l1')
#clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
clf = GradientBoostingClassifier(n_estimators=200)
clf.fit(train_scale_x, train_y)
#scores = cross_val_score(clf, train_scale_x, train_y, scoring='mean_squared_error', cv=3)
score = cross_val_score(clf, train_scale_x, train_y, scoring='roc_auc', cv=3)
#score = math.sqrt(-scores.mean())
## 做了这么多， 我们终于拿到了一个
## 骑士特征还算丰富的模型
## 欧巴， 我们开始预测吧， 希望不会又是然并卵
## 可不可以， rmse降到0.03以下
fx = get_trainset(sd, pred_date, is_predict=True)
rxy = rxy.drop('label', 1)
tmpx = rxy.append(fx)
cfx = tmpx.tail(1)
cfx.fillna(0, inplace=True)
cfx.to_csv('data/pred_x.csv', index=None)
cfx = pd.read_csv('data/pred_x.csv')
pred_x = np.asarray(cfx)
pred_scale_x = ss.fit_transform(pred_x)
pred = clf.predict_proba(pred_scale_x)
print  id, score, pred




