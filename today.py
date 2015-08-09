# -*- coding: utf-8 -*-
import tushare as ts
import datetime as dt
import pandas as pd
import sys

for id in sys.stdin:
    id = id.strip()
    try:
        today = ts.get_today_ticks(id).sort('amount', ascending=False)
    except:
        continue
    if today.ix[0].amount > 20000000:
        today['id'] = id
        today.head(10).to_csv('./money.csv', mode='a', index=False, encoding='utf-8')
    
