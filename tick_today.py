import tushare
import sys

code = sys.argv[1]
s = tushare.get_today_ticks(code)
s = s.sort('amount')
s.to_csv('%s.csv'%code, encoding='utf-8')
