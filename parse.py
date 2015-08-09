import pandas as pd
s = pd.read_csv('pred_min', sep=' ', error_bad_lines=False)
s.columns = ['code', 'rmse', 'pred']
s.rmse = s.rmse.astype(float).abs()
s.pred = s.pred.astype(float)
s['delta'] = s.pred - s.rmse
print s.sort('delta')
