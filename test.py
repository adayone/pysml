import tushare as ts
date = '2015-06-01'
tick = ts.get_tick_data('000002', date)
ft = tick.sort('amount', ascending=False).head(10).reset_index().drop(['index', 'time', 'change'], 1).stack().reset_index()
ft['fea'] = ft.level_0.map(str) + '_' + ft.level_1
fea_pd = ft.drop(['level_0', 'level_1'], 1)
fea_pd.index = fea_pd.fea
fT = fea_pd.T
