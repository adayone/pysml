# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.style.use('ggplot')

# <codecell>

id = '300220'
df = ts.get_h_data(id, autype='qfq', start='2013-06-10')

# <codecell>

df = pd.DataFrame(df.query('date > "2015-07-01"'))

# <codecell>

df['m1'] = pd.rolling_mean(df['close'], window=15, min_periods=1, center=True)
df['m2'] = pd.rolling_mean(df['close'], window=30, min_periods=1, center=True)
df['m4'] = pd.rolling_mean(df['close'], window=45, min_periods=1, center=True)

# <codecell>

print df.plot(y=['close', 'm1', 'm2', 'm4', 'volume'], title=id,  secondary_y='volume', grid=True, legend=True, figsize=(16, 10))

# <codecell>

df

# <codecell>


