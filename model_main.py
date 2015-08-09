import sys
from model_tool import *
# 到此为止， 获取一条训练数据的操作已经结束了
# 在整个过程中， 随着目标时间的推移， 训练数据逐步增多
# 那么， 让我们把上面的过程，包装成一个函数吧
id = sys.argv[1]
train_start_date = '2015-07-15'
sd_start_date = '2015-08-09'
pred_date = '2015-08-10'
train_delta = -900
sd = get_sd(id, sd_start_date)
date_list = dt_tool.dt_range(train_start_date, train_delta)
rxy = get_trainset(sd, date_list[0])
rxy.to_csv('data/rxy_1.csv', index=None)
for d in date_list:
    rs = get_trainset(sd, d)
    if rs is None:
        continue    
    rxy = rxy.append(rs)
rxy.fillna(0, inplace=True)
rxy.to_csv('data/rxy.csv', index=None)


