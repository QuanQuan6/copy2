from read_data import read_data
from population import population
import time
'''
1:自适应初始化 未完成
2:自适应交叉概率和变异概率
2:新种群选择策略  
    1)锦标赛
'''
begin_time = time.time()
data_path = 'data.txt'

data = read_data(data_path)
# data.display_info(True)
s = 0
for i in range (100):
    peoples = population(data, 60, 30, 10)
    best = peoples.GA(max_step=100,max_no_new_best=100)
    s += peoples.best_score
print('平均运行时间: ', (time.time()-begin_time)/100, '秒')
print('平均结果： ', s/100, '秒')
peoples.print(peoples.best_MS,peoples.best_OS)
# peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
