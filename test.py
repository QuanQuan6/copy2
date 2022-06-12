from read_data import read_data
from population import population, uniform_crossover_random
from population import uniform_crossover
import time
import numba
import numpy as np
'''
1:自适应初始化 未完成
2:自适应交叉概率和变异概率   改锦标
2:新种群选择策略  
    1)锦标赛
3:交叉方法
    1) 均匀交叉
    2) 工序码
'''
data_path = 'data.txt'
data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk01.fjs'

data = read_data(data_path)
# data.display_info(True)

peoples = population(data, 60, 30, 10)
begin_time = time.time()
for i in range(2000):
    uniform_crossover(peoples.MS, 1, 1)
print(time.time()-begin_time)
print(peoples.best_score)
begin_time = time.time()
for i in range(2000):
    positions = np.array([range(peoples.length)]).flatten()
    uniform_crossover_random(peoples.MS, 1, 1,positions)
print(time.time()-begin_time)
#encode.print(encode.best_MS, encode.best_OS)
#peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
