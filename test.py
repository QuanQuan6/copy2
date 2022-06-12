import code
from read_data import read_data
from population import *
import time
import numpy as np
import os
import psutil
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


codes = population(data, 2)
print(codes.OS)
random_crossover(codes.OS, 0.1)
print(codes.OS)
begin_time = time.time()
for i in range(1000):
    pass


end_time = time.time()
print("运行了{}秒".format(end_time-begin_time))


#encode.print(encode.best_MS, encode.best_OS)
#peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
