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

peoples = population(data, 5)
jobs_operations = peoples.data.jobs_operations
jobs_num = peoples.data.jobs_num
machines_num = peoples.data.machines_num
max_operations = jobs_operations.max()
jobs_order = np.arange(jobs_num)
MS_positions = np.empty(
    shape=(jobs_num, max_operations), dtype=int)
jobs_operations_detail = peoples.data.jobs_operations_detail
jobs_operations_detail = initial_jobs_operations_detail(jobs_operations_detail)
candidate_machine = peoples.data.candidate_machine
MS = np.empty(shape=(peoples.size, peoples.length), dtype=int)
candidate_machine_index = peoples.data.candidate_machine_index
global_first_index = 0
global_rear_index = peoples.global_size
machine_time = np.zeros(
    shape=(1, machines_num), dtype=int).flatten()
print(MS)
initial_MS_global_selection(MS, global_first_index, global_rear_index, MS_positions, jobs_order,
                            jobs_operations, jobs_operations_detail, machine_time)
print(MS)
begin_time = time.time()
for i in range(100000):
    initial_MS_position(MS_positions, jobs_operations)
end_time = time.time()
print("运行了{}秒".format(end_time-begin_time))
print(MS_positions.shape[0])


#encode.print(encode.best_MS, encode.best_OS)
#peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
