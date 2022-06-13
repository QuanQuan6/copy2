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

data = read_data(data_path)
# data.display_info(True)


codes = population(data, 60, 30, 10)
jobs_num = codes.data.jobs_num
machines_num = codes.data.machines_num
jobs_operations = codes.data.jobs_operations
max_operations = jobs_operations.max()
jobs_operations_detail = codes.data.jobs_operations_detail

candidate_machine = codes.data.candidate_machine.copy()
candidate_machine_index = codes.data.candidate_machine_index


selected_machine = np.empty(
    shape=(jobs_num, max_operations), dtype=np.int32)
selected_machine_time = np.empty(
    shape=(jobs_num, max_operations), dtype=np.int32)
begin_time = np.empty(
    shape=(machines_num, jobs_num, max_operations), dtype=np.int32)
end_time = np.empty(
    shape=(machines_num, jobs_num, max_operations), dtype=np.int32)
jobs_operation = np.empty(shape=(1, jobs_num), dtype=np.int32).flatten()
machine_operationed = np.empty(
    shape=(1, machines_num), dtype=np.int32).flatten()
begin_time_lists = np.empty(
    shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
end_time_lists = np.empty(
    shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
machine_operations = np.empty(
    shape=(1, machines_num), dtype=np.int32).flatten()

list = np.ones(shape=(1, 10), dtype=np.int32).flatten()
list.fill(-1)
rear = 0

codes.GA()
begin_time_ = time.time()
for i in range(100):
    decode_one(codes.best_MS, codes.best_OS, jobs_operations,
               jobs_operations_detail, begin_time, end_time,
               selected_machine_time, selected_machine, jobs_operation,
               machine_operationed, begin_time_lists,  end_time_lists,
               machine_operations, candidate_machine)


end_time_ = time.time()
print("运行了{}秒".format(end_time_-begin_time_))


# encode.print(encode.best_MS, encode.best_OS)
codes.show_gantt_chart(codes.best_MS, codes.best_OS, figsize=(14, 4))
