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
codes_list = []
index_list = []
for i in range(9):
    data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk0{}.fjs'.format(
        i+1)
    data = read_data(data_path)
    codes = population(data, 200)
    codes_list.append(codes)
    index = 'Mk0{}'.format(i+1)
    index_list.append(index)

columns_index = []
data = pd.DataFrame(data=None, index=index_list, columns=columns_index)

com_times = 50
max_step = 200
max_no_new_best = 50
tournament_M = 3
# UPNR
for i in range(9):
    total_time = 0
    total_score = 0
    total_step = 0
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial()
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament', tournament_M=3,
                         find_type='max', V_C_ratio=0.2,
                         crossover_MS_type='uniform', crossover_OS_type='POX',
                         mutation_MS_type='best', mutation_OS_type='random')
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
    data.loc['Mk0{}'.format(i+1), 'UPBR_M'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRT_M'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRS_M'] = total_step/com_times

# UPBR
for i in range(9):
    total_time = 0
    total_score = 0
    total_step = 0
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial()
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament', tournament_M=tournament_M,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type='uniform', crossover_OS_type='POX',
                         mutation_MS_type='best', mutation_OS_type='random')
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
    data.loc['Mk0{}'.format(i+1), 'UPBR'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRT'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRS'] = total_step/com_times
print(data)
