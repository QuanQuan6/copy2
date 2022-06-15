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
    codes = population(data, 60, 30, 10)
    codes_list.append(codes)
    index = 'Mk0{}'.format(i+1)
    index_list.append(index)

columns_index = []
data = pd.DataFrame(data=None, index=index_list, columns=columns_index)

com_times = 50
max_step = 50
max_no_new_best = 100
crossover_P = 1
uniform_P = 1
tournament_M = 2
# UPNR
for i in range(9):
    total_time = 0
    total_score = 0
    total_step = 0
    for _ in range(com_times):
        print('{}-{}'.format(i,_))
        begin_time = time.time()
        codes_list[i].initial()
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament', tournament_M=tournament_M,
                         crossover_P=crossover_P, uniform_P=uniform_P,
                         crossover_MS_type='uniform', crossover_OS_type='POX',
                         mutation_MS_type='not', mutation_OS_type='random')
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
    data.loc['Mk0{}'.format(i+1), 'UPNR'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'UPNRT'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'UPNRS'] = total_step/com_times

# UPNN
for i in range(9):
    total_time = 0
    total_score = 0
    total_step = 0
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial()
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament', tournament_M=tournament_M,
                         crossover_P=crossover_P, uniform_P=uniform_P,
                         crossover_MS_type='uniform', crossover_OS_type='POX',
                         mutation_MS_type='not', mutation_OS_type='not')
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
    data.loc['Mk0{}'.format(i+1), 'UPNN'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'UPNNT'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'UPNNS'] = total_step/com_times

# UPBN
for i in range(9):
    total_time = 0
    total_score = 0
    total_step = 0
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial()
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament', tournament_M=tournament_M,
                         crossover_P=crossover_P, uniform_P=uniform_P,
                         crossover_MS_type='uniform', crossover_OS_type='POX',
                         mutation_MS_type='best', mutation_OS_type='not')
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
    data.loc['Mk0{}'.format(i+1), 'UPBN'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'UPNBT'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'UPNBS'] = total_step/com_times

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
                         crossover_P=crossover_P, uniform_P=uniform_P,
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
