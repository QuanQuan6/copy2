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
    codes = population(data, 100)
    codes_list.append(codes)
    index = 'Mk0{}'.format(i+1)
    index_list.append(index)

columns_index = []
data = pd.DataFrame(data=None, index=index_list, columns=columns_index)

com_times = 30
max_step = 100
max_no_new_best = 200
tournament_M = 3

for i in range(9):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial()
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament', tournament_M=tournament_M,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type='uniform', crossover_OS_type='POX',
                         mutation_MS_type='best', mutation_OS_type='random',
                         VNS_type='not', VNS_ratio=0.2)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'UPBRM'] = min_score
    data.loc['Mk0{}'.format(i+1), 'UPBR'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRT'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRS'] = total_step/com_times

for i in range(9):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial()
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament', tournament_M=tournament_M,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type='uniform', crossover_OS_type='POX',
                         mutation_MS_type='best', mutation_OS_type='random',
                         VNS_type='one', VNS_ratio=0.2)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'UPBRM_one'] = min_score
    data.loc['Mk0{}'.format(i+1), 'UPBR_one'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRT_one'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRS_one'] = total_step/com_times

for i in range(9):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial()
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament', tournament_M=tournament_M,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type='uniform', crossover_OS_type='POX',
                         mutation_MS_type='best', mutation_OS_type='random',
                         VNS_type='two', VNS_ratio=0.2)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'UPBRM_two'] = min_score
    data.loc['Mk0{}'.format(i+1), 'UPBR_two'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRT_two'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRS_two'] = total_step/com_times

for i in range(9):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial()
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament', tournament_M=tournament_M,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type='uniform', crossover_OS_type='POX',
                         mutation_MS_type='best', mutation_OS_type='random',
                         VNS_type='both', VNS_ratio=0.2)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'UPBRM_both'] = min_score
    data.loc['Mk0{}'.format(i+1), 'UPBR_both'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRT_both'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRS_both'] = total_step/com_times

data.to_csv('测试图片\one\data4.csv')
