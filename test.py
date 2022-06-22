from soupsieve import select
from read_data import read_data
from population import *
import time
import numpy as np
import os
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
# for i in range(9):
#     data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk0{}.fjs'.format(
#         i+1)
#     data = read_data(data_path)
#     codes = population(data, 200)
#     codes_list.append(codes)
#     index = 'Mk0{}'.format(i+1)
#     index_list.append(index)
data_path = 'data.txt'
data = read_data(data_path)
codes = population(data, 500)
codes_list.append(codes)
index = 'data'
index_list.append(index)
columns_index = []
data = pd.DataFrame(data=None, index=index_list, columns=columns_index)

com_times = 500
max_step = 50
max_no_new_best = 200
tournament_M = 3
memory_size = 0.1
select_type = 'tournament'
crossover_MS_type = 'not'
mutation_MS_type = 'not'
VNS_type = 'two'
file_path = '测试图片\\test\data_比较VNS_方式的影响_500_50_3_0.1.csv'

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial()
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type=select_type, tournament_M=tournament_M,
                         memory_size=memory_size,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='not', VNS_type=VNS_type, VNS_ratio=1)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'UPBRM'] = min_score
    data.loc['Mk0{}'.format(i+1), 'UPBR'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRT'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRS'] = total_step/com_times

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial()
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type=select_type, tournament_M=tournament_M,
                         memory_size=memory_size,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='normal', VNS_type=VNS_type, VNS_ratio=1)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'UPBRM_normal'] = min_score
    data.loc['Mk0{}'.format(i+1), 'UPBR_normal'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRT_normal'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRS_normal'] = total_step/com_times

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial()
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type=select_type, tournament_M=tournament_M,
                         memory_size=memory_size,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='quick', VNS_type=VNS_type, VNS_ratio=1)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'UPBRM_quick'] = min_score
    data.loc['Mk0{}'.format(i+1), 'UPBR_quick'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRT_quick'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'UPBRS_quick'] = total_step/com_times


data.to_csv(file_path)
