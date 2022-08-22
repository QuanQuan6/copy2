
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
for i in range(9):
    data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk0{}.fjs'.format(
        i+1)
    data = read_data(data_path)
    codes = population(data)
    codes_list.append(codes)
    index = 'Mk0{}'.format(i+1)
    index_list.append(index)
data_path = 'data.txt'
data = read_data(data_path)
codes = population(data)
codes_list.append(codes)
index = 'data'
index_list.append(index)
columns_index = []
data = pd.DataFrame(data=None, index=index_list, columns=columns_index)

com_times = 20
max_step = 50
max_no_new_best = 20
tournament_M = 4
memory_size = 0.1
select_type = 'tournament'
crossover_MS_type = 'uniform'
mutation_MS_type = 'random'
VNS_type = 'two'
file_path = '测试图片\\test\测试.csv'

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial(50)
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament', tournament_M=tournament_M,
                         memory_size=0,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='not', VNS_type=VNS_type, VNS_ratio=1)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'tournament_min'] = min_score
    data.loc['Mk0{}'.format(i+1), 'tournament_av'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'tournament_time'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'tournament_step'] = total_step/com_times

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial(50)
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament_memory', tournament_M=tournament_M,
                         memory_size=0.2,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='not', VNS_type=VNS_type, VNS_ratio=1)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'tournament_memory_min'] = min_score
    data.loc['Mk0{}'.format(i+1), 'tournament_memory_av'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'tournament_memory_time'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'tournament_memory_step'] = total_step/com_times

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial(50)
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament_random', tournament_M=tournament_M,
                         memory_size=0,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='not', VNS_type=VNS_type, VNS_ratio=1)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'tournament_random_min'] = min_score
    data.loc['Mk0{}'.format(i+1), 'tournament_random_av'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'tournament_random_time'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'tournament_random_step'] = total_step/com_times

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial(50)
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament_memory', tournament_M=tournament_M,
                         memory_size=0.2,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='normal', VNS_type='one', VNS_ratio=1)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'one_min'] = min_score
    data.loc['Mk0{}'.format(i+1), 'one_av'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'one_time'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'one_step'] = total_step/com_times

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial(50)
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament_memory', tournament_M=tournament_M,
                         memory_size=0.2,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='normal', VNS_type='two', VNS_ratio=1)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'two_min'] = min_score
    data.loc['Mk0{}'.format(i+1), 'two_av'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'two_time'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'two_step'] = total_step/com_times

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial(50)
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament_memory', tournament_M=tournament_M,
                         memory_size=0.2,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='normal', VNS_type='both', VNS_ratio=1)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'both_min'] = min_score
    data.loc['Mk0{}'.format(i+1), 'both_av'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'both_time'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'both_step'] = total_step/com_times

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial(50)
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament_memory', tournament_M=tournament_M,
                         memory_size=0.2,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='normal', VNS_type='both', VNS_ratio=0.5)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), '0.5_min'] = min_score
    data.loc['Mk0{}'.format(i+1), '0.5_av'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), '0.5_time'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), '0.5_step'] = total_step/com_times

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial(50)
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament_memory', tournament_M=tournament_M,
                         memory_size=0.2,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='normal', VNS_type='both', VNS_ratio=0.1)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), '0.1_min'] = min_score
    data.loc['Mk0{}'.format(i+1), '0.1_av'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), '0.1_time'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), '0.1_step'] = total_step/com_times

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial(50)
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament_memory', tournament_M=tournament_M,
                         memory_size=0.2,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='quick', VNS_type='both', VNS_ratio=0.1)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'q0.1_min'] = min_score
    data.loc['Mk0{}'.format(i+1), 'q0.1_av'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'q0.1_time'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'q0.1_step'] = total_step/com_times

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial(50)
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament_memory', tournament_M=tournament_M,
                         memory_size=0.2,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='quick', VNS_type='both', VNS_ratio=0.5)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'q0.5_min'] = min_score
    data.loc['Mk0{}'.format(i+1), 'q0.5_av'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'q0.5_time'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'q0.5_step'] = total_step/com_times

for i in range(len(codes_list)):
    total_time = 0
    total_score = 0
    total_step = 0
    min_score = float('inf')
    for _ in range(com_times):
        begin_time = time.time()
        codes_list[i].initial(50)
        codes_list[i].GA(max_step=max_step, max_no_new_best=max_no_new_best,
                         select_type='tournament_memory', tournament_M=tournament_M,
                         memory_size=0.2,
                         find_type='auto', V_C_ratio=0.2,
                         crossover_MS_type=crossover_MS_type, crossover_OS_type='POX',
                         mutation_MS_type=mutation_MS_type, mutation_OS_type='random',
                         VNS_='quick', VNS_type='both', VNS_ratio=1)
        end_time = time.time()
        total_time += end_time-begin_time
        total_score += codes_list[i].best_score
        total_step += codes_list[i].best_step
        min_score = min(codes_list[i].best_score, min_score)
    data.loc['Mk0{}'.format(i+1), 'q_min'] = min_score
    data.loc['Mk0{}'.format(i+1), 'q_av'] = total_score/com_times
    data.loc['Mk0{}'.format(i+1), 'q_time'] = total_time/com_times
    data.loc['Mk0{}'.format(i+1), 'q_step'] = total_step/com_times

data.to_csv(file_path)


data_path = 'data_final.txt'
data = read_data(data_path)
# data.display_info(True)
count = 5
score = []
step = []
time_list = []
peoples = population(data)

s = 0
s_t = 0
begin_time = time.time()
for i in range(count):
    peoples.initial(500)
    peoples.GA(max_step=150, max_no_new_best=5,
               select_type='tournament_memory', tournament_M=4,
               memory_size=0.2,
               find_type='auto', V_C_ratio=0.1,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='random', mutation_OS_type='random',
               VNS_='not', VNS_type='both', VNS_ratio=1)
    s += peoples.best_score
    s_t += peoples.best_step
score.append(s)
step.append(s_t)
time_list.append(time.time()-begin_time)

s = 0
s_t = 0
begin_time = time.time()
for i in range(count):
    peoples.initial(300)
    peoples.GA(max_step=150, max_no_new_best=5,
               select_type='tournament_memory', tournament_M=4,
               memory_size=0.2,
               find_type='auto', V_C_ratio=0.1,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='random', mutation_OS_type='random',
               VNS_='not', VNS_type='both', VNS_ratio=1)
    s += peoples.best_score
    s_t += peoples.best_step
score.append(s)
step.append(s_t)
time_list.append(time.time()-begin_time)

s = 0
s_t = 0
begin_time = time.time()
for i in range(count):
    peoples.initial(1000)
    peoples.GA(max_step=150, max_no_new_best=5,
               select_type='tournament_memory', tournament_M=4,
               memory_size=0.2,
               find_type='auto', V_C_ratio=0.1,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='random', mutation_OS_type='random',
               VNS_='not', VNS_type='both', VNS_ratio=1)
    s += peoples.best_score
    s_t += peoples.best_step
score.append(s)
step.append(s_t)
time_list.append(time.time()-begin_time)

s = 0
s_t = 0
begin_time = time.time()
for i in range(count):
    peoples.initial(500)
    peoples.GA(max_step=150, max_no_new_best=5,
               select_type='tournament', tournament_M=4,
               memory_size=0.2,
               find_type='auto', V_C_ratio=0.1,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='random', mutation_OS_type='random',
               VNS_='not', VNS_type='both', VNS_ratio=1)
    s += peoples.best_score
    s_t += peoples.best_step
score.append(s)
step.append(s_t)
time_list.append(time.time()-begin_time)

s = 0
s_t = 0
begin_time = time.time()
for i in range(count):
    peoples.initial(300)
    peoples.GA(max_step=150, max_no_new_best=5,
               select_type='tournament', tournament_M=4,
               memory_size=0.2,
               find_type='auto', V_C_ratio=0.1,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='random', mutation_OS_type='random',
               VNS_='not', VNS_type='both', VNS_ratio=1)
    s += peoples.best_score
    s_t += peoples.best_step
score.append(s)
step.append(s_t)
time_list.append(time.time()-begin_time)

s = 0
s_t = 0
begin_time = time.time()
for i in range(count):
    peoples.initial(1000)
    peoples.GA(max_step=150, max_no_new_best=5,
               select_type='tournament', tournament_M=4,
               memory_size=0.2,
               find_type='auto', V_C_ratio=0.1,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='random', mutation_OS_type='random',
               VNS_='not', VNS_type='both', VNS_ratio=1)
    s += peoples.best_score
    s_t += peoples.best_step
score.append(s)
step.append(s_t)
time_list.append(time.time()-begin_time)

s = 0
s_t = 0
begin_time = time.time()
for i in range(count):
    peoples.initial(500)
    peoples.GA(max_step=150, max_no_new_best=5,
               select_type='tournament_memory', tournament_M=4,
               memory_size=0.2,
               find_type='auto', V_C_ratio=0.1,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='random', mutation_OS_type='random',
               VNS_='not', VNS_type='both', VNS_ratio=1)
    s += peoples.best_score
    s_t += peoples.best_step
score.append(s)
step.append(s_t)
time_list.append(time.time()-begin_time)

s = 0
s_t = 0
begin_time = time.time()
for i in range(count):
    peoples.initial(500)
    peoples.GA(max_step=150, max_no_new_best=5,
               select_type='tournament_memory', tournament_M=4,
               memory_size=0.2,
               find_type='auto', V_C_ratio=0.1,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='best', mutation_OS_type='random',
               VNS_='not', VNS_type='both', VNS_ratio=1)
    s += peoples.best_score
    s_t += peoples.best_step
score.append(s)
step.append(s_t)
time_list.append(time.time()-begin_time)

s = 0
s_t = 0
begin_time = time.time()
for i in range(count):
    peoples.initial(500)
    peoples.GA(max_step=150, max_no_new_best=5,
               select_type='tournament_memory', tournament_M=4,
               memory_size=0.2,
               find_type='auto', V_C_ratio=0.1,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='best', mutation_OS_type='random',
               VNS_='normal', VNS_type='both', VNS_ratio=0.1)
    s += peoples.best_score
    s_t += peoples.best_step
score.append(s)
step.append(s_t)
time_list.append(time.time()-begin_time)

s = 0
s_t = 0
begin_time = time.time()
for i in range(count):
    peoples.initial(500)
    peoples.GA(max_step=150, max_no_new_best=5,
               select_type='tournament_memory', tournament_M=4,
               memory_size=0.2,
               find_type='auto', V_C_ratio=0.1,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='random', mutation_OS_type='random',
               VNS_='not', VNS_type='both', VNS_ratio=0.1)
    s += peoples.best_score
    s_t += peoples.best_step
score.append(s)
step.append(s_t)
time_list.append(time.time()-begin_time)

print(score)
print(step)
print(time_list)