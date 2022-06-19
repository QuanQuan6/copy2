import time
from turtle import pencolor
from read_data import read_data
from population import population
import numpy as np
from auxiliary import *


data_path = 'data.txt'
data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk06.fjs'
data = read_data(data_path)
peoples = population(data,100)
begin_time = time.time()
peoples.initial()
peoples.GA(max_step=200, max_no_new_best=100,
           select_type='tournament', tournament_M=3,
           find_type='auto', crossover=0.8, V_C_ratio=0.2,
           crossover_MS_type='uniform', crossover_OS_type='POX',
           mutation_MS_type='best', mutation_OS_type='random',
           VNS_type='both', VNS_ratio=0.2)
print(time.time()-begin_time)
print(peoples.best_score)
print(peoples.best_step)
peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
# max_step = 0
# max_score = 0
# total_step = 0
# total_time = 0
# count = 0
# for i in range(100):
#     begin_time = time.time()
#     peoples.initial()
#     peoples.GA(max_step=200, max_no_new_best=200,
#                select_type='tournament', tournament_M=3,
#                find_type='auto', V_C_ratio=0.2,
#                crossover_MS_type='uniform', crossover_OS_type='POX',
#                mutation_MS_type='not', mutation_OS_type='random',
#                VNS_type='not', VNS_ratio=0.2)
#     total_time += time.time()-begin_time
#     max_score = max(max_score, peoples.best_score)
#     max_step = max(max_step, peoples.best_step)
#     total_step += peoples.best_step
#     if peoples.best_score == 51:
#         count += 1
# print(count)
# print(max_score)
# print(max_step)
# print(total_step/100)
# print(total_time/100)
# print()
# for i in range(100):
#     begin_time = time.time()
#     peoples.initial()
#     peoples.GA(max_step=200, max_no_new_best=200,
#                select_type='tournament', tournament_M=3,
#                find_type='auto', V_C_ratio=0.2,
#                crossover_MS_type='uniform', crossover_OS_type='POX',
#                mutation_MS_type='not', mutation_OS_type='random',
#                VNS_type='two', VNS_ratio=0.2)
#     total_time += time.time()-begin_time
#     max_score = max(max_score, peoples.best_score)
#     max_step = max(max_step, peoples.best_step)
#     total_step += peoples.best_step
#     if peoples.best_score == 51:
#         count += 1
# print(count)
# print(max_score)
# print(max_step)
# print(total_step/100)
# print(total_time/100)
