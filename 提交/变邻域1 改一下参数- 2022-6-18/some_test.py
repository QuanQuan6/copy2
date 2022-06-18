
from read_data import read_data
from population import population
import numpy as np
from auxiliary import *


data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk07.fjs'
data_path = 'data.txt'
data = read_data(data_path)
peoples = population(data, 200)
max_step = 0
max_score = 0
total_step = 0
count = 0
for i in range(100):
    peoples.initial()
    peoples.GA( max_step=100,max_no_new_best=30,
               select_type='tournament', tournament_M=3,
               find_type='auto', V_C_ratio=0.2,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='not', mutation_OS_type='random',
               VNS_type='not', VNS_ratio=0.1)
    max_step = max(peoples.best_step, max_step)
    total_step += peoples.best_step
    max_score = max(peoples.best_score, max_score)
    if peoples.best_score == 51:
        count += 1

print(max_score)
print(max_step)
print(total_step/100)
print(count)


# print(peoples.best_score)
# print(peoples.best_step)
# peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS)
