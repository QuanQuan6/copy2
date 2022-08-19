from read_data import read_data
from population import population
import time
import numba


begin_time = time.time()
data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk06.fjs'
data_path = 'data_final.txt'
data = read_data(data_path)
# data.display_info(True)
count = 1
score = []
step = []
time_list = []
peoples = population(data)

s = 0
s_t = 0
begin_time = time.time()
for i in range(count):
    peoples.initial(500)
    peoples.GA(max_step=200, max_no_new_best=5,
               select_type='tournament_memory', tournament_M=3,
               memory_size=0.1,
               find_type='auto', V_C_ratio=0.2,
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
# peoples.print(peoples.best_MS, peoples.best_OS)
# print('运行时间: ', time.time()-begin_time, '秒')
# print('求解最短时间: ', peoples.best_score, '秒')

# print('收敛次数: ', peoples.best_step)
# peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
