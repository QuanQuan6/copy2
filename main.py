from read_data import read_data
from population import population
import time
import numba



data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk06.fjs'
data_path = 'data.txt'
data = read_data(data_path)
# data.display_info(True)


peoples = population(data)
begin_time = time.time()
peoples.initial(100)
peoples.GA(max_step=100, max_no_new_best=30,
           select_type='tournament_memory', tournament_M=3,
           memory_size = 0.05,
           find_type='auto', V_C_ratio=0.2,
           crossover_MS_type='not', crossover_OS_type='POX',
           mutation_MS_type='not', mutation_OS_type='random',
           VNS_='normal',VNS_type='two', VNS_ratio=0.05)
print('运行时间: ', time.time()-begin_time, '秒')
print('求解最短时间: ', peoples.best_score, '秒')
print('收敛次数: ', peoples.best_step)
#peoples.print(peoples.best_MS, peoples.best_OS)
#peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
