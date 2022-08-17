from read_data import read_data
from population import population
import time
import numba


begin_time = time.time()
data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk06.fjs'
data_path = 'data.txt'
data = read_data(data_path)
# data.display_info(True)


peoples = population(data)

peoples.initial(500)
peoples.GA(max_step=20, max_no_new_best=100,
            select_type='tournament', tournament_M=3,
            memory_size = 0,
            find_type='auto', V_C_ratio=1,
            crossover_MS_type='not', crossover_OS_type='POX',
            mutation_MS_type='not', mutation_OS_type='random',
            VNS_='not',VNS_type='two', VNS_ratio=0.05)
# print('运行时间: ', time.time()-begin_time, '秒')
# print('求解最短时间: ', peoples.best_score, '秒')

# print('收敛次数: ', peoples.best_step)
peoples.print(peoples.best_MS, peoples.best_OS)
#peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
