from read_data import read_data
from population import population
import time
import numba


begin_time = time.time()
data_path = 'data.txt'
data = read_data(data_path)
# data.display_info(True)
s = 0
for i in range(100):
    peoples = population(data, 500)
    peoples.initial()
    peoples.GA(max_step=50, max_no_new_best=30,
               select_type='tournament', tournament_M=3,
               find_type='auto', V_C_ratio=0.2,
               crossover_MS_type='not', crossover_OS_type='POX',
               mutation_MS_type='not', mutation_OS_type='random',
               VNS_='normal', VNS_type='two', VNS_ratio=1)
    peoples.check()
    s += peoples.best_score
print('平均运行时间: ', (time.time()-begin_time)/100, '秒')
print('平均结果： ', s/100, '秒')
print('求解最短时间: ', peoples.best_score, '秒')
# print('收敛次数: ', peoples.best_step)
# peoples.print(peoples.best_MS, peoples.best_OS)
# peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
