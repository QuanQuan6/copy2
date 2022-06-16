from read_data import read_data
from population import population
import time
import numba
'''
2:自适应交叉概率和变异概率  
3.邻域搜索变异
4.精英策略
5.改进锦标赛
6.变邻域搜索以及自适应
'''


data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk06.fjs'
data_path = 'data.txt'
data = read_data(data_path)
# data.display_info(True)


peoples = population(data, 200)
begin_time = time.time()
peoples.initial()
peoples.GA(max_step=200, max_no_new_best=50,
           select_type='tournament', tournament_M=3,
            find_type='auto', V_C_ratio=0.2,
           crossover_MS_type='uniform', crossover_OS_type='POX',
           mutation_MS_type='best', mutation_OS_type='random')
print('运行时间: ', time.time()-begin_time, '秒')
print('求解最短时间: ', peoples.best_score, '秒')
print('收敛次数: ', peoples.best_step)
peoples.print(peoples.best_MS, peoples.best_OS)
# peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
