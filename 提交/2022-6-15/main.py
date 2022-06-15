from read_data import read_data
from population import population
from population import uniform_crossover
import time
import numba
'''
1:自适应初始化 未完成
2:自适应交叉概率和变异概率   改锦标
2:新种群选择策略  
    1)锦标赛
3:交叉方法
    1) 均匀交叉
    2) 工序码
'''

data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk09.fjs'
data_path = 'data.txt'


data = read_data(data_path)
# data.display_info(True)


peoples = population(data, 60, 30, 10)
begin_time = time.time()
peoples.initial()
peoples.GA(max_step=80, max_no_new_best=50,
           select_type='tournament', tournament_M=3,
           crossover_P=1, uniform_P=1,
           crossover_MS_type='uniform', crossover_OS_type='POX',
           mutation_MS_type='best', mutation_OS_type='random')
print('运行时间: ',time.time()-begin_time,'秒')
print('求解最短时间: ',peoples.best_score,'秒')
print('收敛次数: ',peoples.best_step)
#encode.print(encode.best_MS, encode.best_OS)
peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
