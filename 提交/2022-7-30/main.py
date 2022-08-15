from read_data import read_data
from population import population
import time
'''
1:自适应初始化 未完成
2:自适应交叉概率和变异概率
2:新种群选择策略  
    1)锦标赛
'''
begin_time = time.time()
data_path = 'data.txt'

data = read_data(data_path)
# data.display_info(True)


peoples = population(data, 500)
peoples.GA(max_step=20, max_no_new_best=10, tournament_M=3, memory_size=0.02)

peoples.print(peoples.best_OS)
print(time.time()-begin_time)
#peoples.show_gantt_chart(peoples.best_OS, figsize=(14, 4))
