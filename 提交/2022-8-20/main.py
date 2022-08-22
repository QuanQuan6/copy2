from read_data import read_data
from population import population
import time
import numba
import sys

def main(input, output):
    begin_time = time.time()

    data_path = 'data_final.txt'
    data_path = input
    data = read_data(data_path)
    # data.display_info(True)

    peoples = population(data)

    begin_time = time.time()
    peoples.initial(500)
    peoples.GA(max_step=100, max_no_new_best=20,
               select_type='tournament_memory', tournament_M=4,
               memory_size=0.2,
               find_type='auto', V_C_ratio=0.15,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='random', mutation_OS_type='random',
               VNS_='normal', VNS_type='both', VNS_ratio=0.03)

    final = peoples.print(peoples.best_MS, peoples.best_OS)
    final.append('运行时间: '+str(time.time()-begin_time)+'秒'+'\n')
    final.append('求解时间: '+str(peoples.best_score)+'秒'+'\n')
    final.append('收敛次数: '+str(peoples.best_step))
    with open(output, 'w', encoding='utf-8') as f:
        f.writelines(final)
    for line in final:
        print(line)


#peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
if __name__ == "__main__":
    # main('data_final.txt', 'output.txt')
    main(sys.argv[1], sys.argv[2])
