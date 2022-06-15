
from read_data import read_data
from population import population


data_path = 'data.txt'
data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk07.fjs'
data = read_data(data_path)
# data.display_info(True)


peoples = population(data, 200)
peoples.initial()
peoples.GA(max_step=100)
print(peoples.best_step)
print(peoples.best_score)
peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))



    
