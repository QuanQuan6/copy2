from data_read import read_data, FJSP_Data
import geatpy as ga

data_path = 'Monaldo\Fjsp\Job_Data\Barnes\Text\mt10c1.txt'

data = read_data(data_path)
data.display_info()
