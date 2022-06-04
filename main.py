from data_read_copy import read_data
from FJSP_Data import FJSP_Data
from Gene import Gene

data_path = 'Monaldo\Fjsp\Job_Data\Barnes\Text\mt10c1.txt'

data = read_data(data_path)
#data.display_info(True)
encode = Gene(data)
encode.show()
