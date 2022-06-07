from colorama import init
from read_data import read_data
from FJSP_Data import FJSP_Data
from population import population
import time
import numpy as np
data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk09.fjs'

data = read_data(data_path)
#data.display_info(True)
begin_time = time.time()
encode = population(data, 2000)
print(time.time()-begin_time)
