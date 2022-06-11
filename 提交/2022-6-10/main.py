from colorama import init
from read_data import read_data
from FJSP_Data import FJSP_Data
from population import population
import time
import numpy as np



if __name__ == '__main__':
    data_path = 'data.txt'
    data = read_data(data_path)
    # data.display_info(True)
    begin_time = time.time()
    encode = population(data, 1000, 1000, 1000)
    best = encode.GA()
    encode.show_gantt_chart(encode.MS[best], encode.OS[best])
    encode.print(encode.MS[best], encode.OS[best])
