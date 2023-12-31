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
    encode = population(data, 6000, 3000, 3000)
    best = encode.GA()
    print('运行时间: ', time.time()-begin_time, '秒')
    encode.show_gantt_chart(encode.MS[best], encode.OS[best])
    encode.print(encode.MS[best], encode.OS[best])
