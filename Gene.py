
import numpy as np
from FJSP_Data import FJSP_Data


class Gene:
    '''
    编码
    '''
    MS = None
    OS = None
    length = None
    data = None

    def __init__(self, data):
        self.data = data
        self.length = data.jobs_operations.sum()
        self.MS = np.zeros(shape=(1, self.length))
        self.OS = np.zeros(shape=(1, self.length))

    def global_selection(self):
        machine_time = np.zeros(shape=(1, self.length))

    def show(self):
        print(self.MS)
        print(self.OS)
