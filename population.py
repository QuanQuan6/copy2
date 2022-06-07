
import numpy as np
from FJSP_Data import FJSP_Data
import random


class population:
    '''
    种群
    '''
    MS = None
    '''
    机器码
    '''
    OS = None
    '''
    工序码
    '''
    length = None
    '''
    长度
    '''
    data = None
    '''
    数据
    '''
    size = None
    '''
    种群规模
    '''
    global_size = None
    '''
    全局方式生成的规模
    '''
    local_size = None
    '''
    局部生成的规模
    '''
    random_size = None
    '''
    全局随机生成的规模
    '''

    def __init__(self, data, global_size=0, local_size=0, random_size=0):
        self.data = data
        self.global_size = global_size
        self.local_size = local_size
        self.random_size = random_size
        self.size = global_size+local_size+random_size
        self.length = data.jobs_operations.sum()
        self.__initial_OS()
        self.__initial_MS()

    def __initial_MS_global_selection(self):
        '''
        全局选择MS
        初始化慢
        '''
        jobs_operations_detail = self.data.jobs_operations_detail
        jobs_operations_detail = np.where(
            jobs_operations_detail == 0, 1000000000, jobs_operations_detail)
        people_index = 0
        machine_time = np.zeros(
            shape=(1, self.data.machines_num), dtype=int).flatten()
        jobs_order = np.arange(self.data.jobs_num)
        for people in range(self.global_size):
            np.random.shuffle(jobs_order)
            MS_position = 0
            for i in range(self.data.jobs_num):
                job_num = jobs_order[i]
                candidate_machine_j = self.data.candidate_machine[job_num]
                candidate_machine_time_j = self.data.candidate_machine_time[job_num]
                candidate_machine_index = self.data.candidate_machine_index[job_num]
                for operation_num in range(self.data.jobs_operations[job_num]):
                    rear_index = operation_num+1
                    candidate_machine_o = candidate_machine_j[
                        candidate_machine_index[operation_num]:candidate_machine_index[rear_index]]
                    candidate_machine_time_o = jobs_operations_detail[job_num][operation_num]
                    tem = machine_time + candidate_machine_time_o
                    selected_machine = np.argmin(tem)
                    MS_code = np.where(candidate_machine_o ==
                                       selected_machine)[0][0]
                    # index = np.ix_(candidate_machine_o.tolist())
                    # tem_machine_time = machine_time[index].copy()
                    # tem_machine_time = tem_machine_time + candidate_machine_time_o
                    # MS_code = np.argmin(tem_machine_time)
                    #selected_machine = candidate_machine_o[MS_code]
                    operation_time = candidate_machine_time_o[selected_machine]
                    machine_time[selected_machine] += operation_time
                    self.MS[people_index][MS_position] = MS_code
                    MS_position += 1
            people_index += 1
            machine_time.fill(0)

    def __initial_MS_local_selection(self):
        '''
        局部选择MS
        '''
        machine_time = np.zeros(
            shape=(1, self.data.machines_num), dtype=int).flatten()
        MS_ = np.empty(shape=(1, self.length), dtype=int).flatten()
        MS_position = 0
        for job_num in range(self.data.jobs_num):
            candidate_machine_j = self.data.candidate_machine[job_num]
            candidate_machine_time_j = self.data.candidate_machine_time[job_num]
            candidate_machine_index = self.data.candidate_machine_index[job_num]
            for operation_num in range(self.data.jobs_operations[job_num]):
                rear_index = operation_num + 1
                candidate_machine_o = candidate_machine_j[
                    candidate_machine_index[operation_num]:candidate_machine_index[rear_index]]
                candidate_machine_time_o = candidate_machine_time_j[
                    candidate_machine_index[operation_num]:candidate_machine_index[rear_index]]
                index = np.ix_(candidate_machine_o.tolist())
                tem_machine_time = machine_time[index].copy()
                tem_machine_time = tem_machine_time + candidate_machine_time_o
                MS_code = np.argmin(tem_machine_time)
                selected_machine = candidate_machine_o[MS_code]
                operation_time = candidate_machine_time_o[MS_code]
                machine_time[selected_machine] += operation_time
                MS_[MS_position] = MS_code
                MS_position += 1
            machine_time.fill(0)
        MS_.reshape(-1, 1)
        self.MS[self.global_size:self.global_size+self.local_size][:].fill(1)
        self.MS[self.global_size:self.global_size +
                self.local_size] = MS_*self.MS[self.global_size:self.global_size+self.local_size]

    def __initial_random_selection(self):
        '''
        随机随机选择
        '''
        people_index = self.global_size+self.local_size
        for people in range(self.random_size):
            MS_position = 0
            for job_num in range(self.data.jobs_num):
                candidate_machine_j = self.data.candidate_machine[job_num]
                candidate_machine_index = self.data.candidate_machine_index[job_num]
                for operation_num in range(self.data.jobs_operations[job_num]):
                    rear_index = operation_num + 1
                    candidate_machine_o = candidate_machine_j[
                        candidate_machine_index[operation_num]:candidate_machine_index[rear_index]]
                    MS_code = random.randint(0, len(candidate_machine_o)-1)
                    self.MS[people_index][MS_position] = MS_code
                    MS_position += 1
            people_index += 1

    def __initial_MS(self):
        '''
        初始化MS
        '''
        self.MS = np.empty(shape=(self.size, self.length), dtype=int)
        self.__initial_MS_global_selection()
        self.__initial_MS_local_selection()
        self.__initial_random_selection()

    def __initial_OS(self):
        '''
        随机排列OS
        '''
        jobs_operations = self.data.jobs_operations
        self.OS = np.empty(shape=(self.size, self.length), dtype=int)
        left_index = 0
        right_index = 0
        for job_num in range(len(jobs_operations)):
            operations = jobs_operations[job_num]
            left_index = right_index
            right_index += operations
            for people in range(self.size):
                self.OS[people][left_index:right_index] = job_num
        for people in range(self.size):
            np.random.shuffle(self.OS[people])

    def __decode(self, MS, OS):
        '''
        解码
        '''
        begin_time = np.array(shape=(self.data.machines_num, self.length),dtype=int)

    def show(self):
        print(self.MS)
        print(self.OS)
