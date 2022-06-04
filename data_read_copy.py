import os
from tkinter import N
import copy
import numpy as np


class FJSP_Data:
    '''
    数据
    '''

    path = None
    '''
    文件路径
    '''
    jobs_num = None
    '''
    工件数量
    '''
    machines_num = None
    '''
    机器数量
    '''
    jobs_id = None
    '''
    工件编号
    '''
    origial_data = None
    '''
    原始数据
    '''
    jobs_operations = None
    '''
    零件的工序
    '''
    jobs_operations_detile = None
    '''
    零件加工的详细信息
    '''

    def __init__(self, path, jobs_num, machines_num, max_machine_per_operation, jobs_id, origial_data, jobs_operations, jobs_operations_detile):
        self.path = path
        self.jobs_num = jobs_num
        self.machines_num = machines_num
        self.max_machine_per_operation = max_machine_per_operation
        self.jobs_id = jobs_id
        self.origial_data = origial_data
        self.jobs_operations = jobs_operations
        self.jobs_operations_detile = jobs_operations_detile

    def display_info(self, show_origial_data=False):
        print()
        print('一共有{}个零件,{}台机器,每个步骤只能在{}台机器上进行加工。'.format(
            self.jobs_num, self.machines_num, self.max_machine_per_operation))
        if show_origial_data:
            print('原始数据为:', self.origial_data[0])
        print()
        print('零件的编号列表:', self.jobs_id)
        print()
        for i in range(len(self.jobs_id)):
            id = self.jobs_id[i]
            print('零件{}有{}道工序:'.format(id, self.jobs_operations[0][i]))
            if show_origial_data:
                print('原始数据为:')
                print(self.origial_data[i+1])
            print('零件工序详细信息矩阵为:')
            print(self.jobs_operations_detile[i])
            print()


def read_data(path):
    if os.path.exists(path):
        with open(path, 'r') as data:
            lines = data.readlines()
            data_lists = []
            for line in lines:
                data_lists.append(get_list(line))
            origial_data = copy.deepcopy(data_lists)
            origial_data = np.array(origial_data, dtype=object)
            jobs_num = data_lists[0][0]
            machines_num = data_lists[0][1]
            max_machine_per_operation = data_lists[0][2]
            data_lists.pop(0)
            jobs_id = [i for i in range(jobs_num)]
            jobs_operations = np.zeros(shape=(1, jobs_num), dtype=int)
            for i in range(len(data_lists)):
                jobs_operations[0][i] = data_lists[i][0]
            jobs_operations_detile = np.zeros(
                shape=(jobs_num, jobs_operations.max(), machines_num), dtype=int)
            for i in range(len(data_lists)):
                position = 1
                list_length = len(data_lists[i])
                operation_number = 0
                while position < list_length:
                    machine_can_operation_num = data_lists[i][position]
                    position += 1
                    for j in range(machine_can_operation_num):
                        machine_number_can_operation = data_lists[i][position]-1
                        position += 1
                        jobs_operations_detile[i][operation_number][machine_number_can_operation] = data_lists[i][position]
                        position += 1
                    operation_number += 1
        return FJSP_Data(path, jobs_num, machines_num, max_machine_per_operation, jobs_id, origial_data, jobs_operations, jobs_operations_detile)


def get_list(line):
    list = []
    i = 0
    while i < len(line):
        try:
            if line[i].isdigit():
                j = i+1
                while(line[j].isdigit() and j <= len(line)):
                    j = j+1
                num = get_num(line, i, j-1)
                i = j
                list.append(num)
            else:
                i = i+1
        except Exception as e:
            print("出现异常")
    return list


def get_num(line, i, j):
    num = 0
    while(i <= j):
        num = num*10 + int(line[i])
        i = i+1
    return num
