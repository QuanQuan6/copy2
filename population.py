import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sortedcontainers import SortedList
import math
from numba import njit, prange, jit, int32, int64


# @njit(parallel=False)
def POX_crossover(code, crossover_P, jobs_num):
    '''
    基于工件优先顺序的交叉  未修正
    '''
    people_size, code_length = code.shape
    jobs = np.array([range(jobs_num)]).flatten()
    # 遍历每条基因
    for people in range(people_size-1):
        # 判断是否发生交叉
        if np.random.random() >= crossover_P:
            continue
        # 交叉基因 ### begin
        people_1 = code[people]
        people_2 = code[people+1]
        # 选取工件集
        np.random.shuffle(jobs)
        jobs_set = set(jobs[0:np.random.randint(1, jobs_num)])
        print(jobs_set)
        # 开始交叉
        positions_2 = 0
        for positions_1 in range(code_length):
            if people_1[positions_1] in jobs_set:
                continue
            while(people_2[positions_2] in jobs_set):
                positions_2 += 1
            people_1[positions_1] = people_2[positions_2]
            positions_2 += 1
        # 交叉基因 ### begin
    # 随机排序最后一个
    np.random.shuffle(code[people_size-1])





@njit((int32[:, :], int32[:]), parallel=False)
def initial_MS_position(MS_positions, jobs_operations):
    '''
    初始化对应工序的MS码矩阵

    参数:F
        MS_positions:对应工序的MS码矩阵
        jobs_operations:各个工件的工序数
    '''
    # 工件个数
    jobs_num = jobs_operations.shape[0]
    # 辅助位置
    position = 0
    # 初始化MS码对应位置矩阵 ### begin
    for job_num in range(jobs_num):
        for operation_num in prange(jobs_operations[job_num]):
            # 更新MS码对应位置矩阵
            MS_positions[job_num][operation_num] = position
            # 更新辅助位置
            position += 1


@njit(parallel=False)
def initial_jobs_operations_detail(jobs_operations_detail):
    '''
    初始化jobs_operations_detail变量方便计算
    '''
    return np.where(
        jobs_operations_detail == 0, 1000000000, jobs_operations_detail)


@njit((int32[:, :], int64, int64, int32[:, :], int32[:], int32[:], int64[:, :, :], int32[:, :, :], int32[:, :], int32[:]), parallel=False)
def initial_MS_global_selection(MS, first_index, rear_index, MS_positions, jobs_order, jobs_operations, jobs_operations_detail, candidate_machine, candidate_machine_index, machine_time):
    '''
    全局选择方法初始化对应MS码
    '''
    # 工件个数
    jobs_num = jobs_operations.shape[0]
    # 初始化机器加工时间辅助矩阵
    machine_time.fill(0)
    # 生成子种群 ### begin
    for position in range(rear_index-first_index):
        # 生成MS码的位置
        MS_index = position+first_index
        # 打乱随机工件矩阵
        np.random.shuffle(jobs_order)
        # 初始化机器加工时间辅助矩阵
        machine_time.fill(0)
        # 生成并更新MS码 ### begin
        for i in range(jobs_num):
            # 当前工件编号
            job_num = jobs_order[i]
            # 当前工件候选机器矩阵
            candidate_machine_j = candidate_machine[job_num]
            # 当前机器候选机器矩阵的索引矩阵
            candidate_machine_index_j = candidate_machine_index[job_num]
            for operation_num in range(jobs_operations[job_num]):
                # 当前工序的候选机器
                candidate_machine_o = candidate_machine_j[operation_num][0:
                                                                         candidate_machine_index_j[operation_num]]
                # 当前工序的加工时间
                machine_time_o = jobs_operations_detail[job_num][operation_num]
                # 临时机器加工时间
                tem_machine_time = machine_time + machine_time_o
                # 选取的最优机器编号
                selected_machine = np.argmin(tem_machine_time)
                # 最优机器对应的MS码
                MS_code = np.where(candidate_machine_o ==
                                   selected_machine)[0][0]
                # 最优机器对应的加工时间
                operation_time = machine_time_o[selected_machine]
                # 更新机器加工时间辅助矩阵
                machine_time[selected_machine] += operation_time
                # 当前工序对应的MS码位置
                MS_position = MS_positions[job_num][operation_num]
                # 更新MS码
                MS[MS_index][MS_position] = MS_code
        # 生成并更新MS码 ### end
    # 生成子种群 ### end


@njit((int32[:, :], int32[:], int64, int64, int32[:], int64[:, :, :], int32[:, :, :], int32[:, :], int32[:]), parallel=False)
def initial_MS_local_selection(MS, MS_, first_index, rear_index, jobs_operations, jobs_operations_detail, candidate_machine, candidate_machine_index, machine_time):
    '''
    局部选择方法初始化对应MS码
    '''
    # 工件个数
    jobs_num = jobs_operations.shape[0]
    # MS码MS码副本的位置
    MS_position = 0
    # 计算生成的MS码 ### begin
    for job_num in range(jobs_num):
        # 当前工件候选机器矩阵
        candidate_machine_j = candidate_machine[job_num]
        # 当前机器候选机器矩阵的索引矩阵
        candidate_machine_index_j = candidate_machine_index[job_num]
        for operation_num in range(jobs_operations[job_num]):
            # 当前工序的候选机器
            candidate_machine_o = candidate_machine_j[operation_num][0:
                                                                     candidate_machine_index_j[operation_num]]
            # 当前工序的加工时间
            machine_time_o = jobs_operations_detail[job_num][operation_num]
            # 临时机器加工时间
            tem_machine_time = machine_time + machine_time_o
            # 选取的最优机器编号
            selected_machine = np.argmin(tem_machine_time)
            # 最优机器对应的MS码
            MS_code = np.where(candidate_machine_o ==
                               selected_machine)[0][0]
            # 最优机器对应的加工时间
            operation_time = machine_time_o[selected_machine]
            # 更新机器加工时间辅助矩阵
            machine_time[selected_machine] += operation_time
            # 更新生成的MS码副本
            MS_[MS_position] = MS_code
            # 更新MS码副本的位置
            MS_position += 1
        # 初始化机器加工时间
        machine_time.fill(0)
    # 计算生成的MS码 ### end
    # 将该方法对应的MS码都设为1
    MS[first_index:rear_index][:].fill(1)
    # 将所有MS码变为和MS码副本一样的数值
    MS[first_index:rear_index] = MS_*MS[first_index:rear_index]


@njit((int32[:, :],  int64, int64, int32[:],  int32[:, :, :], int32[:, :]), parallel=False)
def initial_MS_random_selection(MS, first_index, rear_index, jobs_operations, candidate_machine, candidate_machine_index):
    '''
    随机选择方法初始化对应MS码
    '''
    # 工件个数
    jobs_num = jobs_operations.shape[0]
    # 生成子种群 ### begin
    for position in range(rear_index-first_index):
        # 个体的索引
        MS_index = first_index+position
        # 对应的MS码位置
        MS_position = 0
        # 生成并更新MS码 ### begin
        for job_num in range(jobs_num):
            # 当前工件候选机器矩阵
            candidate_machine_j = candidate_machine[job_num]
            # 当前机器候选机器矩阵的索引矩阵
            candidate_machine_index_j = candidate_machine_index[job_num]
            for operation_num in range(jobs_operations[job_num]):
                # 当前工序的候选机器
                candidate_machine_o = candidate_machine_j[operation_num][0:
                                                                         candidate_machine_index_j[operation_num]]
                # 随机选取的MS码
                MS_code = np.random.randint(0, len(candidate_machine_o))
                # 更新MS码
                MS[MS_index][MS_position] = MS_code
                # 更新对应的MS码位置
                MS_position += 1
        # 生成并更新MS码 ### end
    # 生成子种群 ### end


@njit((int32[:, :],  int64,  int32[:]), parallel=False)
def initial_OS(OS, size, jobs_operations):
    '''
    随机排列OS
    '''
    # 工件个数
    jobs_num = jobs_operations.shape[0]
    # OS左索引
    left_index = 0
    # OS右索引
    right_index = 0
    for job_num in range(jobs_num):
        # 当前工件的工序数
        operations = jobs_operations[job_num]
        # 更新左索引
        left_index = right_index
        # 更新右索引
        right_index += operations
        # 赋值编码
        for OS_index in range(size):
            OS[OS_index][left_index:right_index] = job_num
    # 随机打乱个体编码
    for OS_index in range(size):
        np.random.shuffle(OS[OS_index])


@njit((int32[:, :], int64, int64), parallel=False)
def uniform_crossover(code, crossover_P, uniform_P):
    '''
    均匀交叉

    参数：
        code:基因
        crossover_P:基因对交叉交叉概率
        uniform_P:交换基因序列概率

    '''
    code_size, code_length = code.shape
    # 遍历每组基因对
    for code_pair in range(math.floor(code_size/2)):
        # 判断是否发生交叉
        if np.random.random() >= crossover_P:
            continue
        # 交叉基因 ### begin
        code_index = code_pair*2
        code_1 = code[code_index]
        code_2 = code[code_index+1]
        # 遍历基因的每个位置
        for position in range(code_length):
            # 相同位置如果基因相同就不交叉
            if code_1[position] == code_2[position]:
                continue
            # 判断要不要交叉这个位置的基因
            if np.random.random() >= uniform_P:
                continue
            # 交叉该位置的基因
            code_1[position], code_2[position] = code_2[position], code_1[position]
        # 交叉基因 ### end


@njit((int32[:, :], int64), parallel=False)
def random_crossover(code, crossover_P):
    '''
    随机交叉
    '''
    # 判断是否发生交叉
    code_size = code.shape[0]
    for code_index in range(code_size):
        if np.random.random() >= crossover_P:
            continue
        np.random.shuffle(code[code_index])

def tournament(self, M, results):
    '''
    锦标赛选择
    '''
    MS = self.MS
    OS = self.OS
    # 新种群
    new_MS = np.empty(shape=(MS.shape), dtype=int)
    new_OS = np.empty(shape=(OS.shape), dtype=int)
    # 生成新种族 ### begin
    for new_people in range(self.size):
        selected_M_people = random.sample(range(self.size), k=M)
        selected_best_people = selected_M_people[np.argmin(
            results[selected_M_people])]
        new_MS[new_people] = MS[selected_best_people]
        new_OS[new_people] = OS[selected_best_people]
    # 生成新种族 ### end
    self.MS = new_MS
    self.OS = new_OS

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
    best_score = float('inf')
    '''
    最短加工时间
    '''
    best_MS = None
    '''
    最好的MS码
    '''
    best_OS = None
    '''
    最好的OS码
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

    def __initial_MS(self):
        '''
        初始化MS
        '''
        # 初始化MS码
        self.MS = np.empty(shape=(self.size, self.length), dtype=int)
        # 工件数量
        jobs_num = self.data.jobs_num
        # 各工件的工序数
        jobs_operations = self.data.jobs_operations
        # 最大工序数
        max_operations = jobs_operations.max()
        # 机器数量
        machines_num = self.data.machines_num
        # 各个工序的详细信息矩阵
        jobs_operations_detail = self.data.jobs_operations_detail
        jobs_operations_detail = initial_jobs_operations_detail(
            jobs_operations_detail)
        # 各个工序的候选机器矩阵
        candidate_machine = self.data.candidate_machine
        # 各个工序的候选机器矩阵和加工时间的索引矩阵
        candidate_machine_index = self.data.candidate_machine_index
        # 机器加工时间辅助矩阵
        machine_time = np.zeros(
            shape=(1, machines_num), dtype=int).flatten()
        # 全局选择初始化 ### begin
        # 随机工件矩阵
        jobs_order = np.arange(jobs_num)
        # 初始化对应工序的MS码矩阵
        MS_positions = np.empty(
            shape=(jobs_num, max_operations), dtype=int)
        initial_MS_position(MS_positions, jobs_operations)
        global_first_index = 0
        global_rear_index = self.global_size
        initial_MS_global_selection(self.MS, global_first_index, global_rear_index, MS_positions, jobs_order,
                                    jobs_operations, jobs_operations_detail, candidate_machine, candidate_machine_index, machine_time)
        # 全局选择初始化 ### end
        # 局部选择初始化 ### begin
        local_first_index = global_rear_index
        local_rear_index = local_first_index+self.local_size
        MS_ = np.empty(shape=(1, self.length), dtype=int).flatten()
        initial_MS_local_selection(self.MS, MS_, local_first_index, local_rear_index, jobs_operations,
                                   jobs_operations_detail, candidate_machine, candidate_machine_index, machine_time)
        # 局部选择初始化 ### end
        # 随机选择初始化 ### begin
        random_first_index = local_rear_index
        random_rear_index = random_first_index+self.random_size
        initial_MS_random_selection(self.MS, random_first_index, random_rear_index,
                                    jobs_operations, candidate_machine, candidate_machine_index)
        # 随机选择初始化 ### end

    def __initial_OS(self):
        '''
        初始化OS
        '''
        # 初始化OS码
        self.OS = np.empty(shape=(self.size, self.length), dtype=int)
        # 各工件的工序数
        jobs_operations = self.data.jobs_operations
        # 种群规模
        size = self.size
        initial_OS(self.OS, size, jobs_operations)

    def __decode(self):
        '''
        解码族群
        '''
        # 工件个数
        jobs_num = self.data.jobs_num
        # 机器数量
        machines_num = self.data.machines_num
        # 各工件的工序数
        jobs_operations = self.data.jobs_operations
        # 各个工序的候选机器矩阵
        candidate_machine = self.data.candidate_machine
        # 各个工序的候选机器矩阵和加工时间的索引矩阵
        candidate_machine_index = self.data.candidate_machine_index
        # 各个工序的详细信息矩阵
        jobs_operations_detail = self.data.jobs_operations_detail
        jobs_operations_detail = np.where(
            jobs_operations_detail == 0, 1000000000, jobs_operations_detail)
        # 最大工序数
        max_operations = jobs_operations.max()
        # 机器矩阵
        selected_machine = np.zeros(
            shape=(jobs_num, max_operations), dtype=int)
        # 时间矩阵
        selected_machine_time = np.zeros(
            shape=(jobs_num, max_operations), dtype=int)
        # 开工时间矩阵
        begin_time = np.zeros(
            shape=(machines_num, jobs_num, max_operations), dtype=int)
        # 结束时间矩阵
        end_time = np.zeros(
            shape=(machines_num, jobs_num, max_operations), dtype=int)
        # 记录该哪个工序的矩阵
        jobs_operation = np.zeros(shape=(1, jobs_num), dtype=int).flatten()
        # 记录机器是否开始加工过的矩阵
        machine_operationed = np.zeros(
            shape=(1, machines_num), dtype=bool).flatten()
        # 每个个体的解
        results = np.zeros(shape=(1, self.size)).flatten()
        # 求解 ### begin
        for people in range(self.size):
            # 计算时间矩阵和机器矩阵 ### begin
            # 初始化开始时间矩阵
            begin_time.fill(-1)
            # 初始化结束时间矩阵
            end_time.fill(-1)
            # 初始化该哪个工序的矩阵
            jobs_operation.fill(0)
            # 初始化记录机器是否开始加工过的矩阵
            machine_operationed.fill(False)
            # 初始化各机器开始时间和结束时间列表列表 ### begin
            begin_time_lists = []
            end_time_lists = []
            for machine_num in range(machines_num):
                begin_time_lists.append(SortedList())
                end_time_lists.append(SortedList())
            # 初始化各机器开始时间和结束时间列表列表 ### end
            # 当前的MS码
            MS = self.MS[people]
            # 依次读取的位置
            MS_position = 0
            # 计算时间矩阵和机器矩阵 ### begin
            for job_num in range(jobs_num):
                # 当前工件的候选机器矩阵
                candidate_machine_j = candidate_machine[job_num]
                # 当前工件的详细信息矩阵
                jobs_operations_detail_j = jobs_operations_detail[job_num]
                # 当前工件候选机器矩阵和加工时间的索引矩阵
                candidate_machine_index_j = candidate_machine_index[job_num]
                for operation_num in range(jobs_operations[job_num]):
                    # MS码上的机器码
                    candidate_machine_num = MS[MS_position]
                    # 当前工序的候选机器矩阵
                    candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]                                                              :candidate_machine_index_j[operation_num + 1]]
                    # 当前工序所用机器
                    selected_machine_num = candidate_machine_o[candidate_machine_num]
                    # 更新机器矩阵
                    selected_machine[job_num][operation_num] = selected_machine_num
                    # 更新时间矩阵
                    selected_machine_time[job_num][operation_num] = jobs_operations_detail_j[operation_num][selected_machine_num]
                    # 更新位置
                    MS_position += 1
            # 计算时间矩阵和机器矩阵 ### end
            # 当前的OS码
            OS = self.OS[people]
            # 生成调度 ### begin
            for OS_position in range(len(OS)):
                # 当前的工件编号
                job_num = OS[OS_position]
                # 当前的工序编号
                operation_num = jobs_operation[job_num]
                # 更新记录该哪个工序的矩阵
                jobs_operation[job_num] += 1
                # 加工所用的机器编号
                selected_machine_num = selected_machine[job_num][operation_num]
                # 加工所用的时间
                selected_machine_time_o = selected_machine_time[job_num][operation_num]
                # 记录所选机器加工开始时间的堆
                begin_time_list = begin_time_lists[selected_machine_num]
                # 记录所选机器加工结束时间的堆
                end_time_list = end_time_lists[selected_machine_num]
                # 机器没有开工过 ### begin
                if machine_operationed[selected_machine_num] == False:
                    # 是该工件的第一道工序，从0时刻开始 ### begin
                    if operation_num == 0:
                        # 更新开始时间矩阵
                        begin_time[selected_machine_num][job_num][operation_num] = 0
                        # 更新该机器开始时间列表
                        begin_time_list.add(0)
                        # 更新结束时间矩阵
                        end_time[selected_machine_num][job_num][operation_num] = selected_machine_time_o
                        # 更新该机器结束时间列表
                        end_time_list.add(selected_machine_time_o)
                    # 是该工件的第一道工序，从0时刻开始 ### end
                    # 从上道工序的结束时间开始 ### begin
                    else:
                        # 上一道工序所用的机器编号
                        last_machine_num = selected_machine[job_num][operation_num-1]
                        # 当前工序的开始时间
                        begin_time_o = end_time[last_machine_num][job_num][operation_num-1]
                        # 更新开始时间矩阵
                        begin_time[selected_machine_num][job_num][operation_num] = begin_time_o
                        # 更新该机器开始时间列表
                        begin_time_list.add(begin_time_o)
                        # 当前工序的结束时间
                        end_time_o = begin_time_o + selected_machine_time_o
                        # 更新结束时间矩阵
                        end_time[selected_machine_num][job_num][operation_num] = end_time_o
                        # 更新该机器结束时间列表
                        end_time_list.add(end_time_o)
                    # 从上道工序的结束时间开始 ### end
                    # 更新记录机器是否开始加工过的矩阵
                    machine_operationed[selected_machine_num] = True
                # 机器没有开工过 ### begin
                # 机器开工过则寻找最优时间段 ### begin
                else:
                    # 初始化当前工序的最早的开始时间
                    operation_begin_time = 0
                    # 如果该工序不是第一道工序更新operation_begin_time ### begin
                    if operation_num != 0:
                        # 上一个工序所用的机器编号
                        last_machine_num = selected_machine[job_num][operation_num-1]
                        # 最早从上一道工序的结束时间开始
                        operation_begin_time = end_time[last_machine_num][job_num][operation_num-1]
                    # 如果该工序不是第一道工序更新operation_begin_time ### end
                    # 初始化当前工序的开始时间（上一道工序的结束的时间）
                    begin_time_o = operation_begin_time
                    # 遍历寻找最优开始时间 ### begin
                    for (begin, end) in zip(begin_time_list, end_time_list):
                        if begin_time_o >= end:
                            continue
                        # 找到了，跳出
                        if begin - begin_time_o >= selected_machine_time_o:
                            break
                        # 更新进行下一次寻找
                        begin_time_o = end
                    # 遍历寻找最优开始时间 ### end
                    # 更新开始时间矩阵
                    begin_time[selected_machine_num][job_num][operation_num] = begin_time_o
                    # 更新该机器开始时间列表
                    begin_time_list.add(begin_time_o)
                    # 当前工序的结束时间
                    end_time_o = begin_time_o + selected_machine_time_o
                    # 更新结束时间矩阵
                    end_time[selected_machine_num][job_num][operation_num] = end_time_o
                    # 更新该机器结束时间列表
                    end_time_list.add(end_time_o)
                # 机器开工过则寻找最优时间段 ### end
            # 生成调度 ### end
            # 初始化最短加工时间
            min_time = 0
            # 计算最短加工时间 ### begin
            for machine_num in range(machines_num):
                # 当前机器的结束时间列表
                end_time_list = end_time_lists[machine_num]
                # 如果机器没有加工过就换下一个机器
                if machine_operationed[machine_num] == False:
                    continue
                # 当前机器的最晚完工时间
                min_time_m = end_time_lists[machine_num].pop()
                # 更新最短加工时间
                min_time = max(min_time, min_time_m)
            # 计算最短加工时间 ### end
            # 更新解
            results[people] = min_time
        # 求解 ### end
        # 返回解
        return results

    def __decode_one(self, MS, OS):
        '''
        解码一个个体
        '''
        # 工件个数
        jobs_num = self.data.jobs_num
        # 机器数量
        machines_num = self.data.machines_num
        # 各工件的工序数
        jobs_operations = self.data.jobs_operations
        # 各个工序的候选机器矩阵
        candidate_machine = self.data.candidate_machine
        # 各个工序的候选机器矩阵和加工时间的索引矩阵
        candidate_machine_index = self.data.candidate_machine_index
        # 各个工序的详细信息矩阵
        jobs_operations_detail = self.data.jobs_operations_detail
        jobs_operations_detail = np.where(
            jobs_operations_detail == 0, 1000000000, jobs_operations_detail)
        # 最大工序数
        max_operations = jobs_operations.max()
        # 机器矩阵
        selected_machine = np.zeros(
            shape=(jobs_num, max_operations), dtype=int)
        # 时间矩阵
        selected_machine_time = np.zeros(
            shape=(jobs_num, max_operations), dtype=int)
        # 开工时间矩阵
        begin_time = np.zeros(
            shape=(machines_num, jobs_num, max_operations), dtype=int)
        # 结束时间矩阵
        end_time = np.zeros(
            shape=(machines_num, jobs_num, max_operations), dtype=int)
        # 记录该哪个工序的矩阵
        jobs_operation = np.zeros(shape=(1, jobs_num), dtype=int).flatten()
        # 记录机器是否开始加工过的矩阵
        machine_operationed = np.zeros(
            shape=(1, machines_num), dtype=bool).flatten()
        # 初始化开始时间矩阵
        begin_time.fill(-1)
        # 初始化结束时间矩阵
        end_time.fill(-1)
        # 初始化该哪个工序的矩阵
        jobs_operation.fill(0)
        # 初始化记录机器是否开始加工过的矩阵
        machine_operationed.fill(False)
        # 初始化各机器开始时间和结束时间列表列表 ### begin
        begin_time_lists = []
        end_time_lists = []
        for machine_num in range(machines_num):
            begin_time_lists.append(SortedList())
            end_time_lists.append(SortedList())
        # 初始化各机器开始时间和结束时间列表列表 ### end
        # 依次读取的位置
        MS_position = 0
        # 计算时间矩阵和机器矩阵 ### begin
        for job_num in range(jobs_num):
            # 当前工件的候选机器矩阵
            candidate_machine_j = candidate_machine[job_num]
            # 当前工件的详细信息矩阵
            jobs_operations_detail_j = jobs_operations_detail[job_num]
            # 当前工件候选机器矩阵和加工时间的索引矩阵
            candidate_machine_index_j = candidate_machine_index[job_num]
            for operation_num in range(jobs_operations[job_num]):
                # MS码上的机器码
                candidate_machine_num = MS[MS_position]
                # 当前工序的候选机器矩阵
                candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]                                                          :candidate_machine_index_j[operation_num + 1]]
                # 当前工序所用机器
                selected_machine_num = candidate_machine_o[candidate_machine_num]
                # 更新机器矩阵
                selected_machine[job_num][operation_num] = selected_machine_num
                # 更新时间矩阵
                selected_machine_time[job_num][operation_num] = jobs_operations_detail_j[operation_num][selected_machine_num]
                # 更新位置
                MS_position += 1
        # 计算时间矩阵和机器矩阵 ### end
        # 生成调度 ### begin
        for OS_position in range(len(OS)):
            # 当前的工件编号
            job_num = OS[OS_position]
            # 当前的工序编号
            operation_num = jobs_operation[job_num]
            # 更新记录该哪个工序的矩阵
            jobs_operation[job_num] += 1
            # 加工所用的机器编号
            selected_machine_num = selected_machine[job_num][operation_num]
            # 加工所用的时间
            selected_machine_time_o = selected_machine_time[job_num][operation_num]
            # 记录所选机器加工开始时间的堆
            begin_time_list = begin_time_lists[selected_machine_num]
            # 记录所选机器加工结束时间的堆
            end_time_list = end_time_lists[selected_machine_num]
            # 机器没有开工过 ### begin
            if machine_operationed[selected_machine_num] == False:
                # 是该工件的第一道工序，从0时刻开始 ### begin
                if operation_num == 0:
                    # 更新开始时间矩阵
                    begin_time[selected_machine_num][job_num][operation_num] = 0
                    # 更新该机器开始时间列表
                    begin_time_list.add(0)
                    # 更新结束时间矩阵
                    end_time[selected_machine_num][job_num][operation_num] = selected_machine_time_o
                    # 更新该机器结束时间列表
                    end_time_list.add(selected_machine_time_o)
                # 是该工件的第一道工序，从0时刻开始 ### end
                # 从上道工序的结束时间开始 ### begin
                else:
                    # 上一道工序所用的机器编号
                    last_machine_num = selected_machine[job_num][operation_num-1]
                    # 当前工序的开始时间
                    begin_time_o = end_time[last_machine_num][job_num][operation_num-1]
                    # 更新开始时间矩阵
                    begin_time[selected_machine_num][job_num][operation_num] = begin_time_o
                    # 更新该机器开始时间列表
                    begin_time_list.add(begin_time_o)
                    # 当前工序的结束时间
                    end_time_o = begin_time_o + selected_machine_time_o
                    # 更新结束时间矩阵
                    end_time[selected_machine_num][job_num][operation_num] = end_time_o
                    # 更新该机器结束时间列表
                    end_time_list.add(end_time_o)
                # 从上道工序的结束时间开始 ### end
                # 更新记录机器是否开始加工过的矩阵
                machine_operationed[selected_machine_num] = True
            # 机器没有开工过 ### begin
            # 机器开工过则寻找最优时间段 ### begin
            else:
                # 初始化当前工序的最早的开始时间
                operation_begin_time = 0
                # 如果该工序不是第一道工序更新operation_begin_time ### begin
                if operation_num != 0:
                    # 上一个工序所用的机器编号
                    last_machine_num = selected_machine[job_num][operation_num-1]
                    # 最早从上一道工序的结束时间开始
                    operation_begin_time = end_time[last_machine_num][job_num][operation_num-1]
                # 如果该工序不是第一道工序更新operation_begin_time ### end
                # 初始化当前工序的开始时间（上一道工序的结束的时间）
                begin_time_o = operation_begin_time
                # 遍历寻找最优开始时间 ### begin
                for (begin, end) in zip(begin_time_list, end_time_list):
                    # 结束时间大于等于最优开始时间
                    if begin_time_o >= end:
                        continue
                    # 找到了，跳出
                    if begin - begin_time_o >= selected_machine_time_o:
                        break
                    # 更新进行下一次寻找
                    begin_time_o = end
                # 遍历寻找最优开始时间 ### end
                # 更新开始时间矩阵
                begin_time[selected_machine_num][job_num][operation_num] = begin_time_o
                # 更新该机器开始时间列表
                begin_time_list.add(begin_time_o)
                # 当前工序的结束时间
                end_time_o = begin_time_o + selected_machine_time_o
                # 更新结束时间矩阵
                end_time[selected_machine_num][job_num][operation_num] = end_time_o
                # 更新该机器结束时间列表
                end_time_list.add(end_time_o)
            # 机器开工过则寻找最优时间段 ### end
        # 生成调度 ### end
        # 初始化最短加工时间
        min_time = 0
        # 计算最短加工时间 ### begin
        for machine_num in range(machines_num):
            # 当前机器的结束时间列表
            end_time_list = end_time_lists[machine_num]
            # 如果机器没有加工过就换下一个机器
            if machine_operationed[machine_num] == False:
                continue
            # 当前机器的最晚完工时间
            min_time_m = end_time_lists[machine_num].pop()
            # 更新最短加工时间
            min_time = max(min_time, min_time_m)
        # 计算最短加工时间 ### end
        # 返回解
        return begin_time, end_time

    def GA(self, max_step=2000, max_no_new_best=15, select_type='tournament', tournament_M=2, crossover_P=1, uniform_P=1, crossover_MS_type='uniform', crossover_OS_type='random'):
        '''
        遗传算法
        '''
        no_new_best = 0
        # tournament_M大于种群个数的修正
        if tournament_M > self.size:
            tournament_M = self.size
        # 繁殖一代 ### begin
        for step in range(max_step):
            # 找不到更好的个体,结束迭代
            if max_no_new_best <= no_new_best:
                break
            results = self.__decode()
            # 更新最优解 ### begin
            best_poeple = np.argmin(results)
            if self.best_score > results[best_poeple]:
                self.best_MS = self.MS[best_poeple]
                self.best_OS = self.OS[best_poeple]
                self.best_score = results[best_poeple]
                no_new_best = 0
                # print('step: ', step)
                # print("best_score: ", self.best_score)
            else:
                no_new_best += 1
            # 更新最优解 ### end
            # 生成新种群 ### begin
            if select_type == 'tournament':
                tournament(tournament_M, results)
            else:
                print('select_type参数生成出错')
                return
            # 生成新种群 ### end
            # 交叉MS ### begin
            if crossover_MS_type == 'uniform':
                uniform_crossover(self.MS, crossover_P, uniform_P)
            else:
                print('crossover_MS_type参数生成出错')
                return
            # 交叉MS ### end
            # 交叉OS ### begin
            if crossover_OS_type == 'random':
                random_crossover(self.OS, crossover_P)
            else:
                print('crossover_OS_type参数生成出错')
                return
            # 交叉OS ### end
            # 变异MS ### begin
            # 变异MS ### end
            # 变异OS ### begin
            # 变异OS ### end
        # 繁殖一代 ### end

    def __get_data(self, MS, OS):
        # 初始化索引列表
        columns_index = ['machine_num', 'job_num',
                         'operation_num', 'begin_time', 'end_time']
        # 绘制甘特图的数据
        data = pd.DataFrame(data=None, columns=columns_index)
        # 初始化开始时间矩阵和结束时间矩阵
        begin_time, end_time = self.__decode_one(MS, OS)
        # 工件个数
        jobs_num = self.data.jobs_num
        # 机器数量
        machines_num = self.data.machines_num
        # 各工件的工序数
        jobs_operations = self.data.jobs_operations
        # 更新绘图数据 ### begin
        for machine_num in range(machines_num):
            begin_time_m = begin_time[machine_num]
            for job_num in range(jobs_num):
                begin_time_j = begin_time_m[job_num]
                for operation_num in range(jobs_operations[job_num]):
                    if begin_time_j[operation_num] >= 0:
                        new_data = []
                        new_data.append(machine_num)
                        new_data.append(job_num)
                        new_data.append(operation_num)
                        new_data.append(begin_time_j[operation_num])
                        new_data.append(
                            end_time[machine_num][job_num][operation_num])
                        new_data = pd.DataFrame(
                            [new_data], columns=columns_index)
                        data = pd.concat([data, new_data], copy=False)
        data['operation_time'] = data['end_time']-data['begin_time']
        return data

    def show_gantt_chart(self, MS, OS, figsize):
        '''
        绘制甘特图
        '''
        # 工件个数
        jobs_num = self.data.jobs_num
        # 获取数据
        data = self.__get_data(MS, OS)
        # 生成颜色列表
        colors = self.__make_colors(jobs_num)
        # 更新绘图数据 ### end
        plt.figure(figsize=figsize)
        for index, row in data.iterrows():
            plt.barh(y=row['machine_num'],
                     width=row['operation_time'], left=row['begin_time'], color=colors[row['job_num']])
            plt.text(x=row['begin_time'], y=row['machine_num'], s='J{}{}'.format(
                row['job_num'], row['operation_num']))
        plt.show()

    def __make_colors(self, numbers):
        colors = []
        COLOR_BITS = ['1', '2', '3', '4', '5', '6',
                      '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        for i in range(numbers):
            colorBit = ['#']
            colorBit.extend(random.sample(COLOR_BITS, 6))
            colors.append(''.join(colorBit))
        return colors

    def print(self, MS, OS):
        # 机器数量
        machines_num = self.data.machines_num
        # 获取数据
        data = self.__get_data(MS, OS)
        for machine_num in range(machines_num):
            result_list = []
            result_list.append('{}:'.format(machine_num))
            data_m = data[data['machine_num'] == machine_num]
            data_m = data_m.sort_values(by='begin_time')
            for index, row in data_m.iterrows():
                result_list.append('({},{}-{},{},{}),'.format(
                    row['job_num'], row['job_num'], row['operation_num'], row['begin_time'], row['end_time']))
            result_m = ''.join(result_list)
            print(result_m[:-1])
