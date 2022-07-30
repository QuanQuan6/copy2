from re import M
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sortedcontainers import SortedList
import math


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
        # self.__initial_MS()

    def __initial_MS_global_selection(self):
        '''
        全局选择方法初始化对应MS码
        '''
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
        jobs_operations_detail = np.where(
            jobs_operations_detail == 0, 1000000000, jobs_operations_detail)
        # 各个工序的候选机器矩阵
        candidate_machine = self.data.candidate_machine
        # 各个工序的候选机器矩阵和加工时间的索引矩阵
        candidate_machine_index = self.data.candidate_machine_index
        # 个体的索引
        people_index = 0
        # 机器加工时间辅助矩阵
        machine_time = np.zeros(
            shape=(1, machines_num), dtype=int).flatten()
        # 随机工件矩阵
        jobs_order = np.arange(jobs_num)
        # MS码对应位置矩阵
        MS_positions = np.zeros(
            shape=(jobs_num, max_operations), dtype=int)
        # 辅助位置
        position = 0
        # 初始化MS码对应位置矩阵 ### begin
        for job_num in range(jobs_num):
            for operation_num in range(jobs_operations[job_num]):
                # 更新MS码对应位置矩阵
                MS_positions[job_num][operation_num] = position
                # 更新辅助位置
                position += 1
        # 初始化MS码对应位置矩阵 ### end
        # 生成子种群 ### begin
        for people in range(self.global_size):
            # 打乱随机工件矩阵
            np.random.shuffle(jobs_order)
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
                    candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]
                        :candidate_machine_index_j[operation_num + 1]]
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
                    self.MS[people_index][MS_position] = MS_code
            # 生成并更新MS码 ### end
            # 更新的个体的索引
            people_index += 1
            # 初始化机器加工时间辅助矩阵
            machine_time.fill(0)
        # 生成子种群 ### end

    def __initial_MS_local_selection(self):
        '''
        局部选择方法初始化对应MS码
        '''
        # 各个工序的候选机器矩阵
        candidate_machine = self.data.candidate_machine
        # 各个工序的候选机器矩阵和加工时间的索引矩阵
        candidate_machine_index = self.data.candidate_machine_index
        # 各个工序的加工时间矩阵
        candidate_machine_time = self.data.candidate_machine_time
        # 各工件的工序数
        jobs_operations = self.data.jobs_operations
        # 机器加工时间辅助矩阵
        machine_time = np.zeros(
            shape=(1, self.data.machines_num), dtype=int).flatten()
        # MS码副本
        MS_ = np.empty(shape=(1, self.length), dtype=int).flatten()
        # MS码MS码副本的位置
        MS_position = 0
        # 计算生成的MS码 ### begin
        for job_num in range(self.data.jobs_num):
            # 当前工件候选机器矩阵
            candidate_machine_j = candidate_machine[job_num]
            # 当前机器的加工时间矩阵
            candidate_machine_time_j = candidate_machine_time[job_num]
            # 当前机器候选机器矩阵的索引矩阵
            candidate_machine_index_j = candidate_machine_index[job_num]
            for operation_num in range(jobs_operations[job_num]):
                # 结尾索引
                rear_index = operation_num + 1
                # 当前工序的候选机器
                candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]
                    :candidate_machine_index_j[rear_index]]
                # 当前工序的加工时间
                candidate_machine_time_o = candidate_machine_time_j[
                    candidate_machine_index_j[operation_num]:candidate_machine_index_j[rear_index]]
                # 候选机器的索引
                index = np.ix_(candidate_machine_o.tolist())
                # 初始化临时机器加工时间
                tem_machine_time = machine_time[index].copy()
                # 更新临时机器加工时间
                tem_machine_time = tem_machine_time + candidate_machine_time_o
                # 最优机器对应的MS码
                MS_code = np.argmin(tem_machine_time)
                # 选取的最优机器编号
                selected_machine = candidate_machine_o[MS_code]
                # 最优机器对应的加工时间
                operation_time = candidate_machine_time_o[MS_code]
                # 更新机器加工时间辅助矩阵
                machine_time[selected_machine] += operation_time
                # 更新生成的MS码副本
                MS_[MS_position] = MS_code
                # 更新MS码副本的位置
                MS_position += 1
            # 初始化机器加工时间
            machine_time.fill(0)
        # 计算生成的MS码 ### end
        # 重定形MS副本
        MS_.reshape(-1, 1)
        # 将该方法对应的MS码都设为1
        self.MS[self.global_size:self.global_size+self.local_size][:].fill(1)
        # 将所有MS码变为和MS码副本一样的数值
        self.MS[self.global_size:self.global_size +
                self.local_size] = MS_*self.MS[self.global_size:self.global_size+self.local_size]

    def __initial_random_selection(self):
        '''
        随机选择方法初始化对应MS码
        '''
        # 各个工序的候选机器矩阵
        candidate_machine = self.data.candidate_machine
        # 各个工序的候选机器矩阵和加工时间的索引矩阵
        candidate_machine_index = self.data.candidate_machine_index
        # 工件数量
        jobs_num = self.data.jobs_num
        # 各工件的工序数
        jobs_operations = self.data.jobs_operations
        # 个体的索引
        people_index = self.global_size+self.local_size
        # 生成子种群 ### begin
        for people in range(self.random_size):
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
                    candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]
                        :candidate_machine_index_j[operation_num + 1]]
                    # 随机选取的MS码
                    MS_code = random.randint(0, len(candidate_machine_o)-1)
                    # 更新MS码
                    self.MS[people_index][MS_position] = MS_code
                    # 更新对应的MS码位置
                    MS_position += 1
            # 生成并更新MS码 ### end
            # 更新的个体的索引
            people_index += 1
        # 生成子种群 ### end

    def __initial_MS(self):
        '''
        初始化MS
        '''
        self.MS = np.zeros(shape=(self.size, self.length), dtype=int)
        # self.__initial_MS_global_selection()
        # self.__initial_MS_local_selection()
        # self.__initial_random_selection()

    def __initial_OS(self):
        '''
        随机排列OS
        '''
        # 种群规模
        size = self.size
        # 各工件的工序数
        jobs_operations = self.data.jobs_operations
        # 初始化OS码
        self.OS = np.empty(shape=(size, self.length), dtype=int)
        # OS左索引
        left_index = 0
        # OS右索引
        right_index = 0
        for job_num in range(len(jobs_operations)):
            # 当前工件的工序数
            operations = jobs_operations[job_num]
            # 更新左索引
            left_index = right_index
            # 更新右索引
            right_index += operations
            # 赋值编码
            for people in range(size):
                self.OS[people][left_index:right_index] = job_num
        # 随机打乱个体编码
        for people in range(size):
            np.random.shuffle(self.OS[people])

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
                candidate_machine_num = 0
                # 当前工序的候选机器矩阵
                candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]:candidate_machine_index_j[operation_num + 1]]
                # 当前工序所用机器
                selected_machine_num = candidate_machine_o[candidate_machine_num]
                # 更新机器矩阵
                selected_machine[job_num][operation_num] = selected_machine_num
                # 更新时间矩阵
                selected_machine_time[job_num][operation_num] = jobs_operations_detail_j[operation_num][selected_machine_num]
            # 计算时间矩阵和机器矩阵 ### end
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
            # MS = self.MS[people]
            # # 依次读取的位置
            # MS_position = 0
            # # 计算时间矩阵和机器矩阵 ### begin
            # for job_num in range(jobs_num):
            #     # 当前工件的候选机器矩阵
            #     candidate_machine_j = candidate_machine[job_num]
            #     # 当前工件的详细信息矩阵
            #     jobs_operations_detail_j = jobs_operations_detail[job_num]
            #     # 当前工件候选机器矩阵和加工时间的索引矩阵
            #     candidate_machine_index_j = candidate_machine_index[job_num]
            #     for operation_num in range(jobs_operations[job_num]):
            #         # MS码上的机器码
            #         candidate_machine_num = MS[MS_position]
            #         # 当前工序的候选机器矩阵
            #         candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]:candidate_machine_index_j[operation_num + 1]]
            #         # 当前工序所用机器
            #         selected_machine_num = candidate_machine_o[candidate_machine_num]
            #         # 更新机器矩阵
            #         selected_machine[job_num][operation_num] = selected_machine_num
            #         # 更新时间矩阵
            #         selected_machine_time[job_num][operation_num] = jobs_operations_detail_j[operation_num][selected_machine_num]
            #         # 更新位置
            #         MS_position += 1
            # # 计算时间矩阵和机器矩阵 ### end
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

    def __decode_one(self, OS):
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
                candidate_machine_num = 0
                # 当前工序的候选机器矩阵
                candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]
                    :candidate_machine_index_j[operation_num + 1]]
                # 当前工序所用机器
                selected_machine_num = candidate_machine_o[candidate_machine_num]
                # 更新机器矩阵
                selected_machine[job_num][operation_num] = selected_machine_num
                # 更新时间矩阵
                selected_machine_time[job_num][operation_num] = jobs_operations_detail_j[operation_num][selected_machine_num]
                # 更新位置
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

    def __tournament(self, M, results):
        '''
        锦标赛选择
        '''
        OS = self.OS
        # 新种群
        new_OS = np.empty(shape=(OS.shape), dtype=int)
        # 生成新种族 ### begin
        for new_people in range(self.size):
            selected_M_people = random.sample(range(self.size), k=M)
            selected_best_people = selected_M_people[np.argmin(
                results[selected_M_people])]
            new_OS[new_people] = OS[selected_best_people]
        # 生成新种族 ### end
        self.OS = new_OS

    def __POX_crossover(self, code, crossover_P):
        '''
        基于工件优先顺序的交叉
        '''
        people_size, code_length = code.shape
        jobs_num = self.data.jobs_num
        jobs = np.array([range(jobs_num)]).flatten()
        # 遍历每组基因对
        for people in range(people_size-1):
            # 判断是否发生交叉
            if random.random() >= crossover_P[people]:
                continue
            # 交叉基因 ### begin
            people_1 = code[people]
            people_2 = code[people+1]
            # 选取工件集
            np.random.shuffle(jobs)
            jobs_set = set(jobs[0:random.randint(1, jobs_num-1)])
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

    def random_mutation(self, code, Mutation_P):
        '''
        随机变异
        '''
        code_size, code_length = code.shape
        for code_index in range(code_size):
            if np.random.random() >= Mutation_P[code_index]:
                continue
            position_1 = np.random.randint(0, code_length)
            position_2 = np.random.randint(0, code_length)
            code[code_index][position_1], code[code_index][position_2] = code[code_index][position_2], code[code_index][position_1]

    def update_memory_lib(self, memory_OS, memory_results,
                          OS, decode_results):
        #code_length = OS.shape[1]
        sorted_decode_results = np.argsort(decode_results)
        max_index = np.argmax(memory_results)
        for code_index in sorted_decode_results:
            if decode_results[code_index] < memory_results[max_index]:
                memory_OS[max_index] = OS[code_index]
                memory_results[max_index] = decode_results[code_index]
                max_index = np.argmax(memory_results)
            # elif decode_results[code_index] == memory_results[max_index]:
            #     same_indexs = np.where(
            #         memory_results == decode_results[code_index])[0]
            #     exchange_index = max_index
            #     excheng_H = code_length+1
            #     for same_index in same_indexs:
            #         _H = 0
            #         for position in range(code_length):
            #             if memory_OS[same_index][position] == OS[code_index][position]:
            #                 _H += 1
            #         if _H == 0:
            #             break
            #         if _H < excheng_H:
            #             excheng_H == _H
            #             exchange_index = same_index
                # for position in range(code_length):
                #     memory_OS[exchange_index][position] = OS[code_index][position]
                # memory_results[exchange_index] = decode_results[code_index]
                #max_index = np.argmax(memory_results)
            else:
                break

    def get_crossover_P(self, decode_results, crossover_P):
        best_score = self.best_score
        av_score = decode_results.mean()
        for position in range(decode_results.shape[0]):
            if decode_results[position] >= av_score:
                crossover_P[position] = 1
            elif av_score-best_score == 0:
                crossover_P[position] = 1
            else:
                crossover_P[position] = (av_score-decode_results[position])/(
                    av_score-best_score)
        p_mean = crossover_P.mean()
        if p_mean < 0.5:
            crossover_P = crossover_P * (0.5/p_mean)

    def tournament_memory(self, OS,
                          memory_OS,
                          new_OS,
                          tournament_M,
                          decode_results, memory_results,
                          step, max_step):
        '''
        锦标赛选择
        '''
        size = OS.shape[0]
        # 生成新种族 ### begin
        P =  0.5#np.cos(math.pi*(step)/max_step/2)
        memory_size = memory_OS.shape[0]
        for index in range(size):
            if np.random.random() > P:
                selected_codes = np.random.choice(size, tournament_M)
                selected_codes = np.unique(selected_codes)
                selected_best_code = selected_codes[(np.argmin(
                    decode_results[selected_codes]))]
                new_OS[index] = OS[selected_best_code]
            else:
                selected_codes = np.random.choice(size, tournament_M-1)
                selected_codes = np.unique(selected_codes)
                selected_best_code = selected_codes[(np.argmin(
                    decode_results[selected_codes]))]
                memory_code = np.random.randint(0, memory_size)
                if decode_results[selected_best_code] > memory_results[memory_code]:
                    new_OS[index] = OS[selected_best_code]
                else:
                    new_OS[index] = memory_OS[memory_code]

    def POX_crossover(self, code, Crossover_P,  jobs):
        '''
        基于工件优先顺序的交叉
        '''
        code_size, code_length = code.shape
        jobs_num = jobs.shape[0]
        # 遍历每条基因
        for code_index in range(code_size-1):
            # 判断是否发生交叉
            if np.random.random() >= Crossover_P[code_index]:
                continue
            # 交叉基因 ### begin
            code_1 = code[code_index]
            code_2 = code[code_index+1]
            # 选取工件集
            jobs_set = np.random.choice(
                jobs_num, np.random.randint(1, jobs_num))
            # 开始交叉
            positions_2 = 0
            for positions_1 in range(code_length):
                if code_1[positions_1] in jobs_set:
                    continue
                while(code_2[positions_2] in jobs_set):
                    positions_2 += 1
                code_1[positions_1] = code_2[positions_2]
                positions_2 += 1

    def GA(self, max_step=20, max_no_new_best=15, memory_size=0.2, tournament_M=4):
        '''
        遗传算法
        '''
        jobs_num = self.data.jobs_num
        Crossover_P = np.empty(shape=(1, self.size),
                               dtype=np.float64).flatten()
        jobs = np.array([range(jobs_num)]).flatten()
        memory_OS = np.empty(
            shape=(math.ceil(self.size*memory_size), self.length), dtype=np.int32)
        # 记忆库的解
        memory_results = np.empty(
            shape=(1, math.ceil(self.size*memory_size)), dtype=np.int32).flatten()
        memory_results.fill(np.int32(200000000))

        no_new_best = 0
        # tournament_M大于种群个数的修正
        if tournament_M > self.size:
            tournament_M = self.size
        # 繁殖一代 ### begin
        for step in range(max_step):
            # 找不到更好的个体,结束迭代
            results = self.__decode()
            # 更新最优解 ### begin
            best_poeple = np.argmin(results)
            if self.best_score > results[best_poeple]:
                # self.best_MS = self.MS[best_poeple]
                self.best_OS = self.OS[best_poeple]
                self.best_score = results[best_poeple]
                self.best_step = step
                no_new_best = 0
            else:
                no_new_best += 1
            # if no_new_best > max_no_new_best:
            #     memory_results.fill(np.int32(200000000))
            # 记忆库的解
            self.get_crossover_P(results, Crossover_P)
            
            # 更新最优解 ### end
            # 生成新种群
            self.update_memory_lib(memory_OS, memory_results, self.OS, results)
            new_OS = np.empty(shape=self.OS.shape, dtype=int)
            self.tournament_memory(
                self.OS, memory_OS, new_OS, tournament_M, results, memory_results, step, max_step)
            self.POX_crossover(new_OS, Crossover_P, jobs)
            # print(memory_results)
            #self.__tournament(tournament_M, results)
            # # 交叉MS
            # self.__uniform_crossover(self.MS, crossover_P, uniform_P)
            # 交叉OS
            self.random_mutation(new_OS,Crossover_P*0.3)
            self.OS = new_OS

    def show_best(self):
        print('MS: ', self.best_MS)
        print('OS: ', self.best_OS)
        print('time: ', self.best_time)

    def __get_data(self, OS):
        # 初始化索引列表
        columns_index = ['machine_num', 'job_num',
                         'operation_num', 'begin_time', 'end_time']
        # 绘制甘特图的数据
        data = pd.DataFrame(data=None, columns=columns_index)
        # 初始化开始时间矩阵和结束时间矩阵
        begin_time, end_time = self.__decode_one(OS)
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

    def show_gantt_chart(self,  OS, figsize):
        '''
        绘制甘特图
        '''
        # 工件个数
        jobs_num = self.data.jobs_num
        # 获取数据
        data = self.__get_data(OS)
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

    def print(self, OS):
        # 机器数量
        machines_num = self.data.machines_num
        # 获取数据
        data = self.__get_data(OS)
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
