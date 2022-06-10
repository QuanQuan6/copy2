
from platform import machine
from turtle import position
import numpy as np
from scipy import rand
from FJSP_Data import FJSP_Data
import random
import heapq
import matplotlib.pyplot as plt
import pandas as pd


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
                    candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]:candidate_machine_index_j[operation_num + 1]]
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
                candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]:candidate_machine_index_j[rear_index]]
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
                    candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]:candidate_machine_index_j[operation_num + 1]]
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
        self.MS = np.empty(shape=(self.size, self.length), dtype=int)
        self.__initial_MS_global_selection()
        self.__initial_MS_local_selection()
        self.__initial_random_selection()

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
            # 初始化各机器开始时间和结束时间堆列表 ### begin
            begin_time_heapqs = []
            end_time_heapqs = []
            for machine_num in range(machines_num):
                begin_time_heapq = []
                end_time_heapq = []
                begin_time_heapqs.append(begin_time_heapq)
                end_time_heapqs.append(end_time_heapq)
            # 初始化各机器开始时间和结束时间堆列表 ### end
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
                    candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]:candidate_machine_index_j[operation_num + 1]]
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
                begin_time_heapq = begin_time_heapqs[selected_machine_num]
                # 记录所选机器加工结束时间的堆
                end_time_heapq = end_time_heapqs[selected_machine_num]
                # 机器没有开工过 ### begin
                if machine_operationed[selected_machine_num] == False:
                    # 是该工件的第一道工序，从0时刻开始 ### begin
                    if operation_num == 0:
                        # 更新开始时间矩阵
                        begin_time[selected_machine_num][job_num][operation_num] = 0
                        # 更新该机器开始时间堆
                        heapq.heappush(begin_time_heapq, 0)
                        # 更新结束时间矩阵
                        end_time[selected_machine_num][job_num][operation_num] = selected_machine_time_o
                        # 更新该机器结束时间堆
                        heapq.heappush(end_time_heapq, selected_machine_time_o)
                    # 是该工件的第一道工序，从0时刻开始 ### end
                    # 从上道工序的结束时间开始 ### begin
                    else:
                        # 上一道工序所用的机器编号
                        last_machine_num = selected_machine[job_num][operation_num-1]
                        # 当前工序的开始时间
                        begin_time_o = end_time[last_machine_num][job_num][operation_num-1]
                        # 更新开始时间矩阵
                        begin_time[selected_machine_num][job_num][operation_num] = begin_time_o
                        # 更新该机器开始时间堆
                        heapq.heappush(begin_time_heapq, begin_time_o)
                        # 当前工序的结束时间
                        end_time_o = begin_time_o + selected_machine_time_o
                        # 更新结束时间矩阵
                        end_time[selected_machine_num][job_num][operation_num] = end_time_o
                        # 更新该机器结束时间堆
                        heapq.heappush(end_time_heapq, end_time_o)
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
                    for (begin, end) in zip(begin_time_heapq, end_time_heapq):
                        # 找到了，跳出
                        if begin - begin_time_o >= selected_machine_time_o:
                            break
                        # 更新进行下一次寻找
                        begin_time_o = end
                    # 遍历寻找最优开始时间 ### end
                    # 更新开始时间矩阵
                    begin_time[selected_machine_num][job_num][operation_num] = begin_time_o
                    # 更新该机器开始时间堆
                    heapq.heappush(begin_time_heapq, begin_time_o)
                    # 当前工序的结束时间
                    end_time_o = begin_time_o + selected_machine_time_o
                    # 更新结束时间矩阵
                    end_time[selected_machine_num][job_num][operation_num] = end_time_o
                    # 更新该机器结束时间堆
                    heapq.heappush(end_time_heapq, end_time_o)
                # 机器开工过则寻找最优时间段 ### end
            # 生成调度 ### end
            # 初始化最短加工时间
            min_time = 0
            # 计算最短加工时间 ### begin
            for machine_num in range(machines_num):
                # 当前机器的结束时间堆
                end_time_heapq = end_time_heapqs[machine_num]
                # 如果机器没有加工过就换下一个机器
                if machine_operationed[machine_num] == False:
                    continue
                # 当前机器的最晚完工时间
                min_time_m = heapq.nlargest(
                    1,  end_time_heapqs[machine_num])[0]
                # 如果完工时间大于当前的最短加工时间
                if min_time < min_time_m:
                    # 更新最短加工时间
                    min_time = min_time_m
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
        # 每个个体的解
        results = np.zeros(shape=(1, self.size)).flatten()
        # 初始化开始时间矩阵
        begin_time.fill(-1)
        # 初始化结束时间矩阵
        end_time.fill(-1)
        # 初始化该哪个工序的矩阵
        jobs_operation.fill(0)
        # 初始化记录机器是否开始加工过的矩阵
        machine_operationed.fill(False)
        # 初始化各机器开始时间和结束时间堆列表 ### begin
        begin_time_heapqs = []
        end_time_heapqs = []
        for machine_num in range(machines_num):
            begin_time_heapq = []
            end_time_heapq = []
            begin_time_heapqs.append(begin_time_heapq)
            end_time_heapqs.append(end_time_heapq)
        # 初始化各机器开始时间和结束时间堆列表 ### end
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
                candidate_machine_o = candidate_machine_j[candidate_machine_index_j[operation_num]:candidate_machine_index_j[operation_num + 1]]
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
            begin_time_heapq = begin_time_heapqs[selected_machine_num]
            # 记录所选机器加工结束时间的堆
            end_time_heapq = end_time_heapqs[selected_machine_num]
            # 机器没有开工过 ### begin
            if machine_operationed[selected_machine_num] == False:
                # 是该工件的第一道工序，从0时刻开始 ### begin
                if operation_num == 0:
                    # 更新开始时间矩阵
                    begin_time[selected_machine_num][job_num][operation_num] = 0
                    # 更新该机器开始时间堆
                    heapq.heappush(begin_time_heapq, 0)
                    # 更新结束时间矩阵
                    end_time[selected_machine_num][job_num][operation_num] = selected_machine_time_o
                    # 更新该机器结束时间堆
                    heapq.heappush(end_time_heapq, selected_machine_time_o)
                # 是该工件的第一道工序，从0时刻开始 ### end
                # 从上道工序的结束时间开始 ### begin
                else:
                    # 上一道工序所用的机器编号
                    last_machine_num = selected_machine[job_num][operation_num-1]
                    # 当前工序的开始时间
                    begin_time_o = end_time[last_machine_num][job_num][operation_num-1]
                    # 更新开始时间矩阵
                    begin_time[selected_machine_num][job_num][operation_num] = begin_time_o
                    # 更新该机器开始时间堆
                    heapq.heappush(begin_time_heapq, begin_time_o)
                    # 当前工序的结束时间
                    end_time_o = begin_time_o + selected_machine_time_o
                    # 更新结束时间矩阵
                    end_time[selected_machine_num][job_num][operation_num] = end_time_o
                    # 更新该机器结束时间堆
                    heapq.heappush(end_time_heapq, end_time_o)
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
                for (begin, end) in zip(begin_time_heapq, end_time_heapq):
                    # 找到了，跳出
                    if begin - begin_time_o >= selected_machine_time_o:
                        break
                    # 更新进行下一次寻找
                    begin_time_o = end
                # 遍历寻找最优开始时间 ### end
                # 更新开始时间矩阵
                begin_time[selected_machine_num][job_num][operation_num] = begin_time_o
                # 更新该机器开始时间堆
                heapq.heappush(begin_time_heapq, begin_time_o)
                # 当前工序的结束时间
                end_time_o = begin_time_o + selected_machine_time_o
                # 更新结束时间矩阵
                end_time[selected_machine_num][job_num][operation_num] = end_time_o
                # 更新该机器结束时间堆
                heapq.heappush(end_time_heapq, end_time_o)
            # 机器开工过则寻找最优时间段 ### end
        # 生成调度 ### end
        # 初始化最短加工时间
        min_time = 0
        # 计算最短加工时间 ### begin
        for machine_num in range(machines_num):
            # 当前机器的结束时间堆
            end_time_heapq = end_time_heapqs[machine_num]
            # 如果机器没有加工过就换下一个机器
            if machine_operationed[machine_num] == False:
                continue
            # 当前机器的最晚完工时间
            min_time_m = heapq.nlargest(
                1,  end_time_heapqs[machine_num])[0]
            # 如果完工时间大于当前的最短加工时间
            if min_time < min_time_m:
                # 更新最短加工时间
                min_time = min_time_m
        # 计算最短加工时间 ### end
        # 返回解
        return begin_time, end_time

    def GA(self):
        '''
        遗传算法
        '''
        results = self.__decode()
        index = np.argmin(results)
        return index

    def show_best(self):
        print('MS: ', self.best_MS)
        print('OS: ', self.best_OS)
        print('time: ', self.best_time)

    def show_gantt_chart(self, MS, OS, figsize):
        '''
        绘制甘特图
        '''
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
        colors = self.__make_colors(jobs_num)
        print(colors)
        # 更新绘图数据 ### end
        plt.figure(figsize=figsize)
        for index, row in data.iterrows():
            plt.barh(y=row['machine_num'],
                     width=row['operation_time'], left=row['begin_time'],color=colors[row['\
                         
                         ']])
        plt.show()

    def __make_colors(self,numbers):
        colors = []
        COLOR_BITS = ['1', '2', '3', '4', '5', '6',
                          '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        for i in range(numbers):
            colorBit = ['#']
            colorBit.extend(random.sample(COLOR_BITS, 6))
            colors.append(''.join(colorBit))
        return colors
