import math
from numba import njit,  int32, int64, float64
import numpy as np


@njit(
    (int32[:, :], int32[:]),
    parallel=False)
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
        for operation_num in range(jobs_operations[job_num]):
            # 更新MS码对应位置矩阵
            MS_positions[job_num][operation_num] = position
            # 更新辅助位置
            position += 1


@njit(
    int64[:, :, :](int32[:, :, :]),
    parallel=False)
def initial_jobs_operations_detail(jobs_operations_detail):
    '''
    初始化jobs_operations_detail变量方便计算
    '''
    return np.where(
        jobs_operations_detail == 0, 1000000000, jobs_operations_detail)


@njit(
    (int32[:, :],
     int64, int64,
     int32[:, :], int32[:],
     int32[:], int64[:, :, :],
     int32[:, :, :], int32[:, :],
     int32[:]),
    parallel=False)
def initial_MS_global_selection(MS,
                                first_index, rear_index,
                                MS_positions, jobs_order,
                                jobs_operations, jobs_operations_detail,
                                candidate_machine, candidate_machine_index,
                                machine_time):
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


@njit(
    (int32[:, :], int32[:],
     int64, int64,
     int32[:], int64[:, :, :],
     int32[:, :, :], int32[:, :],
     int32[:]),
    parallel=False)
def initial_MS_local_selection(MS, MS_,
                               first_index, rear_index,
                               jobs_operations, jobs_operations_detail,
                               candidate_machine, candidate_machine_index,
                               machine_time):
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


@njit(
    (int32[:, :],
     int64, int64,
     int32[:],
     int32[:, :, :], int32[:, :]),
    parallel=False)
def initial_MS_random_selection(MS,
                                first_index, rear_index,
                                jobs_operations,
                                candidate_machine, candidate_machine_index):
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


@njit(
    (int32[:, :],  int64,  int32[:]),
    parallel=False)
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


@njit(
    (int32[:], int64, int64),
    parallel=False)
def add_one_item(list, value, length):
    if length == 0:
        list[0] = value
        return
    index = np.searchsorted(list[0:length], value)
    list[index+1:length+1] = list[index: length]
    list[index] = value


@njit(int32(int32[:], int32[:],
            int32[:], int32[:, :, :],
            int32[:, :, :], int32[:, :, :],
            int32[:, :], int32[:, :],
            int32[:], int32[:], int32[:],
            int32[:, :], int32[:, :],
            int32[:, :, :]),
      parallel=False)
def decode_one(MS, OS,
               jobs_operations, jobs_operations_detail,
               begin_time, end_time,
               selected_machine_time, selected_machine,
               jobs_operation, machine_operationed, machine_operations,
               begin_time_lists,  end_time_lists,
               candidate_machine):
    '''
    解码一个个体
    '''
    # 工件个数
    jobs_num = jobs_operations.shape[0]
    # 机器个数
    machines_num = begin_time_lists.shape[0]
    # 各个工序的详细信息矩阵
    # jobs_operations_detail = initial_jobs_operations_detail(
    #     jobs_operations_detail)
    # 初始化开始时间矩阵
    begin_time.fill(-1)
    # 初始化结束时间矩阵
    end_time.fill(-1)
    # 初始化该哪个工序的矩阵
    jobs_operation.fill(0)
    # 初始化记录机器是否开始加工过的矩阵
    machine_operationed.fill(0)
    # 初始化各工序所选机器矩阵
    selected_machine.fill(0)
    # 初始化各工序所用时间矩阵
    selected_machine_time.fill(0)
    # 初始化各机器开始时间列表
    begin_time_lists.fill(0)
    # 初始化各机器结束时间列表
    end_time_lists.fill(0)
    # 初始各个机器加工步骤数
    machine_operations.fill(0)
    # 依次读取的位置
    MS_position = 0
    # 计算时间矩阵和机器矩阵 ### begin
    for job_num in range(jobs_num):
        # 当前工件的候选机器矩阵
        candidate_machine_j = candidate_machine[job_num]
        # 当前工件的详细信息矩阵
        jobs_operations_detail_j = jobs_operations_detail[job_num]

        selected_machine_j = selected_machine[job_num]

        for operation_num in range(jobs_operations[job_num]):
            # MS码上的机器码
            candidate_machine_num = MS[MS_position]
            # 当前工序所用机器
            selected_machine_num = candidate_machine_j[operation_num][candidate_machine_num]
            # 更新机器矩阵
            selected_machine_j[operation_num] = selected_machine_num
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
        if machine_operationed[selected_machine_num] == 0:
            # 是该工件的第一道工序，从0时刻开始 ### begin
            if operation_num == 0:
                # 更新开始时间矩阵
                begin_time[selected_machine_num][job_num][operation_num] = 0
                # 更新该机器开始时间列表
                add_one_item(begin_time_list, 0,
                             machine_operations[selected_machine_num])
                # 更新结束时间矩阵
                end_time[selected_machine_num][job_num][operation_num] = selected_machine_time_o
                # 更新该机器结束时间列表
                add_one_item(end_time_list, selected_machine_time_o,
                             machine_operations[selected_machine_num])

                # 更新机器的操作数
                machine_operations[selected_machine_num] += 1
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
                add_one_item(begin_time_list, begin_time_o,
                             machine_operations[selected_machine_num])
                # 当前工序的结束时间
                end_time_o = begin_time_o + selected_machine_time_o
                # 更新结束时间矩阵
                end_time[selected_machine_num][job_num][operation_num] = end_time_o
                # 更新该机器结束时间列表
                add_one_item(end_time_list, end_time_o,
                             machine_operations[selected_machine_num])
                # 更新机器的操作数
                machine_operations[selected_machine_num] += 1
            # 从上道工序的结束时间开始 ### end
            # 更新记录机器是否开始加工过的矩阵
            machine_operationed[selected_machine_num] = 1
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
            for i in range(machine_operations[selected_machine_num]):
                # 结束时间大于等于最优开始时间
                if begin_time_o >= end_time_list[i]:
                    continue
                # 找到了，跳出
                if begin_time_list[i] - begin_time_o >= selected_machine_time_o:
                    break
                # 更新进行下一次寻找
                begin_time_o = end_time_list[i]
            # 遍历寻找最优开始时间 ### end
            # 更新开始时间矩阵
            begin_time[selected_machine_num][job_num][operation_num] = begin_time_o
            # 更新该机器开始时间列表
            add_one_item(begin_time_list, begin_time_o,
                         machine_operations[selected_machine_num])
            # 当前工序的结束时间
            end_time_o = begin_time_o + selected_machine_time_o
            # 更新结束时间矩阵
            end_time[selected_machine_num][job_num][operation_num] = end_time_o
            # 更新该机器结束时间列表
            add_one_item(end_time_list, end_time_o,
                         machine_operations[selected_machine_num])
            # 更新记录机器是否开始加工过的矩阵
            machine_operations[selected_machine_num] += 1
        # 机器开工过则寻找最优时间段 ### end
    # 生成调度 ### end
    # 初始化最短加工时间
    min_time = 0
    # 计算最短加工时间 ### begin
    for machine_num in range(machines_num):
        # 如果机器没有加工过就换下一个机器
        if machine_operationed[machine_num] == 0:
            continue
        # 当前机器的最晚完工时间
        min_time_m = end_time_lists[machine_num][machine_operations[machine_num]-1]
        # 更新最短加工时间
        min_time = max(min_time, min_time_m)

    # 计算最短加工时间 ### end
    # 返回解
    return min_time


def decode(MS, OS,
           decode_results,
           jobs_operations, jobs_operations_detail,
           begin_time, end_time,
           selected_machine_time, selected_machine,
           jobs_operation, machine_operationed, machine_operations,
           begin_time_lists,  end_time_lists,
           candidate_machine):
    '''
    解码 可修改加速但没必要
    '''
    for code_index in range(decode_results.shape[0]):
        decode_results[code_index] = decode_one(MS[code_index], OS[code_index],
                                                jobs_operations, jobs_operations_detail,
                                                begin_time, end_time,
                                                selected_machine_time, selected_machine,
                                                jobs_operation, machine_operationed, machine_operations,
                                                begin_time_lists,  end_time_lists,
                                                candidate_machine)


@njit(
    (int32[:, :], int32[:, :],
     int32[:, :], int32[:, :],
     int64,
     int32[:],
     int32[:]),
    parallel=False)
def tournament(MS, OS,
               new_MS, new_OS,
               tournament_M,
               results,
               recodes):
    '''
    锦标赛选择
    '''
    size = MS.shape[0]
    recodes.fill(-1)
    # 生成新种族 ### begin
    for new_code_index in range(size):
        selected_codes = np.random.choice(size, tournament_M)
        selected_codes = np.unique(selected_codes)
        selected_best_code = selected_codes[(np.argmin(
            results[selected_codes]))]
        recodes[new_code_index] = selected_best_code
    recodes_size = np.sum(recodes >= 0)
    for new_code_index in range(recodes_size):
        selected_code = recodes[new_code_index]
        new_MS[new_code_index] = MS[selected_code]
        new_OS[new_code_index] = OS[selected_code]
    # 生成新种族 ### end


@njit(
    (int32[:], int64, float64[:]),
    parallel=False
)
def get_crossover_P(decode_results, best_score, crossover_P):
    av_score = decode_results.mean()
    min_score = min(decode_results)
    for position in range(decode_results.shape[0]):
        if decode_results[position] > av_score:
            crossover_P[position] = 1
        elif av_score-best_score == 0:
            crossover_P[position] = 1
        else:
            crossover_P[position] = (av_score-decode_results[position])/(
                av_score-best_score)


@njit(
    (int32[:, :], float64[:]),
    parallel=False)
def uniform_crossover(code, Crossover_P):
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
        code_index_1 = code_pair*2
        code_index_2 = code_index_1+1
        crossover_P = max(Crossover_P[code_index_1], Crossover_P[code_index_2])
        # 判断是否发生交叉
        if np.random.random() >= crossover_P:
            continue
        # 交叉基因 ### begin
        code_1 = code[code_index_1]
        code_2 = code[code_index_2]
        # 遍历基因的每个位置
        for position in range(code_length):
            # 相同位置如果基因相同就不交叉
            if code_1[position] == code_2[position]:
                continue
            if np.random.random() >= crossover_P:
                continue
            # 判断要不要交叉这个位置的基因
            # 交叉该位置的基因
            code_1[position], code_2[position] = code_2[position], code_1[position]
        # 交叉基因 ### end


@njit(
    (int32[:, :], float64[:]),
    parallel=False)
def random_crossover(code, Crossover_P):
    '''
    随机交叉
    '''
    # 判断是否发生交叉
    code_size = code.shape[0]
    for code_index in range(code_size):
        if np.random.random() >= Crossover_P[code_index]:
            continue
        np.random.shuffle(code[code_index])


@njit(
    (int32[:, :], float64[:], int32[:]),
    parallel=False)
def POX_crossover(code, Crossover_P,  jobs):
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
        jobs_set = np.random.choice(jobs_num, np.random.randint(1, jobs_num))
        # 开始交叉
        positions_2 = 0
        for positions_1 in range(code_length):
            if code_1[positions_1] in jobs_set:
                continue
            while(code_2[positions_2] in jobs_set):
                positions_2 += 1
            code_1[positions_1] = code_2[positions_2]
            positions_2 += 1
        # 交叉基因 ### begin
    # 随机排序最后一个


@njit(
    (int32[:, :], float64[:]),
    parallel=False)
def random_mutation(code, Mutation_P):
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


@njit(
    (int32[:, :],
     int32[:, :, :], int32[:],
     int32[:]),
    parallel=False
)
def initial_best_MS(best_MS,
                    jobs_operations_detail, jobs_operations,
                    position_auxiliary):
    # 工件个数
    jobs_num = jobs_operations.shape[0]
    for job_num in range(jobs_num):
        for operation_num in range(jobs_operations[job_num]):
            min_machine = np.argmin(
                jobs_operations_detail[job_num][operation_num])
            best_MS[job_num][operation_num] = position_auxiliary[min_machine]


@njit(
    (int32[:, :], float64[:],
     int32[:, :], int32[:, :],
     int32[:]),
    parallel=False
)
def best_MS_mutations(MS, Mutation_P,
                      MS_positions, best_MS,
                      jobs_operations):
    '''
    最优机器变异
    '''
    MS_size, MS_length = MS.shape
    # 工件个数 
    jobs_num = jobs_operations.shape[0]
    # jobs_operations_detail = initial_jobs_operations_detail(
    #     jobs_operations_detail)
    for MS_index in range(MS_size):
        if np.random.random() >= Mutation_P[MS_index]:
            continue
        for job_num in range(jobs_num):
            for operation_num in range(jobs_operations[job_num]):
                if np.random.random() >= Mutation_P[MS_index]:
                    continue
                # 当前工序对应的MS码位置
                MS_position = MS_positions[job_num][operation_num]
                # 更新MS码
                MS[MS_index][MS_position] = best_MS[job_num][operation_num]
