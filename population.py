
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from auxiliary import*
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
    best_score = float('inf')
    '''
    最短加工时间
    '''
    best_step = float('inf')
    '''
    收敛次数
    '''
    best_MS = None
    '''
    最好的MS码
    '''
    best_OS = None
    '''
    最好的OS码
    '''

    def __init__(self, data,):
        self.data = data
        
        self.length = data.jobs_operations.sum()

    def initial(self,size):
        '''
        自适应初始化
        '''
        self.size = size
        self.best_MS = None
        self.best_OS = None
        self.best_score = float('inf')
        self.best_step = float('inf')
        real_size = self.size
        self.global_size = 100
        self.local_size = 100
        self.random_size = 100
        self.size = 300
        self.__initial_OS()
        self.__initial_MS()
        # 工件个数
        jobs_num = self.data.jobs_num
        # 机器数量
        machines_num = self.data.machines_num
        # 各工件的工序数
        jobs_operations = self.data.jobs_operations
        # 最大工序数
        max_operations = jobs_operations.max()
        # 各个工序的详细信息矩阵
        jobs_operations_detail = self.data.jobs_operations_detail
        # 各个工序的候选机器矩阵
        candidate_machine = self.data.candidate_machine
        candidate_machine_index = self.data.candidate_machine_index
        # 各个工序的候选机器矩阵和加工时间的索引矩阵
        # 选用机器矩阵
        selected_machine = np.empty(
            shape=(jobs_num, max_operations), dtype=np.int32)
        # 选用机器时间矩阵
        selected_machine_time = np.empty(
            shape=(jobs_num, max_operations), dtype=np.int32)
        # 开工时间矩阵
        begin_time = np.empty(
            shape=(machines_num, jobs_num, max_operations), dtype=np.int32)
        # 结束时间矩阵
        end_time = np.empty(
            shape=(machines_num, jobs_num, max_operations), dtype=np.int32)
        # 记录工件加工到哪道工序的矩阵
        jobs_operation = np.empty(
            shape=(1, jobs_num), dtype=np.int32).flatten()
        # 记录机器是否开始加工过的矩阵
        machine_operationed = np.empty(
            shape=(1, machines_num), dtype=np.int32).flatten()
        # 各机器开始时间矩阵
        begin_time_lists = np.empty(
            shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
        # 各机器结束时间矩阵
        end_time_lists = np.empty(
            shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
        job_lists = np.empty(
            shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
        operation_lists = np.empty(
            shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
        # 记录机器加工的步骤数
        machine_operations = np.empty(
            shape=(1, machines_num), dtype=np.int32).flatten()
        # 解码结果矩阵
        decode_results = np.empty(
            shape=(1, self.size), dtype=np.int32).flatten()
        # 记录最短时间的辅助列表
        result = np.empty(shape=(1, 1), dtype=np.int32).flatten()
        decode(self.MS, self.OS,
               decode_results, np.arange(self.size, dtype=np.int32).flatten(),
               jobs_operations, jobs_operations_detail,
               begin_time, end_time,
               selected_machine_time, selected_machine,
               jobs_operation, machine_operationed, machine_operations,
               begin_time_lists,  end_time_lists,
               job_lists, operation_lists,
               candidate_machine, candidate_machine_index,
               result)
        # 计算自适应初始化的数据 ### begin
        global_score = 1/decode_results[0:100].sum()
        local_score = 1/decode_results[100:200].sum()
        random_score = 1/decode_results[200:].sum()
        global_size = math.ceil(
            real_size*global_score/(global_score+local_score+random_score))
        self.global_size = global_size
        local_size = math.ceil((real_size-global_size)*local_score /
                               (local_score+random_score))
        self.local_size = local_size
        self.random_size = real_size-local_size-global_size
        self.size = real_size
        # 计算自适应初始化的数据 ### end
        # 执行初始化
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
            shape=(1, machines_num), dtype=np.int32).flatten()
        # 全局选择初始化 ### begin
        # 随机工件矩阵
        jobs_order = np.arange(jobs_num, dtype=np.int32).flatten()
        # 初始化对应工序的MS码矩阵
        MS_positions = np.empty(
            shape=(jobs_num, max_operations), dtype=int)
        initial_MS_position(MS_positions, jobs_operations)
        global_first_index = 0
        global_rear_index = self.global_size
        initial_MS_global_selection(self.MS,
                                    global_first_index, global_rear_index,
                                    MS_positions, jobs_order,
                                    jobs_operations, jobs_operations_detail,
                                    candidate_machine, candidate_machine_index,
                                    machine_time)
        # 全局选择初始化 ### end
        # 局部选择初始化 ### begin
        local_first_index = global_rear_index
        local_rear_index = local_first_index+self.local_size
        MS_ = np.empty(shape=(1, self.length), dtype=int).flatten()
        initial_MS_local_selection(self.MS, MS_,
                                   local_first_index, local_rear_index,
                                   jobs_operations, jobs_operations_detail,
                                   candidate_machine, candidate_machine_index,
                                   machine_time)
        # 局部选择初始化 ### end
        # 随机选择初始化 ### begin
        random_first_index = local_rear_index
        random_rear_index = random_first_index+self.random_size
        initial_MS_random_selection(self.MS,
                                    random_first_index, random_rear_index,
                                    jobs_operations,
                                    candidate_machine, candidate_machine_index)
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

    def GA(self, max_step=2000, max_no_new_best=100,
           select_type='tournament', tournament_M=3,
           memory_size=0.2,
           find_type='auto', crossover=1.0, V_C_ratio=0.2,
           crossover_MS_type='uniform', crossover_OS_type='POX',
           mutation_MS_type='best', mutation_OS_type='random',
           VNS_ratio=0.2, VNS_='not', VNS_type='not'):
        '''
        遗传算法
        '''
        # 初始化辅助参数 ### begin
        # 记录迭代多少次没有更新最优解
        no_new_best = 0
        # tournament_M大于种群个数的修正
        if tournament_M > self.size:
            tournament_M = self.size
        # 工件个数
        jobs_num = self.data.jobs_num
        # 机器数量
        machines_num = self.data.machines_num
        # 各工件的工序数
        jobs_operations = self.data.jobs_operations
        # 最大工序数
        max_operations = jobs_operations.max()
        # 各个工序的详细信息矩阵
        jobs_operations_detail = self.data.jobs_operations_detail
        # 各个工序的候选机器矩阵
        candidate_machine = self.data.candidate_machine
        # 各个工序的候选机器矩阵和加工时间的索引矩阵
        candidate_machine_index = self.data.candidate_machine_index
        # VNS数量
        # 选用机器矩阵
        selected_machine = np.empty(
            shape=(jobs_num, max_operations), dtype=np.int32)
        # 选用机器时间矩阵
        selected_machine_time = np.empty(
            shape=(jobs_num, max_operations), dtype=np.int32)
        # 开工时间矩阵
        begin_time = np.empty(
            shape=(machines_num, jobs_num, max_operations), dtype=np.int32)
        # 结束时间矩阵
        end_time = np.empty(
            shape=(machines_num, jobs_num, max_operations), dtype=np.int32)
        # 记录工件加工到哪道工序的矩阵
        jobs_operation = np.empty(
            shape=(1, jobs_num), dtype=np.int32).flatten()
        # 记录机器是否开始加工过的矩阵
        machine_operationed = np.empty(
            shape=(1, machines_num), dtype=np.int32).flatten()
        # 各机器开始时间矩阵
        begin_time_lists = np.empty(
            shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
        # 各机器结束时间矩阵
        end_time_lists = np.empty(
            shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
        # 记录各机器加工的工件矩阵
        job_lists = np.empty(
            shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
        # 记录各机器加工的工序矩阵
        operation_lists = np.empty(
            shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
        # 记录机器加工的步骤数
        machine_operations = np.empty(
            shape=(1, machines_num), dtype=np.int32).flatten()
        # 解码结果矩阵
        decode_results = np.empty(
            shape=(1, self.size), dtype=np.int32).flatten()
        # 初始化对应工序的MS码矩阵
        MS_positions = np.empty(
            shape=(jobs_num, max_operations), dtype=np.int32)
        initial_MS_position(MS_positions, jobs_operations)
        # 初始化各个工序最短时间矩阵
        best_MS = np.empty(shape=(jobs_num, max_operations), dtype=np.int32)
        initial_best_MS(best_MS, jobs_operations_detail,
                        jobs_operations, np.array(range(machines_num), dtype=np.int32).flatten())
        # 初始化交叉概率
        Crossover_P = np.empty(shape=(1, self.size),
                               dtype=np.float64).flatten()
        # 记录最短时间的辅助列表
        result = np.empty(shape=(1, 1), dtype=np.int32).flatten()
        # VNS辅助列表
        VNS_num = math.ceil(self.size*VNS_ratio)
        # 随机工件矩阵
        jobs_order = np.arange(jobs_num, dtype=np.int32).flatten()
        # 记忆库
        memory_MS = np.empty(
            shape=(math.ceil(self.size*memory_size), self.length), dtype=np.int32)
        memory_OS = np.empty(
            shape=(math.ceil(self.size*memory_size), self.length), dtype=np.int32)
        # 记忆库的解
        memory_results = np.empty(
            shape=(1, math.ceil(self.size*memory_size)), dtype=np.int32).flatten()
        memory_results.fill(int32(200000000))
        # 初始化辅助参数 ### end
        # 繁殖一代 ### begin
        for step in range(max_step):
            # 找不到更好的个体,结束迭代
            if max_no_new_best <= no_new_best:
                break
            # 邻域搜索 begin
            decode(self.MS, self.OS,
                       decode_results, np.arange(
                           self.size, dtype=np.int32).flatten(),
                       jobs_operations, jobs_operations_detail,
                       begin_time, end_time,
                       selected_machine_time, selected_machine,
                       jobs_operation, machine_operationed, machine_operations,
                       begin_time_lists,  end_time_lists,
                       job_lists, operation_lists,
                       candidate_machine, candidate_machine_index,
                       result)
            if VNS_ == 'normal':
                sorted_index = np.argsort(decode_results)
                VNS(self.MS, self.OS,
                    decode_results, sorted_index[:VNS_num],
                    jobs_operations, jobs_operations_detail,
                    begin_time, end_time,
                    selected_machine_time, selected_machine,
                    jobs_operation, machine_operationed, machine_operations,
                    begin_time_lists,  end_time_lists,
                    job_lists, operation_lists,
                    candidate_machine, candidate_machine_index,
                    result, VNS_type,
                    jobs_order, MS_positions)
            elif VNS_ == 'quick':
                sorted_index = np.argsort(decode_results)
                quick_VNS(self.MS, self.OS,
                          decode_results, sorted_index[:VNS_num],
                          jobs_operations, jobs_operations_detail,
                          begin_time, end_time,
                          selected_machine_time, selected_machine,
                          jobs_operation, machine_operationed, machine_operations,
                          begin_time_lists,  end_time_lists,
                          job_lists, operation_lists,
                          candidate_machine, candidate_machine_index,
                          result, VNS_type,
                          jobs_order, MS_positions)
            # 邻域搜索 end
            # 更新最优解 ### begin
            best_poeple = np.argmin(decode_results)
            if self.best_score > decode_results[best_poeple]:
                self.best_MS = self.MS[best_poeple]
                self.best_OS = self.OS[best_poeple]
                self.best_score = decode_results[best_poeple]
                self.best_step = step+1
                no_new_best = 0
            else:
                no_new_best += 1
            # 更新最优解 ### end
            # 记忆库
            if memory_size > 0:
                update_memory_lib(memory_MS, memory_OS, memory_results,
                                  self.MS, self.OS, decode_results)
            # 记忆库
            # 生成新种群 ### begin
            new_OS = np.empty(shape=self.MS.shape, dtype=int)
            new_MS = np.empty(shape=self.MS.shape, dtype=int)
            # 选取下一代 ### begin
            recodes = np.empty(shape=(1, self.size), dtype=int).flatten()
            if select_type == 'tournament':
                tournament(self.MS, self.OS,
                           new_MS, new_OS,
                           tournament_M,
                           decode_results, recodes)
            elif select_type == 'tournament_random':
                initial_OS(new_OS, self.size, jobs_operations)
                tournament_random(self.MS, self.OS,
                                  new_MS, new_OS,
                                  tournament_M,
                                  decode_results,
                                  recodes,
                                  jobs_operations,
                                  candidate_machine, candidate_machine_index)
            elif select_type == 'tournament_memory':
                tournament_memory(self.MS, self.OS,
                                  memory_MS, memory_OS,
                                  new_MS, new_OS,
                                  tournament_M,
                                  decode_results, memory_results,
                                  step, max_step

                                  )
            else:
                print('select_type参数生成出错')
                return
            # 选取下一代 ### end
            # 计算交叉概率 ### begin
            if find_type == 'auto':
                get_crossover_P(decode_results, self.best_score, Crossover_P)
            elif find_type == 'const':
                Crossover_P.fill(crossover)
            else:
                print('find_type参数生成出错')
                return
            # 计算交叉概率 ### end
            # 交叉MS ### begin
            if crossover_MS_type == 'uniform':
                uniform_crossover(new_MS, Crossover_P)
            elif crossover_MS_type == 'not':
                pass
            else:
                print('crossover_MS_type参数生成出错')
                return
            # 交叉MS ### end
            # 交叉OS ### begin
            if crossover_OS_type == 'random':
                random_crossover(new_OS, Crossover_P)
            elif crossover_OS_type == 'POX':
                jobs = np.array([range(self.data.jobs_num)]).flatten()
                POX_crossover(new_OS, Crossover_P, jobs)
            elif crossover_OS_type == 'not':
                pass
            else:
                print('crossover_OS_type参数生成出错')
                return
            # 交叉OS ### end
            Mutation_P = Crossover_P*V_C_ratio
            # 变异MS ### begin
            if mutation_MS_type == 'not':
                pass
            elif mutation_MS_type == 'best':
                best_MS_mutations(new_MS, Mutation_P,
                                  MS_positions, best_MS,
                                  jobs_operations)
            elif mutation_MS_type == 'random':
                random_MS_mutations(new_MS, Mutation_P,
                                    MS_positions, candidate_machine_index,
                                    jobs_operations)
            else:
                print('mutation_MS_type参数生成出错')
                return
            # 变异MS ### end
            # 变异OS ### begin
            if mutation_OS_type == 'random':
                random_mutation(new_OS, Mutation_P)
            elif mutation_OS_type == 'not':
                pass
            else:
                print('mutation_OS_type参数生成出错')
                return
            # 变异OS ### end
            self.MS = new_MS
            self.OS = new_OS
            # 生成新种群 ### end
        # 繁殖一代 ### end


    def initial_new(self, size):
        '''
        自适应初始化
        '''
        self.best_MS = None
        self.best_OS = None
        self.best_score = float('inf')
        self.best_step = float('inf')
        self.size = size
        self.__initial_OS()

    def GA_new_no_numba(self, max_step=20000, memory_size=0.2, tournament_M=3, max_no_new_best=50):
        
        jobs_operations = self.data.jobs_operations
        decode_results = np.empty(
            shape=(1, self.size), dtype=np.int64).flatten()
        begin_time = np.empty(
            shape=(self.data.machines_num, self.data.jobs_num, jobs_operations.max()), dtype=np.int64)
        end_time = np.empty(
            shape=(self.data.machines_num, self.data.jobs_num, jobs_operations.max()), dtype=np.int64)
        begin_time_lists = np.empty(
            shape=(self.data.machines_num, jobs_operations), dtype=np.int64)
        end_time_lists = np.empty(
            shape=(self.data.machines_num, jobs_operations), dtype=np.int64)
        job_lists = np.empty(
            shape=(self.data.machines_num, jobs_operations), dtype=np.int64)
        operation_lists = np.empty(
            shape=(self.data.machines_num, jobs_operations), dtype=np.int64)
        product_operation = np.empty(
            shape=(1, self.data.jobs_num), dtype=np.int64).flatten()
        selected_machine = np.empty(
            shape=(self.data.jobs_num, jobs_operations.max()), dtype=np.int64)
        machine_operations = np.empty(
            shape=(1, self.data.machines_num), dtype=np.int64).flatten()
        result = np.empty(shape=(1, 1), dtype=np.int64).flatten()
        memory_OS = np.empty(
            shape=(math.ceil(self.size*memory_size), self.length), dtype=np.int64)
        # 记忆库的解
        memory_results = np.empty(
            shape=(1, math.ceil(self.size*memory_size)), dtype=np.int64).flatten()
        memory_results.fill(np.int64(200000000))
        Crossover_P = np.empty(shape=(1, self.size),
                               dtype=np.float64).flatten()
        no_new_best = 0
        for step in range(max_step):
            if max_no_new_best <= no_new_best:
                break
            decode_new(self.OS,
                       decode_results, np.arange(
                           self.size, dtype=np.int64).flatten(),
                       self.data.machines_num,
                       self.data.product_operation_detail,
                       self.data.candidate_machine, self.data.candidate_machine_index,
                       begin_time, end_time,
                       begin_time_lists,  end_time_lists,
                       job_lists, operation_lists,
                       product_operation,
                       selected_machine,
                       machine_operations,
                       result)
            best_poeple = np.argmin(decode_results)
            if self.best_score > decode_results[best_poeple]:
                self.best_OS = self.OS[best_poeple]
                self.best_score = decode_results[best_poeple]
                self.best_step = step+1
                no_new_best = 0
                print(self.best_step)
                print(self.best_score)
                print()
            else:
                no_new_best += 1
            # if no_new_best > 20:
            #     self.OS_fill(self.best_OS)
            if memory_size > 0:
                update_memory_lib_new(memory_OS, memory_results,
                                      self.OS, decode_results)
            new_OS = np.empty(shape=self.OS.shape, dtype=np.int64)
            recodes = np.empty(shape=(1, self.size), dtype=int).flatten()
            tournament_memory_new(self.OS,
                                  memory_OS,
                                  new_OS,
                                  tournament_M,
                                  decode_results, memory_results,
                                  step, max_step
                                  )
            get_crossover_P(decode_results, self.best_score, Crossover_P)
            # Crossover_P.fill(1)
            jobs = np.array([range(self.data.jobs_num)],
                            dtype=np.int64).flatten()
            POX_crossover(new_OS, Crossover_P, jobs)
            random_mutation(new_OS, Crossover_P)
            self.OS = new_OS

    def __get_data(self, MS, OS):
        # 初始化索引列表
        columns_index = ['machine_num', 'job_num',
                         'operation_num', 'begin_time', 'end_time']
        # 绘制甘特图的数据
        data = pd.DataFrame(data=None, columns=columns_index)
        # 解码 ### begin
        # 工件个数
        jobs_num = self.data.jobs_num
        # 机器数量
        machines_num = self.data.machines_num
        # 各工件的工序数
        jobs_operations = self.data.jobs_operations
        # 最大工序数
        max_operations = jobs_operations.max()
        # 各个工序的详细信息矩阵
        jobs_operations_detail = self.data.jobs_operations_detail
        # 各个工序的候选机器矩阵
        candidate_machine = self.data.candidate_machine
        # 记录最短时间的辅助列表
        result = np.empty(shape=(1, 1), dtype=np.int32).flatten()
        # 选用机器矩阵
        selected_machine = np.empty(
            shape=(jobs_num, max_operations), dtype=np.int32)
        # 选用机器时间矩阵
        selected_machine_time = np.empty(
            shape=(jobs_num, max_operations), dtype=np.int32)
        # 开工时间矩阵
        begin_time = np.empty(
            shape=(machines_num, jobs_num, max_operations), dtype=np.int32)
        # 结束时间矩阵
        end_time = np.empty(
            shape=(machines_num, jobs_num, max_operations), dtype=np.int32)
        # 记录工件加工到哪道工序的矩阵
        jobs_operation = np.empty(
            shape=(1, jobs_num), dtype=np.int32).flatten()
        # 记录机器是否开始加工过的矩阵
        machine_operationed = np.empty(
            shape=(1, machines_num), dtype=np.int32).flatten()
        # 各机器开始时间矩阵
        begin_time_lists = np.empty(
            shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
        # 各机器结束时间矩阵
        end_time_lists = np.empty(
            shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
        # 记录各机器的工件列表
        job_lists = np.empty(
            shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
        # 记录各机器的工序列表
        operation_lists = np.empty(
            shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
        # 记录机器加工的步骤数
        machine_operations = np.empty(
            shape=(1, machines_num), dtype=np.int32).flatten()
        decode_one(MS, OS,
                   jobs_operations, jobs_operations_detail,
                   begin_time, end_time,
                   selected_machine_time, selected_machine,
                   jobs_operation, machine_operationed, machine_operations,
                   begin_time_lists,  end_time_lists,
                   job_lists, operation_lists,
                   candidate_machine, result)
        # 解码 ### end
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

    def show_gantt_chart(self, MS, OS, figsize=(16, 4)):
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
            plt.text(x=row['begin_time'], y=row['machine_num']-0.2, s='{}-{}'.format(
                row['job_num'], row['operation_num']), rotation=90)
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
