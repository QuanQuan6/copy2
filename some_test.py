
from read_data import read_data
from population import population
import numpy as np
from auxiliary import *

data_path = 'data.txt'
data_path = 'Monaldo\Fjsp\Job_Data\Brandimarte_Data\Text\Mk02.fjs'
data = read_data(data_path)
# data.display_info(True)


peoples = population(data, 200)
# 工件个数
jobs_num = peoples.data.jobs_num
# 机器数量
machines_num = peoples.data.machines_num
# 各工件的工序数
jobs_operations = peoples.data.jobs_operations
# 最大工序数
max_operations = jobs_operations.max()
# 各个工序的详细信息矩阵
jobs_operations_detail = peoples.data.jobs_operations_detail
# 各个工序的候选机器矩阵
candidate_machine = peoples.data.candidate_machine
# 各个工序的候选机器矩阵和加工时间的索引矩阵
candidate_machine_index = peoples.data.candidate_machine_index
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

job_lists = np.empty(
    shape=(machines_num, jobs_operations.sum()), dtype=np.int32)
operation_lists = np.empty(
    shape=(machines_num, jobs_operations.sum()), dtype=np.int32)

# 记录机器加工的步骤数
machine_operations = np.empty(
    shape=(1, machines_num), dtype=np.int32).flatten()
# 解码结果矩阵
decode_results = np.empty(
    shape=(1, peoples.size), dtype=np.int32).flatten()
# 初始化对应工序的MS码矩阵
MS_positions = np.empty(
    shape=(jobs_num, max_operations), dtype=np.int32)
initial_MS_position(MS_positions, jobs_operations)
# 初始化各个工序最短时间矩阵
best_MS = np.empty(shape=(jobs_num, max_operations), dtype=np.int32)
initial_best_MS(best_MS, jobs_operations_detail,
                jobs_operations, np.array(range(machines_num), dtype=np.int32).flatten())
# 初始化交叉概率
Crossover_P = np.empty(shape=(1, peoples.size),
                       dtype=np.float64).flatten()
# 记录最短时间的辅助列表
result = np.empty(shape=(1, 1), dtype=np.int32).flatten()


peoples.initial()

peoples.GA(max_step=200, max_no_new_best=200)
print(peoples.best_step)
print(peoples.best_score)
decode_one(peoples.best_MS, peoples.best_OS,
           jobs_operations, jobs_operations_detail,
           begin_time, end_time,
           selected_machine_time, selected_machine,
           jobs_operation, machine_operationed, machine_operations,
           begin_time_lists,  end_time_lists,
           job_lists, operation_lists,
           candidate_machine, result)
for machine in range(machines_num):
    print(machine)
    print(begin_time_lists[machine][:machine_operations[machine]])
    print(end_time_lists[machine][:machine_operations[machine]])
    print(job_lists[machine][:machine_operations[machine]])
    print(operation_lists[machine][:machine_operations[machine]])
    print()

peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
