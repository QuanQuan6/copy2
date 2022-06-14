import time
import os
import psutil


def count_time(function):
    '''
    记录函数运行时间
    '''
    def record_time():
        begin_time = time.time()
        function()
        end_time = time.time()
        print("{}运行了{}秒".format(str(function), end_time-begin_time))
    return record_time


def count_memory(function):
    '''
    记录函数占用的内存
    '''
    def record_memory():
        pid = os.getpid()
        p = psutil.Process(pid)
        memory_begin = p.memory_full_info.uss/1024
        function()
        memory_end = p.memory_full_info.uss/1024
        print("{}占用了了{}KB的内存".format(str(function), memory_begin-memory_end))
    return record_memory
