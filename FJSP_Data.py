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
        '''
        显示信息
        '''
        print()
        print('一共有{}个零件,{}台机器,每个步骤只能在{}台机器上进行加工。'.format(
            self.jobs_num, self.machines_num, self.max_machine_per_operation))
        if show_origial_data:
            print('原始数据为:', self.origial_data[0])
        print()
        print('零件的编号列表:', self.jobs_id)
        print(self.jobs_operations.shape)
        for i in range(len(self.jobs_id)):
            id = self.jobs_id[i]
            print('零件{}有{}道工序:'.format(id, self.jobs_operations[i]))
            if show_origial_data:
                print('原始数据为:')
                print(self.origial_data[i+1])
            print('零件工序详细信息矩阵为:')
            print(self.jobs_operations_detile[i][:][:])
            print()
