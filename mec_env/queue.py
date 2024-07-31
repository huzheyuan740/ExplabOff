from queue import Queue
import threading
import time
import numpy as np
from mec_env import task
from config import GlobalConfig


class TaskQueue:
    def __init__(self, item, global_config: GlobalConfig):
        self.item = item  # item_id为当前这个队列属于的哪个设备
        self.shared_task_execute_queue = []  # 这个queue目前想的是直接存入task对象, 一个epoch更新一次
        self.seen_mobile_device_id = []  # 记录当前队列曾经见过的MD设备的id, 一个epoch更新一次

    def get_task_queue_top(self):
        pass

    def get_out_task_queue(self):
        pass

    def pop_finished_task(self):  # TODO 对于出队列 执行完的任务需要与其最大容忍时延进行对比
        new_list = []
        for task_item in self.shared_task_execute_queue:
            if task_item.task_current_process_time_in_queue != 0:
                new_list.append(task_item)
        self.shared_task_execute_queue = new_list

    def put_task_queue(self):
        pass

    def clean_task_queue(self):
        pass

    # def get_current_task_queue_task_size(self):  # 统计当前加入的这个任务的数据量以及之前没处理完的任务的数据量
    #     current_task_queue_task_size = 0
    #     for task_item in self.shared_task_execute_queue:
    #         current_task_queue_task_size += task_item.task_current_data_size_in_queue
    #     return current_task_queue_task_size

    def get_task_current_sum_process_time(self):  # 统计当前加入的这个任务的数据量以及之前没处理完的任务的数据量
        task_current_sum_process_time = 0
        for task_item in self.shared_task_execute_queue:
            # print("id:{}, cur_p_time:{}".format(task_item.task_from_mobile_device_id, task_item.task_current_process_time_in_queue))
            task_current_sum_process_time += task_item.task_current_process_time_in_queue
        return task_current_sum_process_time

    def get_queue_wait_time(self):
        pass

    def update_task_sum_process_time(self, time_step_max):  # update queue这个操作感觉只有当前step遍历完当前BS队列中所有任务再进行更新
        remain_task_process_time = time_step_max
        for task_item in self.shared_task_execute_queue:
            remain_task_process_time = task_item.task_current_process_time_in_queue - remain_task_process_time
            if remain_task_process_time <= 0:  # 队列中的这个任务已经执行完了
                task_item.task_current_process_time_in_queue = 0
                remain_task_process_time = abs(remain_task_process_time)
                # TODO 晚了几个时间片才完成的任务如何考虑
            else:
                task_item.task_current_process_time_in_queue = remain_task_process_time
                self.pop_finished_task()  # 清除这些完成的任务(Task_item.task_current_process_time_in_queue = 0的任务)
                return

    def update_task_queue_data_size(self, computed_task_size, computing_ability_now):
        remain_task_size = computed_task_size
        for task_item in self.shared_task_execute_queue:
            remain_task_size = task_item.task_current_data_size_in_queue - remain_task_size
            if remain_task_size <= 0:  # 队列中的这个任务已经执行完了
                task_item.task_offload_finish_time += (
                        task_item.task_current_data_size_in_queue / computing_ability_now)
                task_item.task_current_data_size_in_queue = 0
                # TODO 之后统一使用pop_finished_task()操作 清除执行完成的任务 也就是task_item.task_current_data_size_in_queue = 0的任务
            else:  # 该任务还没执行完
                task_item.task_offload_finish_time += (
                        remain_task_size / computing_ability_now)
                task_item.task_current_data_size_in_queue -= remain_task_size
                return  # TODO 好像不能直接return
            remain_task_size = abs(remain_task_size)

    def cur_task_s_mobile_device_appeared_in_queue(self, task):
        # for task_item in self.shared_task_execute_queue:
        #     if task_item.task_from_mobile_device_id == task.task_from_mobile_device_id:
        #         return True
        # return False
        # 上面的方法可能会占用大量资源 暂时废弃

        if task.task_from_mobile_device_id not in self.seen_mobile_device_id:
            self.seen_mobile_device_id.append(task.task_from_mobile_device_id)
            return False
        else:
            return True
