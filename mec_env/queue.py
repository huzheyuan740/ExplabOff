from queue import Queue
import threading
import time
import numpy as np
from mec_env import task
from config import GlobalConfig


class TaskQueue:
    def __init__(self, item, global_config: GlobalConfig):
        self.item = item  # which md
        self.shared_task_execute_queue = []
        self.seen_mobile_device_id = []

    def get_task_queue_top(self):
        pass

    def get_out_task_queue(self):
        pass

    def pop_finished_task(self):
        new_list = []
        for task_item in self.shared_task_execute_queue:
            if task_item.task_current_process_time_in_queue != 0:
                new_list.append(task_item)
        self.shared_task_execute_queue = new_list

    def put_task_queue(self):
        pass

    def clean_task_queue(self):
        pass

    def get_task_current_sum_process_time(self):
        task_current_sum_process_time = 0
        for task_item in self.shared_task_execute_queue:
            # print("id:{}, cur_p_time:{}".format(task_item.task_from_mobile_device_id, task_item.task_current_process_time_in_queue))
            task_current_sum_process_time += task_item.task_current_process_time_in_queue
        return task_current_sum_process_time

    def get_queue_wait_time(self):
        pass

    def update_task_sum_process_time(self, time_step_max):
        remain_task_process_time = time_step_max
        for task_item in self.shared_task_execute_queue:
            remain_task_process_time = task_item.task_current_process_time_in_queue - remain_task_process_time
            if remain_task_process_time <= 0:  # finish the task processed in current queue
                task_item.task_current_process_time_in_queue = 0
                remain_task_process_time = abs(remain_task_process_time)
            else:
                task_item.task_current_process_time_in_queue = remain_task_process_time
                self.pop_finished_task()
                return

    def update_task_queue_data_size(self, computed_task_size, computing_ability_now):
        remain_task_size = computed_task_size
        for task_item in self.shared_task_execute_queue:
            remain_task_size = task_item.task_current_data_size_in_queue - remain_task_size
            if remain_task_size <= 0:
                task_item.task_offload_finish_time += (
                        task_item.task_current_data_size_in_queue / computing_ability_now)
                task_item.task_current_data_size_in_queue = 0
            else:
                task_item.task_offload_finish_time += (
                        remain_task_size / computing_ability_now)
                task_item.task_current_data_size_in_queue -= remain_task_size
                return
            remain_task_size = abs(remain_task_size)

    def cur_task_s_mobile_device_appeared_in_queue(self, task):

        if task.task_from_mobile_device_id not in self.seen_mobile_device_id:
            self.seen_mobile_device_id.append(task.task_from_mobile_device_id)
            return False
        else:
            return True
