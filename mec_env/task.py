from config import GlobalConfig
import numpy as np


class Task:
    def __init__(self, mobile_device_id, global_config: GlobalConfig):
        self.task_config = global_config.base_station_set_config.task_config
        self.task_from_mobile_device_id = mobile_device_id

        self.task_data_size = np.random.normal(self.task_config.task_data_size_now[mobile_device_id],
                                               self.task_config.task_date_size_std).item()
        self.task_data_size_max = self.task_config.task_data_size_max
        # self.task_current_data_size_in_queue = self.task_data_size
        self.task_current_process_time_in_queue = 0

        self.task_local_finish_time = 0
        self.task_offload_finish_time = 0
        self.task_tolerance_delay = self.task_config.task_tolerance_delay_list[mobile_device_id]
        self.task_tolerance_delay_max = self.task_config.task_tolerance_delay_max

        self.step_count_begin = -1
        self.task_switch_time_list_on_base_station = self.task_config.task_switch_time_matrix_on_base_station[mobile_device_id]
        # print("__init__task_size:", self.task_data_size)
        # print("__init__task_tolerance_delay:", self.task_tolerance_delay)
        # print("__init__task_from_mobile_device_id:", self.task_from_mobile_device_id)

    def get_task_info_list(self):
        # print("__function__task_size:", self.task_data_size)
        # print("__function__task_tolerance_delay:", self.task_tolerance_delay)
        # print("__function__task_from_mobile_device_id:", self.task_from_mobile_device_id)
        return [self.task_data_size, self.task_tolerance_delay]
