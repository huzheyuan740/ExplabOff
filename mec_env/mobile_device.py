from config import GlobalConfig
from mec_env.task import Task
from mec_env.queue import TaskQueue


class MobileDevice:
    def __init__(self, mobile_device_id, global_config: GlobalConfig):
        self.mobile_device_id = mobile_device_id
        self.belong_base_station = None
        self.global_config = global_config
        self.mobile_device_config = global_config.base_station_set_config.mobile_device_config
        self.last_base_station_offload_choice = -1

        self.transmitting_time_to_all_base_station = \
            self.mobile_device_config.transmitting_time_to_all_base_station_array[self.mobile_device_id]
        self.transmitting_time_to_base_station_max = self.mobile_device_config.transmitting_time_to_base_station_max

        self.computing_ability_max = self.mobile_device_config.mobile_device_ability_max
        self.computing_ability_now = self.mobile_device_config.mobile_device_ability

        self.task = None
        self.task_queue = TaskQueue(self, global_config)

        self.task_queue_current_data_size = 0
        self.task_queue_size_max = self.mobile_device_config.task_queue_size_max

    def create_task(self, mobile_device_id):
        self.task = Task(mobile_device_id, self.global_config)

    def update_task(self, mobile_device_id):
        self.task = Task(mobile_device_id, self.global_config)
