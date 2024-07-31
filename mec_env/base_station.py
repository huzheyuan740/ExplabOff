from mec_env.queue import TaskQueue
from config import GlobalConfig


class BaseStation:
    def __init__(self, base_station_id, global_config: GlobalConfig) -> None:
        self.base_station_config = global_config.base_station_set_config.base_station_config
        self.base_station_id = base_station_id
        # self.belong_base_station_set = belong_base_station_set  # 感觉用不到

        self.computing_ability_max = self.base_station_config.base_station_computing_ability_max
        self.computing_ability_now = self.computing_ability_max

        self.global_config = global_config
        self.priority_task_list = []  # 当前step传到本BS的任务依上传时间排序的优先队列 每个step需要清空
        self.task_queue = TaskQueue(self, global_config)
        # 下面这两行感觉可以在TaskQueue类中进行功能实现
        self.task_queue_current_data_size = 0
        self.task_queue_size_max = self.base_station_config.task_queue_size_max


