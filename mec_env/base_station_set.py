import numpy as np
import random
from config import GlobalConfig
from mec_env.base_station import BaseStation
from mec_env.mobile_device import MobileDevice


class BaseStationSet:
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self.base_station_set_config = global_config.base_station_set_config

        self.task_data_size_list = self.base_station_set_config.task_config.task_data_size_now
        self.task_tolerance_delay_list = self.base_station_set_config.task_config.task_tolerance_delay_list
        self.task_data_index_list = self.base_station_set_config.task_config.task_data_index_list

        self.base_station_list = []
        self.all_mobile_device_list = []

        self.base_station_num = self.base_station_set_config.base_station_num
        self.mobile_device_num = self.base_station_set_config.mobile_device_num
        self.base_station0 = BaseStation(0, global_config)
        self.base_station1 = BaseStation(1, global_config)
        self.base_station2 = BaseStation(2, global_config)
        # set BS' computational capacity
        self.base_station0.computing_ability_now = \
            self.base_station_set_config.base_station_config.base_station_computing_ability_list[0]
        self.base_station1.computing_ability_now = \
            self.base_station_set_config.base_station_config.base_station_computing_ability_list[1]
        # self.base_station2.computing_ability_now = \
        #     self.base_station_set_config.base_station_config.base_station_computing_ability_list[2]
        # 在这里建立MD
        self.mobile_device0 = MobileDevice(0, global_config)
        self.mobile_device1 = MobileDevice(1, global_config)
        self.mobile_device2 = MobileDevice(2, global_config)
        # self.mobile_device3 = MobileDevice(3, global_config)
        # self.mobile_device4 = MobileDevice(4, global_config)
        # self.mobile_device5 = MobileDevice(5, global_config)
        # self.mobile_device6 = MobileDevice(6, global_config)

        self.base_station_list = [self.base_station0, self.base_station1]
        # self.base_station_list = [self.base_station0, self.base_station1, self.base_station2]
        self.all_mobile_device_list = [self.mobile_device0, self.mobile_device1, self.mobile_device2]
        # self.all_mobile_device_list = [self.mobile_device0, self.mobile_device1, self.mobile_device2, self.mobile_device3, self.mobile_device4]
        # self.all_mobile_device_list = [self.mobile_device0, self.mobile_device1, self.mobile_device2,
        #                                self.mobile_device3, self.mobile_device4, self.mobile_device5,
        #                                self.mobile_device6]

        assert len(self.base_station_list) == self.base_station_num
        assert len(self.all_mobile_device_list) == self.mobile_device_num

    def update_state(self):
        pass

    def shuffle_task_size_list(self):
        assert len(self.task_data_size_list) == len(self.task_tolerance_delay_list)
        shuffled_list = random.sample(self.task_data_index_list, len(self.task_data_index_list))
        self.task_data_size_list = self.base_station_set_config.task_config.task_data_size_now = [
            self.task_data_size_list[self.task_data_index_list.index(i)] for i in shuffled_list]
        self.task_tolerance_delay_list = self.base_station_set_config.task_config.task_tolerance_delay_list = [
            self.task_tolerance_delay_list[self.task_data_index_list.index(i)] for i in shuffled_list]
        self.task_data_index_list = self.base_station_set_config.task_config.task_data_index_list = shuffled_list

    def update_all_mobile_device_message(self):
        for mobile_device_id, mobile_device in enumerate(self.all_mobile_device_list):
            mobile_device.create_task(mobile_device_id)

    def get_state_per_mobile_device(self):
        pass

    def draw_image(self):
        pass
