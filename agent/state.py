import numpy as np

from mec_env.base_station_set import BaseStationSet
from mec_env.base_station import BaseStation
from mec_env.mobile_device import MobileDevice


class State:
    def __init__(self, mobile_device: MobileDevice, base_station_set: BaseStationSet):
        self.mobile_device = mobile_device
        self.mobile_device_id = mobile_device.mobile_device_id

        self.base_station_list = base_station_set.base_station_list
        self.mobile_device_computing_ability = mobile_device.computing_ability_now  # for BS_com_ability_list
        self.mobile_device_list = base_station_set.all_mobile_device_list  # for other task size and mask
        self.task_data_size = mobile_device.task.task_data_size
        self.task_tolerance_delay = mobile_device.task.task_tolerance_delay
        self.task_data_index_list = base_station_set.base_station_set_config.task_config.task_data_index_list

        self.mobile_device_task_queue_current_data_size = mobile_device.task_queue_current_data_size
        self.mobile_device_task_queue_size_max = mobile_device.task_queue_size_max
        self.base_station_task_current_sum_process_time_list = []
        self.base_station_task_queue_size_max_list = [base_station.task_queue_size_max for base_station
                                                      in self.base_station_list]

        self.base_station_set_computing_ability_list = self.get_base_station_set_computing_ability_list(
            self.base_station_list)

        self.all_task_size_list, self.task_size_mask_list = self.get_other_task_size(self.mobile_device_list,
                                                                                     self.mobile_device_id)
        self.transmitting_time_to_all_base_station_list = self.mobile_device.transmitting_time_to_all_base_station.tolist()
        self.base_station_task_current_sum_process_time_list = self.get_base_station_task_current_sum_process_time_list(
            self.base_station_list)
        self.last_base_station_offload_choice = mobile_device.last_base_station_offload_choice

    def get_base_station_set_computing_ability_list(self, base_station_list):
        base_station_set_computing_ability_list = []
        for base_station in base_station_list:
            base_station_set_computing_ability_list.append(base_station.computing_ability_now)
        return base_station_set_computing_ability_list

    def get_other_task_size(self, mobile_device_list, mobile_device_id):
        all_task_size_list = []
        task_size_mask_list = []
        mask_item = 0
        for idx, mobile_device in enumerate(mobile_device_list):
            if self.task_data_index_list[mobile_device_id] == idx:
                mask_item = 1
            else:
                mask_item = 0
            all_task_size_list.append(mobile_device_list[self.task_data_index_list.index(idx)].task.task_data_size)
            task_size_mask_list.append(mask_item)
        return all_task_size_list, task_size_mask_list

    def get_base_station_task_current_sum_process_time_list(self, base_station_list):
        base_station_task_current_sum_process_time_list = []
        for base_station in base_station_list:
            task_current_sum_process_time = base_station.task_queue.get_task_current_sum_process_time()
            base_station_task_current_sum_process_time_list.append(task_current_sum_process_time)
        return base_station_task_current_sum_process_time_list

    def get_state_list(self):
        state_list = []
        base_station_set_computing_ability_list = self.get_base_station_set_computing_ability_list(
            self.base_station_list)
        state_list.extend(base_station_set_computing_ability_list)
        state_list.append(self.mobile_device_computing_ability)
        state_list.append(self.task_data_size)
        state_list.append(self.task_tolerance_delay)
        self.all_task_size_list, self.task_size_mask_list = self.get_other_task_size(self.mobile_device_list,
                                                                                     self.mobile_device_id)
        self.transmitting_time_to_all_base_station_list = self.mobile_device.transmitting_time_to_all_base_station.tolist()
        self.base_station_task_current_sum_process_time_list = self.get_base_station_task_current_sum_process_time_list(
            self.base_station_list)

        state_list += self.all_task_size_list + self.task_size_mask_list + self.transmitting_time_to_all_base_station_list + self.base_station_task_current_sum_process_time_list
        self.last_base_station_offload_choice = self.mobile_device.last_base_station_offload_choice
        state_list.append(self.last_base_station_offload_choice)

        return state_list

    def get_normalized_state_array(self):
        state_array = self.get_state_array()
        normalized_state_array = np.zeros_like(state_array)

        base_station_computing_ability_max = self.base_station_list[0].computing_ability_max
        normalized_state_array[:2] = state_array[:2] / base_station_computing_ability_max

        mobile_device_computing_ability_max = self.mobile_device_list[0].computing_ability_max
        normalized_state_array[2] = state_array[2] / mobile_device_computing_ability_max

        task_data_size_max = self.mobile_device.task.task_data_size_max
        normalized_state_array[3] = state_array[3] / task_data_size_max
        normalized_state_array[4] = state_array[4] / self.mobile_device.task.task_tolerance_delay_max
        normalized_state_array[5:8] = state_array[5:8] / task_data_size_max
        normalized_state_array[8:11] = state_array[8:11]

        normalized_state_array[11:13] = state_array[11:13] / self.mobile_device_list[0].transmitting_time_to_base_station_max

        normalized_state_array[-3:] = state_array[-3:]

        return normalized_state_array

    def get_state_array(self):
        import numpy as np
        return np.array(self.get_state_list())
