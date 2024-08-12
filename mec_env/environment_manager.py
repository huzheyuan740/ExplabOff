import copy
import numpy as np
from config import GlobalConfig
from agent.state import State
from agent.action import Action
from mec_env.base_station_set import BaseStationSet


# 负责与环境交互
class EnvironmentManager:
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self.base_station_set_config = global_config.base_station_set_config
        self.task_config = self.base_station_set_config.task_config
        self.interface_config = global_config.interface_config

        self.writer = None
        self.step_real_count = None
        self.episode_num_now = 0
        self.is_save_json = global_config.train_config.is_save_json
        self.load_model_path = global_config.train_config.load_model_path
        self.load_model_name = global_config.train_config.load_model_name

        self.base_station_set = BaseStationSet(self.global_config)
        self.cost_array_all_available_step_in_episode = np.zeros(
            (self.global_config.train_config.step_num, self.base_station_set.mobile_device_num))  # 记录所有当前episode
        self.reward_array_all_available_step_in_episode = np.zeros(
            (self.global_config.train_config.step_num, self.base_station_set.mobile_device_num))

    def reset(self):  # 只对 base_station_set 一个对象操作
        self.base_station_set = BaseStationSet(self.global_config)
        # self.next_base_station_set = copy.deepcopy(self.base_station_set)

        self.base_station_set.shuffle_task_size_list()
        self.base_station_set.update_all_mobile_device_message()
        self.cost_array_all_available_step_in_episode = np.zeros(
            (self.global_config.train_config.step_num, self.base_station_set.mobile_device_num))
        self.reward_array_all_available_step_in_episode = np.zeros(
            (self.global_config.train_config.step_num, self.base_station_set.mobile_device_num))

    def create_task_per_step(self):
        self.base_station_set.update_all_mobile_device_message()
        for base_station in self.base_station_set.base_station_list:
            base_station.priority_task_list.clear()

    def step(self, state_class_list, action_class_list, step_count):
        assert len(state_class_list) == len(action_class_list)
        cost_array = np.zeros(len(state_class_list))
        reward_array = np.zeros(len(state_class_list))
        done_list = [False for _ in range(len(state_class_list))]
        info = {}
        done = False
        for idx, each_mobile_device_state in enumerate(state_class_list):
            cur_mobile_device_offload_choice_idx = action_class_list[idx].offload_choice_idx
            if cur_mobile_device_offload_choice_idx < 0 or cur_mobile_device_offload_choice_idx == self.base_station_set.base_station_num:
                print("cur_mobile_device_offload_choice_idx:", cur_mobile_device_offload_choice_idx)
                print("error!")
                exit()
            transmit_time = each_mobile_device_state.transmitting_time_to_all_base_station_list[
                cur_mobile_device_offload_choice_idx]
            task = self.base_station_set.all_mobile_device_list[idx].task
            assert task.task_from_mobile_device_id == idx
            self.base_station_set.base_station_list[cur_mobile_device_offload_choice_idx].priority_task_list.append(
                {'transmit_time': transmit_time, 'task': task})  # BS receive MDs' task
            # print("res:", self.base_station_set.base_station_list[cur_mobile_device_offload_choice_idx].priority_task_list)
            self.base_station_set.all_mobile_device_list[
                idx].last_base_station_offload_choice = cur_mobile_device_offload_choice_idx

        for idx, base_station in enumerate(self.base_station_set.base_station_list):
            base_station.priority_task_list.sort(key=lambda x: x['transmit_time'])  # each BS sort by the priority_task_list
            for task_info in base_station.priority_task_list:
                task = task_info['task']
                base_station.task_queue.shared_task_execute_queue.append(
                    task)
                offload_task_percentage = action_class_list[
                    task.task_from_mobile_device_id].offload_task_percentage

                if 1 - offload_task_percentage > 0:  # local
                    local_time = task.task_data_size * (1 - offload_task_percentage) / \
                                 self.base_station_set.all_mobile_device_list[
                                     task.task_from_mobile_device_id].computing_ability_now
                    task.task_local_finish_time = local_time
                if offload_task_percentage > 0:  # edge
                    task_switch_time = 0
                    if not base_station.task_queue.cur_task_s_mobile_device_appeared_in_queue(task):
                        task_switch_time = \
                            self.task_config.task_switch_time_matrix_on_base_station[task.task_from_mobile_device_id][
                                base_station.base_station_id]
                        assert task_switch_time == task.task_switch_time_list_on_base_station[
                            base_station.base_station_id]
                    task.task_offload_finish_time += task_info['transmit_time']
                    task_exe_time = task.task_data_size * offload_task_percentage / base_station.computing_ability_now
                    task.task_current_process_time_in_queue = task_switch_time + task_exe_time
                    task_current_sum_process_time = base_station.task_queue.get_task_current_sum_process_time()
                    task.task_offload_finish_time += task_current_sum_process_time

                task_total_time = max(task.task_local_finish_time, task.task_offload_finish_time)
                cost = self.interface_config.cost_config.time_cost_weight * task_total_time
                reward = self.interface_config.reward_config.init_reward
                done = False
                cost_array[task.task_from_mobile_device_id] = cost
                if task_total_time > task.task_tolerance_delay:
                    reward = 0
                    reward += self.cost_to_reward_bad(cost)
                    done = True
                else:
                    reward = 0
                    reward += self.cost_to_reward_add_adjust_bias_normal(cost)
                if step_count == self.global_config.train_config.step_num - 1:
                    done = True
                reward_array[task.task_from_mobile_device_id] = reward
                done_list[task.task_from_mobile_device_id] = done
                # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            base_station.task_queue.update_task_sum_process_time(self.base_station_set_config.time_step_max)

        self.cost_array_all_available_step_in_episode[step_count] = cost_array
        self.reward_array_all_available_step_in_episode[step_count] = reward_array
        next_state_class_list = []
        for mobile_device_id in range(len(self.base_station_set.all_mobile_device_list)):
            each_state = self.get_state_per_mobile_device(mobile_device_id)
            next_state_class_list.append(each_state)
            next_state_list = each_state.get_state_list()
        print("reward_array:", reward_array)
        reward_array = np.zeros_like(reward_array)
        cost_array_max = cost_array
        if any(done_list) or step_count == self.global_config.train_config.step_num - 1:
            cost_array, reward_array, cost_array_max = self.cost_reward_function_v2(reward_array, step_count)
        return next_state_class_list, cost_array, reward_array, cost_array_max, done_list, info

    # def cost_to_reward_add_adjust_bias_all(self, cost):
    #     reward = self.interface_config.reward_config.cost_to_reward * cost + self.interface_config.reward_config.adjust_bias
    #     return reward

    def cost_to_reward_add_adjust_bias_normal(self, cost):
        # reward = self.interface_config.reward_config.cost_to_reward * (cost**2) + self.interface_config.reward_config.adjust_bias_for_normal
        reward = self.interface_config.reward_config.adjust_bias_for_normal * (cost ** (-2))
        return reward

    def cost_to_reward_bad(self, cost):
        reward = self.interface_config.reward_config.cost_to_reward * cost
        return reward

    def cost_reward_function_v2(self, reward_array, step_count):
        cost_array_mean_in_step = np.mean(self.cost_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_mean_in_step = np.mean(self.reward_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_in_last_step = self.reward_array_all_available_step_in_episode[step_count]
        cost_array_in_last_step = self.cost_array_all_available_step_in_episode[step_count]

        cost_array_mean_in_step_in_MD = np.ones_like(cost_array_mean_in_step)
        cost_array_max_in_step_in_MD = np.ones_like(cost_array_mean_in_step)
        reward_array_mean_in_step_in_MD = np.ones_like(reward_array_mean_in_step)

        cost_mean = None
        cost_max = None
        if step_count == self.global_config.train_config.step_num - 1:
            cost_mean = np.mean(cost_array_mean_in_step)
            cost_max = np.max(cost_array_mean_in_step)
            reward_mean = self.cost_to_reward_add_adjust_bias_normal(cost_mean)
        else:
            reward_array_in_last_step = np.where(reward_array_in_last_step > 0, 0, reward_array_in_last_step)
            reward_mean = np.mean(reward_array_in_last_step)
            cost_mean = np.max(cost_array_in_last_step)
            cost_max = np.max(cost_array_in_last_step)
        cost_array_mean_in_step_in_MD = cost_mean * cost_array_mean_in_step_in_MD
        cost_array_max_in_step_in_MD = cost_max * cost_array_mean_in_step_in_MD
        reward_array_mean_in_step_in_MD = reward_mean * reward_array_mean_in_step_in_MD
        # reward_array = reward_array + self.global_config.interface_config.reward_config.cost_to_reward * cost_array_mean_in_step_in_MD
        return cost_array_mean_in_step_in_MD, reward_array_mean_in_step_in_MD, cost_array_max_in_step_in_MD

    def cost_reward_function(self, reward_array, step_count):
        cost_array_mean_in_step = np.mean(self.cost_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_mean_in_step = np.mean(self.reward_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_in_last_step = self.reward_array_all_available_step_in_episode[step_count]
        cost_array_mean_in_step_in_MD = np.ones_like(cost_array_mean_in_step)
        reward_array_mean_in_step_in_MD = np.ones_like(reward_array_mean_in_step)
        cost_mean = np.mean(cost_array_mean_in_step)
        if step_count == self.global_config.train_config.step_num - 1:
            reward_mean = np.mean(reward_array_mean_in_step)
        else:
            reward_array_in_last_step = np.where(reward_array_in_last_step > 0, 0, reward_array_in_last_step)
            reward_mean = np.mean(reward_array_in_last_step)
        cost_array_mean_in_step_in_MD = cost_mean * cost_array_mean_in_step_in_MD
        reward_array_mean_in_step_in_MD = reward_mean * reward_array_mean_in_step_in_MD
        # reward_array = reward_array + self.global_config.interface_config.reward_config.cost_to_reward * cost_array_mean_in_step_in_MD
        return cost_array_mean_in_step_in_MD, reward_array_mean_in_step_in_MD

    def cost_reward_function_last_step(self, reward_array, step_count):
        cost_array_mean_in_step = np.mean(self.cost_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_in_last_step = self.reward_array_all_available_step_in_episode[step_count]
        cost_array_mean_in_step_in_MD = np.ones_like(cost_array_mean_in_step)
        reward_array_in_last_step_in_MD = np.ones_like(reward_array_in_last_step)
        cost_mean = np.mean(cost_array_mean_in_step)
        if step_count == self.global_config.train_config.step_num - 1:
            reward_mean = np.mean(reward_array_in_last_step)
        else:
            reward_array_in_last_step = np.where(reward_array_in_last_step > 0, 0, reward_array_in_last_step)
            reward_mean = np.mean(reward_array_in_last_step)
        cost_array_mean_in_step_in_MD = cost_mean * cost_array_mean_in_step_in_MD
        reward_array_mean_in_step_in_MD = reward_mean * reward_array_in_last_step_in_MD
        # reward_array = reward_array + self.global_config.interface_config.reward_config.cost_to_reward * cost_array_mean_in_step_in_MD
        return cost_array_mean_in_step_in_MD, reward_array_mean_in_step_in_MD

    def get_state_per_mobile_device(self, mobile_device_id):
        mobile_device = self.base_station_set.all_mobile_device_list[mobile_device_id]
        state = State(mobile_device, self.base_station_set)
        return state

    def get_random_action(self, global_config):
        action = Action([1, 7 / 8], global_config)
        return action
