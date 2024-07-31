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
        # 初始化结果列表 对应结果的次序按mobile_device_state idx对应的次序
        cost_array = np.zeros(len(state_class_list))
        reward_array = np.zeros(len(state_class_list))
        done_list = [False for _ in range(len(state_class_list))]
        info = {}
        done = False
        for idx, each_mobile_device_state in enumerate(state_class_list):
            cur_mobile_device_offload_choice_idx = action_class_list[idx].offload_choice_idx
            if cur_mobile_device_offload_choice_idx < 0 or cur_mobile_device_offload_choice_idx == self.base_station_set.base_station_num:
                print("cur_mobile_device_offload_choice_idx:", cur_mobile_device_offload_choice_idx)
                print("Boom!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                exit()
            transmit_time = each_mobile_device_state.transmitting_time_to_all_base_station_list[
                cur_mobile_device_offload_choice_idx]
            task = self.base_station_set.all_mobile_device_list[idx].task
            assert task.task_from_mobile_device_id == idx
            self.base_station_set.base_station_list[cur_mobile_device_offload_choice_idx].priority_task_list.append(
                {'transmit_time': transmit_time, 'task': task})  # BS 接受 MD产生的任务
            # print("res:", self.base_station_set.base_station_list[cur_mobile_device_offload_choice_idx].priority_task_list)
            self.base_station_set.all_mobile_device_list[
                idx].last_base_station_offload_choice = cur_mobile_device_offload_choice_idx

        for idx, base_station in enumerate(self.base_station_set.base_station_list):
            base_station.priority_task_list.sort(key=lambda x: x['transmit_time'])  # 每个BS的priority_task_list根据传输时间排序
            for task_info in base_station.priority_task_list:  # task_info指包含上传到选择的BS的传输时间和对应task对象的总体任务信息
                task = task_info['task']
                # print("queue_item:",  base_station.task_queue.item)
                base_station.task_queue.shared_task_execute_queue.append(
                    task)
                # 从这里正式开始对于存入shared_task_execute_queue的每个任务 BS对其进行计算
                offload_task_percentage = action_class_list[
                    task.task_from_mobile_device_id].offload_task_percentage  # offload_task_percentage记录的是当前task卸载/本地分配的百分比

                if 1 - offload_task_percentage > 0:  # 本地计算部分的task的相关操作
                    local_time = task.task_data_size * (1 - offload_task_percentage) / \
                                 self.base_station_set.all_mobile_device_list[
                                     task.task_from_mobile_device_id].computing_ability_now
                    task.task_local_finish_time = local_time
                    # print("task.task_local_finish_time:", task.task_local_finish_time)
                    # TODO 可能后续要加入本地的任务计算队列
                if offload_task_percentage > 0:  # 边缘计算部分的task的相关操作
                    task_switch_time = 0
                    if not base_station.task_queue.cur_task_s_mobile_device_appeared_in_queue(task):
                        task_switch_time = \
                            self.task_config.task_switch_time_matrix_on_base_station[task.task_from_mobile_device_id][
                                base_station.base_station_id]
                        assert task_switch_time == task.task_switch_time_list_on_base_station[
                            base_station.base_station_id]  # 这个base_station.base_station_id必是这个任务的offload动作选择
                    task.task_offload_finish_time += task_info['transmit_time']  # 加上数据传输时间
                    # print("transmit_time:", task_info['transmit_time'])
                    # 以下开始计算任务执行时间
                    task_exe_time = task.task_data_size * offload_task_percentage / base_station.computing_ability_now
                    # print("off_data_size:", task.task_data_size * offload_task_percentage)
                    task.task_current_process_time_in_queue = task_switch_time + task_exe_time
                    # print("task_switch_time：{}，task_exe_time:{}".format(task_switch_time, task_exe_time))
                    task_current_sum_process_time = base_station.task_queue.get_task_current_sum_process_time()  # task_current_sum_process_time包含了当前加入的这个任务的数据量以及之前没处理完的任务的处理时间
                    # 上面这个值超过时间片大小 会在队列中 累计到下一时间片 所以说我事先算出了这个step下该任务的完成时间 但是这个任务在实际情况下可能是之后几个时间片这个任务才会完成
                    # TODO 这个问题我感觉主要是在cost和reward的影响上 所以感觉是要具体看cost和reward是要想怎么设置了
                    # print("task_current_sum_process_time:", task_current_sum_process_time)
                    task.task_offload_finish_time += task_current_sum_process_time
                    # print("task.task_offload_finish_time:", task.task_offload_finish_time)

                task_total_time = max(task.task_local_finish_time, task.task_offload_finish_time)
                cost = self.interface_config.cost_config.time_cost_weight * task_total_time
                reward = self.interface_config.reward_config.init_reward  # TODO 查看gym中有关的处理 具体的reward设置需要仔细考虑 (是负数关系/倒数关系？)
                done = False
                cost_array[task.task_from_mobile_device_id] = cost
                if task_total_time > task.task_tolerance_delay:
                    # print("task_total_time:", task_total_time)
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

            # update_queue只有当前step遍历完当前BS队列中的所有任务再进行更新
            base_station.task_queue.update_task_sum_process_time(self.base_station_set_config.time_step_max)

        self.cost_array_all_available_step_in_episode[step_count] = cost_array
        self.reward_array_all_available_step_in_episode[step_count] = reward_array
        next_state_class_list = []
        for mobile_device_id in range(len(self.base_station_set.all_mobile_device_list)):
            each_state = self.get_state_per_mobile_device(mobile_device_id)  # 只有调用这个方法才能得到state对象
            next_state_class_list.append(each_state)
            next_state_list = each_state.get_state_list()
            # print("next_state_list:", next_state_list)
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
        # print("self.cost_array_all_available_step_in_episode:\n", self.cost_array_all_available_step_in_episode)
        # print("self.reward_array_all_available_step_in_episode:\n", self.reward_array_all_available_step_in_episode)
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
            # reward_mean = np.mean(reward_array_mean_in_step)  #
            reward_mean = self.cost_to_reward_add_adjust_bias_normal(cost_mean)
        else:
            reward_array_in_last_step = np.where(reward_array_in_last_step > 0, 0, reward_array_in_last_step)
            # print("reward_array_mean_in_step:", reward_array_in_last_step)
            reward_mean = np.mean(reward_array_in_last_step)
            cost_mean = np.max(cost_array_in_last_step)
            cost_max = np.max(cost_array_in_last_step)
        cost_array_mean_in_step_in_MD = cost_mean * cost_array_mean_in_step_in_MD
        cost_array_max_in_step_in_MD = cost_max * cost_array_mean_in_step_in_MD
        reward_array_mean_in_step_in_MD = reward_mean * reward_array_mean_in_step_in_MD
        # reward_array = reward_array + self.global_config.interface_config.reward_config.cost_to_reward * cost_array_mean_in_step_in_MD
        return cost_array_mean_in_step_in_MD, reward_array_mean_in_step_in_MD, cost_array_max_in_step_in_MD

    def cost_reward_function(self, reward_array, step_count):
        # print("self.cost_array_all_available_step_in_episode:\n", self.cost_array_all_available_step_in_episode)
        # print("self.reward_array_all_available_step_in_episode:\n", self.reward_array_all_available_step_in_episode)
        cost_array_mean_in_step = np.mean(self.cost_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_mean_in_step = np.mean(self.reward_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_in_last_step = self.reward_array_all_available_step_in_episode[step_count]
        cost_array_mean_in_step_in_MD = np.ones_like(cost_array_mean_in_step)
        reward_array_mean_in_step_in_MD = np.ones_like(reward_array_mean_in_step)
        cost_mean = np.mean(cost_array_mean_in_step)
        if step_count == self.global_config.train_config.step_num - 1:
            reward_mean = np.mean(reward_array_mean_in_step)  # 不能算平均 如果出现没跑满的情况 连带惩罚
        else:
            reward_array_in_last_step = np.where(reward_array_in_last_step > 0, 0, reward_array_in_last_step)
            # print("reward_array_mean_in_step:", reward_array_in_last_step)
            reward_mean = np.mean(reward_array_in_last_step)
        cost_array_mean_in_step_in_MD = cost_mean * cost_array_mean_in_step_in_MD
        reward_array_mean_in_step_in_MD = reward_mean * reward_array_mean_in_step_in_MD
        # reward_array = reward_array + self.global_config.interface_config.reward_config.cost_to_reward * cost_array_mean_in_step_in_MD
        return cost_array_mean_in_step_in_MD, reward_array_mean_in_step_in_MD

    def cost_reward_function_last_step(self, reward_array, step_count):
        # print("self.cost_array_all_available_step_in_episode:\n", self.cost_array_all_available_step_in_episode)
        # print("self.reward_array_all_available_step_in_episode:\n", self.reward_array_all_available_step_in_episode)
        cost_array_mean_in_step = np.mean(self.cost_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_in_last_step = self.reward_array_all_available_step_in_episode[step_count]
        cost_array_mean_in_step_in_MD = np.ones_like(cost_array_mean_in_step)
        reward_array_in_last_step_in_MD = np.ones_like(reward_array_in_last_step)
        cost_mean = np.mean(cost_array_mean_in_step)
        if step_count == self.global_config.train_config.step_num - 1:
            reward_mean = np.mean(reward_array_in_last_step)  # 不能算平均 如果出现没跑满的情况 连带惩罚
        else:
            reward_array_in_last_step = np.where(reward_array_in_last_step > 0, 0, reward_array_in_last_step)
            print("reward_array_in_last_step:", reward_array_in_last_step)
            reward_mean = np.mean(reward_array_in_last_step)
        cost_array_mean_in_step_in_MD = cost_mean * cost_array_mean_in_step_in_MD
        reward_array_mean_in_step_in_MD = reward_mean * reward_array_in_last_step_in_MD
        # reward_array = reward_array + self.global_config.interface_config.reward_config.cost_to_reward * cost_array_mean_in_step_in_MD
        print("reward_array_mean_in_step_in_MD:", reward_array_mean_in_step_in_MD)
        return cost_array_mean_in_step_in_MD, reward_array_mean_in_step_in_MD

    def get_state_per_mobile_device(self, mobile_device_id):
        mobile_device = self.base_station_set.all_mobile_device_list[mobile_device_id]
        state = State(mobile_device, self.base_station_set)
        return state

    def get_random_action(self, global_config):
        action = Action([1, 7 / 8], global_config)
        return action
