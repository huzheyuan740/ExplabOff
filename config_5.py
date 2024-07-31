import numpy as np
import copy


class TrainConfig:
    def __init__(self):
        self.algorithm = "ExplabOff"
        self.total_timesteps = 90e5
        self.step_num = 10
        self.episode_num = self.total_timesteps / self.step_num
        self.part = 1
        self.seed = 0
        self.gpu_index = 0
        self.tensorboard = True
        self.is_eval_mode = False
        self.dynamic_update_a_b_episode_begin = 300e3
        self.dynamic_update_a_b_episode_range = 1000
        self.dynamic_update_a_b_fluctuate = 27
        self.is_save_json = False
        self.load_model_path = None
        self.load_model_name = None


class BaseStationSetConfig:
    def __init__(self):
        self.base_station_num = 2
        self.mobile_device_num = 5
        self.time_step_max = 1  # 每个时间片的长度

        self.base_station_config = BaseStationConfig()
        self.mobile_device_config = MobileDeviceConfig()
        self.task_config = TaskConfig()


class BaseStationConfig:
    def __init__(self):
        self.base_station_computing_ability = 6.0
        self.base_station_computing_ability_list = [10.0, 19.0]  # 单位改为GHz
        self.base_station_computing_ability_eval = 8.8
        self.base_station_computing_ability_eval_list = [8.6]  # [8.8 * 10 ** 9, 9.3 * 10 ** 9]
        self.base_station_computing_ability_max = 25.0
        self.base_station_computing_ability_eval_max = 18
        self.base_station_energy = 200.0
        self.base_station_height = 20.0
        self.task_queue_size_max = 100  # TODO 这里需要根据任务的大小 MD设备的数量和时间片的数量等因素进行计算确定


class MobileDeviceConfig:
    def __init__(self):
        self.mobile_device_ability = 1.0
        self.mobile_device_ability_max = 3.0
        self.queue_time = 0.0
        # TODO 之后需要根据具体的计算进行调整
        self.transmitting_time_to_all_base_station_array = np.array([
            [0.3011, 0.303],
            [0.3021, 0.302],
            [0.3031, 0.301],
            [0.3021, 0.302],
            [0.3031, 0.301]
        ])
        self.transmitting_time_to_base_station_max = 0.5  # TODO 待定
        # 能量目前暂时用不到
        self.user_equipment_energy = (10 ** -27) * ((1 * 10 ** 9) ** 2) * 700000 * 1000 * 10
        self.task_queue_size_max = 100  # TODO 这里需要根据任务的大小 MD设备的数量和时间片的数量等因素进行计算确定


class TaskConfig:
    def __init__(self):
        self.task_data_size_min = 0
        self.task_data_size_max = 20
        self.task_data_size_now = [6 + 1, 5 + 1, 2 + 1, 3 + 1, 4 + 1]  # 任务的计算量的均值好和标准差改为量化的消耗计算能力的份数
        self.origin_task_data_size_now = copy.deepcopy(self.task_data_size_now)  #
        self.task_data_index_list = list(range(len(self.task_data_size_now)))
        self.task_data_size_now_eval = [7]
        self.task_date_size_std = [0.1]  # [200, 220]
        self.task_date_size_std_eval = [0.1, 0.2]
        self.task_date_size_std_max = 0.5
        self.task_date_size_std_min = 0
        self.task_switch_time_matrix_on_base_station = np.array([  # TODO 之后需要根据具体的计算进行修改
            [0.101, 0.102],
            [0.103, 0.104],
            [0.105, 0.106],
            [0.101, 0.102],
            [0.103, 0.104],
        ])
        self.task_tolerance_delay_list = [2.208, 2.206, 2.205, 2.206, 2.205]  # TODO 之后需要根据具体的计算进行修改
        # self.task_tolerance_delay_list = [1.808, 1.806, 1.805, 1.806, 1.805]  # TODO 之后需要根据具体的计算进行修改
        self.task_tolerance_delay_max = 5


class CostConfig:
    def __init__(self):
        self.time_cost_weight = 1


class RewardConfig:
    def __init__(self):
        # self.penalty_over_time = -1000
        self.cost_to_reward = -10
        self.init_reward = 0  #
        self.mean_reward = self.init_reward
        self.lowest_reward = -10000
        # self.adjust_bias = 10000  # 暂时用不到了 这是把不过是否惩罚的-reward全都平移到了大于0的区间
        self.adjust_bias_for_normal = 60  # 只对正常的 没违反约束的任务加入偏执奖励


class EnvInterfaceConfig:
    def __init__(self):
        self.cost_config = CostConfig()
        self.reward_config = RewardConfig()


class StateConfig:
    def __init__(self):
        self.control_config = ControlConfig()
        self.train_config = TrainConfig()
        self.dimension = 9  # (self.hexagon_network_config.user_equipment_num + 1) * 2  # 20
        # if self.train_config.algorithm == 'ddpg':
        #     self.dimension *= self.ue_num


class ActionConfig:
    def __init__(self):
        # self.hexagon_network_config = HexagonNetworkConfig()
        self.control_config = ControlConfig()
        self.train_config = TrainConfig()
        # self.ue_num = self.hexagon_network_config.user_equipment_num
        self.dimension = 2
        self.action_noise = np.random.uniform(0, 1, self.dimension)
        self.action_noise_decay = 0.995
        self.threshold_to_offload = 0.0  # 0.5


class TorchConfig:
    def __init__(self):
        self.gamma = 0.98
        self.hidden_sizes = (128, 128)
        self.buffer_size = int(4e3)  # int(4e3)
        self.max_seq_length = 50
        # self.buffer_size = int(64)
        self.batch_size = 4
        self.policy_learning_rate = 1e-4
        self.critic_learning_rate = 1e-3
        self.policy_gradient_clip = 0.5
        self.critic_gradient_clip = 1.0
        self.epsilon_max = 1.0
        self.epsilon_decay = 0.998
        self.action_limit = 1.0


class AgentConfig:
    def __init__(self):
        self.state_config = StateConfig()
        self.action_config = ActionConfig()
        self.torch_config = TorchConfig()


class DebugConfig:
    def __init__(self):
        self.whether_output_finish_episode_reason = 1
        self.whether_output_replay_buffer_message = 0


class ControlConfig:
    def __init__(self):
        self.save_runs = True
        self.save_save_model = True

        self.output_network_config = True
        self.output_action_config = True
        self.output_other_config = False

        self.easy_output_mode = True
        self.easy_output_cycle = 100
        self.env_without_D2D = True
        self.bandwidth_disturb = True


class GlobalConfig:
    def __init__(self):
        self.train_config = TrainConfig()
        self.agent_config = AgentConfig()
        self.base_station_set_config = BaseStationSetConfig()  # 相当于HexagonNetworkConfig()
        self.interface_config = EnvInterfaceConfig()