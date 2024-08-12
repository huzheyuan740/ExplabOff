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
        self.base_station_num = 3
        self.mobile_device_num = 7
        self.time_step_max = 1

        self.base_station_config = BaseStationConfig()
        self.mobile_device_config = MobileDeviceConfig()
        self.task_config = TaskConfig()


class BaseStationConfig:
    def __init__(self):
        self.base_station_computing_ability = 6.0
        self.base_station_computing_ability_list = [10.0, 19.0, 26.0]
        self.base_station_computing_ability_eval = 8.8
        self.base_station_computing_ability_eval_list = [8.6]  # [8.8 * 10 ** 9, 9.3 * 10 ** 9]
        self.base_station_computing_ability_max = 25.0
        self.base_station_computing_ability_eval_max = 18
        self.base_station_energy = 200.0
        self.base_station_height = 20.0
        self.task_queue_size_max = 100


class MobileDeviceConfig:
    def __init__(self):
        self.mobile_device_ability = 1.0
        self.mobile_device_ability_max = 3.0
        self.queue_time = 0.0
        self.transmitting_time_to_all_base_station_array = np.array([
            [0.3011, 0.303, 0.302],
            [0.3021, 0.302, 0.301],
            [0.3031, 0.301, 0.302],
            [0.3021, 0.302, 0.303],
            [0.3031, 0.301, 0.302],
            [0.3021, 0.302, 0.301],
            [0.3031, 0.301, 0.302],
        ])
        self.transmitting_time_to_base_station_max = 0.5
        self.user_equipment_energy = (10 ** -27) * ((1 * 10 ** 9) ** 2) * 900000 * 1000 * 10
        self.task_queue_size_max = 100


class TaskConfig:
    def __init__(self):
        self.task_data_size_min = 0
        self.task_data_size_max = 20
        self.task_data_size_now = [6 + 1, 5 + 1, 2 + 1, 3 + 1, 4 + 1, 3.5 + 1, 4.5 + 1]
        self.origin_task_data_size_now = copy.deepcopy(self.task_data_size_now)
        self.task_data_index_list = list(range(len(self.task_data_size_now)))
        self.task_data_size_now_eval = [7]
        self.task_date_size_std = [0.1]
        self.task_date_size_std_eval = [0.1, 0.2]
        self.task_date_size_std_max = 0.5
        self.task_date_size_std_min = 0
        self.task_switch_time_matrix_on_base_station = np.array([
            [0.101, 0.102, 0.103],
            [0.103, 0.104, 0.102],
            [0.105, 0.106, 0.104],
            [0.101, 0.102, 0.103],
            [0.103, 0.104, 0.105],
            [0.103, 0.104, 0.102],
            [0.105, 0.106, 0.104],
        ])
        self.task_tolerance_delay_list = [2.208, 2.206, 2.205, 2.206, 2.205, 2.206, 2.205]
        self.task_tolerance_delay_max = 5


class CostConfig:
    def __init__(self):
        self.time_cost_weight = 1


class RewardConfig:
    def __init__(self):
        self.cost_to_reward = -10
        self.init_reward = 0  #
        self.mean_reward = self.init_reward
        self.lowest_reward = -10000
        self.adjust_bias_for_normal = 60


class EnvInterfaceConfig:
    def __init__(self):
        self.cost_config = CostConfig()
        self.reward_config = RewardConfig()


class StateConfig:
    def __init__(self):
        self.control_config = ControlConfig()
        self.train_config = TrainConfig()
        self.dimension = 9


class ActionConfig:
    def __init__(self):
        self.control_config = ControlConfig()
        self.train_config = TrainConfig()
        self.dimension = 2
        self.action_noise = np.random.uniform(0, 1, self.dimension)
        self.action_noise_decay = 0.995
        self.threshold_to_offload = 0.0


class TorchConfig:
    def __init__(self):
        self.gamma = 0.98
        self.hidden_sizes = (128, 128)
        self.buffer_size = int(4e3)
        self.max_seq_length = 50
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
        self.base_station_set_config = BaseStationSetConfig()
        self.interface_config = EnvInterfaceConfig()
