from config import GlobalConfig


class Action:
    def __init__(self, action_list, global_config: GlobalConfig):
        self.offload_choice_idx = int(action_list[0])
        self.offload_task_percentage = action_list[1]

        self.global_config = global_config
        self.action_config = global_config.agent_config.action_config

    def get_action_list(self):
        return [self.offload_choice_idx,
                self.offload_task_percentage]

    def get_action_array(self):
        import numpy as np
        return np.array([self.offload_choice_idx,
                         self.offload_task_percentage])

    def get_random_action(self):
        pass

    def get_determined_action(self, offload_choice_idx, offload_task_percentage):
        pass
