import wandb
import os
from typing import Dict
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter # tensorboardX to work with macos
from baselines.onpolicy.utils.shared_buffer import SharedReplayBuffer
from baselines.onpolicy.utils.graph_buffer import GraphReplayBuffer

from config import GlobalConfig
from agent.state import State
from agent.action import Action
from mec_env.base_station_set import BaseStationSet
from mec_env.environment_manager import EnvironmentManager

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config:Dict):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.ue_num = self.num_agents
        self.mec_config = config["mec_config"]
        # total entites is agents + goals + obstacles  agent和goals的数量是一致的
        # self.num_entities = (self.num_agents + self.num_agents +
        #                     self.all_args.num_obstacles)
        self.num_entities = self.num_agents
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']
        self.env_manager = self.envs

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        # if not testing model
        if not self.use_render:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        if self.all_args.env_name == "GraphMPE" or self.all_args.env_name == "MEC":
        #     from baselines.onpolicy.algorithms.graph_mappo import GR_MAPPO as TrainAlgo
        #     from baselines.onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy as Policy
        # else:
            from baselines.onpolicy.algorithms.mappo import R_MAPPO as TrainAlgo
            from baselines.onpolicy.algorithms.MAPPOPolicy import R_MAPPOPolicy as Policy

        # get state base message
        ue_state_list = []
        ue_state_array = np.array([])
        ue_action_list = []
        ue_state = None
        ue_action_array = None

        # 2024 new baseline
        train_config = self.mec_config.train_config

        env = EnvironmentManager(self.mec_config)
        env.reset()
        num_agents = env.base_station_set.mobile_device_num
        assert num_agents == self.num_agents
        obs_shape_n = []
        action_shape_n = []

        for i in range(num_agents):
            each_state = env.get_state_per_mobile_device(i)
            obs_shape_n.append(len(each_state.get_state_list()))
            action_shape_n.append(len(env.get_random_action(self.mec_config).get_action_list()))

        # for ue_id in range(self.ue_num):
        #     ue_state = self.env_manager.get_state_per_user_equipment(ue_id)
        #     ue_state_list.append(ue_state)
        #     ue_state_array = np.append(ue_state_array, ue_state.get_state_array())
        #     random_index = np.random.randint(0, 2)  # TODO: delete
        #     # ue_action_array = np.eye(self.global_config.agent_config.action_config.dimension)[random_index]
        #     # ue_action_array = np.ones_like(ue_action_array)  # it's available action
        #     ue_action_array = np.ones(self.mec_config.agent_config.action_config.dimension)
        #     ue_action_list.append(ue_action_array.tolist())

        # NOTE change variable input here
        if self.use_centralized_V:
            share_observation_space = sum(obs_shape_n)
        else:
            share_observation_space = obs_shape_n[0]

        # policy network
        if self.all_args.env_name == 'GraphMPE':
            # print("!!!!!!!!!!!!:", np.array(range(ue_action_array.shape[0]+1)).shape)
            print("share_observation_space:", share_observation_space)
            self.policy = Policy(self.all_args,
                                ue_state.get_state_array().shape,  # self.envs.observation_space[0]
                                share_observation_space,
                                 (self.ue_num, ue_state.get_state_array().shape[0]),  # 点邻居暂定为状态的维度 self.envs.node_observation_space[0],
                                np.array(range(1)).shape,  # 边邻居暂定为状态的维度 self.envs.edge_observation_space[0],
                                ue_action_array.shape,  # self.envs.action_space[0],
                                device=self.device)
        elif self.all_args.env_name == "MEC":
            self.policy = Policy(self.all_args,
                                obs_shape_n[0],
                                share_observation_space,
                                action_shape_n[0],
                                device = self.device)

        if self.model_dir is not None:
            print(f'Restoring from checkpoint stored in {self.model_dir}')
            self.restore()
            self.gif_dir = self.model_dir

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
        # buffer
        if self.all_args.env_name == "GraphMPE":
            self.buffer = GraphReplayBuffer(self.all_args, 
                                            self.num_agents,
                                            ue_state.get_state_array().shape,  # self.envs.observation_space[0]
                                            share_observation_space,
                                            (self.ue_num, ue_state.get_state_array().shape[0]),  # 点邻居暂定为状态的维度 self.envs.node_observation_space[0],
                                            np.array(range(1)).shape, #self.envs.agent_id_observation_space[0],
                                            np.array(range(self.ue_num)).shape,  # self.envs.share_agent_id_observation_space[0], 实际shared应该不可能这么多
                                            (self.ue_num, self.ue_num),  # self.envs.adj_observation_space[0], 这个不知道是干嘛的 暂定为state的维数
                                            ue_action_array.shape,  # self.envs.action_space[0]
                                            )
        elif self.all_args.env_name == "MEC":
            self.buffer = SharedReplayBuffer(self.all_args,
                                            num_agents,
                                            obs_shape_n[0],
                                            share_observation_space,
                                            action_shape_n[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        raise NotImplementedError
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        # print("policy_critic.state_dict():\n", policy_critic.state_dict().keys())
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt', 
                                    map_location=torch.device('cpu'))
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt',
                                        map_location=torch.device('cpu'))
            self.policy.critic.v_out.update(np.zeros((10, 1)))
            # print(">>>policy_critic_state_dict:", self.policy.critic.state_dict().keys())  #
            print("---hhhhhhhhhhh----------")
            self.policy.critic.load_state_dict(policy_critic_state_dict)
    
    def process_infos(self, infos):
        """Process infos returned by environment."""
        env_infos = {}
        for agent_id in range(self.num_agents):
            idv_rews = []
            dist_goals, time_to_goals, min_times_to_goal = [], [], []
            idv_collisions, obst_collisions = [], []
            for info in infos:
                if 'individual_reward' in info[agent_id].keys():
                    idv_rews.append(info[agent_id]['individual_reward'])
                if 'Dist_to_goal' in info[agent_id].keys():
                    dist_goals.append(info[agent_id]['Dist_to_goal'])
                if 'Time_req_to_goal' in info[agent_id].keys():
                    times = info[agent_id]['Time_req_to_goal']
                    if times == -1:
                        times = self.all_args.episode_length * self.dt    # NOTE: Hardcoding `dt`
                    time_to_goals.append(times)
                if 'Num_agent_collisions' in info[agent_id].keys():
                    idv_collisions.append(info[agent_id]['Num_agent_collisions'])
                if 'Num_obst_collisions' in info[agent_id].keys():
                    obst_collisions.append(info[agent_id]['Num_obst_collisions'])
                if 'Min_time_to_goal' in info[agent_id].keys():
                    min_times_to_goal.append(info[agent_id]['Min_time_to_goal'])
            
            agent_rew = f'agent{agent_id}/individual_rewards'
            times     = f'agent{agent_id}/time_to_goal'
            dists     = f'agent{agent_id}/dist_to_goal'
            agent_col = f'agent{agent_id}/num_agent_collisions'
            obst_col  = f'agent{agent_id}/num_obstacle_collisions'
            min_times = f'agent{agent_id}/min_time_to_goal'

            env_infos[agent_rew] = idv_rews
            env_infos[times]     = time_to_goals
            env_infos[min_times] = min_times_to_goal
            env_infos[dists]     = dist_goals
            env_infos[agent_col] = idv_collisions
            env_infos[obst_col]  = obst_collisions
        return env_infos

    def log_train(self, train_infos:Dict, total_num_steps:int):
        """
        Log training info.
        train_infos: (dict) 
            information about training update.
        total_num_steps: (int) 
            total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos:Dict, total_num_steps:int):
        """
        Log env info.
        env_infos: (dict) 
            information about env state.
        total_num_steps: (int) 
            total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def get_collisions(self, env_infos:Dict):
        """
            Get the collisions from the env_infos
            Example: {'agent0/individual_rewards': [5], 
                    'agent0/time_to_goal': [0.6000000000000001], 
                    'agent0/min_time_to_goal': [0.23632679886748278], 
                    'agent0/dist_to_goal': [0.03768003822249384], 
                    'agent0/num_agent_collisions': [1.0], 
                    'agent0/num_obstacle_collisions': [0.0], 
                    'agent1/individual_rewards': [5], 
                    'agent1/time_to_goal': [0.6000000000000001], 
                    'agent1/min_time_to_goal': [0.3067362645187025], 
                    'agent1/dist_to_goal': [0.0387233764393595], 
                    'agent1/num_agent_collisions': [1.0], 
                    'agent1/num_obstacle_collisions': [0.0]}

        """
        collisions = 0
        for k, v in env_infos.items():
            if 'collision' in k:
                collisions += v[0]
        return collisions

    def get_fraction_episodes(self, env_infos:Dict):
        """
            Get the fraction of episode required to get to the goals
            from env_infos
            Example: {'agent0/individual_rewards': [5], 
                    'agent0/time_to_goal': [0.6000000000000001], 
                    'agent0/min_time_to_goal': [0.23632679886748278], 
                    'agent0/dist_to_goal': [0.03768003822249384], 
                    'agent0/num_agent_collisions': [1.0], 
                    'agent0/num_obstacle_collisions': [0.0], 
                    'agent1/individual_rewards': [5], 
                    'agent1/time_to_goal': [0.6000000000000001], 
                    'agent1/min_time_to_goal': [0.3067362645187025], 
                    'agent1/dist_to_goal': [0.0387233764393595], 
                    'agent1/num_agent_collisions': [1.0], 
                    'agent1/num_obstacle_collisions': [0.0]}
        """
        fracs = []
        success = []
        for k, v in env_infos.items():
            if 'time_to_goal' in k and 'min_time_to_goal' not in k:
                fracs.append(v[0] / (self.all_args.episode_length * self.dt))
                # if didn't reach goal then time_to_goal >= episode_len * dt
                if v[0] < self.all_args.episode_length * self.dt:
                    success.append(1)
                else:
                    success.append(0)
        assert len(success) == self.all_args.num_agents
        if sum(success) == self.all_args.num_agents:
            success = True
        else:
            success = False
                    

        return fracs, success
        
