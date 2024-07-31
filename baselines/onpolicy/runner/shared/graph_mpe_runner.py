import time
import numpy as np
from numpy import ndarray as arr
from typing import Tuple
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
import tensorboardX
import shutil
import os

def _t2n(x):
    return x.detach().cpu().numpy()

class GMPERunner(Runner):
    """
        Runner class to perform training, evaluation and data 
        collection for the MPEs. See parent class for details
    """
    dt = 0.1
    def __init__(self, config):
        super(GMPERunner, self).__init__(config)
        tensorboard_path = self.envs.global_config.train_config.tensorboard_path
        tensorboard_id = self.all_args.id
        tensorboard_seed = self.all_args.seed
        print("tensorboard_path:", tensorboard_path)
        print("tensorboard_id:", tensorboard_id)
        print("tensorboard_seed:", tensorboard_seed)
        self.work_dir = os.path.join(tensorboard_path, tensorboard_id, str(tensorboard_seed))
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        self.tf_writer = tensorboardX.SummaryWriter(self.work_dir)

    def run(self):
        self.warmup()

        start = time.time()
        # episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        episodes = self.envs.global_config.train_config.episode_num
        
        # This is where the episodes are actually run.
        for episode in range(episodes):  # episodes为625
            print("==========================================episode:", episode)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            infos = {}
            train_rews = []
            train_cost = []
            train_epoch_reward = 0

            cost_list = []
            reward_list = []
            offload_count_avg_list = []
            current_step = 0
            self.warmup()
            # print("vvvvvvvvvvvvep0_obs:\n", obs)
            for step in range(self.envs.max_step_num):  # 25 self.episode_length
                # Sample actions
                values, actions, action_log_probs, rnn_states, \
                    rnn_states_critic, actions_env = self.collect(step)
                # actions_env: (128, 3, 5)
                # actions: (128, 3, 1)
                # rnn_states: (128, 3, 1, 64)
                # values: (128, 3, 1)
                # rnn_states_critic: (128, 3, 1, 64)

                obs = self.buffer.obs[step]
                print("----------------------step:", step)
                print("obs_step:\n", obs)
                ue_queue_time_now = np.zeros(self.ue_num)
                offloading_count_list = []
                reward_list_step = []
                cost_avg_list = []
                cost_baseline_avg_list = []
                # Obs reward and next obs

                # obs, agent_id, node_obs, adj, rewards, \
                #     dones, infos = self.envs.step(actions_env)

                ue_state_list = self.envs.set_array_to_class_list(obs)

                reward, cost_avg, cost_baseline_avg, ue_done_all, offloading_count_list, ue_next_state_list, next_goal_list = \
                    self.envs.one_step_for_all_ue(actions,
                                                  ue_state_list,
                                                  ue_queue_time_now,
                                                  offloading_count_list)

                # obs: (128, 3, 6)
                # node_obs: (128, 3, 9, 7)
                reward_list_step.append(reward)
                cost_avg_list.append(cost_avg)
                cost_baseline_avg_list.append(cost_baseline_avg)
                done = ue_done_all

                if step == self.envs.max_step_num - 1:
                    done = True
                    # print("done?", done)
                if bool(self.envs.reason_to_finish_this_episode) or done:
                    done = True
                    pass
                else:
                    # print("hii")
                    self.envs.update_the_hexagon_network()

                obs, agent_id, node_obs, adj = self.envs.get_state_and_class_list()
                # print(">>>next2:\n", np.array(obs).reshape(self.ue_num, -1))

                offload_count_avg = np.mean(offloading_count_list)
                reward_list_step, mask = self.envs.cost_to_reward_episode(cost_avg_list, reward_list_step,
                                                                     self.envs.interface_config.reward_config.reward_weight_of_cost)

                reward_list_step = self.envs.reward_gift_episode(cost_baseline_avg_list, cost_avg_list,
                                                            self.envs.max_step_num, mask, reward_list_step)
                assert len(reward_list_step) == 1
                rewards = np.ones((self.n_rollout_threads, self.num_agents, 1)) * reward_list_step[0]
                dones = np.full((self.n_rollout_threads, self.num_agents), done, dtype=bool)
                infos = {}

                data = (obs, agent_id, node_obs, adj, agent_id, rewards,
                        dones, infos, values, actions, action_log_probs,
                        rnn_states, rnn_states_critic)

                # insert data into buffer
                self.insert(data)

                reward_list.append(reward_list_step[0])
                cost_list.append(cost_avg)
                offload_count_avg_list.append(offload_count_avg)
                if bool(self.envs.reason_to_finish_this_episode) or done:
                    done = True
                    print("reward_list:", reward_list)
                    train_rews.append(np.mean(reward_list))
                    cost_step_avg_magnify = np.mean(cost_list) * 3
                    self.tf_writer.add_scalar("1_Train/" + 'step', step + 1, episode)
                    i = step + 1
                    while i < self.envs.max_step_num:
                        cost_list.append(cost_step_avg_magnify)
                        i += 1
                    train_cost.append(np.mean(cost_list))
                    self.buffer.step = 0
                    break

                # if step >= 3:
                #     exit()

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            print("train_rews:", train_rews)
            train_rews_tensorboard = np.mean(np.array(train_rews))
            self.tf_writer.add_scalar("1_Train/" + 'reward', train_rews_tensorboard, episode)
            train_cost_tensorboard = np.mean(np.array(train_cost))
            self.tf_writer.add_scalar("1_Train/" + 'cost', train_cost_tensorboard, episode)
            offload_count_tensorboard = np.mean(np.array(offload_count_avg_list))
            self.tf_writer.add_scalar("1_Train/" + 'offload_count', offload_count_tensorboard, episode)
            # log information
            if episode % self.log_interval == 0:
                end = time.time()

                env_infos = self.process_infos(infos)

                avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_rewards"] = avg_ep_rew
                print(f"Average episode rewards is {avg_ep_rew:.3f} \t"
                    f"Total timesteps: {total_num_steps} \t "
                    f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}")
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            print("self.eval_interval:", self.eval_interval)
            print("self.use_eval:", self.use_eval)
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        # print("1111111111")
        obs, agent_id, node_obs, adj = self.envs.reset(self.n_rollout_threads)  # env: GraphSubprocVecEnv
        # print("vvvv")
        # print("obs:", obs.shape)
        # print("agent_id:", agent_id.shape)
        # print("node_obs:", node_obs.shape)
        # print("adj:", adj.shape)  # 正常的话，这原本是节点之间相互距离的对称阵
        # print("^^^^")

        # TODO 在MEC环境中 n_rollout_threads 由128暂时设置为1
        # obs: (128, 3, 6) -> batch, agent, obs_item? 相对距离 速度 点的信息等
        # agent_id: (128, 3, 1)
        # node_obs: (128, 3, 9, 7)
        # adj: (128, 3, 9, 9)

        # replay buffer
        if self.use_centralized_V:
            # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, 
                                                                    axis=1)
            # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
            share_agent_id = np.expand_dims(share_agent_id, 
                                            1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
            share_agent_id = agent_id

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        # print("warm0:\n", obs)
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.agent_id[0] = agent_id.copy()
        self.buffer.share_agent_id[0] = share_agent_id.copy()

    @torch.no_grad()
    def collect(self, step:int) -> Tuple[arr, arr, arr, arr, arr, arr]:
        self.trainer.prep_rollout()
        print("self.trainer.policy:", self.trainer.policy)
        exit()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(        # get_actions会过actor网络和critic网络
                            np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.node_obs[step]),
                            np.concatenate(self.buffer.adj[step]),
                            np.concatenate(self.buffer.agent_id[step]),
                            np.concatenate(self.buffer.share_agent_id[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), 
                                            self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), 
                                self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), 
                                            self.n_rollout_threads))
        # rearrange action
        # if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
        #     for i in range(self.envs.action_space[0].shape):
        #         uc_actions_env = np.eye(self.envs.action_space[0].high[i] +
        #                                                     1)[actions[:, :, i]]
        #         if i == 0:
        #             actions_env = uc_actions_env
        #         else:
        #             actions_env = np.concatenate((actions_env,
        #                                         uc_actions_env), axis=2)
        # elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
        #     actions_env = np.squeeze(np.eye(
        #                             self.envs.action_space[0].n)[actions], 2)
        # else:
        #     raise NotImplementedError
        actions_env = np.array([])  # hu: 不清楚意义 先设置为空

        return (values, actions, action_log_probs, rnn_states, 
                rnn_states_critic, actions_env)

    def insert(self, data):
        obs, agent_id, node_obs, adj, agent_id, rewards, dones, \
            infos, values, actions, action_log_probs, \
            rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), 
                                                self.recurrent_N, 
                                                self.hidden_size), 
                                                dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), 
                                        *self.buffer.rnn_states_critic.shape[3:]), 
                                        dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, 
                        self.num_agents, 1), 
                        dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), 
                                        dtype=np.float32)

        # if centralized critic, then shared_obs is concatenation of obs from all agents
        if self.use_centralized_V:
            # TODO stack agent_id as well for agent specific information
            # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
            share_obs = np.expand_dims(share_obs, 
                                        1).repeat(self.num_agents, axis=1)
            # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
            share_agent_id = np.expand_dims(share_agent_id, 
                                            1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
            share_agent_id = agent_id

        self.buffer.insert(share_obs, obs, node_obs, adj, agent_id, share_agent_id, 
                        rnn_states, rnn_states_critic, actions, action_log_probs, 
                        values, rewards, masks)

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
                            np.concatenate(self.buffer.share_obs[-1]),
                            np.concatenate(self.buffer.node_obs[-1]),
                            np.concatenate(self.buffer.adj[-1]),
                            np.concatenate(self.buffer.share_agent_id[-1]),
                            np.concatenate(self.buffer.rnn_states_critic[-1]),
                            np.concatenate(self.buffer.masks[
                                               -1]))
        next_values = np.array(np.split(_t2n(next_values), 
                                self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    @torch.no_grad()
    def eval(self, total_num_steps:int):
        eval_episode_rewards = []
        eval_obs, eval_agent_id, eval_node_obs, eval_adj = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, 
                                    *self.buffer.rnn_states.shape[2:]), 
                                    dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, 
                                self.num_agents, 1), 
                                dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                                                np.concatenate(eval_obs),
                                                np.concatenate(eval_node_obs),
                                                np.concatenate(eval_adj),
                                                np.concatenate(eval_agent_id),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), 
                                            self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), 
                                            self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(
                                self.eval_envs.action_space[0].high[i] + 
                                                        1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, 
                                                            eval_uc_actions_env), 
                                                            axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(
                            self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_agent_id, eval_node_obs, eval_adj, eval_rewards, \
                eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros((
                                                    (eval_dones == True).sum(), 
                                                    self.recurrent_N, 
                                                    self.hidden_size), 
                                                    dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, 
                                self.num_agents, 1), 
                                dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros((
                                                (eval_dones == True).sum(), 1), 
                                                dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(
                                                np.array(eval_episode_rewards), 
                                                axis=0)
        eval_average_episode_rewards = np.mean(
                                    eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + 
                                            str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self, get_metrics:bool=False):
        """
            Visualize the env.
            get_metrics: bool (default=False)
                if True, just return the metrics of the env and don't render.
        """
        envs = self.envs
        
        all_frames = []
        rewards_arr, success_rates_arr, num_collisions_arr, frac_episode_arr = [], [], [], []

        for episode in range(self.all_args.render_episodes):
            obs, agent_id, node_obs, adj = envs.reset()
            if not get_metrics:
                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                else:
                    envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, 
                                    self.num_agents, 
                                    self.recurrent_N, 
                                    self.hidden_size), 
                                    dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, 
                            self.num_agents, 1), 
                            dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                                                    np.concatenate(obs),
                                                    np.concatenate(node_obs),
                                                    np.concatenate(adj),
                                                    np.concatenate(agent_id),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), 
                                    self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(
                                envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, 
                                                        uc_actions_env), 
                                                        axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(
                                            envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, agent_id, node_obs, adj, \
                    rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), 
                                                    self.recurrent_N, 
                                                    self.hidden_size), 
                                                    dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, 
                                self.num_agents, 1), 
                                dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), 
                                                dtype=np.float32)

                if not get_metrics:
                    if self.all_args.save_gifs:
                        image = envs.render('rgb_array')[0][0]
                        all_frames.append(image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                    else:
                        envs.render('human')

            env_infos = self.process_infos(infos)
            # print('_'*50)
            num_collisions = self.get_collisions(env_infos)
            frac, success = self.get_fraction_episodes(env_infos)
            rewards_arr.append(np.mean(np.sum(np.array(episode_rewards), axis=0)))
            frac_episode_arr.append(np.mean(frac))
            success_rates_arr.append(success)
            num_collisions_arr.append(num_collisions)
            # print(np.mean(frac), success)
            # print("Average episode rewards is: " + 
                    # str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
        
        print(rewards_arr)
        print(frac_episode_arr)
        print(success_rates_arr)
        print(num_collisions_arr)

        if not get_metrics:
            if self.all_args.save_gifs:
                imageio.mimsave(str(self.gif_dir) + '/render.gif', 
                                all_frames, duration=self.all_args.ifi)
