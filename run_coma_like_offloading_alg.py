'''
    Demo run file for different algorithms and Mujoco tasks.
'''
import random

import numpy as np
import pandas as pd
import torch
import gym
import argparse

import ma_utils
import algorithms.mec_maxminMADDPG as MA_MINE_DDPG
import algorithms.dynamic_mec_new_maxminMADDPG as DYNAMIC_DDPG
import math
import os

from tensorboardX import SummaryWriter

from multiprocessing import cpu_count
from maddpg.utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import time

from config import GlobalConfig
from agent.state import State
from agent.action import Action
from mec_env.base_station_set import BaseStationSet
from mec_env.environment_manager import EnvironmentManager

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

# Runs policy for X episodes and returns average reward


if __name__ == "__main__":
    global_config = GlobalConfig()
    train_config = global_config.train_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="DYNAMIC")  # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v1")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--batch_size", default=1024, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.95, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.01, type=float)  # Target network update rate
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--freq_test_mine", default=5e3, type=float)
    parser.add_argument("--gpu-no", default='-1', type=str)  # GPU number, -1 means CPU
    parser.add_argument("--MI_update_freq", default=1, type=int)
    parser.add_argument("--max_adv_c", default=0.0, type=float)
    parser.add_argument("--min_adv_c", default=0.0, type=float)
    parser.add_argument("--discrete_action", action='store_true')
    parser.add_argument("--id", default="default")
    args = parser.parse_args()

    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    file_name_origin = "ExplabOff_%s_%s_%s_%s_%s" % (
        args.MI_update_freq, args.max_adv_c, args.min_adv_c, args.env_name, args.seed)
    file_name = "ExplabOff_%s_%s_%s_%s_%s_%s" % (
        args.MI_update_freq, args.max_adv_c, args.min_adv_c, args.env_name, args.seed, str(args.id))

    writer = SummaryWriter(log_dir="./tensorboard/" + file_name_origin + '/' + str(args.id))

    output_dir = "./output/" + file_name
    model_dir = "./model/" + file_name
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    if os.path.exists(model_dir) is False:
        os.makedirs(model_dir)

    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = EnvironmentManager(global_config)
    env.reset()

    n_agents = env.base_station_set.mobile_device_num
    print("n_agent:", n_agents)
    obs_shape_n = []
    action_shape_n = []
    for i in range(n_agents):
        each_state = env.get_state_per_mobile_device(i)
        obs_shape_n.append(len(each_state.get_state_list()))
        action_shape_n.append(len(env.get_random_action(global_config).get_action_list()))

    print("obs_shape_n ", obs_shape_n)
    print("action_shape_n:", action_shape_n)

    print("env .n ", n_agents)

    if args.policy_name == "MA_MINE_DDPG":
        policy = MA_MINE_DDPG.MA_T_DDPG(n_agents, obs_shape_n, sum(obs_shape_n), action_shape_n, 1.0, device, 0.0, 0.0)
    else:
        policy = DYNAMIC_DDPG.MA_T_DDPG(n_agents, obs_shape_n, sum(obs_shape_n), action_shape_n, 1.0, device, 0.0, 0.0)

    replay_buffer = ma_utils.ReplayBuffer(1e6)

    good_data_buffer = ma_utils.embedding_Buffer(1e3)
    bad_data_buffer = ma_utils.embedding_Buffer(1e3)

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = global_config.interface_config.reward_config.init_reward
    episode_cost_list = []
    episode_cost_max_list = []
    done = True

    get_epoch_Mi = False
    Mi_list = []
    data_recorder = []
    replay_buffer_recorder = []
    moving_avg_reward_list = []
    avg_reward_list_for_best = []
    avg_cost_list_for_best = []
    embedding_recorder = []

    best_reward_start = -1
    reward_list = []
    eposide_reward_list = []
    compare_list = []
    compare_range_count = 0
    recorder_reward = 0
    best_reward = -100000000000
    best_reward_for_save_model = -100000000000
    eposide_num = -1
    current_policy_performance = -1000
    episode_timesteps = train_config.step_num
    start_time = time.time()

    t1 = time.time()
    while total_timesteps < train_config.total_timesteps:
        if episode_timesteps == train_config.step_num or done:

            print("===================================================episode_count:", episode_num)
            print("episode_timesteps:", episode_timesteps)
            eposide_num += 1
            for d in replay_buffer_recorder:
                replay_buffer.add(d, episode_reward)
            print("episode_reward:", episode_reward)
            if total_timesteps != 0:
                lowest_reward = global_config.interface_config.reward_config.lowest_reward
                mean_reward = global_config.interface_config.reward_config.mean_reward

                writer.add_scalar("data/0episode_reward", episode_reward, episode_num)
                writer.add_scalar("data/0episode_cost", episode_cost_list[-1], episode_num)
                writer.add_scalar("data/0episode_cost_max", episode_cost_max_list[-1], episode_num)
                writer.add_scalar("data/0timesteps_in_episode", episode_timesteps, episode_num)
                writer.add_scalar("data/0success_rate_episode", (episode_timesteps - 1) / (train_config.step_num - 1),
                                  episode_num)
                avg_reward_list_for_best.append(episode_reward)
                avg_cost_list_for_best.append(episode_cost_list[-1])
                if len(avg_reward_list_for_best) > 10000:
                    if np.mean(avg_reward_list_for_best) > best_reward_for_save_model and np.std(
                            avg_reward_list_for_best) < 100 and np.mean(avg_reward_list_for_best) > 500:
                        best_reward_for_save_model = np.mean(avg_reward_list_for_best)
                        policy.save("best_" + str(total_timesteps) + "cost_" + str(
                            np.around(np.mean(avg_cost_list_for_best), 5)) + "reward_" + str(
                            np.around(np.mean(avg_reward_list_for_best), 3)), "model/" + file_name)
                    avg_reward_list_for_best = []
                    avg_cost_list_for_best = []

                if len(moving_avg_reward_list) % 10 == 0:
                    writer.add_scalar("data/reward", np.mean(moving_avg_reward_list[-1000:]), total_timesteps)

                if episode_num % 1000 == 0:
                    print('Total T:', total_timesteps, 'Episode Num:', episode_num, 'Episode T:', episode_timesteps,
                          'Reward:', np.mean(moving_avg_reward_list[-1000:]) / 3.0, " time cost:", time.time() - t1)
                    t1 = time.time()

            if total_timesteps >= 1024 and total_timesteps % 100 == 0:
                sp_actor_loss_list = []
                process_Q_list = []
                process_min_MI_list = []
                process_max_MI_list = []
                process_min_MI_loss_list = []
                process_max_MI_loss_list = []
                Q_grads_list = []
                MI_grads_list = []
                MI_upper_bound_list_list = []
                MI_lower_bound_list_list = []
                training_reward_list_list = []
                training_reward_Q_list_list = []
                training_r_ExplabOff_list_list = []

                for i in range(1):
                    update_signal = False
                    if len(good_data_buffer.pos_storage) < 500:
                        print("coma!")
                        process_Q = policy.train(replay_buffer, 1, args.batch_size, args.discount, args.tau)

                        process_min_MI = 0
                        process_min_MI_loss = 0
                        min_mi = 0.0
                        min_mi_loss = 0.0
                        process_max_MI = 0
                        pr_sp_loss = 0.0
                        Q_grads = 0.0
                        MI_grads = 0.0
                        process_max_MI_loss = 0.0
                    else:
                        print("MADDPG!")
                        exit()
                        if total_timesteps % (args.MI_update_freq * 100) == 0:
                            update_signal = True
                            if args.min_adv_c > 0.0:
                                process_min_MI_loss = policy.train_club(bad_data_buffer, 1, batch_size=args.batch_size)
                            else:
                                process_min_MI_loss = 0.0

                            if args.max_adv_c > 0.0:
                                process_max_MI_loss, _ = policy.train_mine(good_data_buffer, 1,
                                                                           batch_size=args.batch_size)
                            else:
                                process_max_MI_loss = 0.0

                        else:
                            process_min_MI_loss = 0.0
                            process_max_MI_loss = 0.0
                        process_Q, process_min_MI, process_max_MI, Q_grads, MI_grads, MI_upper_bound_list, MI_lower_bound_list, training_reward_list, training_reward_Q_list, training_r_ExplabOff_list = policy.train_actor_with_mine(
                            replay_buffer, 1, args.batch_size, args.discount, args.tau, max_mi_c=0.0, min_mi_c=0.0,
                            min_adv_c=args.min_adv_c, max_adv_c=args.max_adv_c, total_timesteps=total_timesteps,
                            update_signal=update_signal)

                        MI_upper_bound_list_list.append(MI_upper_bound_list)
                        MI_lower_bound_list_list.append(MI_lower_bound_list)
                        training_reward_list_list.append(training_reward_list)
                        training_reward_Q_list_list.append(training_reward_Q_list)
                        training_r_ExplabOff_list_list.append(training_r_ExplabOff_list)

                        upper_bound = np.mean(MI_upper_bound_list_list)
                        lower_bound = np.mean(MI_lower_bound_list_list)
                        training_reward = np.mean(training_reward_list_list)
                        training_reward_Q = np.mean(training_reward_Q_list_list)
                        training_r_ExplabOff = np.mean(training_r_ExplabOff_list_list)
                        writer.add_scalar("data/upper_bound", upper_bound, total_timesteps)
                        writer.add_scalar("data/lower_bound", lower_bound, total_timesteps)
                        writer.add_scalar("data/upper_bound-lower_bound", upper_bound - lower_bound, total_timesteps)
                        writer.add_scalar("data/training_reward", training_reward, total_timesteps)
                        writer.add_scalar("data/training_reward_Q", training_reward_Q, total_timesteps)
                        writer.add_scalar("data/training_r_ExplabOff", training_r_ExplabOff, total_timesteps)

                    process_max_MI_list.append(process_max_MI)
                    process_Q_list.append(process_Q)
                    Q_grads_list.append(Q_grads)
                    MI_grads_list.append(MI_grads)
                    process_max_MI_loss_list.append(process_max_MI_loss)
                    process_min_MI_list.append(process_min_MI)

                    process_min_MI_loss_list.append(process_min_MI_loss)
                if len(moving_avg_reward_list) % 10 == 0:
                    writer.add_scalar("data/MINE_lower_bound_loss", np.mean(process_max_MI_loss_list), total_timesteps)

                    writer.add_scalar("data/process_Q", np.mean(process_Q_list), total_timesteps)
                    writer.add_scalar("data/club_upper_bound_loss", np.mean(process_min_MI_loss_list), total_timesteps)

            env.reset()
            obs = []
            for mobile_device_id in range(len(env.base_station_set.all_mobile_device_list)):
                each_state = env.get_state_per_mobile_device(mobile_device_id)
                state_array = each_state.get_normalized_state_array()
                obs.append(state_array)
            state = np.concatenate(obs, -1)

            # print("ep reward ", episode_reward)
            moving_avg_reward_list.append(episode_reward)

            done = False

            explr_pct_remaining = max(0, 25000 - episode_num) / 25000
            policy.scale_noise(0.3 * explr_pct_remaining)
            policy.reset_noise()
            episode_reward = 0
            episode_cost_list = []
            episode_cost_max_list = []
            reward_list = []
            eposide_reward_list = []
            episode_timesteps = 0
            episode_num += 1
            data_recorder = []
            replay_buffer_recorder = []
            best_reward_start = -1
            best_reward = -1000000
            Mi_list = []

            if total_timesteps % (4e5 - 1) == 0:
                policy.save(total_timesteps, "model/" + file_name)

        # 这里是每一个step的开始
        env.create_task_per_step()
        print("----------------------------------------step_count:", episode_timesteps)
        state_class_list = []
        obs = []
        for mobile_device_id in range(len(env.base_station_set.all_mobile_device_list)):
            each_state = env.get_state_per_mobile_device(mobile_device_id)
            state_class_list.append(each_state)
            state_list = each_state.get_state_list()
            state_array = each_state.get_normalized_state_array()
            obs.append(state_array)
        state = np.concatenate(obs, -1)

        # Select action randomly or according to policy
        scaled_a_list = []
        action_class_list = []

        for i in range(n_agents):
            a = policy.select_action(obs[i], i)
            scaled_a = np.multiply(a, 1.0)
            scaled_a = np.clip(scaled_a, -0.9999, 0.9999)
            meaningful_scaled_a = scaled_a
            # print("scaled_a:", scaled_a)
            meaningful_scaled_a[0] = int(np.floor((scaled_a[0] + 1) * env.base_station_set.base_station_num / 2))
            meaningful_scaled_a[1] = (scaled_a[1] + 1) / 2
            # print("meaningful_scaled_a:", meaningful_scaled_a)
            action_class = Action(meaningful_scaled_a, global_config)
            scaled_a_list.append(scaled_a)
            action_class_list.append(action_class)
        next_state_class_list, cost_array, reward_array, cost_array_max, done_list, _ = env.step(state_class_list,
                                                                                                 action_class_list,
                                                                                                 episode_timesteps)
        reward = reward_array[0]

        next_obs = []
        for mobile_device_id in range(len(env.base_station_set.all_mobile_device_list)):
            each_state = env.get_state_per_mobile_device(mobile_device_id)
            state_array = each_state.get_normalized_state_array()
            next_obs.append(state_array)
        next_state = np.concatenate(next_obs, -1)

        done = any(done_list)

        terminal = False
        terminal = (episode_timesteps + 1 >= train_config.step_num)
        done = done or terminal
        done_bool = float(done or terminal)

        episode_reward += reward
        episode_cost_list.append(cost_array[0])
        episode_cost_max_list.append(cost_array_max[0])
        eposide_reward_list.append(reward)

        replay_buffer_recorder.append(
            (obs, state, next_state, next_obs, np.concatenate(scaled_a_list, -1), reward, done))

        obs = next_obs
        state = next_state

        episode_timesteps += 1
        print("episode_timesteps:", episode_timesteps)
        total_timesteps += 1
        timesteps_since_eval += 1

    print("total time ", time.time() - start_time)
    policy.save(total_timesteps, "model/" + file_name)

    writer.close()
