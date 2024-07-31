# ExplabOff
ExplabOff: Towards Explorative and Collaborative Task Offloading via Mutual Information-Enhanced MARL
## Usage:
To train Explaboff:
```bash
python -u run_mec_offloading_alg.py  --MI_update_freq=1 --min_adv_c=1 --max_adv_c=3.5  --env_name=mec_env  --policy_name=MA_MINE_DDPG --seed=4 --gpu-no=1 --id Explaboff
```
To train MDOff:
```bash
python baselines/offpolicy/scripts/train/train_mpe.py --env_name "MEC" --algorithm_name "maddpg" --experiment_name "test" --scenario_name "MEC_multi" --seed 2 --actor_train_interval_step 1 --episode_length 25 --use_soft_update --lr 7e-4 --hard_update_interval_episode 200 --num_env_steps 10000000 --id MDOff
```
To train CMOff:
```bash
python -u run_coma_like_offloading_alg.py  --MI_update_freq=1 --min_adv_c=1 --max_adv_c=4  --env_name=mec_env  --policy_name=MA_MINE_DDPG --seed=1 --gpu-no=0 --id test_coma 
```
To train MPOff:
```bash
python baselines/onpolicy/scripts/train_mpe.py --use_valuenorm     --use_popart --env_name 'MEC' --algorithm_name 'mappo'     --experiment_name "test" --scenario_name "simple_tag"     --num_agents 3 --num_landmarks 3     --seed 0 --n_training_threads 1 --n_rollout_threads 1     --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000     --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "marl" --id MPOff
```

To train MTOff:
```bash
python baselines/offpolicy/scripts/train/train_mpe.py --env_name "MEC" --algorithm_name "matd3" --experiment_name "test" --scenario_name "MEC_multi" --seed 0 --actor_train_interval_step 1 --episode_length 25 --use_soft_update --lr 7e-4 --hard_update_interval_episode 200 --num_env_steps 10000000 --id MTOff
```