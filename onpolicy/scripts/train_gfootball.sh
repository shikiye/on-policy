#!/bin/sh
env="GoogleFootball"
scenario="5_vs_5"
algo="mappo"
exp="mlp" #to change
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_gfootball.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 500 --num_env_steps 10000 --ppo_epoch 10 --use_value_active_masks --use_eval --use_recurrent_policy
done
