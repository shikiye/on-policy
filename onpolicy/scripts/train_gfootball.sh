#!/bin/sh
env="GoogleFootball"
scenario="academy_counterattack_hard" #"academy_3_vs_1_with_keeper" #"5_vs_5"
algo="mappo"
exp="mlp" #to change
seed_max=1
num_a=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_gfootball.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_a} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 500 --num_env_steps 500000 --ppo_epoch 10 --use_value_active_masks --use_eval --use_recurrent_policy
done
