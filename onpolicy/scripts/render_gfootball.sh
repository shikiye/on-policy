#!/bin/sh
env="GoogleFootball"
scenario="5_vs_5"
algo="mappo"
num_agents=4
exp="mlp" #to change
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_gfootball.py --save_gifs --share_policy --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 500 --render_episodes 5 --model_dir "results\GoogleFootball\mappo\mlp\run332\models"
done
