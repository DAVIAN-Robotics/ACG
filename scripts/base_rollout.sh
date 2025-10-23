#!/bin/bash

config_path=${1:-"libs/Isaac-GR00T-N1/robomimic_configs/robocasa_mg100.json"}
model_path=${2:-"DAVIAN-Robotics/GR00T-N1-2B-tuned-RoboCasa-MG100-FrankaPandaGripper"}
seed=${3:-"123"}
n_rollouts=${4:-"24"}
num_batch_envs=${5:-"8"}
note=${6:-""}
additional_args=${@:7}

output_dir="outputs/${model_path}/rollout_results"
save_name="n=${n_rollouts}_seed=${seed}${note}"

echo "config_path: ${config_path}"
echo "model_path: ${model_path}"
echo "seed: ${seed}"
echo "n_rollouts: ${n_rollouts}"
echo "num_batch_envs: ${num_batch_envs}"
echo "note: ${note}"
echo "additional_args: ${additional_args}"
echo "output_dir: ${output_dir}"
echo "save_name: ${save_name}"

python libs/Isaac-GR00T-N1/scripts/rollout_with_robomimic.py \
    --config "${config_path}" \
    --config_add \
    algo_name=gr00t_guidance \
    algo.model_path="${model_path}" \
    train.output_dir="${output_dir}" \
    experiment.name="${save_name}" \
    experiment.rollout.n="${n_rollouts}" \
    train.seed="${seed}" \
    experiment.rollout.num_batch_envs=${num_batch_envs} \
    experiment.rollout.batched=True \
    ${additional_args}
