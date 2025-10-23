# 🛠️ How to Train and Evaluate GR00T-N1-2B

You can **train** and **rollout** the GR00T-N1-2B models using the scripts below.
Each section includes practical notes on environment setup, parameters, and reproducibility.

## 🔧 Training (Post-Training Phase)

Fine-tunes **GR00T-N1-2B** on the **RoboCasa** and **DexMimicGen** datasets using a modified Robomimic DataLoader.
Supports **multi-embodiment training** for DexMimicGen, including _bimanual gripper_, _bimanual hand_, and _GR1_ configurations.

> Before starting, make sure to download each dataset from its **official repository**: [RoboCasa](https://github.com/robocasa/robocasa) and [DexMimicGen](https://github.com/NVlabs/dexmimicgen).
> Then, update the dataset paths in the [Robomimic config files](libs/Isaac-GR00T-N1/robomimic_configs) accordingly.
>
> Unlike RoboCasa, **DexMimicGen** does not provide split files through `filter_key`.
> Therefore, we include our own split files under [`data/splits/dexmimicgen`](data/splits/dexmimicgen).
> Copy this directory to your DexMimicGen installation at `dexmimicgen/generated/splits`,
> and update the `filter_key` field in your Robomimic config to point to this path.

### RoboCasa

- You can adjust the dataset scale by modifying the `n_mg` parameter.
- The arguments `--data-configs` and `--embodiment_tag` determine the model embodiment used for fine-tuning.  
  This example uses the **SinglePandaGripper** embodiment.
- ⏱️ _Estimated training time: ~20 hours on a single NVIDIA H200 GPU._

```bash
# === Experiment setup ===
n_mg="100"                 # MG100 dataset
ngpu="1"                   # number of GPUs
bs="64"                    # per-GPU batch size
ga="2"                     # gradient accumulation steps
steps="60000"              # total training steps
training_seed="42"
exp_name="MG${n_mg}/LR=1e-4_Bs=${ngpu}x${bs}x${ga}_Steps=${steps}_Seed=${training_seed}${note}"

# === Logging (Weights & Biases) ===
export WANDB_ENTITY="your-entity"
export WANDB_PROJECT="Your Robot Project"

# === Launch training ===
python libs/Isaac-GR00T-N1/scripts/gr00t_finetune_robocasa.py \
  --num-gpus ${ngpu} \
  --output-dir checkpoints/robocasa/${exp_name} \
  --data-configs robocasa_single_panda_gripper \
  --video-backend decord \
  --embodiment_tag single_panda_gripper \
  --exp_name ${exp_name} \
  --batch_size ${bs} \
  --robomimic_config_json libs/Isaac-GR00T-N1/robomimic_configs/robocasa_mg${n_mg}.json \
  --gradient_accumulation_steps ${ga} \
  --no-save-only-model \
  --dataloader_num_workers 16 \
  --pin_memory \
  --max-steps ${steps} \
  --save_steps 1000 \
  --save_total_limit 3 \
  --training_seed ${training_seed}
```

### DexMG

1. **Quick start** — without preprocessing (_low-resolution 84×84_, as provided in DexMG).
2. **Recommended** — with preprocessed videos (_256×256 high-resolution_, which might have been used by the GR00T-N1 team).

#### 1️⃣ Skip Preprocessing (Quick Start)

- Suitable for quick baseline runs or debugging — lower visual fidelity but faster startup.
- The argument `--data-configs` combines multiple embodiments into a single dataset stream.
- The flag `--embodiment_tag` is only used for loading the GR00T policy and will automatically change per sample when `--dataset_cls=dexmg` is specified.
- ⏱️ _Estimated training time: ~20 hours on a single NVIDIA H200 GPU._

```bash
# === Quick path: no preprocessing (84x84 decoding) ===
n_mg="100"
export DEXMG_VIDEO_RESOLUTION="84x84"   # fast decoding, lower resolution
note="_${DEXMG_VIDEO_RESOLUTION}"

# === Experiment setup ===
steps="60000"
ngpu="1"
bs="64"
ga="2"
training_seed="42"
exp_name="MG${n_mg}/LR=1e-4_Bs=${ngpu}x${bs}x${ga}_Steps=${steps}_Seed=${training_seed}${note}"

# === Logging (Weights & Biases) ===
export WANDB_ENTITY="your-entity"
export WANDB_PROJECT="Your Robot Project"

# === Launch training ===
python libs/Isaac-GR00T-N1/scripts/gr00t_finetune_robocasa.py \
  --num-gpus ${ngpu} \
  --output-dir checkpoints/dexmg/${exp_name} \
  --data-configs dexmg_bimanual_panda_gripper dexmg_bimanual_panda_hand dexmg_gr1_arms_only dexmg_gr1_arms_only \
  --video-backend decord \
  --embodiment_tag single_panda_gripper \
  --exp_name ${exp_name} \
  --batch_size ${bs} \
  --robomimic_config_json libs/Isaac-GR00T-N1/robomimic_configs/dexmg_mg${n_mg}.json \
  --gradient_accumulation_steps ${ga} \
  --no-save-only-model \
  --dataloader_num_workers 16 \
  --max-steps ${steps} \
  --save_steps 1000 \
  --save_total_limit 3 \
  --dataset_cls=dexmg \
  --pin_memory \
  --training_seed ${training_seed}
```

#### 2️⃣ With Preprocessing (Recommended)

- Recommended for stable training and higher visual quality.
  - This version might be similar to what the **GR00T-N1** team used.
- ⏱️ _Estimated training time: ~20 hours on a single NVIDIA H200 GPU._

<details>

<summary>Preprocessing similar to the GR00T-N1 team’s dataset</summary>

> Based on the released dataset on [Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim),
> the GR00T-N1 team replays videos at a **160 × 256** resolution and pads the top and bottom regions,
> whereas the original **DexMimicGen** dataset provides **84 × 84** frames.

**Step 1**: Replay the existing datasets.

```bash
export DATASET_ROOT="/datasets"  # fill in your dataset root path

# Replay and save videos for each task and camera view
task_name="two_arm_threading"               ; render_image_name="agentview"             ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_threading"               ; render_image_name="robot0_eye_in_hand"    ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_threading"               ; render_image_name="robot1_eye_in_hand"    ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_three_piece_assembly"    ; render_image_name="agentview"             ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_three_piece_assembly"    ; render_image_name="robot0_eye_in_hand"    ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_three_piece_assembly"    ; render_image_name="robot1_eye_in_hand"    ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_transport"               ; render_image_name="agentview"             ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_transport"               ; render_image_name="robot0_eye_in_hand"    ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_transport"               ; render_image_name="robot1_eye_in_hand"    ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_box_cleanup"             ; render_image_name="agentview"             ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_box_cleanup"             ; render_image_name="robot0_eye_in_hand"    ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_box_cleanup"             ; render_image_name="robot1_eye_in_hand"    ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_drawer_cleanup"          ; render_image_name="agentview"             ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_drawer_cleanup"          ; render_image_name="robot0_eye_in_hand"    ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_drawer_cleanup"          ; render_image_name="robot1_eye_in_hand"    ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_lift_tray"               ; render_image_name="agentview"             ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_lift_tray"               ; render_image_name="robot0_eye_in_hand"    ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_lift_tray"               ; render_image_name="robot1_eye_in_hand"    ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_can_sort_random"         ; render_image_name="agentview"             ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_coffee"                  ; render_image_name="agentview"             ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
task_name="two_arm_pouring"                 ; render_image_name="agentview"             ; python scripts/playback_datasets_save_videos.py --dataset ${DATASET_ROOT}/dexmimicgen/generated/${task_name}.hdf5 --video_skip 1 --video_dir ${DATASET_ROOT}/dexmimicgen/generated/videos/${task_name}/obs/${render_image_name}_image --n 1000 --render_image_names ${render_image_name}
```

**Step 2**: Convert videos to HDF5

After replaying, convert the generated videos into HDF5 format to reduce I/O overhead during training.

```bash
export DATASET_ROOT="/datasets"  # fill in your dataset root path

# Each index (task index) corresponds to one parallel conversion job.
python libs/dexmimicgen/scripts/convert_videos_to_hdf5.py 0
python libs/dexmimicgen/scripts/convert_videos_to_hdf5.py 1
python libs/dexmimicgen/scripts/convert_videos_to_hdf5.py 2
python libs/dexmimicgen/scripts/convert_videos_to_hdf5.py 3
python libs/dexmimicgen/scripts/convert_videos_to_hdf5.py 4
python libs/dexmimicgen/scripts/convert_videos_to_hdf5.py 5
python libs/dexmimicgen/scripts/convert_videos_to_hdf5.py 6
python libs/dexmimicgen/scripts/convert_videos_to_hdf5.py 7
python libs/dexmimicgen/scripts/convert_videos_to_hdf5.py 8
```

</details>

```bash
# === Recommended path: preprocessed videos (256x256) ===
n_mg="100"
export DEXMG_VIDEO_RESOLUTION="256x256" # preprocessed, high-quality frames
note="_${DEXMG_VIDEO_RESOLUTION}"

# === Logging (Weights & Biases) ===
steps="60000"
ngpu="1"
bs="64"
ga="2"
training_seed="42"
exp_name="MG${n_mg}/LR=1e-4_Bs=${ngpu}x${bs}x${ga}_Steps=${steps}_Seed=${training_seed}${note}"

# === Launch training ===
export WANDB_ENTITY="your-entity"
export WANDB_PROJECT="Your Robot Project"

python libs/Isaac-GR00T-N1/scripts/gr00t_finetune_robocasa.py \
  --num-gpus ${ngpu} \
  --output-dir checkpoints/dexmg/${exp_name} \
  --data-configs dexmg_bimanual_panda_gripper dexmg_bimanual_panda_hand dexmg_gr1_arms_only dexmg_gr1_arms_only \
  --video-backend decord \
  --embodiment_tag single_panda_gripper \
  --exp_name ${exp_name} \
  --batch_size ${bs} \
  --robomimic_config_json libs/Isaac-GR00T-N1/robomimic_configs/dexmg_mg${n_mg}.json \
  --gradient_accumulation_steps ${ga} \
  --no-save-only-model \
  --dataloader_num_workers 16 \
  --max-steps ${steps} \
  --save_steps 1000 \
  --save_total_limit 3 \
  --dataset_cls=dexmg \
  --pin_memory \
  --training_seed ${training_seed}
```

## 🤖 Inference (Rollout)

### Preprocessing

1. Change the dataset path (`train.data.path`) in robomimic config.

### RoboCasa

- Compares vanilla GR00T-N1 and ACG-guided versions on RoboCasa.
- Numbers below are averaged over three random seeds.
- `MAX_NUM_EMBODIMENTS=32` means it newly add 1 additional embodiment (31 + 1).
- ⏱️ Estimated rollout time: ~2.5 hours on a single NVIDIA A6000 GPU.

| Method           |  Seeds  |  Success Rate  |
| ---------------- | :-----: | :------------: |
| Vanilla GR00T-N1 | 123–125 | 32.6% (±2.07%) |
| ACG              | 123–125 | 39.3% (±3.02%) |

```bash
# === RoboCasa rollout: without guidance ===
note=""
n_rollouts="24"                   # number of rollout episodes
num_batch_envs="8"                # number of parallel environments
export MAX_NUM_EMBODIMENTS="32"
dataset_name="robocasa_mg100"
config_path="libs/Isaac-GR00T-N1/robomimic_configs/${dataset_name}.json"
model_path="DAVIAN-Robotics/GR00T-N1-2B-tuned-RoboCasa-MG100-FrankaPandaGripper"
seed="123"

bash scripts/base_rollout.sh ${config_path} ${model_path} ${seed} ${n_rollouts} ${num_batch_envs} "${note}"
```

```bash
# === RoboCasa rollout: with ACG ===
note=""
n_rollouts="24"
num_batch_envs="8"
export MAX_NUM_EMBODIMENTS="32"
dataset_name="robocasa_mg100"
config_path="libs/Isaac-GR00T-N1/robomimic_configs/${dataset_name}.json"
model_path="DAVIAN-Robotics/GR00T-N1-2B-tuned-RoboCasa-MG100-FrankaPandaGripper"

acg_options="
algo.guidance.name=acg
algo.guidance.scale=3.0
algo.guidance.skip_blocks=7,9,11
"  # layers 7,9,11 correspond to the 4th–6th self-attention layers

note="${note}_acg"
seed="123"

bash scripts/base_rollout.sh ${config_path} ${model_path} ${seed} ${n_rollouts} ${num_batch_envs} "${note}" ${acg_options}
```

### DexMG

- Compares vanilla GR00T-N1 and ACG-guided versions on DexMG.
- Numbers below are averaged over three random seeds.
- `MAX_NUM_EMBODIMENTS=35` means it newly add 4 additional embodiments (31 + 4).
- `algo_name=gr00t_guidance_dexmg` switches the internal DexMG pipeline logic.
- Smaller `guidance.scale` values (≈ 1.01–1.1) generally yield stable trajectories for dexterous manipulation.
- ⏱️ Estimated rollout time: ~1 hour on a single NVIDIA A6000 GPU.

| Method           |  Seeds  |  Success Rate  |
| ---------------- | :-----: | :------------: |
| Vanilla GR00T-N1 | 123–125 | 40.6% (±3.08%) |
| ACG              | 123–125 | 44.0% (±2.41%) |

```bash
# === DexMG rollout: without guidance ===
note=""
n_rollouts="24"
num_batch_envs="8"
export MAX_NUM_EMBODIMENTS="35"
dataset_name="dexmg_mg100"
config_path="libs/Isaac-GR00T-N1/robomimic_configs/${dataset_name}.json"
model_path="DAVIAN-Robotics/GR00T-N1-2B-tuned-DexMG-MG100-CrossEmbodiments"
seed="123"

bash scripts/base_rollout.sh ${config_path} ${model_path} ${seed} ${n_rollouts} ${num_batch_envs} "${note}" algo_name=gr00t_guidance_dexmg
```

```bash
# === DexMG rollout: with ACG ===
note=""
n_rollouts="24"
num_batch_envs="8"
export MAX_NUM_EMBODIMENTS="35"
dataset_name="dexmg_mg100"
config_path="libs/Isaac-GR00T-N1/robomimic_configs/${dataset_name}.json"
model_path="DAVIAN-Robotics/GR00T-N1-2B-tuned-DexMG-MG100-CrossEmbodiments"

acg_options="
algo.guidance.name=acg
algo.guidance.scale=1.03
algo.guidance.skip_blocks=7,9,11
"

note="${note}_acg"
seed="123"

bash scripts/base_rollout.sh ${config_path} ${model_path} ${seed} ${n_rollouts} ${num_batch_envs} "${note}" algo_name=gr00t_guidance_dexmg ${acg_options}
```
