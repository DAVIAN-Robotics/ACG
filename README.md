# ACG: Action Coherence Guidance for Flow-based VLA Models

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![GitHub Code](https://img.shields.io/badge/Code-GitHub-black.svg?logo=github)](https://github.com/davian-robotics/ACG)
[![Project Page](https://img.shields.io/badge/Project_Page-Visit-blue.svg)](https://davian-robotics.github.io/ACG)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/DAVIAN-Robotics/gr00t-n1-2b-post-trained-models)
[![YouTube Demo](https://img.shields.io/badge/YouTube-Video-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=Fi6VpnPETYU)

> [Minho Park\*](https://pmh9960.github.io/), [Kinam Kim\*](https://kinam0252.github.io/), [Junha Hyung](https://junhahyung.github.io/), [Hyojin Jang](https://github.com/Whit3Snow), [Hoiyeong Jin](https://myyzzzoooo.github.io/), [Jooyeol Yun](https://yeolj00.github.io/), [Hojoon Lee](https://joonleesky.github.io/), and [Jaegul Choo](https://sites.google.com/site/jaegulchoo/)  
> **DAVIAN Robotics, KAIST AI**  
> arXiv 2025. (\* indicates equal contribution)

## 🌐 Overview

**Action Coherence Guidance (ACG)** is a **training-free, test-time guidance algorithm** that improves **temporal and spatial action consistency** in Vision-Language-Action (VLA) models.
It mitigates motion jitter, unintended pauses, and trajectory drift caused by noisy demonstrations, resulting in stable and precise robotic manipulation.

<div align="center">
  <video src="https://github.com/user-attachments/assets/7b60dee5-864a-4e6b-9ad4-3b7ee868c803" width="70%" poster="./assets/teaser_thumbnail.jpg"> </video>
</div>

### 🔑 Key Features

- **Training-Free Guidance**: Enhances action coherence during inference without retraining or fine-tuning.
- **Plug-and-Play Integration**: Seamlessly compatible with existing **diffusion** and **flow-matching** VLA policies across multiple benchmarks.
- **Proven Performance**: Demonstrates consistent improvements in success rate on **RoboCasa**, **DexMimicGen**, and **real-world SO-101** manipulation tasks.

## ⚙️ Getting Started

### 📦 Installation Guide

```bash
# Create conda environment
conda create -n acg python=3.10 -y
conda activate acg

# Install PyTorch (adjust CUDA version if needed)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install core dependencies
pip install -e libs/Isaac-GR00T-N1
pip install --no-build-isolation flash-attn==2.8.3

# Robosuite and RoboCasa
pip install libs/robosuite/
pip install -e libs/robocasa/

# Download RoboCasa assets (~5GB)
python libs/robocasa/robocasa/scripts/download_kitchen_assets.py

# Robomimic and DexMimicGen
pip install -e libs/robomimic/ --no-dependencies
pip install -e libs/dexmimicgen/ --no-dependencies

# Remaining requirements
pip install -r requirements.txt
```

## 🧩 Repository Features

> To the best of our knowledge, this is the **first repository** providing post-trained models and rollout scripts for both **RoboCasa** and **DexMimicGen** benchmarks.
>
> - **RoboCasa**: Reproduced score (**32.6**) closely matches the reported result (**32.1**).
> - **DexMimicGen**: Reproduced score (**40.6**) is lower than the reported (**58.5**); the cause is under investigation.
>
> ⚠️ _This is an unofficial reproduction. Contributions and issue reports are highly welcome._

### Repository Highlights

- **Self-contained Finetuning & Inference** for [GR00T-N1-2B](https://github.com/NVIDIA/Isaac-GR00T).
- **Training script**: [`gr00t_finetune_robocasa.py`](./libs/Isaac-GR00T-N1/scripts/gr00t_finetune_robocasa.py)

  - Utilizes a **modified Robomimic DataLoader** to finetune GR00T-N1-2B on RoboCasa and DexMimicGen datasets.
  - Supports multiple embodiments — _SinglePandaGripper_ for RoboCasa, and _BimanualPandaGripper_, _BimanualPandaHand_, _GR1_ for DexMimicGen.
  - Note: the official GR00T-N1-2B post-training code only supports humanoid (GR1) embodiments.

- **Rollout script**: [`rollout_with_robomimic.py`](./libs/Isaac-GR00T-N1/scripts/rollout_with_robomimic.py)

  - Adapted from the Robomimic rollout framework.
  - You can directly perform rollouts using our finetuned models from [Hugging Face](https://huggingface.co/collections/DAVIAN-Robotics/gr00t-n1-2b-post-trained-models).

## 🚀 Quick Start

We provide RoboCasa examples below.
For DexMimicGen and more detailed scripts, see [`./scripts/run_gr00t.md`](./scripts/run_gr00t.md).

### 🔧 Training (Post-Training Phase)

> Before starting, make sure to download each dataset from its **official repository**: [RoboCasa](https://github.com/robocasa/robocasa), [DexMimicGen](https://github.com/NVlabs/dexmimicgen).  
> Then, update the dataset paths in the [Robomimic config files](libs/Isaac-GR00T-N1/robomimic_configs) accordingly.

```bash
n_mg="100"
ngpu="1"
bs="64"
ga="2"
steps="60000"
training_seed="42"
exp_name="MG${n_mg}/LR=1e-4_Bs=${ngpu}x${bs}x${ga}_Steps=${steps}_Seed=${training_seed}${note}"

export WANDB_ENTITY="your-entity"
export WANDB_PROJECT="Your Robot Project"

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

### 🤖 Inference (Rollout)

Without ACG:

```bash
note=""
n_rollouts="24"
num_batch_envs="8"
export MAX_NUM_EMBODIMENTS="32"
dataset_name="robocasa_mg100"
config_path="libs/Isaac-GR00T-N1/robomimic_configs/${dataset_name}.json"
model_path="DAVIAN-Robotics/GR00T-N1-2B-tuned-RoboCasa-MG100-FrankaPandaGripper"
seed="123"

bash scripts/base_rollout.sh ${config_path} ${model_path} ${seed} ${n_rollouts} ${num_batch_envs} "${note}"
```

With ACG enabled:

```bash
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
"  # Corresponding to 4th–6th self-attention layers.

note="${note}_acg"
seed="123"

bash scripts/base_rollout.sh ${config_path} ${model_path} ${seed} ${n_rollouts} ${num_batch_envs} "${note}" ${acg_options}
```

## 🧾 Citation

```bibtex
@article{park2025acg,
  title={ACG: Action Coherence Guidance for Flow-based VLA Models},
  author={Park, Minho and Kim, Kinam and Hyung, Junha and Jang, Hyojin and Jin, Hoiyeong and Yun, Jooyeol and Lee, Hojoon and Choo, Jaegul},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

## 🙏 Acknowledgement

This repository builds upon the incredible open-source efforts of
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T),
[Robosuite](https://github.com/ARISE-Initiative/robosuite),
[Robomimic](https://github.com/ARISE-Initiative/robomimic),
[RoboCasa](https://github.com/robocasa/robocasa),
[DexMimicGen](https://github.com/NVlabs/dexmimicgen), and
[Diffusers](https://github.com/huggingface/diffusers).  
We sincerely appreciate their outstanding contributions to the robotics and AI community.
