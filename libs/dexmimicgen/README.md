# DexMimicGen

<p align="center">
  <img width="95.0%" src="images/dexmimicgen.gif">
</p>

This repository contains the official release of simulation environments and datasets for the [ICRA 2025](https://2025.ieee-icra.org) paper "DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning".

Website: https://dexmimicgen.github.io

Paper: https://arxiv.org/abs/2410.24185

For business inquiries, please submit this form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

-------

## Getting Started

To use this repository, you need to first install the latest robosuite. For more
information, please refer to [robosuite](https://github.com/ARISE-Initiative/robosuite).

```bash
git clone https://github.com/ARISE-Initiative/robosuite
pip install -e robosuite
```

Then git clone this repository and install.

```bash
git clone https://github.com/NVlabs/dexmimicgen.git
cd dexmimicgen
pip install -e .
```

After installation, you can run the following command to test the environments.

```bash
python scripts/demo_random_action.py --env TwoArmThreading --render
```

Note: If you are on a headless machine, you can run without the `--render` flag.

## Environments

For detailed information about the environments, please refer to [environments.md](environments.md).

## Datasets

You can download the datasets from [HuggingFace](https://huggingface.co/datasets/MimicGen/dexmimicgen_datasets/tree/main).

You can also run the script to download the datasets.

```bash
python scripts/download_hf_dataset.py --path /path/to/save/datasets
```

By default, the datasets will be saved to `./datasets`.

And then, you can playback one demo in the dataset by running:

```bash
python scripts/playback_datasets.py --dataset xxxxx.hdf5 --n 1
```

## Launch Training with robomimic

We provide config and training code to reproduce the BC-RNN result in our paper.

First, you need to install robomimic

```bash

git clone https://github.com/ARISE-Initiative/robomimic.git -b dexmimicgen
cd robomimic
pip install -e .
```

Then you need to generate the config file for the training.

```bash
cd dexmimicgen
python scripts/generate_training_config.py --dataset_dir /path/to/datasets --config_dir /path/to/save/config --output_dir /path/to/save/output
```

By default, it will try to find the datasets in `./datasets`, and save the config and output in `./datasets/train_configs/bcrnn_action_dict` and `./datasets/train_results/bcrnn_action_dict` respectively.

After that, you can run the training script.

```bash
cd robomimic
python scripts/train.py --config /path/to/config
```

## License

The code is released under the [NVIDIA Source Code License](https://github.com/NVlabs/mimicgen/blob/main/LICENSE) and the datasets are released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

Please cite [the DexMimicGen paper](https://arxiv.org/abs/2410.24185) if you use this code in your work:

```bibtex
@inproceedings{jiang2024dexmimicen,
      title     = {DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning},
      author    = {Jiang, Zhenyu and Xie, Yuqi and Lin, Kevin and Xu, Zhenjia and Wan, Weikang and Mandlekar, Ajay and Fan, Linxi and Zhu, Yuke},
      booktitle = {2025 IEEE International Conference on Robotics and Automation (ICRA)},
      year      = {2025}
}
```
