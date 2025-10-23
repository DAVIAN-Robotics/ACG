# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import shlex
import shutil
import tempfile


def set_learning_settings_for_bc_rnn(
    generator,
    group,
    seq_length=10,
    low_dim_keys=None,
    image_keys=None,
    crop_size=None,
    dataset_paths=None,
    dataset_names=None,
    horizon=400,
    output_dir=None,
):
    """
    Sets config generator parameters for robomimic BC-RNN training runs.

    Args:
        generator (robomimic ConfigGenerator instance): config generator object
        group (int): parameter group for these settings
        seq_length (int): BC-RNN context length
        low_dim_keys (list or None): if provided, set low-dim observation keys, else use defaults
        image_keys (list or None): if provided, set image observation keys, else use defaults
        crop_size (tuple or None): if provided, size of crop to use for pixel shift augmentation
    """
    # setup RNN with GMM and desired seq_length
    generator.add_param(
        key="train.seq_length",
        name="",
        group=group,
        values=[seq_length],
    )
    generator.add_param(
        key="train.frame_stack",
        name="",
        group=group,
        values=[1],
    )
    generator.add_param(
        key="algo.rnn.horizon",
        name="",
        group=group,
        values=[seq_length],
    )
    generator.add_param(
        key="algo.rnn.enabled",
        name="",
        group=group,
        values=[True],
    )
    generator.add_param(
        key="algo.gmm.enabled",
        name="",
        group=group,
        values=[False],
    )
    actor_layer_dims = []
    generator.add_param(
        key="algo.actor_layer_dims",
        name="",
        group=group,
        values=[actor_layer_dims],
    )

    # epoch settings
    epoch_every_n_steps = 500
    validation_epoch_every_n_steps = 50
    eval_rate = 100

    # learning settings
    num_epochs = 600
    batch_size = 16
    policy_lr = 1e-4
    rnn_hidden_dim = 1000

    assert low_dim_keys is not None
    assert image_keys is not None
    assert crop_size is not None

    generator.add_param(
        key="observation.encoder.rgb",
        name="",
        group=group,
        values=[
            {
                "core_class": "VisualCore",
                "core_kwargs": {
                    "feature_dimension": 64,
                    "flatten": True,
                    "backbone_class": "ResNet18Conv",
                    "backbone_kwargs": {
                        "pretrained": False,
                        "input_coord_conv": False,
                    },
                    "pool_class": "SpatialSoftmax",
                    "pool_kwargs": {
                        "num_kp": 32,
                        "learnable_temperature": False,
                        "temperature": 1.0,
                        "noise_std": 0.0,
                        "output_variance": False,
                    },
                },
                "obs_randomizer_class": "CropRandomizer",
                "obs_randomizer_kwargs": {
                    "crop_height": crop_size[0],
                    "crop_width": crop_size[1],
                    "num_crops": 1,
                    "pos_enc": False,
                },
            }
        ],
    )

    generator.add_param(
        key="observation.modalities.obs.low_dim",
        name="",
        group=group,
        values=[low_dim_keys],
    )
    generator.add_param(
        key="observation.modalities.obs.rgb",
        name="",
        group=group,
        values=[image_keys],
    )

    # epoch settings
    generator.add_param(
        key="experiment.epoch_every_n_steps",
        name="",
        group=group,
        values=[epoch_every_n_steps],
    )
    generator.add_param(
        key="experiment.validation_epoch_every_n_steps",
        name="",
        group=group,
        values=[validation_epoch_every_n_steps],
    )
    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="",
        group=group,
        values=[eval_rate],
    )
    generator.add_param(
        key="experiment.rollout.rate",
        name="",
        group=group,
        values=[eval_rate],
    )

    # learning settings
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=group,
        values=[num_epochs],
    )
    generator.add_param(
        key="train.batch_size",
        name="",
        group=group,
        values=[batch_size],
    )
    generator.add_param(
        key="algo.optim_params.policy.learning_rate.initial",
        name="",
        group=group,
        values=[policy_lr],
    )
    generator.add_param(
        key="algo.rnn.hidden_dim",
        name="",
        group=group,
        values=[rnn_hidden_dim],
    )

    # 4 data workers and low-dim cache mode seems to work well for both low-dim and image observations
    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=group,
        values=[4],
    )
    generator.add_param(
        key="train.hdf5_cache_mode",
        name="",
        group=group,
        values=["low_dim"],
    )

    # set dataset
    generator.add_param(
        key="train.data",
        name="ds",
        group=0,
        values=dataset_paths,
        value_names=dataset_names,
    )

    # rollout settings
    generator.add_param(
        key="experiment.rollout.horizon",
        name="",
        group=1,
        values=horizon,
    )

    # output path
    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[output_dir],
    )

    # seed
    generator.add_param(
        key="train.seed",
        name="seed",
        group=100000,
        values=[201, 202, 203],
    )

    # batched rollout
    generator.add_param(
        key="experiment.rollout.batched",
        name="",
        group=3,
        values=[True],
    )

    generator.add_param(
        key="experiment.rollout.num_batch_envs",
        name="",
        group=3,
        values=[25],
    )

    return generator


def config_generator_to_script_lines(generator, config_dir):
    """
    Takes a robomimic ConfigGenerator and uses it to
    generate a set of training configs, and a set of bash command lines
    that correspond to each training run (one per config). Note that
    the generator's script_file will be overridden to be a temporary file that
    will be removed from disk.

    Args:
        generator (ConfigGenerator instance or list): generator(s)
            to use for generating configs and training runs

        config_dir (str): path to directory where configs will be generated

    Returns:
        config_files (list): a list of config files that were generated

        run_lines (list): a list of strings that are training commands, one per config
    """

    # make sure config dir exists
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # support one or more config generators
    if not isinstance(generator, list):
        generator = [generator]

    all_run_lines = []
    for gen in generator:

        # set new config directory by copying base config file from old location to new directory
        base_config_file = gen.base_config_file
        config_name = os.path.basename(base_config_file)
        new_base_config_file = os.path.join(config_dir, config_name)
        shutil.copyfile(
            base_config_file,
            new_base_config_file,
        )
        gen.base_config_file = new_base_config_file

        # we'll write script file to a temp dir and parse it from there to get the training commands
        with tempfile.TemporaryDirectory() as td:
            gen.script_file = os.path.join(td, "tmp.sh")

            # generate configs
            gen.generate()

            # collect training commands
            with open(gen.script_file, "r") as f:
                f_lines = f.readlines()
                run_lines = [line for line in f_lines if line.startswith("python")]
                all_run_lines += run_lines

        os.remove(gen.base_config_file)

    # get list of generated configs too
    config_files = []
    config_file_dict = dict()
    for line in all_run_lines:
        cmd = shlex.split(line)
        config_file_name = cmd[cmd.index("--config") + 1]
        config_files.append(config_file_name)
        assert (
            config_file_name not in config_file_dict
        ), "got duplicate config name {}".format(config_file_name)
        config_file_dict[config_file_name] = 1

    return config_files, all_run_lines
