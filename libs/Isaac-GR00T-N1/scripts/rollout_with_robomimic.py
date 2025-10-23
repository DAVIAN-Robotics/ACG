import argparse
import datetime
import glob
import json
import os
import time

import dexmimicgen
import jsonlines
import numpy as np
import robomimic
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
import torch
from robomimic.algo import RolloutPolicy, RolloutPolicyWOLangEncoder, algo_factory
from robomimic.config import config_factory
from robomimic.utils.script_utils import deep_update
from tools_mpark.dictaction import DictAction


def train(config, device):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    output_dir = None
    if config.experiment.rollout.resume:
        try:
            jsonl_path_regex = os.path.join(os.path.expanduser(config.train.output_dir), config.experiment.name, "*", "results.jsonl")
            jsonl_path = sorted(glob.glob(jsonl_path_regex))[-1]
            output_dir = os.path.dirname(jsonl_path)
            print(f"resuming rollouts to {output_dir}")
        except Exception as e:
            print("EXCEPTION! resuming rollouts led to error:")
            print(e)
            print("starting a new rollout instead")

    if output_dir is None:
        time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        output_dir = os.path.join(os.path.expanduser(config.train.output_dir), config.experiment.name, time_str)

    video_dir = os.path.join(output_dir, "videos") if config.experiment.render_video else None
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True) if config.experiment.render_video else None

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # extract the metadata and shape metadata across all datasets
    env_meta_list, shape_meta_list = [], []
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        ds_format = config.train.data_format
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        # print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)

        # populate language instruction for env in env_meta
        env_meta["env_lang"] = dataset_cfg.get("lang", None)

        if env_meta['env_name'] == 'TwoArmCanSortBlue':  # ! todo hard coding, the dataset file is wrong
            deep_update(env_meta, {'env_name': 'TwoArmCanSortRandom', 'env_kwargs': {'camera_names': ['agentview', 'robot0_eye_in_right_hand', 'robot0_eye_in_left_hand']}})
        deep_update(env_meta, dataset_cfg.get("env_meta_update_dict", {}))
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=False
        )
        shape_meta_list.append(shape_meta)

    envs_done = []
    if config.experiment.rollout.resume and os.path.exists(os.path.join(output_dir, "results.jsonl")):
        with jsonlines.open(os.path.join(output_dir, "results.jsonl")) as f:
            lines = list(f.iter())
        envs_done = [line['env'] for line in lines]

    eval_env_meta_list, eval_shape_meta_list, eval_env_name_list, eval_env_horizon_list = [], [], [], []
    for (dataset_i, dataset_cfg) in enumerate(config.train.data):
        do_eval = dataset_cfg.get("do_eval", True)
        if do_eval is not True:
            continue
        if env_meta_list[dataset_i]["env_name"] in envs_done:
            print(f"skipping {env_meta_list[dataset_i]['env_name']} because already done")
            continue
        eval_env_meta_list.append(env_meta_list[dataset_i])
        eval_shape_meta_list.append(shape_meta_list[dataset_i])
        eval_env_name_list.append(env_meta_list[dataset_i]["env_name"])
        horizon = dataset_cfg.get("horizon", config.experiment.rollout.horizon)
        eval_env_horizon_list.append(horizon)

    # create environments
    def env_iterator():
        for (env_meta, shape_meta, env_name) in zip(eval_env_meta_list, eval_shape_meta_list, eval_env_name_list):
            def create_env_helper(env_i=0):
                env_kwargs = dict(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"],
                    seed=config.train.seed * 1000 + env_i,
                )
                env = EnvUtils.create_env_from_metadata(**env_kwargs)
                # handle environment wrappers
                env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable

                return env

            if config.experiment.rollout.batched:
                from tianshou.env import SubprocVectorEnv
                env_fns = [lambda env_i=i: create_env_helper(env_i) for i in range(config.experiment.rollout.num_batch_envs)]
                env = SubprocVectorEnv(env_fns)
            else:
                env = create_env_helper()
            print(env)
            yield env

    # setup for a new training run
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device,
    )

    # wrap model as a RolloutPolicy to prepare for rollouts
    rollout_model = RolloutPolicyWOLangEncoder(
        model,
        obs_normalization_stats=None,
        action_normalization_stats=None,
        lang_encoder=None,
    )

    TrainUtils.rollout_with_stats(
        policy=rollout_model,
        envs=env_iterator(),
        horizon=eval_env_horizon_list,
        use_goals=config.use_goals,
        num_episodes=config.experiment.rollout.n,
        render=False,
        output_dir=output_dir,
        video_dir=video_dir if config.experiment.render_video else None,
        epoch=None,
        video_skip=config.experiment.get("video_skip", 5),
        terminate_on_success=config.experiment.rollout.terminate_on_success,
        del_envs_after_rollouts=True,
        data_logger=None,
        ds_format=config.train.data_format,
    )


def main(args):

    with open(os.path.expanduser(args.config), 'r') as f:
        ext_cfg = json.load(f)

    config = config_factory("gr00t")  # ! todo hard coding

    with config.values_unlocked():
        config.update(ext_cfg)
    config.update(args.config_add)

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # print(config)
    train(config, device=device)
    print("finished run successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="path to a config json.", required=True)
    parser.add_argument("--config_add", action=DictAction, nargs='+', default=dict())

    args = parser.parse_args()

    main(args)
