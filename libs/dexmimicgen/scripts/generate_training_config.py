# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Robomimic Training Configuration Generator

Args:
    --dataset_dir (str): Path to the dataset directory (default: "../datasets").
    --config_dir (str): Path to store the generated training configurations
        (default: "../datasets/train_configs/bcrnn_action_dict").
    --output_dir (str): Path to store the training results
        (default: "../datasets/training_results/bcrnn_action_dict").

Example usage:
    python script.py --dataset_dir /path/to/dataset --config_dir /path/to/configs --output_dir /path/to/results
"""

import argparse
import json
import os
import shutil

import dexmimicgen
import dexmimicgen.utils.config_utils as ConfigUtils

try:
    import robomimic
    from robomimic.utils.hyperparam_utils import ConfigGenerator
except ImportError:
    raise ImportError(
        "Please make sure to install the robomimic package before running this script."
    )

# set path to folder with mimicgen generated datasets
DATASET_DIR = os.path.join(dexmimicgen.__path__[0], "../datasets")

# path to base config
BASE_CONFIG = os.path.join(robomimic.__path__[0], "exps/templates/bc.json")


def make_generators(base_config, dataset_dir, output_dir):
    """
    An easy way to make multiple config generators by using different
    settings for each.
    """
    panda_settings = [
        # two arm box cleanup
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "generated", "two_arm_box_cleanup.hdf5"),
            ],
            dataset_names=[
                "two_arm_box_cleanup_D0",
            ],
            image_keys=[
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot1_eye_in_hand_image",
            ],
            low_dim_keys=[
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot1_eef_pos",
                "robot1_eef_quat",
                "robot1_gripper_qpos",
            ],
            horizon=[400],
        ),
        # two arm lift tray
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "generated", "two_arm_lift_tray.hdf5"),
            ],
            dataset_names=[
                "two_arm_lift_tray_D0",
            ],
            image_keys=[
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot1_eye_in_hand_image",
            ],
            low_dim_keys=[
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot1_eef_pos",
                "robot1_eef_quat",
                "robot1_gripper_qpos",
            ],
            horizon=[750],
        ),
        # two arm drawer cleanup
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "generated", "two_arm_drawer_cleanup.hdf5"),
            ],
            dataset_names=[
                "two_arm_drawer_cleanup_D0",
            ],
            image_keys=[
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot1_eye_in_hand_image",
            ],
            low_dim_keys=[
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot1_eef_pos",
                "robot1_eef_quat",
                "robot1_gripper_qpos",
            ],
            horizon=[550],
        ),
        # two arm three piece assembly
        dict(
            dataset_paths=[
                os.path.join(
                    dataset_dir, "generated", "two_arm_three_piece_assembly.hdf5"
                ),
            ],
            dataset_names=[
                "two_arm_three_piece_assembly_D0",
            ],
            image_keys=[
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot1_eye_in_hand_image",
            ],
            low_dim_keys=[
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot1_eef_pos",
                "robot1_eef_quat",
                "robot1_gripper_qpos",
            ],
            horizon=[300],
        ),
        # two arm transport
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "generated", "two_arm_transport.hdf5"),
            ],
            dataset_names=[
                "two_arm_transport_D0",
            ],
            image_keys=[
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot1_eye_in_hand_image",
                "shouldercamera0_image",
                "shouldercamera1_image",
            ],
            low_dim_keys=[
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot1_eef_pos",
                "robot1_eef_quat",
                "robot1_gripper_qpos",
            ],
            horizon=[1200],
        ),
        # two arm threading
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "generated", "two_arm_threading.hdf5"),
            ],
            dataset_names=[
                "two_arm_threading_D0",
            ],
            image_keys=[
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot1_eye_in_hand_image",
            ],
            low_dim_keys=[
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot1_eef_pos",
                "robot1_eef_quat",
                "robot1_gripper_qpos",
            ],
            horizon=[400],
        ),
    ]

    humanoid_settings = [
        # two arm pouring humanoid
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "generated", "two_arm_pouring.hdf5"),
            ],
            dataset_names=[
                "two_arm_pouring_humanoid_D0",
            ],
            image_keys=[
                "agentview_image",
                "robot0_eye_in_left_hand_image",
                "robot0_eye_in_right_hand_image",
            ],
            low_dim_keys=[
                "robot0_right_eef_pos",
                "robot0_right_eef_quat",
                "robot0_right_gripper_qpos",
                "robot0_left_eef_pos",
                "robot0_left_eef_quat",
                "robot0_left_gripper_qpos",
            ],
            horizon=[400],
        ),
        # two arm coffee humanoid
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "generated", "two_arm_coffee.hdf5"),
            ],
            dataset_names=[
                "two_arm_coffee_humanoid_D0",
            ],
            image_keys=[
                "agentview_image",
                "robot0_eye_in_left_hand_image",
                "robot0_eye_in_right_hand_image",
            ],
            low_dim_keys=[
                "robot0_right_eef_pos",
                "robot0_right_eef_quat",
                "robot0_right_gripper_qpos",
                "robot0_left_eef_pos",
                "robot0_left_eef_quat",
                "robot0_left_gripper_qpos",
            ],
            horizon=[400],
        ),
        # two arm can sort humanoid
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "generated", "two_arm_can_sort_random.hdf5"),
            ],
            dataset_names=[
                "two_arm_can_sort_humanoid_D0",
            ],
            image_keys=[
                "frontview_image",
                "robot0_eye_in_left_hand_image",
                "robot0_eye_in_right_hand_image",
            ],
            low_dim_keys=[
                "robot0_right_eef_pos",
                "robot0_right_eef_quat",
                "robot0_right_gripper_qpos",
                "robot0_left_eef_pos",
                "robot0_left_eef_quat",
                "robot0_left_gripper_qpos",
            ],
            horizon=[400],
        ),
    ]

    ret = []
    for setting in panda_settings:
        generator = make_gen(os.path.expanduser(base_config), setting, output_dir)
        generator = panda_action_config(generator)
        ret.append(generator)

    for setting in humanoid_settings:
        generator = make_gen(os.path.expanduser(base_config), setting, output_dir)
        generator = humanoid_action_config(generator)
        ret.append(generator)

    return ret


def make_gen(base_config, settings, output_dir):
    """
    Specify training configs to generate here.
    """
    generator = ConfigGenerator(
        base_config_file=base_config,
        script_file="",  # will be overriden in next step
        base_exp_name="bc_rnn_image",
    )

    # set algo settings for bc-rnn
    low_dim_keys = settings.get("low_dim_keys", None)
    image_keys = settings.get("image_keys", None)
    crop_size = settings.get("crop_size", [76, 76])
    assert low_dim_keys is not None, "low_dim_keys must be provided"
    assert image_keys is not None, "image_keys must be provided"
    assert len(crop_size) == 2, "crop_size must be a 2-tuple"

    ConfigUtils.set_learning_settings_for_bc_rnn(
        generator=generator,
        group=-1,
        seq_length=10,
        low_dim_keys=low_dim_keys,
        image_keys=image_keys,
        crop_size=crop_size,
        dataset_paths=settings["dataset_paths"],
        dataset_names=settings["dataset_names"],
        horizon=settings["horizon"],
        output_dir=output_dir,
    )

    return generator


def panda_action_config(generator):
    action_dict_name = "action_dict"
    # action dict
    generator.add_param(
        key="train.action_keys",
        name="",
        group=4,
        values=[
            [
                "{}/right_rel_pos".format(action_dict_name),
                "{}/right_rel_rot_axis_angle".format(action_dict_name),
                "{}/right_gripper".format(action_dict_name),
                "{}/left_rel_pos".format(action_dict_name),
                "{}/left_rel_rot_axis_angle".format(action_dict_name),
                "{}/left_gripper".format(action_dict_name),
            ],
        ],
    )
    generator.add_param(
        key="train.action_config",
        name="",
        group=4,
        values=[
            {
                "{}/right_rel_pos".format(action_dict_name): {
                    "normalization": None,
                },
                "{}/right_rel_rot_axis_angle".format(action_dict_name): {
                    "normalization": None,
                    "format": "rot_axis_angle",
                },
                "{}/right_gripper".format(action_dict_name): {
                    "normalization": "min_max",
                },
                "{}/left_rel_pos".format(action_dict_name): {
                    "normalization": None,
                },
                "{}/left_rel_rot_axis_angle".format(action_dict_name): {
                    "normalization": None,
                    "format": "rot_axis_angle",
                },
                "{}/left_gripper".format(action_dict_name): {
                    "normalization": "min_max",
                },
            }
        ],
    )
    return generator


def humanoid_action_config(generator):
    action_dict_name = "action_dict"
    # action dict
    generator.add_param(
        key="train.action_keys",
        name="",
        group=4,
        values=[
            [
                "{}/right_abs_pos".format(action_dict_name),
                "{}/right_abs_rot_6d".format(action_dict_name),
                "{}/left_abs_pos".format(action_dict_name),
                "{}/left_abs_rot_6d".format(action_dict_name),
                "{}/right_gripper".format(action_dict_name),
                "{}/left_gripper".format(action_dict_name),
            ],
        ],
    )
    generator.add_param(
        key="train.action_config",
        name="",
        group=4,
        values=[
            {
                "{}/right_abs_pos".format(action_dict_name): {
                    "normalization": "min_max",
                },
                "{}/right_abs_rot_6d".format(action_dict_name): {
                    "normalization": None,
                    "format": "rot_6d",
                },
                "{}/left_abs_pos".format(action_dict_name): {
                    "normalization": "min_max",
                },
                "{}/left_abs_rot_6d".format(action_dict_name): {
                    "normalization": None,
                    "format": "rot_6d",
                },
                "{}/right_gripper".format(action_dict_name): {
                    "normalization": "min_max",
                },
                "{}/left_gripper".format(action_dict_name): {
                    "normalization": "min_max",
                },
            }
        ],
    )
    return generator


def main(args):
    # make config generators
    generators = make_generators(
        base_config=BASE_CONFIG,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )

    if os.path.exists(args.config_dir):
        ans = input(
            "Non-empty dir at {}. \nContinue (y / n)? \n".format(args.config_dir)
        )
        if ans != "y":
            exit()
        shutil.rmtree(args.config_dir)

    all_json_files, run_lines = ConfigUtils.config_generator_to_script_lines(
        generators, config_dir=args.config_dir
    )

    run_lines = [line.strip() for line in run_lines]

    print("configs")
    print(json.dumps(all_json_files, indent=4))
    print("runs")
    print(json.dumps(run_lines, indent=4))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DATASET_DIR,
        help="Path to dataset",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=os.path.join(DATASET_DIR, "train_configs/bcrnn_action_dict"),
        help="Path to store the training configs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(DATASET_DIR, "training_results/bcrnn_action_dict"),
        help="Path to store the training results",
    )

    args = parser.parse_args()
    main(args)
