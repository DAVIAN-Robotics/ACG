# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Run random actions in dexmimicgen environments.

Args:
    --env (str): Name of the environment to run (default: "TwoArmThreading").
    --render (bool): Whether to render the environment.

Example usage:
    python script.py --env TwoArmPouring --render
"""

import argparse

import numpy as np
import robosuite
from robosuite import load_composite_controller_config

# IMPORTANT: you need to import the package to register the environments
import dexmimicgen

ENV_ROBOTS = {
    "TwoArmThreading": ["Panda", "Panda"],
    "TwoArmThreePieceAssembly": ["Panda", "Panda"],
    "TwoArmTransport": ["Panda", "Panda"],
    "TwoArmLiftTray": ["PandaDexRH", "PandaDexLH"],
    "TwoArmBoxCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmDrawerCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmCoffee": ["GR1FixedLowerBody"],
    "TwoArmPouring": ["GR1FixedLowerBody"],
    "TwoArmCanSortRandom": ["GR1ArmsOnly"],
}

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="TwoArmThreading",
        help="Name of the environment to run",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Whether to render the environment",
    )

    args = parser.parse_args()

    assert args.env in ENV_ROBOTS, f"Environment {args.env} not found!"

    # Create dict to hold options that will be passed to env creation call
    env_kwargs = {
        "env_name": args.env,
        "robots": ENV_ROBOTS[args.env],
        "controller_configs": load_composite_controller_config(
            robot=ENV_ROBOTS[args.env][0]
        ),
        "has_renderer": args.render,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_camera_obs": True,
        "control_freq": 20,
    }

    # initialize the task
    env = robosuite.make(
        **env_kwargs,
    )
    env.reset()
    if args.render:
        env.render()

    # do visualization
    for i in range(1000):
        # action = np.zeros_like(low)
        action = np.random.randn(*env.action_spec[0].shape)
        obs, reward, done, _ = env.step(action)
        if args.render:
            env.render()
