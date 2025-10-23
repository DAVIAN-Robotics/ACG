# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import dexmimicgen.utils.transform_utils as T
import numpy as np
from dexmimicgen.environments.two_arm_dexmg_env import TwoArmDexMGEnv
from dexmimicgen.models.objects import NeedleObject, RingTripodObject
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from .my_uniform_random_sampler import MyUniformRandomSampler


class TwoArmThreading(TwoArmDexMGEnv):

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        table_offset=(0.0, 0.0, 0.8),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        *args,
        **kwargs,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,  # {None, instance, class, element}
            renderer=renderer,
            renderer_config=renderer_config,
            *args,
            **kwargs,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Dense reward: TODO

        The sparse reward only consists of the threading component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            pass

        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        # self.needle = NeedleObject(name="needle")
        # self.tripod = RingTripodObject(name="tripod")
        self.needle = NeedleObject(name="needle_obj")
        self.tripod = RingTripodObject(name="tripod_obj")
        objects = [self.needle, self.tripod]

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=MyUniformRandomSampler(
                rng=self.rng,
                name="NeedleSampler",
                mujoco_objects=self.needle,
                # x_range=(-0.05278, -0.05278),  # ! sanity check (demo_1)
                # y_range=(0.18989, 0.18989),  # ! sanity check (demo_1)
                # rotation=(-0.39542 * np.pi, -0.39542 * np.pi),  # ! sanity check (demo_1)
                x_range=(-0.2, -0.05),
                y_range=(0.15, 0.25),
                rotation=(-2.0 * np.pi / 3.0, -np.pi / 3.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.0,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=MyUniformRandomSampler(
                rng=self.rng,
                name="TripodSampler",
                mujoco_objects=self.tripod,
                # x_range=(0.03661, 0.03661),  # ! sanity check (demo_1)
                # y_range=(-0.12291, -0.12291),  # ! sanity check (demo_1)
                # rotation=(0.32831 * np.pi, 0.32831 * np.pi),  # ! sanity check (demo_1)
                x_range=(-0.1, 0.15),
                y_range=(-0.2, -0.1),
                rotation=(np.pi / 6.0, np.pi / 2.0),
                # TODO: swap this in to be consistent with D0 in MimicGen
                # x_range=(0.0, 0.0),
                # y_range=(-0.15, -0.15),
                # rotation=(np.pi / 2., np.pi / 2.),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            )
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = dict(
            needle=self.sim.model.body_name2id(self.needle.root_body),
            tripod=self.sim.model.body_name2id(self.tripod.root_body),
        )

    def _check_success(self):
        """
        Check if needle has been inserted into ring.
        """

        # needle_pos = np.array(self.sim.data.geom_xpos[self.sim.model.geom_name2id("needle_needle")])
        needle_pos = np.array(
            self.sim.data.geom_xpos[self.sim.model.geom_name2id("needle_obj_needle")]
        )

        # ring position is average of all the surrounding ring geom positions
        ring_pos = np.zeros(3)
        for i in range(self.tripod.num_ring_geoms):
            # ring_pos += np.array(self.sim.data.geom_xpos[self.sim.model.geom_name2id("tripod_ring_{}".format(i))])
            ring_pos += np.array(
                self.sim.data.geom_xpos[
                    self.sim.model.geom_name2id("tripod_obj_ring_{}".format(i))
                ]
            )
        ring_pos /= self.tripod.num_ring_geoms

        # radius should be the ring size, since we want to check that the bar is within the ring
        radius = self.tripod.ring_size[1]

        # check if the center of the block and the hole are close enough
        return np.linalg.norm(needle_pos - ring_pos) < radius

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the needle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper["right"], target=self.needle
            )
