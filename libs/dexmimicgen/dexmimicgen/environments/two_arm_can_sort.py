# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import dexmimicgen.utils.transform_utils as T
import numpy as np
import robosuite
from dexmimicgen.environments.two_arm_dexmg_env import TwoArmDexMGEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject
from robosuite.models.objects.composite.bin import Bin
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import string_to_array
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from .my_uniform_random_sampler import MyUniformRandomSampler


class TwoArmCanSortRandom(TwoArmDexMGEnv):

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 1.2, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        table_offset=(0.0, 0.0, 0.9),
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
        red_prob=0.5,
        use_cylinder=True,
        *args,
        **kwargs,
    ):

        self.red_prob = red_prob
        self.is_red = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)
        self.use_cylinder = use_cylinder
        self.can_height = 0.18

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        self._initialize_object_states()

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
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            *args,
            **kwargs,
        )

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
        self.red_box = Bin(
            name="red_box",
            bin_size=(0.25, 0.2, 0.05),
            wall_thickness=0.01,
            transparent_walls=False,
            friction=None,
            density=100000.0,
            use_texture=False,
            rgba=(0.58, 0.15, 0.10, 1.0),
        )

        self.blue_box = Bin(
            name="blue_box",
            bin_size=(0.25, 0.2, 0.05),
            wall_thickness=0.01,
            transparent_walls=False,
            friction=None,
            density=100000.0,
            use_texture=False,
            rgba=(0.53, 0.77, 0.95, 1.0),
        )

        if self.rng.uniform(0.0, 1.0) < self.red_prob:
            self.is_red = True
            can_rgba = (0.58, 0.15, 0.10, 1.0)
        else:
            self.is_red = False
            can_rgba = (0.53, 0.77, 0.95, 1.0)
        self.can = CylinderObject(
            name="cube",
            size=[0.03, self.can_height / 2],
            rgba=can_rgba,
        )

        objects = [self.red_box, self.blue_box, self.can]

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

        self._modify_camera_view()

    def _modify_camera_view(self):
        # Modify the agentview camera to have a higher z-axis position
        self.model.mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array("-0.4 0 1.5"),  # Increased z-axis from 1.35 to 1.8
            quat=string_to_array("0.67397475 0.21391128 -0.21391128 -0.6739747"),
            # camera_attribs={'fovy': "60"},
        )

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=MyUniformRandomSampler(
                rng=self.rng,
                name="RedBoxSampler",
                mujoco_objects=self.red_box,
                x_range=(-0.07, -0.07),
                y_range=(0.4, 0.4),
                rotation=(0.0, 0.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.0,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=MyUniformRandomSampler(
                rng=self.rng,
                name="BlueBoxSampler",
                mujoco_objects=self.blue_box,
                x_range=(-0.07, -0.07),
                y_range=(0.2, 0.2),
                rotation=(0.0, 0.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.001,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=MyUniformRandomSampler(
                rng=self.rng,
                name="CanSampler",
                mujoco_objects=self.can,
                x_range=(-0.2, -0.1),
                y_range=(-0.2, 0.0),
                rotation=(0.0, 0.0),
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
            red_box=self.sim.model.body_name2id(self.red_box.root_body),
            blue_box=self.sim.model.body_name2id(self.blue_box.root_body),
            can=self.sim.model.body_name2id(self.can.root_body),
        )

    def _initialize_object_states(self):
        """
        Initialize object states (pose estimates that are observed once at start of episode).
        """
        self.initial_object_states = dict()

    def _check_success(self):
        """
        Check success.
        """
        if self.is_red is None:
            return False
        # TODO: implement success check

        red_box_base_pos = np.array(
            self.sim.data.geom_xpos[self.sim.model.geom_name2id("red_box_base")]
        )
        blue_box_base_pos = np.array(
            self.sim.data.geom_xpos[self.sim.model.geom_name2id("blue_box_base")]
        )
        can_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["can"]])

        x_threshold = self.blue_box.bin_size[0] / 2
        y_threshold = self.blue_box.bin_size[1] / 2
        z_threshold = self.can_height / 2
        x_check_red = np.abs(red_box_base_pos[0] - can_pos[0]) < x_threshold
        x_check_blue = np.abs(blue_box_base_pos[0] - can_pos[0]) < x_threshold
        y_check_red = np.abs(red_box_base_pos[1] - can_pos[1]) < y_threshold
        y_check_blue = np.abs(blue_box_base_pos[1] - can_pos[1]) < y_threshold
        z_check_red = can_pos[2] - red_box_base_pos[2] < z_threshold
        z_check_blue = can_pos[2] - blue_box_base_pos[2] < z_threshold

        if self.is_red:
            return x_check_red and y_check_red and z_check_red
        else:
            return x_check_blue and y_check_blue and z_check_blue

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the object.

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
                gripper=self.robots[0].gripper["right"], target=self.can
            )

    def reward(self, action=None):
        """
        Reward function for the task.

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


class TwoArmCanSortRed(TwoArmCanSortRandom):
    def __init__(self, **kwargs):
        super().__init__(red_prob=1.1, **kwargs)


class TwoArmCanSortBlue(TwoArmCanSortRandom):
    def __init__(self, **kwargs):
        super().__init__(red_prob=-0.1, **kwargs)
