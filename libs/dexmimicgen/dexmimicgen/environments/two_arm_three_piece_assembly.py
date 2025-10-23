# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import random

import dexmimicgen.utils.transform_utils as T
import numpy as np
from dexmimicgen.environments.two_arm_dexmg_env import TwoArmDexMGEnv
from dexmimicgen.models.objects import BoxPatternObject
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from .my_uniform_random_sampler import MyUniformRandomSampler


class TwoArmThreePieceAssembly(TwoArmDexMGEnv):

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
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            *args,
            **kwargs,
        )

    def _get_piece_densities(self):
        """
        Subclasses can override this method to change the weight of the pieces.
        """
        return dict(
            # base=100.,
            # NOTE: changed to make base piece heavier and task easier
            base=10000.0,
            piece_1=100.0,
            piece_2=100.0,
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

    def _get_piece_patterns(self):
        """
        Helper function to get unit-box patterns to make each assembly piece.
        """
        blocks = [[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0]]
        hole_side = [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]
        hole = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]

        # Pick out two sides of block1
        pick = random.randint(0, len(blocks) - 1)
        pick = 0
        side1 = blocks[pick]
        hole1 = hole_side[pick]
        pick = random.randint(0, len(blocks) - 1)
        pick = 0
        side2 = blocks[pick]
        hole2 = hole_side[pick]

        block1 = [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [0.9, 0.9, 0.9], [0, 0, 0]],
            [[0, 0, 0], [0.9, 0.9, 0.9], [0, 0, 0]],
        ]

        block1[0][0] = side1
        block1[0][2] = side2
        hole[0][0] = hole1
        hole[0][2] = hole2

        ### NOTE: we changed base_x from 7 to 5, and hole offset from 2 to 1, to make base piece smaller in size ###

        # Generate hole
        base_x = 5
        base_z = 1
        base = np.ones((base_z, base_x, base_x))

        offset_x = random.randint(1, base_x - 4)
        offset_x = 1
        offset_y = random.randint(1, base_x - 3)
        offset_y = 1

        for z in range(len(hole)):
            for y in range(len(hole[0])):
                for x in range(len(hole[0][0])):
                    base[z][offset_y + y][offset_x + x] = hole[z][y][x]

        block2 = [
            [[1, 1, 1], [0, 0, 0], [1, 1, 1]],
            [[1, 1, 1], [0, 0, 0], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]

        return block1, block2, base

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
        self.piece_1_pattern, self.piece_2_pattern, self.base_pattern = (
            self._get_piece_patterns()
        )

        self.piece_1_size = 0.017
        self.piece_2_size = 0.02
        self.base_size = 0.019

        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        mat = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        piece_densities = self._get_piece_densities()

        self.piece_1 = BoxPatternObject(
            name="piece_1",
            unit_size=[self.piece_1_size, self.piece_1_size, self.piece_1_size],
            pattern=self.piece_1_pattern,
            rgba=None,
            material=mat,
            density=piece_densities["piece_1"],
            friction=None,
        )
        self.piece_2 = BoxPatternObject(
            name="piece_2",
            unit_size=[self.piece_2_size, self.piece_2_size, self.piece_2_size],
            pattern=self.piece_2_pattern,
            rgba=None,
            material=mat,
            density=piece_densities["piece_2"],
            friction=None,
        )
        self.base = BoxPatternObject(
            name="base",
            unit_size=[self.base_size, self.base_size, self.base_size],
            pattern=self.base_pattern,
            rgba=None,
            material=mat,
            density=piece_densities["base"],
            friction=None,
        )

        objects = [self.base, self.piece_1, self.piece_2]

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
                name="BaseSampler",
                mujoco_objects=self.base,
                x_range=(0.0, 0.0),
                y_range=(0.0, 0.0),
                rotation=(0.0, 0.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=MyUniformRandomSampler(
                rng=self.rng,
                name="Piece1Sampler",
                mujoco_objects=self.piece_1,
                x_range=(-0.22, 0.22),
                y_range=(-0.22, 0.0),
                rotation=(1.5708, 1.5708),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="Piece2Sampler",
                mujoco_objects=self.piece_2,
                x_range=(-0.22, 0.22),
                y_range=(0.0, 0.22),
                rotation=(1.5708, 1.5708),
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
            base=self.sim.model.body_name2id(self.base.root_body),
            piece_1=self.sim.model.body_name2id(self.piece_1.root_body),
            piece_2=self.sim.model.body_name2id(self.piece_2.root_body),
        )

    def _check_success(self):
        """
        Check if task is complete.
        """
        metrics = self._get_partial_task_metrics()
        return metrics["task"]

    def _check_first_piece_is_assembled(self, xy_thresh=0.02):
        robot_and_piece_1_in_contact = False
        for robot in self.robots:
            for gripper in robot.gripper:
                robot_and_piece_1_in_contact = (
                    robot_and_piece_1_in_contact
                    or self._check_grasp(
                        gripper=gripper,
                        object_geoms=[g for g in self.piece_1.contact_geoms],
                    )
                )

        piece_1_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["piece_1"]])
        base_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["base"]])

        # assume that first piece is assembled when x-y position is close enough to base piece
        # and gripper is not holding the piece
        first_piece_is_assembled = (
            np.linalg.norm(piece_1_pos[:2] - base_pos[:2]) < xy_thresh
        ) and (not robot_and_piece_1_in_contact)
        return first_piece_is_assembled

    def _check_second_piece_is_assembled(self, xy_thresh=0.02, z_thresh=0.02):
        robot_and_piece_2_in_contact = False
        for robot in self.robots:
            for gripper in robot.gripper:
                robot_and_piece_2_in_contact = (
                    robot_and_piece_2_in_contact
                    or self._check_grasp(
                        gripper=gripper,
                        object_geoms=[g for g in self.piece_2.contact_geoms],
                    )
                )

        piece_1_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["piece_1"]])
        piece_2_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["piece_2"]])
        base_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["base"]])
        z_correct = base_pos[2] + self.piece_2_size * 4

        first_piece_is_assembled = self._check_first_piece_is_assembled(
            xy_thresh=xy_thresh
        )

        # second piece is assembled (and task is complete) when it is close enough to first piece in x-y, close
        # enough to first piece in z (and first piece is assembled) and gripper is not holding the piece
        second_piece_is_assembled = (
            first_piece_is_assembled
            and (np.linalg.norm(piece_1_pos[:2] - piece_2_pos[:2]) < xy_thresh)
            and (np.abs(piece_2_pos[2] - z_correct) < z_thresh)
            and (not robot_and_piece_2_in_contact)
        )
        return second_piece_is_assembled

    def _get_partial_task_metrics(self):
        """
        Check if all three pieces have been assembled together.
        """
        metrics = {
            "first_piece_assembled": self._check_first_piece_is_assembled(),
            "task": self._check_second_piece_is_assembled(),
        }

        return metrics

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
                gripper=self.robots[0].gripper["right"], target=self.piece_1
            )
