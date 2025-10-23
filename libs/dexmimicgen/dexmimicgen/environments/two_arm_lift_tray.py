# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import dexmimicgen.utils.transform_utils as T
import numpy as np
from dexmimicgen.environments.two_arm_dexmg_env import TwoArmDexMGEnv
from dexmimicgen.models.objects import PotWithHandlesObject
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import string_to_array
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from .my_uniform_random_sampler import MyUniformRandomSampler


class TwoArmLiftTray(TwoArmDexMGEnv):
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
        placement_initializer=None,
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

        # object placement initializer
        self.placement_initializer = placement_initializer

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

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 3.0 is provided if the pot is lifted and is parallel within 30 deg to the table

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.5], per-arm component that is proportional to the distance between each arm and its
              respective pot handle, and exactly 0.5 when grasping the handle
              - Note that the agent only gets the lifting reward when flipping no more than 30 degrees.
            - Grasping: in {0, 0.25}, binary per-arm component awarded if the gripper is grasping its correct handle
            - Lifting: in [0, 1.5], proportional to the pot's height above the table, and capped at a certain threshold

        Note that the final reward is normalized and scaled by reward_scale / 3.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0

        # check if the pot is tilted more than 30 degrees
        mat = T.quat2mat(self._pot_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        cos_z = np.dot(z_unit, z_rotated)
        cos_30 = np.cos(np.pi / 6)
        direction_coef = 1 if cos_z >= cos_30 else 0

        # check for goal completion: cube is higher than the table top above a margin
        if self._check_success():
            reward = 3.0 * direction_coef

        # use a shaping reward
        elif self.reward_shaping:
            # lifting reward
            pot_bottom_height = (
                self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
            )
            table_height = self.sim.data.site_xpos[self.table_top_id][2]
            elevation = pot_bottom_height - table_height
            r_lift = min(max(elevation - 0.05, 0), 0.15)
            reward += 10.0 * direction_coef * r_lift

            _gripper0_to_handle0 = self._gripper0_to_handle0
            _gripper1_to_handle1 = self._gripper1_to_handle1

            # gh stands for gripper-handle
            # When grippers are far away, tell them to be closer

            # Get contacts
            (g0, g1) = (
                (self.robots[0].gripper["right"], self.robots[0].gripper["left"])
                if self.env_configuration == "single-robot"
                else (self.robots[0].gripper["right"], self.robots[1].gripper)
            )

            _g0h_dist = np.linalg.norm(_gripper0_to_handle0)
            _g1h_dist = np.linalg.norm(_gripper1_to_handle1)

            # Grasping reward
            if self._check_grasp(gripper=g0, object_geoms=self.pot.handle0_geoms):
                reward += 0.25
            # Reaching reward
            reward += 0.5 * (1 - np.tanh(10.0 * _g0h_dist))

            # Grasping reward
            if self._check_grasp(gripper=g1, object_geoms=self.pot.handle1_geoms):
                reward += 0.25
            # Reaching reward
            reward += 0.5 * (1 - np.tanh(10.0 * _g1h_dist))

        if self.reward_scale is not None:
            reward *= self.reward_scale / 3.0
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
        self.pot = PotWithHandlesObject(
            name="pot",
            body_half_size=[0.16, 0.20, 0.05],
            handle_radius=0.01,
            handle_length=0.06,
            handle_width=0.14,
            use_texture=False,
            rgba_body=(150.0 / 255, 40.0 / 255, 27.0 / 255, 1.0),  # (150, 40, 27);, 1
            rgba_handle_0=(
                45.0 / 255,
                85.0 / 255,
                255.0 / 255,
                1.0,
            ),  # (137, 196, 244);, 1
            rgba_handle_1=(
                0.0 / 255,
                153.0 / 255,
                92.0 / 255,
                1.0,
            ),  # (63, 195, 128);, 1
        )

        self.obj0 = BoxObject(
            name="obj0",
            size_min=[0.03, 0.03, 0.03],
            size_max=[0.03, 0.03, 0.03],
            rgba=(45.0 / 255, 85.0 / 255, 255.0 / 255, 1.0),
        )
        self.obj1 = BoxObject(
            name="obj1",
            size_min=[0.03, 0.03, 0.03],
            size_max=[0.03, 0.03, 0.03],
            rgba=(0.0 / 255, 153.0 / 255, 92.0 / 255, 1.0),
        )

        self.placement_initializer = self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.pot, self.obj0, self.obj1],
        )

        self._modify_camera_view()

    def _modify_camera_view(self):
        # Modify the agentview camera to have a higher z-axis position
        self.model.mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array("-0.5 0 1.65"),  # Increased z-axis from 1.35 to 1.8
            quat=string_to_array("0.67397475 0.21391128 -0.21391128 -0.6739747"),
            # camera_attribs={'fovy': "60"},
        )

    def _get_placement_initializer(self):
        """
        Returns the placement initializer for the objects in this environment.

        Returns:
            ObjectPositionSampler: The ObjectPositionSampler for this environment
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
            sampler=MyUniformRandomSampler(
                rng=self.rng,
                name="TraySampler",
                mujoco_objects=self.pot,
                x_range=(-0.2, -0.15),
                y_range=(-0.03, 0.03),
                rotation=(np.pi + -np.pi / 8, np.pi + np.pi / 8),
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
                name="Obj0Sampler",
                mujoco_objects=self.obj0,
                x_range=(0.0, 0.2),
                y_range=(-0.3, -0.2),
                rotation=(np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.005,
            )
        )

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="Obj1Sampler",
                mujoco_objects=self.obj1,
                x_range=(0.0, 0.2),
                y_range=(0.2, 0.3),
                rotation=(np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.005,
            )
        )
        return self.placement_initializer

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.pot_body_id = self.sim.model.body_name2id(self.pot.root_body)
        self.handle0_site_id = self.sim.model.site_name2id(
            self.pot.important_sites["handle0"]
        )
        self.handle1_site_id = self.sim.model.site_name2id(
            self.pot.important_sites["handle1"]
        )
        self.table_top_id = self.sim.model.site_name2id("table_top")
        self.pot_center_id = self.sim.model.site_name2id(
            self.pot.important_sites["center"]
        )
        self.obj0_id = self.sim.model.body_name2id(self.obj0.root_body)
        self.obj1_id = self.sim.model.body_name2id(self.obj1.root_body)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to each handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to each handle
        if vis_settings["grippers"]:
            handles = [self.pot.important_sites[f"handle{i}"] for i in range(2)]
            grippers = (
                [self.robots[0].gripper[arm] for arm in self.robots[0].arms]
                if self.env_configuration == "single-robot"
                else [robot.gripper for robot in self.robots]
            )
            for gripper, handle in zip(grippers, handles):
                self._visualize_gripper_to_target(
                    gripper=gripper, target=handle, target_type="site"
                )

    def _check_success(self):
        """
        Check if pot is successfully lifted

        Returns:
            bool: True if pot is lifted
        """
        pot_bottom_height = (
            self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
        )
        table_height = self.sim.data.site_xpos[self.table_top_id][2]

        obj0_height = self.sim.data.body_xpos[self.obj0_id][2]
        obj1_height = self.sim.data.body_xpos[self.obj1_id][2]

        # easy way to check for object in drawer - check if object in contact with bottom drawer geom
        drawer_bottom_geom = "pot_base"
        object_in_drawer = self.check_contact(
            drawer_bottom_geom, self.obj0
        ) and self.check_contact(drawer_bottom_geom, self.obj1)

        # cube is higher than the table top above a margin
        return (
            pot_bottom_height > table_height + 0.10
            and obj0_height > table_height + 0.10
            and obj1_height > table_height + 0.10
            and object_in_drawer
        )

    @property
    def _handle0_xpos(self):
        """
        Grab the position of the left (blue) hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.handle0_site_id]

    @property
    def _handle1_xpos(self):
        """
        Grab the position of the right (green) hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.handle1_site_id]

    @property
    def _pot_quat(self):
        """
        Grab the orientation of the pot body.

        Returns:
            np.array: (x,y,z,w) quaternion of the pot body
        """
        return T.convert_quat(self.sim.data.body_xquat[self.pot_body_id], to="xyzw")

    @property
    def _gripper0_to_handle0(self):
        """
        Calculate vector from the left gripper to the left pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._handle0_xpos - self._eef0_xpos

    @property
    def _gripper1_to_handle1(self):
        """
        Calculate vector from the right gripper to the right pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._handle1_xpos - self._eef1_xpos
