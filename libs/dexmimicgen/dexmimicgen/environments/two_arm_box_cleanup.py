# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import dexmimicgen.utils.transform_utils as T
import numpy as np
from dexmimicgen.environments.two_arm_dexmg_env import TwoArmDexMGEnv
from dexmimicgen.models.objects.composite.bin import Bin
from dexmimicgen.models.objects.composite_body.stacked_box import StackedBoxObject
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, string_to_array
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from .my_uniform_random_sampler import MyUniformRandomSampler


class TwoArmBoxCleanup(TwoArmDexMGEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        table_offset=(0.0, 0.0, 0.7),
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
        use_translucent_lid=False,
        # use_translucent_lid=True,
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

        # whether to use translucent lid
        self.use_translucent_lid = use_translucent_lid

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

        # make redwood material to override default
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="lightwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # initialize objects of interest
        self.box = Bin(
            name="box_obj",
            bin_size=(0.2, 0.2, 0.15),
            wall_thickness=0.01,
            transparent_walls=False,
            friction=None,
            density=100000.0,
            use_texture=True,
            # use_texture=False,
            rgba=(0.2, 0.1, 0.0, 1.0),
            material=lightwood,
        )

        # TODO: can try a simple stacked box as a lid
        self.lid = StackedBoxObject(
            name="lid_obj",
            box_1_size=(0.075, 0.075, 0.02),
            box_2_size=(0.125, 0.125, 0.03),
            box_1_rgba=(0.2, 0.1, 0.0, 1.0),
            box_2_rgba=(0.2, 0.1, 0.0, 1.0),
            box_1_material=lightwood,
            box_2_material=lightwood,
            density=100.0,
            make_box_2_transparent=self.use_translucent_lid,
        )

        objects = [self.box, self.lid]

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
            pos=string_to_array("-0.5 0 1.65"),  # Increased z-axis from 1.35 to 1.8
            quat=string_to_array("0.67397475 0.21391128 -0.21391128 -0.6739747"),
            # camera_attribs={'fovy': "60"},
        )

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # TODO: replace

        self.placement_initializer.append_sampler(
            sampler=MyUniformRandomSampler(
                rng=self.rng,
                name="LidSampler",
                mujoco_objects=self.lid,
                # x_range=(-0.2, -0.05),
                # y_range=(0.15, 0.25),
                x_range=(-0.05, 0.05),
                y_range=(0.20, 0.20),
                rotation=(0.0, 0.0),
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
                name="BoxSampler",
                mujoco_objects=self.box,
                x_range=(-0.05, 0.05),
                y_range=(-0.15, -0.15),
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

        # TODO: replace

        # Additional object references from this env
        self.obj_body_id = dict(
            box=self.sim.model.body_name2id(self.box.root_body),
            lid=self.sim.model.body_name2id(self.lid.root_body),
        )

    def _check_success(self):
        """
        Check success.
        """

        # TODO: implement success check

        # NOTE: easiest check is the following. check x and y position alignment being close enough, and z position alignment close enough

        # box base geom: box_obj_base
        box_base_pos = np.array(
            self.sim.data.geom_xpos[self.sim.model.geom_name2id("box_obj_base")]
        )

        # lid base geom: lid_obj_base
        # lid_base_pos = np.array(self.sim.data.geom_xpos[self.sim.model.geom_name2id("lid_obj_base")])

        # TODO: swap this when using stacked box lid
        if self.use_translucent_lid:
            lid_base_pos = np.array(
                self.sim.data.body_xpos[
                    self.sim.model.body_name2id(
                        self.lid.naming_prefix + self.lid.box_2.root_body
                    )
                ]
            )
        else:
            lid_base_pos = np.array(
                self.sim.data.geom_xpos[self.sim.model.geom_name2id("lid_obj_box_2_g0")]
            )
        lid_z = self.lid.box_2_size[2]

        # TODO: tune these thresholds
        xy_threshold = 0.03

        # make it a little more than the height of the box, since that's how much z separation there should be between the lid
        # and the box
        z_threshold = 1.01 * (self.box.bin_size[2] + lid_z)

        xy_check = np.linalg.norm(box_base_pos[:2] - lid_base_pos[:2]) < xy_threshold
        z_check = np.abs(box_base_pos[2] - lid_base_pos[2]) < z_threshold
        return xy_check and z_check

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

        # TODO: replace object ref

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper["right"], target=self.lid
            )
