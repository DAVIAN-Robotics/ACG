# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
from copy import deepcopy

import dexmimicgen
import dexmimicgen.utils.transform_utils as T
import numpy as np
from dexmimicgen.environments.two_arm_dexmg_env import TwoArmDexMGEnv
from dexmimicgen.models.objects import DrawerObject
from robosuite.models.arenas import TableArena
from robosuite.models.objects.composite.bin import Bin
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, add_material, string_to_array
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from .my_uniform_random_sampler import MyUniformRandomSampler


class TwoArmDrawerCleanup(TwoArmDexMGEnv):

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(1.5, 1.5, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
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
        self.table_offset = np.array((0, 0, 0.7))

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

    def _get_drawer_model(self):
        """
        Allow subclasses to override which drawer to use - should load into @self.drawer.
        """

        # Create drawer object
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )
        drawer = DrawerObject(name="DrawerObject")
        obj_body = drawer
        for material in [redwood, ceramic, lightwood]:
            tex_element, mat_element, _, used = add_material(
                root=obj_body.worldbody,
                naming_prefix=obj_body.naming_prefix,
                custom_material=deepcopy(material),
            )
            obj_body.asset.append(tex_element)
            obj_body.asset.append(mat_element)
        return drawer

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
        self.drawer = self._get_drawer_model()

        base_mjcf_path = os.path.join(
            dexmimicgen.__path__[0], "models/assets/objects/objaverse/"
        )
        from dexmimicgen.models.objects.xml_objects import BlenderObject

        def _create_obj(cfg):
            object = BlenderObject(
                name=cfg["name"],
                mjcf_path=cfg["mjcf_path"],
                scale=cfg["scale"],
                solimp=(0.999, 0.999, 0.001),
                solref=(0.001, 1),
                density=100,
                # friction=(0.95, 0.3, 0.1),
                friction=(1, 1, 1),
                margin=0.001,
            )
            return object

        cfg = {
            "name": "cleanup_object",
            "mjcf_path": os.path.join(base_mjcf_path, "mug_1/model.xml"),
            "scale": 1.5,
        }
        self.cleanup_object = _create_obj(cfg)

        objects = [self.drawer, self.cleanup_object]

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
        self.placement_initializer.append_sampler(
            sampler=MyUniformRandomSampler(
                rng=self.rng,
                name="DrawerSampler",
                mujoco_objects=self.drawer,
                # x_range=[0.1, 0.1],
                x_range=[0.1, 0.1],
                y_range=[-0.15, -0.15],
                rotation=-np.pi / 2.0,
                # rotation=-2 * np.pi / 3.,
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
                name="ObjectSampler",
                mujoco_objects=self.cleanup_object,
                # x_range=[0.,  0.],
                # x_range=[-0.15,  -0.15],
                x_range=[-0.3, -0.15],
                # y_range=[-0.25, -0.25],
                y_range=[0.2, 0.4],
                rotation=None,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.obj_body_id = dict(
            object=self.sim.model.body_name2id(self.cleanup_object.root_body),
            drawer=self.sim.model.body_name2id(self.drawer.root_body),
        )
        self.drawer_qpos_addr = self.sim.model.get_joint_qpos_addr(
            self.drawer.joints[0]
        )
        self.drawer_bottom_geom_id = self.sim.model.geom_name2id(
            "DrawerObject_drawer_bottom"
        )

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # skip the implementation in the parent class
        super(TwoArmDexMGEnv, self)._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj is self.drawer:
                    # object is fixture - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    obj_pos_to_set = np.array(obj_pos)
                    obj_pos_to_set[2] = (
                        self.table_offset[2]
                        + 0.005  # hardcode z-value to make sure it lies on table surface
                    )
                    self.sim.model.body_pos[body_id] = obj_pos_to_set
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    # object has free joint - use it to set pose
                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )

        # Drawer should start closed (0.) but can set to open (-0.135) for debugging.
        self.sim.data.qpos[self.drawer_qpos_addr] = 0.0
        # self.sim.data.qpos[self.drawer_qpos_addr] = -0.135
        self.sim.forward()

    def _check_drawer_close(self):
        # check for closed drawer
        drawer_closed = self.sim.data.qpos[self.drawer_qpos_addr] > -0.01
        return drawer_closed

    def _check_object(self):
        # easy way to check for object in drawer - check if object in contact with bottom drawer geom
        drawer_bottom_geom = "DrawerObject_drawer_bottom"
        object_in_drawer = self.check_contact(drawer_bottom_geom, self.cleanup_object)

        return object_in_drawer  # and object_upright

    def _check_success(self):
        """
        Check success.
        """
        drawer_closed = self._check_drawer_close()
        object_in_drawer_upright = self._check_object()

        return object_in_drawer_upright and drawer_closed

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
                gripper=self.robots[0].gripper["right"], target=self.drawer
            )
