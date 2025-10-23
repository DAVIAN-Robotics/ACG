# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import dexmimicgen.utils.transform_utils as T
import numpy as np
from dexmimicgen.environments.two_arm_dexmg_env import TwoArmDexMGEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, string_to_array
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from .my_uniform_random_sampler import MyUniformRandomSampler


class TwoArmPouring(TwoArmDexMGEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
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

    def _get_objects(self):
        """
        Replace pen cap and pen body with cup, sphere in cup, and bowl.
        """
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.pad = BoxObject(
            name="pad",
            size_min=[0.07, 0.07, 0.01],
            size_max=[0.07, 0.07, 0.01],
            rgba=[0, 1, 0, 1],
            material=greenwood,
            joints=None,
        )

        # TODO: tune ball size
        self.ball = BallObject(
            name="ball_obj",
            size=[0.025],
            density=50.0,
            friction=None,
            rgba=None,
            material=redwood,
        )

        import os

        import dexmimicgen
        from dexmimicgen.models.objects.xml_objects import BlenderObject

        base_mjcf_path = os.path.join(
            dexmimicgen.__path__[0], "models/assets/objects/objaverse/"
        )

        def _create_obj(cfg):
            object = BlenderObject(
                name=cfg["name"],
                mjcf_path=cfg["mjcf_path"],
                scale=cfg["scale"],
                solimp=(0.998, 0.998, 0.001),
                solref=(0.001, 1),
                density=cfg.get("density", 100),
                # friction=(0.95, 0.3, 0.1),
                friction=(1, 1, 1),
                margin=0.001,
            )
            return object

        cfg_cup = {
            "name": "cup",
            "mjcf_path": os.path.join(base_mjcf_path, "cup_2/model.xml"),
            "scale": 1.0,
        }

        # red
        cfg_bowl = {
            "name": "bowl",
            "mjcf_path": os.path.join(base_mjcf_path, "bowl_7/model.xml"),
            "scale": 1.5,
            "density": 50,
        }
        self.cup = _create_obj(cfg_cup)
        self.bowl = _create_obj(cfg_bowl)

        return [self.pad, self.cup, self.ball, self.bowl]

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
        objects = self._get_objects()

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
            pos=string_to_array(
                "0.753078462147161 2.062036796036723e-08 1.5194726087166726"
            ),  # Increased z-axis from 1.35 to 1.8
            quat=string_to_array(
                "0.6432409286499023 0.293668270111084 0.2936684489250183 0.6432408690452576"
            ),
            # camera_attribs={"fovy": "60"},
        )

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # TODO: replace

        self.placement_initializer.append_sampler(
            sampler=MyUniformRandomSampler(
                rng=self.rng,
                name="PadSampler",
                mujoco_objects=self.pad,
                x_range=(-0.1, -0.1),
                y_range=(-0.35, -0.35),
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
                name="CupSampler",
                mujoco_objects=self.cup,
                # x_range=(-0.2, -0.05),
                # y_range=(0.15, 0.25),
                x_range=(-0.15, -0.05),
                y_range=(0.1, 0.15),
                # rotation=(-2. * np.pi / 3., -np.pi / 3.),
                rotation=(0.0, 0.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="BowlSampler",
                mujoco_objects=self.bowl,
                x_range=(-0.15, -0.05),
                y_range=(-0.10, -0.15),
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
        TwoArmDexMGEnv._setup_references(self)

        # TODO: replace

        # Additional object references from this env
        self.obj_body_id = dict(
            cup=self.sim.model.body_name2id(self.cup.root_body),
            ball=self.sim.model.body_name2id(self.ball.root_body),
            bowl=self.sim.model.body_name2id(self.bowl.root_body),
            pad=self.sim.model.body_name2id(self.pad.root_body),
        )

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super(TwoArmDexMGEnv, self)._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            cup_pos = None
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj is self.pad:
                    # fixtures - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )
                if obj is self.cup:
                    cup_pos = np.array(obj_pos)

            # TODO: tune this z-value

            # move ball to be placed a little above the cup
            cup_pos[2] += 0.2
            self.sim.data.set_joint_qpos(
                self.ball.joints[0],
                np.concatenate([np.array(cup_pos), np.array([1.0, 0.0, 0.0, 0.0])]),
            )

    def _check_success(self):
        """
        Check success.
        """

        # TODO: implement success check by checking sphere contained within bowl (could probably check contact between
        #       sphere geom and bowl geom and x-y of sphere within x-y of bowl + bowl radius)

        ball_in_bowl = self.check_contact(self.bowl, self.ball)
        ball_xy_th = 0.1
        bowl_pos = self.sim.data.body_xpos[self.obj_body_id["bowl"]]
        ball_pos = self.sim.data.body_xpos[self.obj_body_id["ball"]]
        xy_dist = np.linalg.norm(bowl_pos[:2] - ball_pos[:2])

        bowl_xy_th = 0.06
        bowl_on_pad = self.check_contact(self.pad, self.bowl)
        bowl_rot = self.sim.data.body_xmat[self.obj_body_id["bowl"]].reshape(3, 3)
        z_axis = bowl_rot[:3, 2]
        dist_to_z_axis = 1.0 - z_axis[2]
        bowl_upright = dist_to_z_axis < 0.05
        pad_pos = self.sim.data.body_xpos[self.obj_body_id["pad"]]
        xy_dist_to_pad = np.linalg.norm(pad_pos[:2] - bowl_pos[:2])

        # print("ball_in_bowl", ball_in_bowl, "xy_dist", xy_dist, "bowl_on_pad", bowl_on_pad, "bowl_upright", bowl_upright, "xy_dist_to_pad", xy_dist_to_pad)
        return (
            ball_in_bowl
            and xy_dist < ball_xy_th
            and bowl_on_pad
            and bowl_upright
            and xy_dist_to_pad < bowl_xy_th
        )

    def _get_vis_target_object(self):
        return self.cup

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
                gripper=self.robots[0].gripper["right"],
                target=self._get_vis_target_object(),
            )
