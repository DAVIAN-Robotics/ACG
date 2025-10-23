# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import dexmimicgen.utils.transform_utils as T
import numpy as np
from dexmimicgen.environments.two_arm_dexmg_env import TwoArmDexMGEnv
from robosuite.models.arenas import MultiTableArena
from robosuite.models.objects import BoxObject, HammerObject, TransportGroup
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from .my_uniform_random_sampler import MyUniformRandomSampler


class TwoArmTransport(TwoArmDexMGEnv):

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        tables_boundary=(0.8, 1.2, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        bin_size=(0.3, 0.3, 0.15),
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
        self.tables_boundary = tables_boundary
        self.table_full_size = np.array(tables_boundary)
        self.table_full_size[
            1
        ] *= 0.25  # each table size will only be a fraction of the full boundary
        self.table_friction = table_friction
        self.table_offsets = np.zeros((2, 3))
        self.table_offsets[0, 1] = self.tables_boundary[1] * -3 / 8  # scale y offset
        self.table_offsets[1, 1] = self.tables_boundary[1] * 3 / 8  # scale y offset
        self.table_offsets[:, 2] = 0.8  # scale z offset
        self.bin_size = np.array(bin_size)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.height_threshold = 0.1  # threshold above the table surface which the payload is considered lifted

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

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided when the payload is in the target bin and the trash is in the trash
                bin

        Un-normalized max-wise components if using reward shaping:

            # TODO!

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        # Initialize reward
        reward = 0

        # use a shaping reward if specified
        if self.reward_shaping:
            # TODO! So we print a warning and force sparse rewards
            print(
                "\n\nWarning! No dense reward current implemented for this task. Forcing sparse rewards\n\n"
            )
            self.reward_shaping = False

        # Else this is the sparse reward setting
        else:
            # Provide reward if payload is in target bin and trash is in trash bin
            if self._check_success():
                reward = 1.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Set up robots parallel to each other but offset from the center
        for robot, offset in zip(self.robots, (-0.6, 0.6)):
            xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
            xpos = np.array(xpos) + np.array((0, offset, 0))
            robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = MultiTableArena(
            table_offsets=self.table_offsets,
            table_rots=0,
            table_full_sizes=self.table_full_size,
            table_frictions=self.table_friction,
            has_legs=True,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.8894354364730311, -3.481824231498976e-08, 1.7383813133506494],
            quat=[
                0.6530981063842773,
                0.2710406184196472,
                0.27104079723358154,
                0.6530979871749878,
            ],
        )

        # TODO: Add built-in method into TwoArmEnv so we have an elegant way of automatically adding extra cameras to all these envs
        # Add shoulder cameras
        mujoco_arena.set_camera(
            camera_name="shouldercamera0",
            pos=[0.4430096057365183, -1.0697399743660143, 1.3639950119362048],
            quat=[
                0.804057240486145,
                0.5531665086746216,
                0.11286306381225586,
                0.18644218146800995,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="shouldercamera1",
            pos=[-0.40900713993039983, 0.9613722572245062, 1.3084072951772754],
            quat=[
                0.15484197437763214,
                0.12077208608388901,
                -0.5476858019828796,
                -0.8133130073547363,
            ],
        )

        # Add relevant materials
        # Textures to use
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
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

        # initialize objects of interest
        payload = HammerObject(
            name="payload",
            handle_radius=0.015,
            handle_length=0.20,
            handle_density=150.0,
            handle_friction=4.0,
            head_density_ratio=1.5,
        )
        trash = BoxObject(name="trash", size=[0.02, 0.02, 0.02], material=redwood)
        self.transport = TransportGroup(
            name="transport",
            payload=payload,
            trash=trash,
            bin_size=self.bin_size,
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.transport.objects.values()),
        )

        # Create placement initializer
        self._get_placement_initializer()

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds
        """
        # Create placement initializer
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # Pre-define settings for each object's placement
        object_names = [
            "start_bin",
            "lid",
            "payload",
            "target_bin",
            "trash",
            "trash_bin",
        ]
        table_nums = [0, 0, 0, 1, 1, 1]
        x_centers = [
            self.table_full_size[0] * 0.25,
            0,  # gets overridden anyways
            0,  # gets overridden anyways
            -self.table_full_size[0] * 0.25,
            0,  # gets overridden anyways
            self.table_full_size[0] * 0.25,
        ]
        pos_tol = 0.005
        rot_centers = [0, 0, np.pi / 2, 0, 0, 0]
        rot_tols = [0, 0, np.pi / 6, 0, 0.3 * np.pi, 0]
        rot_axes = ["z", "z", "y", "z", "z", "z"]
        for obj_name, x, r, r_tol, r_axis, table_num in zip(
            object_names, x_centers, rot_centers, rot_tols, rot_axes, table_nums
        ):
            # Get name and table
            obj = self.transport.objects[obj_name]
            table_pos = self.table_offsets[table_num]
            # Create sampler for this object and add it to the sequential sampler
            self.placement_initializer.append_sampler(
                sampler=MyUniformRandomSampler(
                    rng=self.rng,
                    name=f"{obj_name}ObjectSampler",
                    mujoco_objects=obj,
                    x_range=[x - pos_tol, x + pos_tol],
                    y_range=[-pos_tol, pos_tol],
                    rotation=[r - r_tol, r + r_tol],
                    rotation_axis=r_axis,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=table_pos,
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

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super(TwoArmDexMGEnv, self)._reset_internal()

        # Update sim
        self.transport.update_sim(sim=self.sim)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Initialize placeholders that we'll need to override the payload, lid, and trash object locations
            start_bin_pos = None
            target_bin_pos = None

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # If this is toolbox or good bin, store their sampled positions
                if "start_bin" in obj.name and "lid" not in obj.name:
                    start_bin_pos = obj_pos
                elif "target_bin" in obj.name:
                    target_bin_pos = obj_pos
                # Else if this is either the lid, payload, or trash object,
                # we override their positions to match their respective containers' positions
                elif "lid" in obj.name:
                    obj_pos = (
                        start_bin_pos[0],
                        start_bin_pos[1],
                        obj_pos[2] + self.transport.bin_size[2],
                    )
                elif "payload" in obj.name:
                    obj_pos = (
                        start_bin_pos[0],
                        start_bin_pos[1],
                        obj_pos[2] + self.transport.objects["start_bin"].wall_thickness,
                    )
                elif "trash" in obj.name and "bin" not in obj.name:
                    obj_pos = (
                        target_bin_pos[0],
                        target_bin_pos[1],
                        obj_pos[2]
                        + self.transport.objects["target_bin"].wall_thickness,
                    )
                # Set the collision object joints
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def _check_success(self):
        """
        Check if payload is in target in and trash is in trash bin

        Returns:
            bool: True if transport has been completed
        """
        return (
            True
            if self.transport.payload_in_target_bin
            and self.transport.trash_in_trash_bin
            else False
        )

    def _check_lid_on_table(self):
        (g0, g1) = (
            (self.robots[0].gripper["right"], self.robots[0].gripper["left"])
            if self.env_configuration == "single-robot"
            else (self.robots[0].gripper, self.robots[1].gripper)
        )
        self.lid_body_id = self.sim.model.body_name2id(
            self.transport.objects["lid"].root_body
        )

        grasping_lid = self._check_grasp(
            gripper=g0, object_geoms=self.transport.objects["lid"]
        )

        lid_body_pos = self.sim.data.body_xpos[self.lid_body_id]
        lid_body_height = lid_body_pos[2]
        table_height = self.table_offsets[0, 2]
        lid_touching_table = (
            lid_body_height < table_height + 0.1
        )  # initial value of (lid_body_height - table_height) is 0.171

        return (not grasping_lid) and lid_touching_table

    def _check_payload_lifted(self):
        payload_body_id = self.sim.model.body_name2id(
            self.transport.objects["payload"].root_body
        )

        payload_body_pos = self.sim.data.body_xpos[payload_body_id]
        payload_body_height = payload_body_pos[2]
        table_height = self.table_offsets[0, 2]

        return payload_body_height - table_height > 0.1
