# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.utils.mjcf_utils import array_to_string, find_elements, string_to_array

import dexmimicgen
import dexmimicgen.utils.transform_utils as T


class TwoArmDexMGEnv(TwoArmEnv):
    def __init__(self, translucent_robot=False, *args, **kwargs):

        self.translucent_robot = translucent_robot
        super().__init__(*args, **kwargs)

    @property
    def robot_joint_names(self):
        joint_names = []
        for name in self.sim.model.joint_names:
            if "robot0_" in name:
                joint_names.append(name)
        return joint_names

    def set_robot_state(self, init_state):
        """
        Resets the robot to the specified state.

        Args:
            init_state (dict): Dictionary of initial robot state values to set.
        """
        # Reset the robot to the specified state
        for k, v in init_state.items():
            joint_id = self.sim.model.joint_name2id(k)
            self.sim.data.qpos[joint_id] = v
        self.sim.forward()

    def _load_model(self):
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            # Set up robots parallel to each other but offset from the center
            for robot, offset in zip(self.robots, (-0.25, 0.25)):
                xpos = robot.robot_model.base_xpos_offset["table"](
                    self.table_full_size[0]
                )
                xpos = np.array(xpos) + np.array((0, offset, 0))
                robot.robot_model.set_base_xpos(xpos)

        if self.translucent_robot:
            material_names = ["LightGrey", "Black"]
            for robot in self.robots:
                # for material_name in material_names:
                materials = find_elements(
                    root=robot.robot_model.asset,
                    tags="material",
                    # attribs={"name": robot.robot_model.naming_prefix+material_name},
                    return_first=False,
                )
                for material in materials:
                    need_skip = False
                    for name in [
                        "base_material",
                        "thumb_material",
                        "thumb_distal_material",
                        "index_material",
                        "middle_material",
                        "ring_material",
                        "pinky_material",
                    ]:
                        if material.get("name").endswith(name):
                            need_skip = True
                            break
                    if need_skip:
                        continue
                    if material.get("rgba"):
                        rgba = string_to_array(material.get("rgba"))
                        rgba[-1] = 0.1
                        material.set("rgba", array_to_string(rgba))

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()
            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """

        # NOTE: fixed with mujoco >= 2.2
        # # NOTE: we changed this to the robosuite implementation (instead of mujoco binding) because of
        # #       an issue with the DM mujoco binding where .obj meshes dump vertices and increase the
        # #       size of the xml substantially, and also cause some simulation issues when reloading
        # #       from xml (such as slippage during grasping the obj meshes)
        # xml = self.env.model.get_xml()
        xml = self.sim.model.get_xml()  # model xml file
        state = np.array(self.sim.get_state().flatten())  # simulator state
        return dict(model=xml, states=state)

    def edit_model_xml(self, xml_str):
        """
        This function edits the model xml with custom changes, including resolving relative paths,
        applying changes retroactively to existing demonstration files, and other custom scripts.
        Environment subclasses should modify this function to add environment-specific xml editing features.
        Args:
            xml_str (str): Mujoco sim demonstration XML file as string
        Returns:
            str: Edited xml file as string
        """
        xml_str = super().edit_model_xml(xml_str)

        path = os.path.split(dexmimicgen.__file__)[0]
        path_split = path.split("/")

        # replace mesh and texture file paths
        tree = ET.fromstring(xml_str)
        root = tree
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue

            old_path_split = old_path.split("/")
            # maybe replace all paths to robosuite assets
            check_lst = [
                loc
                for loc, val in enumerate(old_path_split)
                if val == "dexmimicgen" or val == "dexmimicgen_environments"
            ]
            if len(check_lst) > 0:
                ind = max(check_lst)  # last occurrence index
                new_path_split = path_split + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

        return ET.tostring(root, encoding="utf8").decode("utf8")
