# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os

import dexmimicgen


def xml_path_completion(xml_path, root=None):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package

    Args:
        xml_path (str): local xml path
        root (str): root folder for xml path. If not specified defaults to robosuite.models.assets_root

    Returns:
        str: Full (absolute) xml path
    """
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        if root is None:
            root = dexmimicgen.models.assets_root
        full_path = os.path.join(root, xml_path)
    return full_path
