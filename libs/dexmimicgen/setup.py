# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()
    long_description = "".join(lines)

# read requirements
with open(path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    lines = f.readlines()
    requirements = [line.strip() for line in lines]

print([package for package in find_packages() if package.startswith("dexmimicgen")])

setup(
    name="dexmimicgen",
    packages=[
        package for package in find_packages() if package.startswith("dexmimicgen")
    ],
    install_requires=requirements,
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3.9",
    description="DexMimicGen environments",
    author="Zhenyu Jiang",
    url="https://github.com/nvglab/dexmimicgen",
    author_email="yukez@cs.utexas.edu",
    version="0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
