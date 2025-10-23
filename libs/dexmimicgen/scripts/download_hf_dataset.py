# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Dataset Downloader for dexmimicgen

This script downloads datasets for specified tasks from the Hugging Face Hub
and saves them to a local directory.

Args:
    --path (str): Path to store the dataset (default: "../datasets/").
    --tasks (list of str): List of task names to download. Task names are
        automatically converted from CamelCase to snake_case.

Example usage:
    # Download all datasets to ./datasets
    python download_hf_dataset.py

    # Download TwoArmBoxCleanup dataset to /your/dataset/path
    python download_hf_dataset.py --path /your/dataset/path --tasks TwoArmBoxCleanup
"""

import argparse
import os
import re

from huggingface_hub import hf_hub_download, list_repo_files

import dexmimicgen

DEXMG_ENV_PATH = os.path.join(dexmimicgen.__path__[0], "../datasets/")
REPO_ID = "MimicGen/dexmimicgen_datasets"


def camel_to_snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=DEXMG_ENV_PATH,
        help="Path to store the dataset",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "TwoArmBoxCleanup",
            "TwoArmCanSortRandom",
            "TwoArmCoffee",
            "TwoArmDrawerCleanup",
            "TwoArmLiftTray",
            "TwoArmPouring",
            "TwoArmThreading",
            "TwoArmThreePieceAssembly",
            "TwoArmTransport",
        ],
        help="List of tasks to download",
    )

    args = parser.parse_args()
    repo_files = list_repo_files(REPO_ID, repo_type="dataset")

    os.makedirs(args.path, exist_ok=True)

    for task in args.tasks:
        task = camel_to_snake(task)
        print(f"Downloading dataset for {task}.hdf5")
        success = False
        for file in repo_files:
            if not file.endswith(".hdf5"):
                continue
            if not task in file:
                continue
            hf_hub_download(
                repo_id=REPO_ID,
                filename=file,
                repo_type="dataset",
                local_dir=args.path,
            )
            success = True
            break
        if not success:
            print(f"Dataset for {task} not found")

    print("Download complete")
    print(f"Dataset downloaded to {args.path}")
