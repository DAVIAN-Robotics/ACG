import os
import sys

import h5py
import numpy as np
import torch
from torchvision import io
from tqdm import tqdm

DATASET_ROOT = os.getenv('DATASET_ROOT', '/datasets')


def main(task_idx=-1):
    root = f'{DATASET_ROOT}/dexmimicgen/generated/videos_256x256'
    tasks = sorted(os.listdir(root))
    tasks = [task for task in tasks if os.path.isdir(os.path.join(root, task))]
    print(f"Found {len(tasks)} tasks in {root}")

    if task_idx >= 0:
        tasks = [tasks[task_idx]]
        print(f"Using task_idx {task_idx}, only processing {tasks}")

    for task in tasks:
        task_path = os.path.join(root, task, 'obs')
        if not os.path.isdir(task_path):
            continue
        views = os.listdir(task_path)
        hdf5_path = os.path.join(root, f"{task}_videos.hdf5")
        # hdf5_file['data/ep/obs/view'] -> video
        if os.path.exists(hdf5_path):
            print(f"{hdf5_path} already exists. Skipping...")
            continue

        # num file check
        num_files = [len(os.listdir(os.path.join(task_path, view))) for view in views]
        if not num_files == [1000] * len(views):
            print(f"[{task}] Number of files in each view do not match: {num_files}")
            continue

        hdf5_file = h5py.File(hdf5_path, 'w')

        for view in views:
            print(f"Processing task {task}, view {view}")
            view_path = os.path.join(task_path, view)
            files = os.listdir(view_path)
            for file in tqdm(files):
                if file.endswith('.mp4'):
                    video_path = os.path.join(view_path, file)

                    # Read video
                    video, _, _ = io.read_video(video_path, pts_unit="sec")
                    video = video.numpy()
                    video = video.astype(np.uint8)

                    ep = file[:-4]
                    hd5key = f"data/{ep}/obs/{view}"
                    hdf5_file.create_dataset(hd5key, data=video, compression="gzip")

        hdf5_file.close()


if __name__ == "__main__":
    task_idx = sys.argv[1] if len(sys.argv) > 1 else -1
    main(int(task_idx))
