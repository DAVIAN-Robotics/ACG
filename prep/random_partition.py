import json
import os
import random
import sys

import h5py

DATASET_ROOT = os.environ.get('DATASET_ROOT', '/datasets')


def main(n_demos: int):

    filenames = os.listdir(f'{DATASET_ROOT}/dexmimicgen/generated')
    indices_all = {k: None for k in filenames}

    for file_idx, h5py_filename in enumerate(filenames):
        h5py_file = h5py.File(f'{DATASET_ROOT}/dexmimicgen/generated/{h5py_filename}', 'r', swmr=True, libver='latest')

        indices = list(range(len(h5py_file['data'].keys())))
        random.seed(42)
        random.shuffle(indices)

        indices_all[h5py_filename] = indices[:n_demos]
        assert len(indices_all[h5py_filename]) == n_demos, len(indices_all[h5py_filename])

        h5py_file.close()

    # with open(f'data/splits/dexmimicgen/{n_demos}_demos.json', 'w') as f:
    #     json.dump(indices_all, f, indent=4)
    os.makedirs(f'{DATASET_ROOT}/dexmimicgen/generated/splits', exist_ok=True)
    with open(f'{DATASET_ROOT}/dexmimicgen/generated/splits/{n_demos}_demos.json', 'w') as f:
        json.dump(indices_all, f, indent=4)


if __name__ == '__main__':
    n_demos = int(sys.argv[1])
    main(n_demos)
