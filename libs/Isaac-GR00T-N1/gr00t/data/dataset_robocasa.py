import torchvision.io as io
import json
import math
import os
import random
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import robomimic.utils.action_utils as AcUtils
import robomimic.utils.lang_utils as LangUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import torch
import torch.utils.data
from pydantic import BaseModel, ValidationError
from robomimic.macros import LANG_EMB_KEY, LANG_STR_KEY
from robomimic.utils.dataset import MetaDataset, SequenceDataset, action_stats_to_normalization_stats

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata, DatasetStatisticalValues, LeRobotModalityMetadata, LeRobotStateActionMetadata
from gr00t.data.transform import ComposedModalityTransform

LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
# LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"


class LeRobotSetup:

    def additional_setup(
        self,
        dataset_path: str,
        embodiment_tag: EmbodimentTag,
        modality_transform: ComposedModalityTransform,
        horizon: dict[str, int],  # robomimic
        null_ratio: float = 0.0,
        null_targets: Optional[List[str]] = None,
        separate_null_targets: bool = False,
        **kwargs
    ):
        # lerobot
        self.dataset_path = Path(dataset_path)
        self.tag = embodiment_tag.value
        self._metadata = self._get_metadata(embodiment_tag)
        self._modality_transform = modality_transform
        self._null_ratio = null_ratio
        self._null_targets = null_targets
        self._separate_null_targets = separate_null_targets

        # robomimic
        self._horizon = horizon

        for key in kwargs:
            setattr(self, key, kwargs[key])
        self._new_keys = list(kwargs.keys())

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"

        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except ValueError:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        try:
            with open(stats_path, "r") as f:
                le_statistics = json.load(f)
            for stat in le_statistics.values():
                DatasetStatisticalValues.model_validate(stat)
        except (FileNotFoundError, ValidationError) as e:
            print(f"Failed to load dataset statistics: {e}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            le_statistics = calculate_dataset_statistics(parquet_files)
            with open(stats_path, "w") as f:
                json.dump(le_statistics, f, indent=4)
        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    @property
    def horizon(self) -> dict[str, int]:
        return self._horizon

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def null_ratio(self) -> float:
        return self._null_ratio

    @property
    def null_targets(self) -> float:
        return self._null_targets

    @property
    def separate_null_targets(self) -> bool:
        return self._separate_null_targets

    @property
    def new_keys(self) -> List[str]:
        return self._new_keys


class RobocasaDataset(SequenceDataset, LeRobotSetup):
    pass


def _aggregate_traj_stats_custom(traj_stats_a, traj_stats_b):
    """
    Helper function to aggregate trajectory statistics.
    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    for more information.
    """
    merged_stats = {}
    for k in traj_stats_a:
        if k not in traj_stats_b:  # ! modified by mpark
            merged_stats[k] = traj_stats_a[k]
            continue

        assert traj_stats_a[k]['mean'].shape == traj_stats_b[k]['mean'].shape, \
            f"Shape mismatch: {traj_stats_a[k]['mean'].shape} vs {traj_stats_b[k]['mean'].shape}"

        n_a, avg_a, M2_a, min_a, max_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"], traj_stats_a[k]["min"], traj_stats_a[k]["max"]
        n_b, avg_b, M2_b, min_b, max_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"], traj_stats_b[k]["min"], traj_stats_b[k]["max"]
        n = n_a + n_b
        mean = (n_a * avg_a + n_b * avg_b) / n
        delta = (avg_b - avg_a)
        M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
        min_ = np.minimum(min_a, min_b)
        max_ = np.maximum(max_a, max_b)
        merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2, min=min_, max=max_)

    for k in traj_stats_b:  # ! modified by mpark
        if k not in traj_stats_a:
            merged_stats[k] = traj_stats_b[k]
    return merged_stats


class DexmgDataset(SequenceDataset, LeRobotSetup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_res = os.environ.get("DEXMG_VIDEO_RESOLUTION", "256x256")
        if self.video_res == "84x84":
            pass
        else:
            self.hdf5_path_video = os.path.join(os.path.dirname(self.hdf5_path), f'videos_{self.video_res}', os.path.basename(self.hdf5_path).replace('.hdf5', '_videos.hdf5'))
            self.hdf5_file_video = h5py.File(self.hdf5_path_video, 'r')

    def load_dataset_in_memory(self, demo_list, hdf5_file, obs_keys, dataset_keys, load_next_obs):
        """
        Loads the hdf5 dataset into memory, preserving the structure of the file. Note that this
        differs from `self.getitem_cache`, which, if active, actually caches the outputs of the
        `getitem` operation.

        Args:
            demo_list (list): list of demo keys, e.g., 'demo_0'
            hdf5_file (h5py.File): file handle to the hdf5 dataset.
            obs_keys (list, tuple): observation keys to fetch, e.g., 'images'
            dataset_keys (list, tuple): dataset keys to fetch, e.g., 'actions'
            load_next_obs (bool): whether to load next_obs from the dataset

        Returns:
            all_data (dict): dictionary of loaded data.
        """
        all_data = dict()
        print("SequenceDataset: loading dataset into memory...")
        for ep in LogUtils.custom_tqdm(demo_list):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {}
            all_data[ep]["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            # get obs
            all_data[ep]["obs"] = {
                k: hdf5_file["data/{}/obs/{}".format(ep, k)][()] for k in obs_keys
                if k in hdf5_file["data/{}/obs".format(ep)]  # ! modified by mpark
            }
            if load_next_obs:
                all_data[ep]["next_obs"] = {k: hdf5_file["data/{}/next_obs/{}".format(ep, k)][()] for k in obs_keys}
            # get other dataset keys
            for k in dataset_keys:
                if k in hdf5_file["data/{}".format(ep)]:
                    all_data[ep][k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype('float32')
                else:
                    all_data[ep][k] = np.zeros((all_data[ep]["attrs"]["num_samples"], 1), dtype=np.float32)

            if "model_file" in hdf5_file["data/{}".format(ep)].attrs:
                all_data[ep]["attrs"]["model_file"] = hdf5_file["data/{}".format(ep)].attrs["model_file"]

        return all_data

    def get_action_traj(self, ep):
        # action_traj = dict()
        # for key in self.action_keys:
        #     action_traj[key] = self.hdf5_file["data/{}/{}".format(ep, key)][()].astype('float32')

        action_traj = {
            k: self.hdf5_file["data/{}/{}".format(ep, k)][()].astype('float32') for k in self.action_keys
            if k in self.hdf5_file["data/{}".format(ep)]  # modified by mpark
        }
        return action_traj

    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """

        # check if this key should be in memory
        key_should_be_in_memory = (self.hdf5_cache_mode in ["all", "low_dim"])
        if key_should_be_in_memory:
            # if key is an observation, it may not be in memory
            if '/' in key:
                key1, key2 = key.split('/')
                assert (key1 in ['obs', 'next_obs', 'action_dict'])
                if key2 not in self.obs_keys_in_memory:
                    key_should_be_in_memory = False

        if key_should_be_in_memory:
            # read cache
            if '/' in key:
                key1, key2 = key.split('/')
                assert (key1 in ['obs', 'next_obs', 'action_dict'])
                ret = self.hdf5_cache[ep][key1][key2]
            else:
                ret = self.hdf5_cache[ep][key]
        else:
            # read from file
            hd5key = "data/{}/{}".format(ep, key)
            if self.video_res == "84x84":
                ret = self.hdf5_file[hd5key]
            else:
                ret = self.hdf5_file_video[hd5key]
            # ret = self.hdf5_file[hd5key]

            # # ! modified by mpark
            # filename = os.path.basename(self.hdf5_path)[:-5]
            # video_path = os.path.join(os.path.dirname(self.hdf5_path), 'videos', filename, key, f"{ep}.mp4")
            # # read video
            # ret, _, _ = io.read_video(video_path, pts_unit="sec")
            # if len(ret) == 0:
            #     raise ValueError(f"Video file {video_path} is empty or cannot be read.")
            # ret = ret.numpy()

        return ret

    def get_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            try:
                data = self.get_dataset_for_ep(demo_id, k)
                seq[k] = data[seq_begin_index: seq_end_index]
            except:
                pass

        seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)

        return seq, pad_mask

    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack - 1,  # note: need to decrement self.n_frame_stack by one
            seq_length=self.seq_length
        )

        # determine goal index
        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_demo - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs"
        )

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs"
            )

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=0,
                seq_length=1,
                prefix="next_obs",
            )
            meta["goal_obs"] = {k: goal[k][0] for k in goal}  # remove sequence dimension for goal

        # get action components
        ac_dict = OrderedDict()
        for k in self.action_keys:
            if k not in meta:
                continue
            ac = meta[k]
            # expand action shape if needed
            if len(ac.shape) == 1:
                ac = ac.reshape(-1, 1)
            ac_dict[k] = ac

        # normalize actions
        action_normalization_stats = self.get_action_normalization_stats()
        ac_dict = ObsUtils.normalize_dict(ac_dict, normalization_stats=action_normalization_stats)

        # concatenate all action components
        meta["actions"] = AcUtils.action_dict_to_vector(ac_dict)

        # also return the sampled index
        meta["index"] = index

        if demo_id in self._demo_id_to_demo_lang_emb:
            # language embedding
            T = meta["actions"].shape[0]
            meta["obs"][LANG_EMB_KEY] = np.tile(
                self._demo_id_to_demo_lang_emb[demo_id],
                (T, 1)
            )
        if demo_id in self._demo_id_to_demo_lang_str:
            meta["obs"][LANG_STR_KEY] = self._demo_id_to_demo_lang_str[demo_id]

        return meta

    def get_embodiment_tag(self):
        filename = os.path.basename(self.hdf5_path)
        if filename in ('two_arm_threading.hdf5', 'two_arm_three_piece_assembly.hdf5', 'two_arm_transport.hdf5'):
            return 31  # bimanual_panda_gripper
        elif filename in ('two_arm_box_cleanup.hdf5', 'two_arm_drawer_cleanup.hdf5', 'two_arm_lift_tray.hdf5'):
            return 32  # bimanual_panda_hand
        elif filename in ('two_arm_can_sort_random.hdf5', ):
            return 33  # gr1_arms_only
        else:
            return 34  # gr1_full_upper_body

    def __getitem__(self, index):
        output = super().__getitem__(index)
        output['embodiment_tag'] = self.get_embodiment_tag()
        return output


class MetaRobocasaDataset(MetaDataset, LeRobotSetup):
    def get_collator_kwargs(self):
        """
        Returns the kwargs to be passed to the data collator.
        """
        collator_kwargs = {
            'horizon': self.horizon,
            'modality_transform': self.modality_transform,
            'null_ratio': self.null_ratio,
            'null_targets': self.null_targets,
            'separate_null_targets': self.separate_null_targets,
        }
        for key in self.new_keys:
            collator_kwargs[key] = getattr(self, key)
        return collator_kwargs


HDF5_NAME_TO_EMBODIMENT_ID = {
    "two_arm_threading.hdf5": 31,
    "two_arm_three_piece_assembly.hdf5": 31,
    "two_arm_transport.hdf5": 31,
    "two_arm_box_cleanup.hdf5": 32,
    "two_arm_drawer_cleanup.hdf5": 32,
    "two_arm_lift_tray.hdf5": 32,
    "two_arm_can_sort_random.hdf5": 33,
    "two_arm_coffee.hdf5": 34,
    "two_arm_pouring.hdf5": 34,
}
EMBODIMENT_ID_TO_NAME = {
    31: "bimanual_panda_gripper",
    32: "bimanual_panda_hand",
    33: "gr1_arms_only",
    # 34: "gr1_full_upper_body",
    34: "gr1_arms_only",
}


class MetaDexmgDataset(MetaRobocasaDataset):
    def __init__(
        self,
        datasets,
        ds_weights,
        normalize_weights_by_ds_size=False,
    ):
        super(MetaDataset, self).__init__()
        self.datasets = datasets
        ds_lens = np.array([len(ds) for ds in self.datasets])
        if normalize_weights_by_ds_size:
            self.ds_weights = np.array(ds_weights) / ds_lens
        else:
            self.ds_weights = ds_weights
        self._ds_ind_bins = np.cumsum([0] + list(ds_lens))

        # cache mode "all" not supported! The action normalization stats of each
        # dataset will change after the datasets are already initialized
        for ds in self.datasets:
            assert ds.hdf5_cache_mode != "all"

        # TODO: comment  # ! modified by mpark
        self.get_action_normalization_stats()
        self.set_action_normalization_stats()

    def get_action_normalization_stats(self):
        """
        Computes a dataset-wide min, max, mean and standard deviation for the actions 
        (per dimension) and returns it.
        """

        # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
        # with the previous statistics.
        if getattr(self, 'action_normalization_stats', None) is None:
            action_stats, self.action_configs = self.get_action_stats()
            self.action_normalization_stats = {
                k: action_stats_to_normalization_stats(action_stats[k], self.action_configs[k])
                for k in action_stats
            }
            # self.action_configs = action_configs
        return self.action_normalization_stats, self.action_configs

    def set_action_normalization_stats(self):
        for ds in self.datasets:
            hdf5_name = os.path.basename(ds.hdf5_path)
            emb_name = EMBODIMENT_ID_TO_NAME[HDF5_NAME_TO_EMBODIMENT_ID[hdf5_name]]
            ds.set_action_normalization_stats(self.action_normalization_stats[emb_name])

    def get_action_stats(self):
        meta_action_stats = {}  # for cross-embodiment
        action_configs = {}
        for dataset in self.datasets:
            hdf5_name = os.path.basename(dataset.hdf5_path)
            emb_name = EMBODIMENT_ID_TO_NAME[HDF5_NAME_TO_EMBODIMENT_ID[hdf5_name]]
            if emb_name not in meta_action_stats:
                meta_action_stats[emb_name] = dataset.get_action_stats()
                action_configs[emb_name] = dataset.action_config
            else:
                ds_action_stats = dataset.get_action_stats()
                meta_action_stats[emb_name] = _aggregate_traj_stats_custom(meta_action_stats[emb_name], ds_action_stats)  # ! modified by mpark

        return meta_action_stats, action_configs

    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack - 1,  # note: need to decrement self.n_frame_stack by one
            seq_length=self.seq_length
        )

        # determine goal index
        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_demo - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs"
        )

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs"
            )

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=0,
                seq_length=1,
                prefix="next_obs",
            )
            meta["goal_obs"] = {k: goal[k][0] for k in goal}  # remove sequence dimension for goal

        # get action components
        ac_dict = OrderedDict()
        for k in self.action_keys:
            ac = meta[k]
            # expand action shape if needed
            if len(ac.shape) == 1:
                ac = ac.reshape(-1, 1)
            ac_dict[k] = ac

        # # normalize actions  # ! modified by mpark
        # action_normalization_stats = self.get_action_normalization_stats()
        # ac_dict = ObsUtils.normalize_dict(ac_dict, normalization_stats=action_normalization_stats)

        # concatenate all action components
        meta["actions"] = AcUtils.action_dict_to_vector(ac_dict)

        # also return the sampled index
        meta["index"] = index

        if demo_id in self._demo_id_to_demo_lang_emb:
            # language embedding
            T = meta["actions"].shape[0]
            meta["obs"][LANG_EMB_KEY] = np.tile(
                self._demo_id_to_demo_lang_emb[demo_id],
                (T, 1)
            )
        if demo_id in self._demo_id_to_demo_lang_str:
            meta["obs"][LANG_STR_KEY] = self._demo_id_to_demo_lang_str[demo_id]

        return meta
