# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field, PrivateAttr
from transformers.data.data_collator import DataCollatorMixin

from gr00t.data.schema import DatasetMetadata, EmbodimentTag
from gr00t.data.transform.base import ComposedModalityTransform, InvertibleModalityTransform
from gr00t.model.backbone.eagle2_hg_model.inference_eagle_repo import EagleProcessor

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
EAGLE_KEYS = ["pixel_values", "input_ids", "attention_mask"]


def collate_gr00t(features: List[dict], processor) -> dict:
    batch = {}
    keys = features[0].keys()
    assert all(
        all(key in feature for key in keys) for feature in features
    ), "All features must have the same keys."

    for key in keys:
        values = [elem[key] for elem in features]
        if key not in EAGLE_KEYS:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))

    vlm_batch = processor.collate_fn(features)
    # merge vlm_batch with batch
    for key in vlm_batch.keys():
        assert key not in batch, f"Key {key} already exists in batch."
        batch[key] = vlm_batch[key]

    return batch


class DefaultDataCollatorGR00T(DataCollatorMixin):
    def __init__(self, processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        return collate_gr00t(features, self.processor)


ROBOCASA_KEY_TO_GR00T_KEY = {
    # video
    'robot0_agentview_left_image': 'video.left_view',
    'robot0_agentview_right_image': 'video.right_view',
    'robot0_eye_in_hand_image': 'video.wrist_view',
    # state
    'robot0_base_to_eef_pos': 'state.end_effector_position_relative',
    'robot0_base_to_eef_quat': 'state.end_effector_rotation_relative',
    'robot0_gripper_qpos': 'state.gripper_qpos',
    'robot0_base_pos': 'state.base_position',
    'robot0_base_quat': 'state.base_rotation',
    # language
    'lang_str': 'annotation.human.action.task_description',
}

ACTION_ORDER = [
    'action.end_effector_position',
    'action.end_effector_rotation',
    'action.gripper_close',
    'action.base_motion',
    'action.control_mode',
]

ACTION_DICT = {
    "action.end_effector_position": list(range(0, 3)),
    "action.end_effector_rotation": list(range(3, 6)),
    "action.gripper_close": list(range(6, 7)),
    "action.base_motion": list(range(7, 11)),
    "action.control_mode": list(range(11, 12)),
}


class RobocasaDataCollatorGR00T(DataCollatorMixin):
    def __init__(
        self,
        processor,
        horizon: Dict[str, int],
        modality_transform: ComposedModalityTransform,
        null_ratio: float,
        null_targets: List[str],
        separate_null_targets: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.To = horizon['observation_horizon']
        self.Tp = horizon['prediction_horizon']
        self.modality_transform = modality_transform
        self.null_ratio = null_ratio
        self.null_targets = null_targets
        self.separate_null_targets = separate_null_targets

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        features_gr00t = self.collate_robocasa_to_gr00t(features)
        return collate_gr00t(features_gr00t, self.processor)

    def collate_robocasa_to_gr00t(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_gr00t = []
        for feature in features:
            obs_dict = feature['obs']
            obs_dict_gr00t = {}
            for k, v in obs_dict.items():
                if k in ROBOCASA_KEY_TO_GR00T_KEY:
                    if ROBOCASA_KEY_TO_GR00T_KEY[k].startswith('video.'):
                        obs_dict_gr00t[ROBOCASA_KEY_TO_GR00T_KEY[k]] = F.interpolate(
                            torch.from_numpy(v[:self.To]).to(torch.float32).permute(0, 3, 1, 2) / 255.,
                            size=(256, 256),
                            mode='bilinear',
                        )
                    elif ROBOCASA_KEY_TO_GR00T_KEY[k].startswith('state.'):
                        obs_dict_gr00t[ROBOCASA_KEY_TO_GR00T_KEY[k]] = torch.from_numpy(v[:self.To]).to(dtype=torch.float32)
                    else:
                        obs_dict_gr00t[ROBOCASA_KEY_TO_GR00T_KEY[k]] = v

            action_dict_gr00t = {}
            for k, v in ACTION_DICT.items():
                action_dict_gr00t[k] = torch.from_numpy(feature['actions'][:self.Tp, v]).to(dtype=torch.float32)

            is_null = random.random() < self.null_ratio

            is_null = random.random() < self.null_ratio if self.separate_null_targets else is_null
            if 'text' in self.null_targets and is_null:
                obs_dict_gr00t['annotation.human.action.task_description'] = ""

            obs_dict_gr00t_new = self.modality_transform({**obs_dict_gr00t, **action_dict_gr00t})

            is_null = random.random() < self.null_ratio if self.separate_null_targets else is_null
            if 'state' in self.null_targets and is_null:
                obs_dict_gr00t_new['state'] = np.zeros_like(obs_dict_gr00t_new['state'])

            is_null = random.random() < self.null_ratio if self.separate_null_targets else is_null
            if 'observation' in self.null_targets and is_null:
                obs_dict_gr00t_new['pixel_values'] = torch.zeros_like(obs_dict_gr00t_new['pixel_values'])

            features_gr00t.append(obs_dict_gr00t_new)

        return features_gr00t


DEXMG_KEY_TO_GR00T_KEY = {
    31: {  # bimanual_panda_gripper
        # video
        'agentview_image': 'video.front_view',
        'robot0_eye_in_hand_image': 'video.right_wrist_view',
        'robot1_eye_in_hand_image': 'video.left_wrist_view',
        # state
        'robot0_eef_pos': 'state.right_arm_eef_pos',
        'robot0_eef_quat': 'state.right_arm_eef_quat',
        'robot0_gripper_qpos': 'state.right_gripper_qpos',
        'robot1_eef_pos': 'state.left_arm_eef_pos',
        'robot1_eef_quat': 'state.left_arm_eef_quat',
        'robot1_gripper_qpos': 'state.left_gripper_qpos',
        # language
        'lang_str': 'annotation.human.action.task_description',
    },
    32: {  # bimanual_panda_hand
        # video
        'agentview_image': 'video.ego_view',
        'robot0_eye_in_hand_image': 'video.right_wrist_view',
        'robot1_eye_in_hand_image': 'video.left_wrist_view',
        # state
        'robot0_eef_pos': 'state.right_arm_eef_pos',
        'robot0_eef_quat': 'state.right_arm_eef_quat',
        'robot0_gripper_qpos': 'state.right_hand_12d',
        'robot1_eef_pos': 'state.left_arm_eef_pos',
        'robot1_eef_quat': 'state.left_arm_eef_quat',
        'robot1_gripper_qpos': 'state.left_hand_12d',
        # language
        'lang_str': 'annotation.human.action.task_description',
    },
    33: {  # gr1_arms_only, gr1_full_upper_body
        # video
        'frontview_image': 'video.ego_view',
        'agentview_image': 'video.ego_view',
        # state
        'robot0_right_eef_pos': 'state.robot0_right_eef_pos',
        'robot0_right_eef_quat': 'state.robot0_right_eef_quat',
        'robot0_left_eef_pos': 'state.robot0_left_eef_pos',
        'robot0_left_eef_quat': 'state.robot0_left_eef_quat',
        'robot0_right_gripper_qpos': 'state.robot0_right_gripper_qpos',
        'robot0_left_gripper_qpos': 'state.robot0_left_gripper_qpos',
        # language
        'lang_str': 'annotation.human.action.task_description',
    },
    34: {  # gr1_arms_only, gr1_full_upper_body
        # video
        'frontview_image': 'video.ego_view',
        'agentview_image': 'video.ego_view',
        # state
        'robot0_right_eef_pos': 'state.robot0_right_eef_pos',
        'robot0_right_eef_quat': 'state.robot0_right_eef_quat',
        'robot0_left_eef_pos': 'state.robot0_left_eef_pos',
        'robot0_left_eef_quat': 'state.robot0_left_eef_quat',
        'robot0_right_gripper_qpos': 'state.robot0_right_gripper_qpos',
        'robot0_left_gripper_qpos': 'state.robot0_left_gripper_qpos',
        # language
        'lang_str': 'annotation.human.action.task_description',
    },
}

DEXMG_ACTION_DICT = {
    31: {
        "action.right_arm_eef_pos": list(range(0, 3)),
        "action.right_arm_eef_rot": list(range(3, 6)),
        "action.right_gripper_close": list(range(6, 7)),
        "action.left_arm_eef_pos": list(range(7, 10)),
        "action.left_arm_eef_rot": list(range(10, 13)),
        "action.left_gripper_close": list(range(13, 14)),
    },
    32: {
        "action.right_arm_eef_pos": list(range(0, 3)),
        "action.right_arm_eef_rot": list(range(3, 6)),
        "action.right_hand": list(range(6, 12)),
        "action.left_arm_eef_pos": list(range(12, 15)),
        "action.left_arm_eef_rot": list(range(15, 18)),
        "action.left_hand": list(range(18, 24)),
    },
    33: {
        "action.right_arm": list(range(0, 6)),
        "action.left_arm": list(range(6, 12)),
        "action.right_hand": list(range(12, 18)),
        "action.left_hand": list(range(18, 24)),
    },
    34: {
        "action.right_arm": list(range(0, 6)),
        "action.left_arm": list(range(6, 12)),
        "action.right_hand": list(range(12, 18)),
        "action.left_hand": list(range(18, 24)),
    }
}


class DexmgDataCollatorGR00T(RobocasaDataCollatorGR00T):

    def collate_robocasa_to_gr00t(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_gr00t = []
        for feature in features:
            embodiment_tag = feature['embodiment_tag']
            obs_dict = feature['obs']
            obs_dict_gr00t = {}
            for k, v in obs_dict.items():
                if k in DEXMG_KEY_TO_GR00T_KEY[embodiment_tag]:
                    if DEXMG_KEY_TO_GR00T_KEY[embodiment_tag][k].startswith('video.'):
                        obs_dict_gr00t[DEXMG_KEY_TO_GR00T_KEY[embodiment_tag][k]] = F.interpolate(
                            torch.from_numpy(v[:self.To]).to(torch.float32).permute(0, 3, 1, 2) / 255.,
                            size=(256, 256),
                            mode='bilinear',
                        )
                    elif DEXMG_KEY_TO_GR00T_KEY[embodiment_tag][k].startswith('state.'):
                        obs_dict_gr00t[DEXMG_KEY_TO_GR00T_KEY[embodiment_tag][k]] = torch.from_numpy(v[:self.To]).to(dtype=torch.float32)
                    else:
                        obs_dict_gr00t[DEXMG_KEY_TO_GR00T_KEY[embodiment_tag][k]] = v

            action_dict_gr00t = {}
            for k, v in DEXMG_ACTION_DICT[embodiment_tag].items():
                action_dict_gr00t[k] = torch.from_numpy(feature['actions'][:self.Tp, v]).to(dtype=torch.float32)

            is_null = random.random() < self.null_ratio

            is_null = random.random() < self.null_ratio if self.separate_null_targets else is_null
            if 'text' in self.null_targets and is_null:
                obs_dict_gr00t['annotation.human.action.task_description'] = ""

            obs_dict_gr00t_new = self.modality_transform[embodiment_tag]({**obs_dict_gr00t, **action_dict_gr00t})

            is_null = random.random() < self.null_ratio if self.separate_null_targets else is_null
            if 'state' in self.null_targets and is_null:
                obs_dict_gr00t_new['state'] = np.zeros_like(obs_dict_gr00t_new['state'])

            is_null = random.random() < self.null_ratio if self.separate_null_targets else is_null
            if 'observation' in self.null_targets and is_null:
                obs_dict_gr00t_new['pixel_values'] = torch.zeros_like(obs_dict_gr00t_new['pixel_values'])

            features_gr00t.append(obs_dict_gr00t_new)

        return features_gr00t


class RobocasaDataCollatorGR00T_forStats(RobocasaDataCollatorGR00T):
    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        features_gr00t = self.collate_robocasa_to_gr00t(features)
        features_gr00t = collate_gr00t(features_gr00t, self.processor)
        features_gr00t['lang_str'] = [feat['obs']['lang_str'] for feat in features]
        return features_gr00t


class GR00TTransform(InvertibleModalityTransform):
    _EMBODIMENT_TAG_MAPPING = {
        "gr1": 24,
        "new_embodiment": 31,  # use the last projector for new embodiment,
        'single_panda_gripper': 31,
        'bimanual_panda_gripper': 31,
        'bimanual_panda_hand': 32,
        'gr1_arms_only': 33,
        'gr1_full_upper_body': 34,
        'so101': 31,
    }

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=_EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    vlm_processor: EagleProcessor = Field(default=EagleProcessor())
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except:  # noqa: E722
                # Handle language annotation special case
                if "annotation" in key:
                    modality = "language"
                else:
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_gr00t_processing(self, batch: dict) -> dict:
        """
        Args:
            batch:
                video: [T, V, H, W, C]
        """
        images = batch["images"]
        assert images.shape[0] == 1, "double check formatting when doing multi-time step"
        # Remove the singleton time dimension.
        images = images[0]
        images = [{"np_array": images[idx]} for idx in range(len(images))]
        if "language" in batch:
            lang = batch["language"]
            if isinstance(lang, list):
                lang = lang[0]
        else:
            lang = self.default_instruction
            raise ValueError("Language not found for {self.embodiment_tag.value}")

        prompt = [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": lang,
                "image": images,
            },
        ]
        inputs = self.vlm_processor.prepare_input({"prompt": prompt})
        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        return data["video"]  # [t v h w c]

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction

        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language = self._prepare_language(data)

        vlm_batch = {"images": images, "language": language}
        vlm_outputs = self._apply_gr00t_processing(vlm_batch)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        # By default, assume regular robot data with only real action.
        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = []
        for i in range(batch_size):
            single_data = {}
            for key, value in data.items():
                # Special handling for string values to prevent character-wise splitting
                if isinstance(value, str):
                    # For string values, keep the entire string instead of indexing
                    single_data[key] = value
                else:
                    # For arrays and other data types, extract the i-th element
                    try:
                        single_data[key] = value[i]
                    except (TypeError, IndexError):
                        # If value is not indexable or index is out of bounds, use the whole value
                        single_data[key] = value
            data_split.append(single_data)

        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate_gr00t(data_split_processed, self.vlm_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
