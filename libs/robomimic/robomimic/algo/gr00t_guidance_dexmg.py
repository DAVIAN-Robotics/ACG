import json
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from gr00t.data.schema import DatasetMetadata
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from robomimic.algo import register_algo_factory_func

from .gr00t_guidance import GR00T_Guidance_Robomimic

# from robomimic.algo.gr00t import ACTION_ORDER, ROBOCASA_KEY_TO_GR00T_KEY


@register_algo_factory_func("gr00t_guidance_dexmg")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return GR00T_Guidance_DexMG_Robomimic, {}


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
    33: {  # gr1_arms_only
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
    34: {  # gr1_full_upper_body but gr1_arms_only
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

DEXMG_ACTION_ORDER = {
    31: [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_gripper_close",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_gripper_close",
    ],
    32: [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_hand",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_hand",
    ],
    33: [
        "action.right_arm",
        "action.left_arm",
        "action.right_hand",
        "action.left_hand",
    ],
    34: [
        "action.right_arm",
        "action.left_arm",
        "action.right_hand",
        "action.left_hand",
    ],
}

ENV_NAME_TO_EMBODIMENT_ID = {
    "TwoArmThreading": 31,
    "TwoArmThreePieceAssembly": 31,
    "TwoArmTransport": 31,
    "TwoArmBoxCleanup": 32,
    "TwoArmDrawerCleanup": 32,
    "TwoArmLiftTray": 32,
    "TwoArmCanSortBlue": 33,
    "TwoArmCanSortRandom": 33,
    "TwoArmCanSortRed": 33,
    "TwoArmCoffee": 34,
    "TwoArmPouring": 34,
}


class GR00T_Guidance_DexMG_Robomimic(GR00T_Guidance_Robomimic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_modality_transform()  # re-init
        self.tf = T.Pad((0, 48), fill=0, padding_mode='constant')

    def set_modality_transform(self):

        data_configs = {
            31: 'dexmg_bimanual_panda_gripper',
            32: 'dexmg_bimanual_panda_hand',
            33: 'dexmg_gr1_arms_only',
            34: 'dexmg_gr1_arms_only',
        }

        self.modality_transform = {}

        for emb_id, data_config in data_configs.items():
            metadata_path = os.path.join('metadatas', data_config + '.json')
            data_config_cls = DATA_CONFIG_MAP[data_config]
            modality_transform = data_config_cls.transform()

            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
            metadata = DatasetMetadata.model_validate(metadata_dict)
            modality_transform.set_metadata(metadata)

            self.modality_transform[emb_id] = modality_transform

    def get_action(self, obs_dict, goal_dict=None, env_name=None):
        emb_id = ENV_NAME_TO_EMBODIMENT_ID[env_name]

        self.policy._modality_transform = self.modality_transform[emb_id]
        self.policy.modality_transform.eval()
        self.policy.model.eval()

        if self.action_queue is None:
            image_keys = [k for k in obs_dict.keys() if k.endswith('image')]
            is_single_obs = obs_dict[image_keys[0]].dim() == 4
            if is_single_obs:
                assert self.To == 1, f"Observation horizon should be 1, but got {self.To}"
            obs_dict_gr00t = {}
            for k, v in obs_dict.items():
                if k in DEXMG_KEY_TO_GR00T_KEY[emb_id]:
                    if DEXMG_KEY_TO_GR00T_KEY[emb_id][k].startswith('video.'):
                        v = self.tf(v)
                        obs_dict_gr00t[DEXMG_KEY_TO_GR00T_KEY[emb_id][k]] = v[:, None, :] if is_single_obs else v[:, :self.To][:, -1:]
                    elif DEXMG_KEY_TO_GR00T_KEY[emb_id][k].startswith('state.'):
                        obs_dict_gr00t[DEXMG_KEY_TO_GR00T_KEY[emb_id][k]] = v[:, None, :].cpu().to(dtype=torch.float32) if is_single_obs else v[:, :self.To].cpu().to(dtype=torch.float32)
                    else:
                        obs_dict_gr00t[DEXMG_KEY_TO_GR00T_KEY[emb_id][k]] = v

            action_dict = self.policy.get_action(obs_dict_gr00t, **self.policy_get_action_kwargs)
            action = torch.cat([action_dict[k][:, None] if action_dict[k].dim() == 1 else action_dict[k] for k in DEXMG_ACTION_ORDER[emb_id]], dim=-1)

            if emb_id == 31:  # bimanual_panda_gripper
                action[..., 6] = action[..., 6] * 2 - 1  # right gripper close
                action[..., 13] = action[..., 13] * 2 - 1  # left gripper close
            self.action_queue = action[:, :self.Ta]  # (B, T, D)

        action = self.action_queue[:, 0, :]
        self.action_queue = self.action_queue[:, 1:, :]
        if self.action_queue.shape[1] == 0:
            self.action_queue = None

        return action
