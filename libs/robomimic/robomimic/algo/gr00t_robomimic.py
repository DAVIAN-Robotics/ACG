import json
import os

import torch
import torchvision.transforms as T
from gr00t.experiment.data_config import DATA_CONFIG_MAP  # todo: cross-import
from gr00t.model.policy import Gr00tPolicy  # todo: cross-import
from robomimic.algo import PolicyAlgo, register_algo_factory_func


@register_algo_factory_func("gr00t")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    return GR00T_Robomimic, {}


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


class GR00T_Robomimic(PolicyAlgo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tf = T.Resize((256, 256))

    def _create_networks(self):
        # action queue
        self.To = self.algo_config.horizon.observation_horizon
        self.Ta = self.algo_config.horizon.action_horizon
        self.Tp = self.algo_config.horizon.prediction_horizon
        # self.action_queue = deque(maxlen=self.Ta)
        self.action_queue = None

        # correction action_order
        self.action_order = ACTION_ORDER

        # create policy
        data_config = DATA_CONFIG_MAP[self.algo_config.data_config]
        modality_config = data_config.modality_config()
        modality_config['action'].delta_indices = list(range(self.Ta))
        modality_transform = data_config.transform()
        with open(self.algo_config.metadata_path, "r") as f:
            metadata_dict = json.load(f)

        self.policy = Gr00tPolicy(
            model_path=self.algo_config.model_path,
            embodiment_tag=self.algo_config.embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=self.algo_config.denoising_steps,
            metadata_dict=metadata_dict,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.policy.model.backbone.set_trainable_parameters(
            tune_visual=self.algo_config.tune_visual, tune_llm=self.algo_config.tune_llm
        )
        self.policy.model.action_head.set_trainable_parameters(
            tune_projector=self.algo_config.tune_projector, tune_diffusion_model=self.algo_config.tune_diffusion_model
        )
        print('### Updated ###')
        print(f"Tune backbone vision tower: {self.algo_config.tune_visual}")
        print(f"Tune backbone LLM: {self.algo_config.tune_llm}")
        print(f"Tune action head projector: {self.algo_config.tune_projector}")
        print(f"Tune action head DiT: {self.algo_config.tune_diffusion_model}")

        self.nets['model'] = self.policy.model

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        self.action_queue = None

    def train_on_batch(self, batch, epoch, validate=False):
        raise NotImplementedError("Training is not supported for GR00TPolicy_Robomimic.")

    def serialize(self):
        raise NotImplementedError("Training is not supported for GR00TPolicy_Robomimic.")

    def deserialize(self, model_dict):
        raise NotImplementedError("Training is not supported for GR00TPolicy_Robomimic.")

    def log_info(self, info):
        raise NotImplementedError("Training is not supported for GR00TPolicy_Robomimic.")

    def get_action(self, obs_dict, goal_dict=None):
        self.policy.modality_transform.eval()
        self.policy.model.eval()

        if len(self.action_queue) == 0:
            is_single_obs = obs_dict['robot0_agentview_left_image'].dim() == 4
            if is_single_obs:
                assert self.To == 1, f"Observation horizon should be 1, but got {self.To}"
            obs_dict_gr00t = {}
            for k, v in obs_dict.items():
                if k in ROBOCASA_KEY_TO_GR00T_KEY:
                    if ROBOCASA_KEY_TO_GR00T_KEY[k].startswith('video.'):
                        obs_dict_gr00t[ROBOCASA_KEY_TO_GR00T_KEY[k]] = v[:, :self.To] if not is_single_obs else v
                    elif ROBOCASA_KEY_TO_GR00T_KEY[k].startswith('state.'):
                        obs_dict_gr00t[ROBOCASA_KEY_TO_GR00T_KEY[k]] = v[:, :self.To].cpu().to(dtype=torch.float32) if not is_single_obs else v.cpu().to(dtype=torch.float32)
                    else:
                        obs_dict_gr00t[ROBOCASA_KEY_TO_GR00T_KEY[k]] = v

            action_dict = self.policy.get_action(obs_dict_gr00t)
            action = torch.cat([action_dict[k][:, None] if action_dict[k].dim() == 1 else action_dict[k] for k in ACTION_ORDER], dim=-1)

            if is_single_obs:
                action = action.unsqueeze(0)
            assert action.shape[0] == 1, "Batch size should be 1, but got {}".format(action.shape)
            action[..., 6] = action[..., 6] * 2 - 1  # gripper close
            action[..., 11] = action[..., 11] * 2 - 1  # control mode
            self.action_queue.extend(action[0, :self.Ta])

        action = self.action_queue.popleft()
        action = action.unsqueeze(0)

        return action
