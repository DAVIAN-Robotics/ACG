import torch
from robomimic.algo import register_algo_factory_func
from robomimic.algo.gr00t_robomimic import ACTION_DICT, ACTION_ORDER, ROBOCASA_KEY_TO_GR00T_KEY, GR00T_Robomimic

from .guidance import modify_gr00t_policy


@register_algo_factory_func("gr00t_guidance")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return GR00T_Guidance_Robomimic, {}


class GR00T_Guidance_Robomimic(GR00T_Robomimic):
    ### not used functions (start) ###
    def serialize(self):
        raise NotImplementedError("Training is not supported for GR00T_Guidance_Robomimic.")

    def deserialize(self, model_dict):
        raise NotImplementedError("Training is not supported for GR00T_Guidance_Robomimic.")

    def train_on_batch(self, batch, epoch, validate=False):
        raise NotImplementedError("Training is not supported for GR00T_Guidance_Robomimic.")

    def log_info(self, info):
        raise NotImplementedError("Training is not supported for GR00T_Guidance_Robomimic.")
    ### not used functions (end) ###

    def _create_networks(self):
        super()._create_networks()

        # modify methods of policy, action_head, etc.
        self.policy = modify_gr00t_policy(self.algo_config.guidance.name, self.policy)

        # additional options
        self.policy_get_action_kwargs = self.algo_config.guidance.to_dict()

    def get_action(self, obs_dict, goal_dict=None):
        self.policy.modality_transform.eval()
        self.policy.model.eval()

        if self.action_queue is None:
            is_single_obs = obs_dict['robot0_agentview_left_image'].dim() == 4
            if is_single_obs:
                assert self.To == 1, f"Observation horizon should be 1, but got {self.To}"
            obs_dict_gr00t = {}
            for k, v in obs_dict.items():
                if k in ROBOCASA_KEY_TO_GR00T_KEY:
                    if ROBOCASA_KEY_TO_GR00T_KEY[k].startswith('video.'):
                        v = self.tf(v)
                        obs_dict_gr00t[ROBOCASA_KEY_TO_GR00T_KEY[k]] = v[:, None, :] if is_single_obs else v[:, :self.To][:, -1:]
                    elif ROBOCASA_KEY_TO_GR00T_KEY[k].startswith('state.'):
                        obs_dict_gr00t[ROBOCASA_KEY_TO_GR00T_KEY[k]] = v[:, None, :].cpu().to(dtype=torch.float32) if is_single_obs else v[:, :self.To].cpu().to(dtype=torch.float32)
                    else:
                        obs_dict_gr00t[ROBOCASA_KEY_TO_GR00T_KEY[k]] = v

            action_dict = self.policy.get_action(obs_dict_gr00t, **self.policy_get_action_kwargs)
            action = torch.cat([action_dict[k][:, None] if action_dict[k].dim() == 1 else action_dict[k] for k in ACTION_ORDER], dim=-1)

            action[..., 6] = action[..., 6] * 2 - 1  # gripper close
            action[..., 11] = action[..., 11] * 2 - 1  # control mode
            self.action_queue = action[:, :self.Ta]  # (B, T, D)

        action = self.action_queue[:, 0, :]
        self.action_queue = self.action_queue[:, 1:, :]
        if self.action_queue.shape[1] == 0:
            self.action_queue = None

        return action
