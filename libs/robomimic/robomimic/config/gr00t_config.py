"""
Config for Diffusion Policy algorithm.
"""
from robomimic.config.base_config import BaseConfig


class GR00TConfig(BaseConfig):
    ALGO_NAME = "gr00t"

    def algo_config(self):
        self.algo.model_path = 'nvidia/GR00T-N1-2B'
        self.algo.metadata_path = 'metadatas/single_panda_gripper.json'
        self.algo.data_config = 'robocasa_single_panda_gripper'
        self.algo.embodiment_tag = 'single_panda_gripper'
        self.algo.denoising_steps = 16  # will be overridden if we use guidance models

        # necessary configs for robomimic
        self.algo.language_conditioned = False

        # action queue
        self.algo.horizon.observation_horizon = 1
        self.algo.horizon.action_horizon = 16
        self.algo.horizon.prediction_horizon = 16

        # optimization parameters
        self.algo.optim_params.model.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.model.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.model.learning_rate.epoch_schedule = []  # epochs where LR decay occurs
        self.algo.optim_params.model.learning_rate.scheduler_type = "constant"
        self.algo.optim_params.model.learning_rate.num_warmup_steps = 0
        self.algo.optim_params.model.regularization.L2 = 0.00          # L2 regularization strength
        self.algo.optim_params.model.optimizer_type = "adam"      # optimizer type
        self.algo.optim_params.model.optimizer_kwargs = {}

        # tuning parameters
        self.algo.tune_visual = True
        self.algo.tune_llm = False
        self.algo.tune_projector = True
        self.algo.tune_diffusion_model = True

        # additional guidance options
        self.algo.guidance.name = "no"

        # tuning-free guidance options
        self.algo.guidance.scale = 3.0
        self.algo.guidance.num_inference_timesteps = 16
        self.algo.guidance.skip_blocks = [7, 9, 11]

        ### configs for baselines ###
        self.algo.guidance.threshold = 0.0
        self.algo.guidance.n_ensemble = 1
        self.algo.guidance.sigma = 1.0
        self.algo.guidance.noise_std = 1.0
        return
