import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.model.policy import Gr00tPolicy, squeeze_dict_values, unsqueeze_dict_values
from transformers.feature_extraction_utils import BatchFeature

COMPUTE_DTYPE = torch.bfloat16


class ACGAttnProcessor2_0:
    r"""
    Processor for implementing the Incoherent Variant of ACG using scaled dot-product attention (enabled by default in PyTorch 2.0).
    ACG reference: https://arxiv.org/abs/xxxx.xxxxx

    We also sincerely appreciate the excellent work on PAG, from which this implementation is derived.
    PAG reference: https://arxiv.org/abs/2403.17377
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:  # False
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        value = attn.to_v(encoder_hidden_states)

        hidden_states = attn.to_v(encoder_hidden_states)
        hidden_states = hidden_states.to(value.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class Gr00tPolicy_ACG(Gr00tPolicy):

    def get_action(
        self, observations: Dict[str, Any],
        scale: float = 3.0,
        skip_blocks: List[int] = [7, 9, 11],
        num_inference_timesteps: int = 16,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # let the get_action handles both batch and single input
        is_batch = self._check_state_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)

        # Apply transforms
        normalized_input = self.apply_transforms(observations)
        normalized_action = self._get_action_from_normalized_input(
            normalized_input,
            scale=scale,
            skip_blocks=skip_blocks,
            num_inference_timesteps=num_inference_timesteps,
        )
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def _get_action_from_normalized_input(
        self, normalized_input: Dict[str, Any],
        scale: float = 3.0,
        skip_blocks: List[int] = [7, 9, 11],
        num_inference_timesteps: int = 16,
    ) -> torch.Tensor:
        # Set up autocast context if needed
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(
                normalized_input,
                scale=scale,
                skip_blocks=skip_blocks,
                num_inference_timesteps=num_inference_timesteps,
            )

        normalized_action = model_pred["action_pred"].float()
        return normalized_action


class GR00T_N1_ACG(GR00T_N1):

    def get_action(
        self,
        inputs: dict,
        scale: float = 3.0,
        skip_blocks: List[int] = [7, 9, 11],
        num_inference_timesteps: int = 16,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(
            backbone_outputs, action_inputs,
            scale=scale,
            skip_blocks=skip_blocks,
            num_inference_timesteps=num_inference_timesteps,
        )
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs


class FlowmatchingActionHead_ACG(FlowmatchingActionHead):

    @torch.no_grad()
    def get_action(
        self, backbone_output: BatchFeature,
        action_input: BatchFeature,
        scale: float = 3.0,
        skip_blocks: List[int] = [7, 9, 11],
        num_inference_timesteps: int = 16,
    ) -> BatchFeature:

        def convert_to_bad_model(skip_blocks: List[int] = []) -> None:
            n_blocks = len(self.model.transformer_blocks)
            for i in range(n_blocks):
                if i in skip_blocks:
                    self.model.transformer_blocks[i].attn1.processor = ACGAttnProcessor2_0()

        def convert_to_original_model(skip_blocks: List[int] = []) -> None:
            n_blocks = len(self.model.transformer_blocks)
            for i in range(n_blocks):
                if i in skip_blocks:
                    self.model.transformer_blocks[i].attn1.processor = AttnProcessor2_0()

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features  # (B, 99, 1536)
        embodiment_id = action_input.embodiment_id  # (B,)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)  # (B, 1, 1536)

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )  # (B, T=16, D=32)

        # num_steps = self.num_inference_timesteps
        num_steps = num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)  # (B, T=16, 1536)
            # Maybe add position embedding.
            if self.config.add_pos_embed:  # True
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            vl_embs = vl_embeds

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)  # (B, T=17, 1536)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_perturb = torch.zeros_like(pred)
            if scale != 1.0:
                convert_to_bad_model(skip_blocks)
                model_output_perturb = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
                pred_perturb = self.action_decoder(model_output_perturb, embodiment_id)
                convert_to_original_model(skip_blocks)

            pred = pred + (scale - 1) * (pred - pred_perturb)

            pred_velocity = pred[:, -self.action_horizon:]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity

        return BatchFeature(data={"action_pred": actions})
