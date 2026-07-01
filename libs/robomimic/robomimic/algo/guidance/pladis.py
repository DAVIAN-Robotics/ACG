import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0
from entmax import entmax15, sparsemax
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.model.policy import Gr00tPolicy, squeeze_dict_values, unsqueeze_dict_values
from transformers.feature_extraction_utils import BatchFeature

COMPUTE_DTYPE = torch.bfloat16


class PLADISAttnProcessor:
    r"""
    PLADIS cross-attention processor (single-pass sparse-guidance).

    Replicates diffusers' ``AttnProcessor2_0`` exactly, but replaces the single
    ``scaled_dot_product_attention`` call with an extrapolation between the
    dense (softmax) and sparse (alpha-entmax / sparsemax) attention maps:

        attn = dense + pladis_scale * (sparse - dense)
             = pladis_scale * sparse + (1 - pladis_scale) * dense

    ``pladis_scale == 1.0`` -> pure sparse, ``0.0`` -> pure dense (no guidance),
    ``> 1.0`` -> extrapolation past the sparse map (the PLADIS regime, ~1.5-2.0).

    PLADIS reference: https://arxiv.org/abs/2503.07677
    Derived from the released PLADIS SparseAttnProcessor and diffusers AttnProcessor2_0.
    """

    def __init__(self, pladis_scale: float = 1.5, method: str = "ent15max"):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("PLADISAttnProcessor requires PyTorch 2.0+.")
        self.pladis_scale = pladis_scale
        self.method = method

    def _sparse(self, logits: torch.Tensor) -> torch.Tensor:
        if self.method == "sparsemax":
            return sparsemax(logits, dim=-1)
        elif self.method == "ent15max":
            return entmax15(logits, dim=-1)
        else:
            raise ValueError(f"Unknown PLADIS method: {self.method}")

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

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # --- PLADIS: dense/sparse extrapolation in place of scaled_dot_product_attention ---
        scale_factor = 1.0 / math.sqrt(query.size(-1))
        logits = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # (B, H, Lq, Lk)
        if attention_mask is not None:
            logits = logits + attention_mask
        # softmax/entmax computed in float32 for numerical stability under bf16 autocast.
        logits = logits.float()
        dense = torch.softmax(logits, dim=-1)
        sparse = self._sparse(logits)
        attn_weight = dense + self.pladis_scale * (sparse - dense)
        attn_weight = attn_weight.to(value.dtype)
        hidden_states = torch.matmul(attn_weight, value)
        # -------------------------------------------------------------------------------

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

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


class Gr00tPolicy_PLADIS(Gr00tPolicy):

    def get_action(
        self, observations: Dict[str, Any],
        pladis_scale: float = 1.5,
        method: str = "ent15max",
        pladis_layers: Optional[List[int]] = None,
        num_inference_timesteps: int = 16,
        **kwargs,
    ) -> Dict[str, Any]:
        # extra guidance-config keys (name, scale, skip_blocks, ...) arrive via **kwargs; ignore them.
        is_batch = self._check_state_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)

        normalized_input = self.apply_transforms(observations)
        normalized_action = self._get_action_from_normalized_input(
            normalized_input,
            pladis_scale=pladis_scale,
            method=method,
            pladis_layers=pladis_layers,
            num_inference_timesteps=num_inference_timesteps,
        )
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def _get_action_from_normalized_input(
        self, normalized_input: Dict[str, Any],
        pladis_scale: float = 1.5,
        method: str = "ent15max",
        pladis_layers: Optional[List[int]] = None,
        num_inference_timesteps: int = 16,
    ) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(
                normalized_input,
                pladis_scale=pladis_scale,
                method=method,
                pladis_layers=pladis_layers,
                num_inference_timesteps=num_inference_timesteps,
            )

        normalized_action = model_pred["action_pred"].float()
        return normalized_action


class GR00T_N1_PLADIS(GR00T_N1):

    def get_action(
        self,
        inputs: dict,
        pladis_scale: float = 1.5,
        method: str = "ent15max",
        pladis_layers: Optional[List[int]] = None,
        num_inference_timesteps: int = 16,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(
            backbone_outputs, action_inputs,
            pladis_scale=pladis_scale,
            method=method,
            pladis_layers=pladis_layers,
            num_inference_timesteps=num_inference_timesteps,
        )
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs


# one-time log guard (which blocks receive PLADIS) — verifies cross-attention targeting at runtime.
_PLADIS_LOGGED = False


class FlowmatchingActionHead_PLADIS(FlowmatchingActionHead):

    @torch.no_grad()
    def get_action(
        self, backbone_output: BatchFeature,
        action_input: BatchFeature,
        pladis_scale: float = 1.5,
        method: str = "ent15max",
        pladis_layers: Optional[List[int]] = None,
        num_inference_timesteps: int = 16,
    ) -> BatchFeature:

        def _cross_block_indices() -> List[int]:
            n_blocks = len(self.model.transformer_blocks)
            interleave = getattr(self.model.config, "interleave_self_attention", False)
            if interleave:
                # odd idx = self-attention (encoder_hidden_states=None), even idx = cross-attention.
                cross = [i for i in range(n_blocks) if i % 2 == 0]
            else:
                cross = list(range(n_blocks))
            if pladis_layers is not None:
                cross = [i for i in cross if i in pladis_layers]
            return cross

        def install_pladis(blocks: List[int]) -> None:
            for i in blocks:
                self.model.transformer_blocks[i].attn1.processor = PLADISAttnProcessor(pladis_scale, method)

        def restore_original(blocks: List[int]) -> None:
            for i in blocks:
                self.model.transformer_blocks[i].attn1.processor = AttnProcessor2_0()

        cross_blocks = _cross_block_indices()

        global _PLADIS_LOGGED
        if not _PLADIS_LOGGED:
            print(
                f"\033[35m\033[1m[PLADIS] scale={pladis_scale} method={method} "
                f"on cross-attention blocks {cross_blocks} "
                f"(of {len(self.model.transformer_blocks)} DiT blocks, single-pass)\033[0m"
            )
            _PLADIS_LOGGED = True

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        num_steps = num_inference_timesteps
        dt = 1.0 / num_steps

        # Install PLADIS cross-attention processors once; single forward per denoising step.
        install_pladis(cross_blocks)
        try:
            for t in range(num_steps):
                t_cont = t / float(num_steps)
                t_discretized = int(t_cont * self.num_timestep_buckets)

                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                vl_embs = vl_embeds
                sa_embs = torch.cat((state_features, action_features), dim=1)

                # Single forward pass — no incoherent/perturbed second pass (PLADIS is single-pass).
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
                pred = self.action_decoder(model_output, embodiment_id)

                pred_velocity = pred[:, -self.action_horizon:]
                actions = actions + dt * pred_velocity
        finally:
            restore_original(cross_blocks)

        return BatchFeature(data={"action_pred": actions})
