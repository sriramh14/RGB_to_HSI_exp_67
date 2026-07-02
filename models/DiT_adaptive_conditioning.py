from models.HSI_VAE import HSIVAE

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)





class RGBUNetBlock(nn.Module):
    """Two-convolution U-Net block using channel-wise LayerNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm1 = nn.LayerNorm(
            out_channels,
            eps=1e-6,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = nn.LayerNorm(
            out_channels,
            eps=1e-6,
        )
        self.activation = nn.SiLU()

    @staticmethod
    def _apply_layer_norm_nchw(
        feature: torch.Tensor,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        feature = feature.permute(0, 2, 3, 1)
        feature = norm(feature)
        return feature.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self._apply_layer_norm_nchw(x, self.norm1)
        x = self.activation(x)

        x = self.conv2(x)
        x = self._apply_layer_norm_nchw(x, self.norm2)
        x = self.activation(x)
        return x


class RGBSkipFusionBlock(nn.Module):
    """
    U-Net top-down fusion block.

    The lower-resolution feature is resized to the skip resolution, projected,
    concatenated with the gated encoder skip, and refined by a convolutional
    block. This explicitly connects global -> regional -> local information.
    """

    def __init__(
        self,
        lower_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.lower_projection = nn.Conv2d(
            lower_channels,
            out_channels,
            kernel_size=1,
        )
        self.skip_projection = nn.Conv2d(
            skip_channels,
            out_channels,
            kernel_size=1,
        )
        self.fusion = RGBUNetBlock(
            2 * out_channels,
            out_channels,
        )

    def forward(
        self,
        lower_feature: torch.Tensor,
        skip_feature: torch.Tensor,
        skip_weight: torch.Tensor,
    ) -> torch.Tensor:
        lower_feature = F.interpolate(
            lower_feature,
            size=skip_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        lower_feature = self.lower_projection(lower_feature)
        skip_feature = self.skip_projection(skip_feature)

        # skip_weight has shape [B, 1, 1, 1]. It controls when the finer
        # encoder feature is allowed to enter the top-down path.
        skip_feature = skip_weight * skip_feature

        return self.fusion(
            torch.cat(
                [lower_feature, skip_feature],
                dim=1,
            )
        )


class RGBConditionEncoder(nn.Module):
    """
    U-Net-style RGB conditioning adapter for the diffusion/DiT backbone.

    Encoder half:
        RGB -> local feature -> regional feature -> global bottleneck.

    Top-down skip path:
        global bottleneck -> regional skip fusion -> local skip fusion.

    The global bottleneck supplies the persistent AdaLN condition. The final
    local-resolution fused feature supplies the spatial DiT residual. Regional
    and local skip strengths are controlled by the diffusion timestep, so
    coarse/global information dominates noisy timesteps while finer RGB detail
    is introduced later in denoising.

    Returns:
        global_condition: [B, hidden_size]
        spatial_update:   [B, token_grid_size**2, hidden_size]
        field_weights:    [B, 3] in [global, regional, local] order
    """

    def __init__(
        self,
        hidden_size: int,
        token_grid_size: int,
        num_timesteps: int = 1000,
        global_grid_size: int = 4,
        regional_grid_size: int = 8,
        global_end: float = 0.65,
        regional_max: float = 0.35,
        regional_end: float = 0.65,
        local_max: float = 0.25,
        local_start: float = 0.85,
        num_global_queries: int = 4,
        num_attention_heads: int = 4,
        max_update_strength: float = 0.25,
    ):
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads, "
                f"but received hidden_size={hidden_size} and "
                f"num_attention_heads={num_attention_heads}."
            )
        if token_grid_size < 1:
            raise ValueError("token_grid_size must be at least 1.")
        if num_timesteps < 2:
            raise ValueError("num_timesteps must be at least 2.")
        if not 0.0 <= global_end <= 1.0:
            raise ValueError("global_end must lie in [0, 1].")
        if not 0.0 <= regional_max <= 1.0:
            raise ValueError("regional_max must lie in [0, 1].")
        if not 0.0 < regional_end <= 1.0:
            raise ValueError("regional_end must lie in (0, 1].")
        if not 0.0 <= local_max <= 1.0:
            raise ValueError("local_max must lie in [0, 1].")
        if not 0.0 <= local_start < 1.0:
            raise ValueError("local_start must lie in [0, 1).")
        if not 0.0 < max_update_strength <= 1.0:
            raise ValueError("max_update_strength must lie in (0, 1].")

        self.hidden_size = hidden_size
        self.token_grid_size = token_grid_size
        self.num_timesteps = num_timesteps
        self.global_grid_size = min(global_grid_size, token_grid_size)
        self.regional_grid_size = min(regional_grid_size, token_grid_size)

        self.global_end = global_end
        self.regional_max = regional_max
        self.regional_end = regional_end
        self.local_max = local_max
        self.local_start = local_start
        self.max_update_strength = max_update_strength

        base_channels = max(hidden_size // 4, 32)
        regional_channels = 2 * base_channels
        global_channels = hidden_size

        # ------------------------------------------------------------------ #
        # U-Net encoder half: local -> regional -> global.
        # ------------------------------------------------------------------ #
        self.local_encoder = RGBUNetBlock(
            in_channels=3,
            out_channels=base_channels,
        )

        self.local_to_regional = nn.Conv2d(
            base_channels,
            regional_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.regional_encoder = RGBUNetBlock(
            in_channels=regional_channels,
            out_channels=regional_channels,
        )

        self.regional_to_global = nn.Conv2d(
            regional_channels,
            global_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.global_encoder = RGBUNetBlock(
            in_channels=global_channels,
            out_channels=global_channels,
        )

        # ------------------------------------------------------------------ #
        # U-Net top-down path. These are the requested skip connections:
        #     global bottleneck + regional encoder skip
        #     fused regional feature + local encoder skip
        # ------------------------------------------------------------------ #
        self.global_to_regional_fusion = RGBSkipFusionBlock(
            lower_channels=global_channels,
            skip_channels=regional_channels,
            out_channels=regional_channels,
        )
        self.regional_to_local_fusion = RGBSkipFusionBlock(
            lower_channels=regional_channels,
            skip_channels=base_channels,
            out_channels=base_channels,
        )

        # ------------------------------------------------------------------ #
        # Learned global pooling from the U-Net bottleneck.
        # ------------------------------------------------------------------ #
        self.global_queries = nn.Parameter(
            torch.randn(
                1,
                num_global_queries,
                hidden_size,
            ) * 0.02
        )
        self.global_token_norm = nn.LayerNorm(
            hidden_size,
            eps=1e-6,
        )
        self.global_attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
        )
        self.global_query_norm = nn.LayerNorm(
            hidden_size,
            eps=1e-6,
        )
        self.global_condition_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Multi-scale contexts used only by the safety gate. The actual spatial
        # update comes from the final local-resolution skip-fused feature.
        self.global_context_projection = nn.Conv2d(
            global_channels,
            hidden_size,
            kernel_size=1,
        )
        self.regional_context_projection = nn.Conv2d(
            regional_channels,
            hidden_size,
            kernel_size=1,
        )
        self.local_context_projection = nn.Conv2d(
            base_channels,
            hidden_size,
            kernel_size=1,
        )

        self.global_context_norm = nn.LayerNorm(
            hidden_size,
            eps=1e-6,
        )
        self.regional_context_norm = nn.LayerNorm(
            hidden_size,
            eps=1e-6,
        )
        self.local_context_norm = nn.LayerNorm(
            hidden_size,
            eps=1e-6,
        )

        self.spatial_gate = nn.Sequential(
            nn.LayerNorm(3 * hidden_size, eps=1e-6),
            nn.Linear(3 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        # This name is retained because DiT.initialize_weights() explicitly
        # restores its zero initialization after applying generic Linear init.
        self.spatial_output_projection = nn.Linear(
            hidden_size,
            hidden_size,
        )
        self.output_norm = nn.LayerNorm(
            hidden_size,
            eps=1e-6,
        )

        self.update_strength_logit = nn.Parameter(
            torch.tensor(-2.0)
        )

        self._initialize_safe_fusion()

    def _initialize_safe_fusion(self) -> None:
        # Start with a zero RGB spatial residual. This preserves the original
        # DiT behaviour at initialization while allowing gradients to train the
        # adapter through this projection.
        nn.init.zeros_(self.spatial_output_projection.weight)
        nn.init.zeros_(self.spatial_output_projection.bias)

    @staticmethod
    def _smoothstep(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    def _field_weights(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        progress = 1.0 - (
            t.float() / float(self.num_timesteps - 1)
        )
        progress = progress.clamp(0.0, 1.0)

        global_weight = (
            self.global_end
            + (1.0 - self.global_end)
            * (1.0 - progress).pow(2)
        )

        regional_progress = progress / self.regional_end
        regional_weight = (
            self.regional_max
            * self._smoothstep(regional_progress)
        )

        local_progress = (
            progress - self.local_start
        ) / max(1.0 - self.local_start, 1e-6)
        local_weight = (
            self.local_max
            * self._smoothstep(local_progress)
        )

        return torch.stack(
            [global_weight, regional_weight, local_weight],
            dim=1,
        )

    @staticmethod
    def _to_tokens(
        feature: torch.Tensor,
    ) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2)

    def _context_to_tokens(
        self,
        feature: torch.Tensor,
        projection: nn.Conv2d,
        norm: nn.LayerNorm,
        source_grid_size: int,
    ) -> torch.Tensor:
        # First preserve the intended receptive-field scale, then align all
        # branches to the fixed DiT token grid.
        feature = F.adaptive_avg_pool2d(
            feature,
            output_size=(source_grid_size, source_grid_size),
        )
        if source_grid_size != self.token_grid_size:
            feature = F.interpolate(
                feature,
                size=(self.token_grid_size, self.token_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        feature = projection(feature)
        return norm(self._to_tokens(feature))

    def _build_global_condition(
        self,
        global_feature: torch.Tensor,
    ) -> torch.Tensor:
        global_tokens = self.global_token_norm(
            self._to_tokens(global_feature)
        )
        queries = self.global_queries.expand(
            global_tokens.shape[0],
            -1,
            -1,
        )
        pooled_queries, _ = self.global_attention_pool(
            query=queries,
            key=global_tokens,
            value=global_tokens,
            need_weights=False,
        )
        pooled_queries = self.global_query_norm(
            pooled_queries + queries
        )
        global_condition = pooled_queries.mean(dim=1)
        return self.global_condition_projection(global_condition)

    def forward(
        self,
        rgb: torch.Tensor,
        t: torch.Tensor,
    ):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(
                "rgb must have shape [B, 3, H, W], "
                f"but received {tuple(rgb.shape)}."
            )
        if t.ndim != 1 or t.shape[0] != rgb.shape[0]:
            raise ValueError(
                "t must have shape [B] and match the RGB batch size. "
                f"Received t={tuple(t.shape)} and rgb={tuple(rgb.shape)}."
            )

        # ----------------------- encoder half ---------------------------- #
        local_skip = self.local_encoder(rgb)

        regional_input = self.local_to_regional(local_skip)
        regional_skip = self.regional_encoder(regional_input)

        global_input = self.regional_to_global(regional_skip)
        global_bottleneck = self.global_encoder(global_input)

        field_weights = self._field_weights(t)
        w_global = field_weights[:, 0].view(-1, 1, 1, 1)
        w_regional = field_weights[:, 1].view(-1, 1, 1, 1)
        w_local = field_weights[:, 2].view(-1, 1, 1, 1)

        # Persistent global vector is kept independent of the spatial skip
        # schedule so RGB scene/material information is always available.
        global_condition = self._build_global_condition(
            global_bottleneck
        )

        # ---------------------- top-down skip path ----------------------- #
        # Global controls the top-down base; regional and local information
        # enters through the corresponding gated encoder skips.
        global_top_down = w_global * global_bottleneck

        regional_fused = self.global_to_regional_fusion(
            lower_feature=global_top_down,
            skip_feature=regional_skip,
            skip_weight=w_regional,
        )

        local_fused = self.regional_to_local_fusion(
            lower_feature=regional_fused,
            skip_feature=local_skip,
            skip_weight=w_local,
        )

        # Build aligned contexts for the safety gate. regional_fused already
        # includes global context; local_fused includes all three levels.
        global_context = self._context_to_tokens(
            feature=global_top_down,
            projection=self.global_context_projection,
            norm=self.global_context_norm,
            source_grid_size=self.global_grid_size,
        )
        regional_context = self._context_to_tokens(
            feature=regional_fused,
            projection=self.regional_context_projection,
            norm=self.regional_context_norm,
            source_grid_size=self.regional_grid_size,
        )
        local_context = self._context_to_tokens(
            feature=local_fused,
            projection=self.local_context_projection,
            norm=self.local_context_norm,
            source_grid_size=self.token_grid_size,
        )

        branch_context = torch.cat(
            [global_context, regional_context, local_context],
            dim=-1,
        )
        safety_gate = self.spatial_gate(branch_context)

        # local_context is the final skip-fused U-Net feature and therefore
        # already contains global, regional, and local information.
        projected_update = self.spatial_output_projection(
            local_context
        )
        projected_update = self.output_norm(projected_update)

        update_strength = (
            self.max_update_strength
            * torch.sigmoid(self.update_strength_logit)
        )
        spatial_update = (
            update_strength
            * safety_gate
            * projected_update
        )

        return global_condition, spatial_update, field_weights


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=4,
        in_channels=16,
        hidden_size=256,
        depth=12,
        num_heads=4,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
      num_timesteps = 1000,
    ):
        super().__init__()
        if input_size % patch_size != 0:
          raise ValueError(
              "input_size must be divisible by patch_size. "
              f"Received input_size={input_size}, "
              f"patch_size={patch_size}."
          )
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        #self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

      
        #self.rgb_encoder = RGBConditionEncoder(hidden_size = hidden_size)

        token_grid_size = input_size // patch_size

        self.rgb_encoder = RGBConditionEncoder(
            hidden_size=hidden_size,
            token_grid_size=token_grid_size,
            num_timesteps=num_timesteps,
        
            # Coarse and regional spatial grids.
            global_grid_size=min(
                4,
                token_grid_size,
            ),
            regional_grid_size=min(
                8,
                token_grid_size,
            ),
        
            # Global remains dominant.
            global_end=0.65,
        
            # Regional stays second.
            regional_max=0.35,
            regional_end=0.65,
        
            # Local activates only during the final 15%.
            local_max=0.25,
            local_start=0.85,
        
            # Used by the current learned global pooling section.
            num_global_queries=4,
            num_attention_heads=num_heads,
        
            # Complete spatial RGB update is bounded by this value.
            max_update_strength=0.25,
        )

        self.global_condition_fuser = nn.Sequential(
            nn.Linear(
                2 * hidden_size,
                hidden_size,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
        )

      
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                  
        self.apply(_basic_init)
        # Fixed positional embedding.
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(
                self.x_embedder.num_patches ** 0.5
            ),
        )
    
        self.pos_embed.data.copy_(
            torch.from_numpy(
                pos_embed
            ).float().unsqueeze(0)
        )
    
        # Patch embedding initialization.
        weight = self.x_embedder.proj.weight.data
    
        nn.init.xavier_uniform_(
            weight.view(
                [
                    weight.shape[0],
                    -1,
                ]
            )
        )
    
        nn.init.constant_(
            self.x_embedder.proj.bias,
            0,
        )
    
        # Timestep embedding initialization.
        nn.init.normal_(
            self.t_embedder.mlp[0].weight,
            std=0.02,
        )
    
        nn.init.normal_(
            self.t_embedder.mlp[2].weight,
            std=0.02,
        )
    
        # AdaLN-Zero initialization.
        for block in self.blocks:
            nn.init.constant_(
                block.adaLN_modulation[-1].weight,
                0,
            )
    
            nn.init.constant_(
                block.adaLN_modulation[-1].bias,
                0,
            )
    
        # Final DiT output initialization.
        nn.init.constant_(
            self.final_layer
            .adaLN_modulation[-1]
            .weight,
            0,
        )
    
        nn.init.constant_(
            self.final_layer
            .adaLN_modulation[-1]
            .bias,
            0,
        )
    
        nn.init.constant_(
            self.final_layer.linear.weight,
            0,
        )
    
        nn.init.constant_(
            self.final_layer.linear.bias,
            0,
        )
    
        # Important: self.apply(_basic_init) overwrote the encoder's own
        # zero initialization, so restore it here.
        nn.init.zeros_(
            self.rgb_encoder
            .spatial_output_projection
            .weight
        )
    
        nn.init.zeros_(
            self.rgb_encoder
            .spatial_output_projection
            .bias
        )
    
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
    
    
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
    
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
    
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        rgb: torch.Tensor,
    ):
        """
        Args:
            x:
                [B, latent_channels, H_latent, W_latent]
                Noisy HSI latent.
    
            t:
                [B]
                Integer diffusion timesteps.
    
            rgb:
                [B, 3, H_rgb, W_rgb]
                Paired RGB condition.
        """
    
        if t.ndim != 1:
            raise ValueError(
                "t must have shape [B], "
                f"but received {tuple(t.shape)}."
            )
    
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(
                "rgb must have shape [B, 3, H, W], "
                f"but received {tuple(rgb.shape)}."
            )
    
        if x.shape[0] != rgb.shape[0]:
            raise ValueError(
                "Latent and RGB batch sizes must match. "
                f"Latent batch={x.shape[0]}, "
                f"RGB batch={rgb.shape[0]}."
            )
    
        # ------------------------------------------------------------- #
        # 1. Patchify the noisy HSI latent.
        # ------------------------------------------------------------- #
        x = self.x_embedder(x)
    
        if x.shape[1] != self.pos_embed.shape[1]:
            raise RuntimeError(
                "The noisy latent produced a different token count "
                "from the fixed positional embedding. "
                f"Latent tokens={x.shape[1]}, "
                f"positional tokens={self.pos_embed.shape[1]}."
            )
    
        x = x + self.pos_embed
    
        # ------------------------------------------------------------- #
        # 2. Embed diffusion timestep.
        # ------------------------------------------------------------- #
        t_embedding = self.t_embedder(t)
    
        # ------------------------------------------------------------- #
        # 3. Obtain RGB conditions.
        #
        # rgb_global_condition:
        #     [B, D]
        #
        # rgb_spatial_update:
        #     [B, N, D]
        #
        # field_weights:
        #     [B, 3]
        # ------------------------------------------------------------- #
        (
            rgb_global_condition,
            rgb_spatial_update,
            field_weights,
        ) = self.rgb_encoder(
            rgb,
            t,
        )
    
        if rgb_spatial_update.shape != x.shape:
            raise RuntimeError(
                "RGB spatial update must match the DiT token shape. "
                f"RGB update={tuple(rgb_spatial_update.shape)}, "
                f"DiT tokens={tuple(x.shape)}."
            )
    
        # ------------------------------------------------------------- #
        # 4. Add the bounded RGB spatial residual once.
        # ------------------------------------------------------------- #
        x = x + rgb_spatial_update
    
        # ------------------------------------------------------------- #
        # 5. Fuse time with persistent global RGB context.
        # ------------------------------------------------------------- #
        c = self.global_condition_fuser(
            torch.cat(
                [
                    t_embedding,
                    rgb_global_condition,
                ],
                dim=-1,
            )
        )
    
        # ------------------------------------------------------------- #
        # 6. Standard DiT processing.
        # ------------------------------------------------------------- #
        for block in self.blocks:
            x = block(
                x,
                c,
            )
    
        x = self.final_layer(
            x,
            c,
        )
    
        x = self.unpatchify(x)
    
        return x




#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class RGB_to_HSI_w_diffusion(nn.Module):
    def __init__(
        self,
        hsi_channels = 31,
        base_channels = 64,
        latent_channels = 16,
        num_res_blocks = 2,
        hidden_size = 256,
        depth = 12,
        num_heads = 4,
        mlp_ratio = 4.0,
        class_dropout_prob = 0.1,
        learn_sigma = False,
        patch_size = 4,
        input_size = 64,
        T = 1000, 
    ):
        super().__init__()  # ← this line is missing in your file

        self.T = T
        self.vae = HSIVAE(
            hsi_channels,
            base_channels,
            latent_channels,
            num_res_blocks
        )
        self.dit = DiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=latent_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            learn_sigma=learn_sigma,
            num_timesteps = T
        )

        
        self.vae.encoder.requires_grad_(False)
        self.vae.decoder.requires_grad_(False)

        self.vae.encoder.eval()
        self.vae.decoder.eval()

        
        # Registered as buffers so they move to the correct device
        # automatically and are saved in the checkpoint.
        self.register_buffer("latent_mean", torch.tensor(0.0))
        self.register_buffer("latent_std",  torch.tensor(1.0))

    def set_latent_stats(self, mean: float, std: float) -> None:
        self.latent_mean.fill_(mean)
        self.latent_std.fill_(std)

    def normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.latent_mean) / self.latent_std.clamp(min=1e-6)

    def denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.latent_std + self.latent_mean
        
    def forward(self, hsi, rgb, t, noise_scheduler):
        """
        Latent diffusion training forward pass.
            Args:
                hsi:             (B, C, H, W)  Ground-truth hyperspectral image.
                rgb:             (B, 3, H, W)  Paired RGB image used as condition.
                t:               (B,)          Diffusion timesteps sampled from U[0, T].
                noise_scheduler:               Scheduler with .add_noise(z0, noise, t) method.
        
            Returns:
                loss:        Scalar — MSE noise prediction loss in latent space.
                hsi_recon:   (B, C, H, W) — decoded HSI reconstruction in pixel space.
                pred_noise:  (B, latent_C, H', W') — raw noise prediction from DiT.
        """
        # ── 1. Encode HSI → clean latent z0 (frozen VAE, no grad needed) ────────
        with torch.no_grad():
            z0, _, _ = self.vae.encode(hsi, sample=False)
            z0 = self.normalize_latent(z0)   # → roughly N(0, 1)
        
        # ── 2. Sample noise and corrupt z0 → zₜ ─────────────────────────────────
        noise = torch.randn_like(z0)
        #noise_rgb = torch.randn_like(rgb)
        #t_idx = (t * (self.T - 1)).long().clamp(0, self.T - 1)   # (B,) int64
        t_idx = t
        
        zt = noise_scheduler.add_noise(z0, noise, t_idx)
        #rgb_t = noise_scheduler.add_noise(rgb,noise_rgb,t_idx)
        # ── 3. DiT predicts noise from zₜ, conditioned on RGB ───────────────────
        pred = self.dit(zt, t_idx, rgb)
        
        if self.dit.learn_sigma:
            # DiT outputs [pred_noise | pred_sigma] — only noise half is trained
            pred_noise = pred[:, :self.dit.in_channels]
        else:
            pred_noise = pred
        
        # ── 4. Noise prediction loss in latent space ─────────────────────────────
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        # ── 5. Reconstruct clean latent from predicted noise, then decode → HSI ──
        # Invert the noise schedule: estimate z0 from zₜ and predicted noise
        # This mirrors what a DDPM sampler does at inference in a single step.
        alpha_bar_t = noise_scheduler.alphas_cumprod[t_idx].to(zt.device)
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1).clamp(min=1e-5)  # prevent /0
        
        #To avoid nan error
        z0_pred   = (zt - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
        z0_pred   = torch.clamp(z0_pred, -10.0, 10.0)

        #Back to latent space normalisation
        z0_pred   = self.denormalize_latent(z0_pred)

        #normalising hsi to 0 to 1
        with torch.no_grad():
            hsi_recon = self.vae.decode(z0_pred)
            hsi_recon = torch.clamp(hsi_recon, 0.0, 1.0)
        
        return loss, hsi_recon, pred_noise, noise

    def train(self, mode: bool = True):
        super().train(mode)
    
        # The DiT follows the requested mode, but the frozen VAE
        # always remains in evaluation mode.
        self.vae.encoder.eval()
        self.vae.decoder.eval()
    
        return self
                
                
        
