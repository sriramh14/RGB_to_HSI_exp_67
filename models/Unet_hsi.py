"""Latent diffusion U-Net for HSI VAE latents.

The model predicts the noise added to a latent tensor. It is intended to be
trained on latents produced by a frozen HSIVAE encoder.

Expected interface
------------------
    predicted_noise = model(noisy_latent, timesteps)

Shapes
------
    noisy_latent:    [B, latent_channels, H, W]
    timesteps:       scalar or [B]
    predicted_noise: [B, latent_channels, H, W]

All normalization layers are torch.nn.LayerNorm applied in channels-last
format, matching the normalization style used by the provided HSIVAE.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm(nn.Module):
    """Apply ``nn.LayerNorm`` over channels of a BCHW tensor."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal diffusion-timestep embedding."""

    def __init__(self, embedding_dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2.")
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.ndim > 1:
            timesteps = timesteps.reshape(timesteps.shape[0])

        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        denominator = max(half_dim - 1, 1)

        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            / denominator
        )

        angles = timesteps[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)

        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class TimeEmbeddingMLP(nn.Module):
    """Project a sinusoidal timestep embedding into the U-Net time space."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class TimeConditionedResBlock(nn.Module):
    """Residual block with LayerNorm and scale-shift timestep conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = ChannelLayerNorm(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels * 2),
        )

        self.norm2 = ChannelLayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

        # The block begins as an approximate identity mapping. This usually
        # stabilizes diffusion training, especially for deeper U-Nets.
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        scale_shift = self.time_projection(time_embedding)
        scale, shift = torch.chunk(scale_shift, chunks=2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return residual + h


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over latent spatial positions."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = ChannelLayerNorm(channels)
        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1,
        )
        self.projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

        # Attention is introduced gradually from an identity mapping.
        nn.init.zeros_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch, channels, height, width = x.shape

        h = self.norm(x)
        query, key, value = torch.chunk(self.qkv(h), chunks=3, dim=1)

        def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.reshape(
                batch,
                self.num_heads,
                self.head_dim,
                height * width,
            )
            return tensor.permute(0, 1, 3, 2).contiguous()

        query = reshape_heads(query)
        key = reshape_heads(key)
        value = reshape_heads(value)

        h = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )

        h = h.permute(0, 1, 3, 2).contiguous()
        h = h.reshape(batch, channels, height, width)
        h = self.projection(h)

        return residual + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if target_size is None:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        else:
            x = F.interpolate(x, size=target_size, mode="nearest")
        return self.conv(x)


class DownStage(nn.Module):
    def __init__(
        self,
        channels: int,
        time_embedding_dim: int,
        num_res_blocks: int,
        add_attention: bool,
        num_heads: int,
        dropout: float,
        next_channels: int | None,
    ) -> None:
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                TimeConditionedResBlock(
                    in_channels=channels,
                    out_channels=channels,
                    time_embedding_dim=time_embedding_dim,
                    dropout=dropout,
                )
                for _ in range(num_res_blocks)
            ]
        )

        self.attention_blocks = nn.ModuleList(
            [
                SpatialSelfAttention(channels, num_heads=num_heads)
                if add_attention
                else nn.Identity()
                for _ in range(num_res_blocks)
            ]
        )

        self.downsample = (
            Downsample(channels, next_channels)
            if next_channels is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for res_block, attention in zip(
            self.res_blocks,
            self.attention_blocks,
        ):
            x = res_block(x, time_embedding)
            x = attention(x)

        skip = x
        if self.downsample is not None:
            x = self.downsample(x)

        return x, skip


class UpStage(nn.Module):
    def __init__(
        self,
        current_channels: int,
        skip_channels: int,
        output_channels: int,
        time_embedding_dim: int,
        num_res_blocks: int,
        add_attention: bool,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = [
            TimeConditionedResBlock(
                in_channels=current_channels + skip_channels,
                out_channels=output_channels,
                time_embedding_dim=time_embedding_dim,
                dropout=dropout,
            )
        ]

        for _ in range(num_res_blocks - 1):
            blocks.append(
                TimeConditionedResBlock(
                    in_channels=output_channels,
                    out_channels=output_channels,
                    time_embedding_dim=time_embedding_dim,
                    dropout=dropout,
                )
            )

        self.res_blocks = nn.ModuleList(blocks)
        self.attention_blocks = nn.ModuleList(
            [
                SpatialSelfAttention(
                    output_channels,
                    num_heads=num_heads,
                )
                if add_attention
                else nn.Identity()
                for _ in range(num_res_blocks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="nearest",
            )

        x = torch.cat([x, skip], dim=1)

        for res_block, attention in zip(
            self.res_blocks,
            self.attention_blocks,
        ):
            x = res_block(x, time_embedding)
            x = attention(x)

        return x


class HSILatentDiffusionUNet(nn.Module):
    """Unconditional latent diffusion prior for HSI VAE latents.

    The model predicts epsilon by default:

        epsilon_hat = model(z_t, timesteps)

    Parameters
    ----------
    latent_channels:
        Number of channels produced by the HSI VAE. The provided VAE uses 16.
    base_channels:
        Width of the first U-Net stage.
    channel_multipliers:
        Per-stage width multipliers. Four stages create three downsamplings.
    num_res_blocks:
        Number of residual blocks in each down/up stage.
    attention_levels:
        Stage indices where spatial self-attention is enabled.
    num_heads:
        Number of attention heads. Every attended channel width must be
        divisible by this value.
    dropout:
        Dropout inside residual blocks.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_levels: Sequence[int] = (2, 3),
        num_heads: int = 8,
        dropout: float = 0.0,
        time_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()

        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive.")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if len(channel_multipliers) < 2:
            raise ValueError("At least two channel multipliers are required.")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be at least 1.")

        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.channel_multipliers = tuple(channel_multipliers)
        self.num_res_blocks = num_res_blocks

        stage_channels = [
            base_channels * multiplier
            for multiplier in self.channel_multipliers
        ]
        attention_level_set = set(attention_levels)

        for level in attention_level_set:
            if level < 0 or level >= len(stage_channels):
                raise ValueError(
                    f"Invalid attention level {level}; valid levels are "
                    f"0 to {len(stage_channels) - 1}."
                )
            if stage_channels[level] % num_heads != 0:
                raise ValueError(
                    f"Stage {level} has {stage_channels[level]} channels, "
                    f"which is not divisible by num_heads={num_heads}."
                )

        if time_embedding_dim is None:
            time_embedding_dim = base_channels * 4

        self.time_position = SinusoidalTimeEmbedding(base_channels)
        self.time_mlp = TimeEmbeddingMLP(
            input_dim=base_channels,
            output_dim=time_embedding_dim,
        )

        self.input_conv = nn.Conv2d(
            latent_channels,
            stage_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.down_stages = nn.ModuleList()
        for level, channels in enumerate(stage_channels):
            next_channels = (
                stage_channels[level + 1]
                if level < len(stage_channels) - 1
                else None
            )
            self.down_stages.append(
                DownStage(
                    channels=channels,
                    time_embedding_dim=time_embedding_dim,
                    num_res_blocks=num_res_blocks,
                    add_attention=level in attention_level_set,
                    num_heads=num_heads,
                    dropout=dropout,
                    next_channels=next_channels,
                )
            )

        middle_channels = stage_channels[-1]
        self.middle_block1 = TimeConditionedResBlock(
            in_channels=middle_channels,
            out_channels=middle_channels,
            time_embedding_dim=time_embedding_dim,
            dropout=dropout,
        )
        self.middle_attention = SpatialSelfAttention(
            middle_channels,
            num_heads=num_heads,
        )
        self.middle_block2 = TimeConditionedResBlock(
            in_channels=middle_channels,
            out_channels=middle_channels,
            time_embedding_dim=time_embedding_dim,
            dropout=dropout,
        )

        self.up_stages = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        current_channels = stage_channels[-1]
        for level in reversed(range(len(stage_channels))):
            skip_channels = stage_channels[level]
            output_channels = stage_channels[level]

            self.up_stages.append(
                UpStage(
                    current_channels=current_channels,
                    skip_channels=skip_channels,
                    output_channels=output_channels,
                    time_embedding_dim=time_embedding_dim,
                    num_res_blocks=num_res_blocks,
                    add_attention=level in attention_level_set,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

            if level > 0:
                next_output_channels = stage_channels[level - 1]
                self.upsamplers.append(
                    Upsample(
                        in_channels=output_channels,
                        out_channels=next_output_channels,
                    )
                )
                current_channels = next_output_channels
            else:
                current_channels = output_channels

        self.output_norm = ChannelLayerNorm(stage_channels[0])
        self.output_conv = nn.Conv2d(
            stage_channels[0],
            latent_channels,
            kernel_size=3,
            padding=1,
        )

        # Begin with a zero noise prediction rather than a random one.
        nn.init.zeros_(self.output_conv.weight)
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_latent.ndim != 4:
            raise ValueError(
                "noisy_latent must have shape [B, C, H, W], "
                f"but received {tuple(noisy_latent.shape)}."
            )
        if noisy_latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, "
                f"but received {noisy_latent.shape[1]}."
            )

        batch_size = noisy_latent.shape[0]
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                timesteps,
                device=noisy_latent.device,
            )
        timesteps = timesteps.to(noisy_latent.device)

        if timesteps.ndim == 0:
            timesteps = timesteps.repeat(batch_size)
        elif timesteps.numel() == 1 and batch_size > 1:
            timesteps = timesteps.reshape(1).repeat(batch_size)
        else:
            timesteps = timesteps.reshape(-1)

        if timesteps.shape[0] != batch_size:
            raise ValueError(
                "timesteps must be a scalar or contain one value per batch "
                f"item. Received {timesteps.shape[0]} values for batch "
                f"size {batch_size}."
            )

        time_embedding = self.time_position(timesteps)
        time_embedding = self.time_mlp(time_embedding)

        x = self.input_conv(noisy_latent)

        skips: list[torch.Tensor] = []
        for down_stage in self.down_stages:
            x, skip = down_stage(x, time_embedding)
            skips.append(skip)

        x = self.middle_block1(x, time_embedding)
        x = self.middle_attention(x)
        x = self.middle_block2(x, time_embedding)

        upsample_index = 0
        for stage_index, up_stage in enumerate(self.up_stages):
            skip = skips.pop()
            x = up_stage(x, skip, time_embedding)

            is_last_up_stage = stage_index == len(self.up_stages) - 1
            if not is_last_up_stage:
                next_skip = skips[-1]
                x = self.upsamplers[upsample_index](
                    x,
                    target_size=next_skip.shape[-2:],
                )
                upsample_index += 1

        x = self.output_norm(x)
        x = F.silu(x)
        return self.output_conv(x)


'''if __name__ == "__main__":
    # Lightweight shape check. Reduce base_channels for a quick CPU test.
    model = HSILatentDiffusionUNet(
        latent_channels=16,
        base_channels=32,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=2,
        attention_levels=(2, 3),
        num_heads=4,
    )

    latent = torch.randn(2, 16, 32, 32)
    timesteps = torch.randint(0, 1000, (2,))
    predicted_noise = model(latent, timesteps)

    print("Input shape: ", tuple(latent.shape))
    print("Output shape:", tuple(predicted_noise.shape))
    print(
        "Trainable parameters:",
        sum(parameter.numel() for parameter in model.parameters()),
    )
'''
