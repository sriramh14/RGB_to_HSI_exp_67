
from __future__ import annotations

import contextlib
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrScalar = Union[torch.Tensor, float, int]


def _group_count(channels: int, maximum: int = 32) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def _modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Adaptive LayerNorm modulation."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _build_2d_sincos_position_embedding(
    embedding_dim: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create a dynamic 2D sinusoidal positional embedding.

    Returns:
        Tensor of shape [1, height * width, embedding_dim].
    """
    if embedding_dim % 4 != 0:
        raise ValueError(
            "The hidden dimension must be divisible by 4 for the "
            "2D sinusoidal positional embedding."
        )

    quarter_dim = embedding_dim // 4

    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    denominator = max(quarter_dim - 1, 1)
    frequencies = torch.arange(
        quarter_dim,
        device=device,
        dtype=torch.float32,
    )
    frequencies = 1.0 / (
        10000.0 ** (frequencies / denominator)
    )

    x_arguments = grid_x.reshape(-1, 1) * frequencies.reshape(1, -1)
    y_arguments = grid_y.reshape(-1, 1) * frequencies.reshape(1, -1)

    position_embedding = torch.cat(
        [
            torch.sin(x_arguments),
            torch.cos(x_arguments),
            torch.sin(y_arguments),
            torch.cos(y_arguments),
        ],
        dim=-1,
    )

    return position_embedding.unsqueeze(0).to(dtype=dtype)


class RMSNorm(nn.Module):
    """RMSNorm used for query/key normalization."""

    def __init__(
        self,
        dimension: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return normalized * self.weight


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a small MLP."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()

        self.frequency_embedding_size = frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal_embedding(
        timestep: torch.Tensor,
        dimension: int,
        max_period: float = 10000.0,
    ) -> torch.Tensor:
        half = dimension // 2

        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(
                half,
                device=timestep.device,
                dtype=torch.float32,
            )
            / max(half, 1)
        )

        # Timesteps are expected in [0, 1].
        arguments = (
            timestep.float().reshape(-1, 1)
            * 1000.0
            * frequencies.reshape(1, -1)
        )

        embedding = torch.cat(
            [torch.cos(arguments), torch.sin(arguments)],
            dim=-1,
        )

        if dimension % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        frequency_embedding = self.sinusoidal_embedding(
            timestep=timestep,
            dimension=self.frequency_embedding_size,
        )
        return self.mlp(frequency_embedding)


class RGBConditionEncoder(nn.Module):
    """
    Simple RGB encoder with a spatial downsampling factor of 4.

    This matches the supplied HSI VAE, whose encoder contains two
    stride-2 downsampling layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        out_channels: int = 256,
    ):
        super().__init__()

        middle_channels = base_channels * 2

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                _group_count(base_channels),
                base_channels,
            ),
            nn.SiLU(),

            nn.Conv2d(
                base_channels,
                middle_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                _group_count(middle_channels),
                middle_channels,
            ),
            nn.SiLU(),

            nn.Conv2d(
                middle_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                _group_count(out_channels),
                out_channels,
            ),
            nn.SiLU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.net(rgb)


class PatchEmbed(nn.Module):
    """Convert a feature map into a sequence of non-overlapping patches."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
    ):
        super().__init__()

        self.patch_size = patch_size

        self.projection = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        x = self.projection(x)

        grid_height = x.shape[-2]
        grid_width = x.shape[-1]

        tokens = x.flatten(2).transpose(1, 2)
        return tokens, grid_height, grid_width


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        inner_size = int(hidden_size * mlp_ratio)

        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointAttention(nn.Module):
    """
    MM-DiT joint attention.

    HSI and RGB use separate QKV/output projections. Their Q, K and V
    tensors are concatenated along the token dimension for one shared
    attention operation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qk_norm: bool = True,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads."
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_dropout = attention_dropout

        self.hsi_qkv = nn.Linear(
            hidden_size,
            hidden_size * 3,
        )
        self.rgb_qkv = nn.Linear(
            hidden_size,
            hidden_size * 3,
        )

        self.hsi_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(projection_dropout),
        )
        self.rgb_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(projection_dropout),
        )

        if qk_norm:
            self.hsi_q_norm = RMSNorm(self.head_dim)
            self.hsi_k_norm = RMSNorm(self.head_dim)
            self.rgb_q_norm = RMSNorm(self.head_dim)
            self.rgb_k_norm = RMSNorm(self.head_dim)
        else:
            self.hsi_q_norm = nn.Identity()
            self.hsi_k_norm = nn.Identity()
            self.rgb_q_norm = nn.Identity()
            self.rgb_k_norm = nn.Identity()

    def _split_qkv(
        self,
        qkv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, token_count, _ = qkv.shape

        qkv = qkv.reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(dim=0)
        return query, key, value

    def forward(
        self,
        hsi_tokens: torch.Tensor,
        rgb_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hsi_token_count = hsi_tokens.shape[1]

        hsi_query, hsi_key, hsi_value = self._split_qkv(
            self.hsi_qkv(hsi_tokens)
        )
        rgb_query, rgb_key, rgb_value = self._split_qkv(
            self.rgb_qkv(rgb_tokens)
        )

        hsi_query = self.hsi_q_norm(hsi_query)
        hsi_key = self.hsi_k_norm(hsi_key)
        rgb_query = self.rgb_q_norm(rgb_query)
        rgb_key = self.rgb_k_norm(rgb_key)

        query = torch.cat(
            [hsi_query, rgb_query],
            dim=2,
        )
        key = torch.cat(
            [hsi_key, rgb_key],
            dim=2,
        )
        value = torch.cat(
            [hsi_value, rgb_value],
            dim=2,
        )

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=(
                self.attention_dropout
                if self.training
                else 0.0
            ),
        )

        hsi_attended = attended[:, :, :hsi_token_count]
        rgb_attended = attended[:, :, hsi_token_count:]

        hsi_attended = (
            hsi_attended.transpose(1, 2)
            .contiguous()
            .reshape(
                hsi_tokens.shape[0],
                hsi_tokens.shape[1],
                self.hidden_size,
            )
        )

        rgb_attended = (
            rgb_attended.transpose(1, 2)
            .contiguous()
            .reshape(
                rgb_tokens.shape[0],
                rgb_tokens.shape[1],
                self.hidden_size,
            )
        )

        return (
            self.hsi_output(hsi_attended),
            self.rgb_output(rgb_attended),
        )


class MMDiTBlock(nn.Module):
    """One two-stream MM-DiT block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        qk_norm: bool = True,
    ):
        super().__init__()

        self.hsi_norm1 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.rgb_norm1 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.attention = JointAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
            qk_norm=qk_norm,
        )

        self.hsi_norm2 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.rgb_norm2 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.hsi_mlp = FeedForward(
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.rgb_mlp = FeedForward(
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.hsi_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )
        self.rgb_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )

        # AdaLN-Zero initialization makes each block initially close
        # to an identity mapping.
        nn.init.zeros_(self.hsi_modulation[-1].weight)
        nn.init.zeros_(self.hsi_modulation[-1].bias)
        nn.init.zeros_(self.rgb_modulation[-1].weight)
        nn.init.zeros_(self.rgb_modulation[-1].bias)

    def forward(
        self,
        hsi_tokens: torch.Tensor,
        rgb_tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            hsi_shift_attention,
            hsi_scale_attention,
            hsi_gate_attention,
            hsi_shift_mlp,
            hsi_scale_mlp,
            hsi_gate_mlp,
        ) = self.hsi_modulation(condition).chunk(6, dim=-1)

        (
            rgb_shift_attention,
            rgb_scale_attention,
            rgb_gate_attention,
            rgb_shift_mlp,
            rgb_scale_mlp,
            rgb_gate_mlp,
        ) = self.rgb_modulation(condition).chunk(6, dim=-1)

        normalized_hsi = _modulate(
            self.hsi_norm1(hsi_tokens),
            hsi_shift_attention,
            hsi_scale_attention,
        )
        normalized_rgb = _modulate(
            self.rgb_norm1(rgb_tokens),
            rgb_shift_attention,
            rgb_scale_attention,
        )

        hsi_attention, rgb_attention = self.attention(
            normalized_hsi,
            normalized_rgb,
        )

        hsi_tokens = (
            hsi_tokens
            + hsi_gate_attention.unsqueeze(1)
            * hsi_attention
        )
        rgb_tokens = (
            rgb_tokens
            + rgb_gate_attention.unsqueeze(1)
            * rgb_attention
        )

        hsi_mlp_input = _modulate(
            self.hsi_norm2(hsi_tokens),
            hsi_shift_mlp,
            hsi_scale_mlp,
        )
        rgb_mlp_input = _modulate(
            self.rgb_norm2(rgb_tokens),
            rgb_shift_mlp,
            rgb_scale_mlp,
        )

        hsi_tokens = (
            hsi_tokens
            + hsi_gate_mlp.unsqueeze(1)
            * self.hsi_mlp(hsi_mlp_input)
        )
        rgb_tokens = (
            rgb_tokens
            + rgb_gate_mlp.unsqueeze(1)
            * self.rgb_mlp(rgb_mlp_input)
        )

        return hsi_tokens, rgb_tokens


class FinalLayer(nn.Module):
    """Map HSI tokens back to latent-space patches."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        latent_channels: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )

        self.output = nn.Linear(
            hidden_size,
            patch_size * patch_size * latent_channels,
        )

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        hsi_tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        hsi_tokens = _modulate(
            self.norm(hsi_tokens),
            shift,
            scale,
        )
        return self.output(hsi_tokens)


class RGBConditionedHSIMMDiT(nn.Module):
    """
    RGB-conditioned MM-DiT for rectified flow in HSI-VAE latent space.

    The model predicts a latent velocity with the same shape as the
    noisy HSI latent.

    Expected HSI VAE latent shape:
        [B, latent_channels, H / 4, W / 4]

    The supplied VAE can be passed directly to this class. The class
    does not depend on a particular import path for HSIVAE.

    Main forward:
        velocity = model(
            noisy_hsi_latent=zt,
            timestep=t,
            rgb=rgb,
        )
    """

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        patch_size: int = 2,
        rgb_base_channels: int = 64,
        rgb_feature_channels: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        qk_norm: bool = True,
        vae: Optional[nn.Module] = None,
        freeze_vae: bool = True,
        latent_mean: TensorOrScalar = 0.0,
        latent_std: TensorOrScalar = 1.0,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads."
            )

        if hidden_size % 4 != 0:
            raise ValueError(
                "hidden_size must be divisible by 4."
            )

        self.latent_channels = latent_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.vae = vae
        self.freeze_vae = freeze_vae

        if self.vae is not None and self.freeze_vae:
            for parameter in self.vae.parameters():
                parameter.requires_grad = False
            self.vae.eval()

        formatted_mean = self._format_latent_stat(
            latent_mean,
            latent_channels,
            "latent_mean",
        )
        formatted_std = self._format_latent_stat(
            latent_std,
            latent_channels,
            "latent_std",
        )

        if torch.any(formatted_std <= 0):
            raise ValueError("Every latent_std value must be positive.")

        self.register_buffer(
            "latent_mean",
            formatted_mean,
            persistent=True,
        )
        self.register_buffer(
            "latent_std",
            formatted_std,
            persistent=True,
        )

        self.rgb_encoder = RGBConditionEncoder(
            in_channels=3,
            base_channels=rgb_base_channels,
            out_channels=rgb_feature_channels,
        )

        self.hsi_patch_embed = PatchEmbed(
            in_channels=latent_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
        )
        self.rgb_patch_embed = PatchEmbed(
            in_channels=rgb_feature_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
        )

        self.hsi_modality_embedding = nn.Parameter(
            torch.zeros(1, 1, hidden_size)
        )
        self.rgb_modality_embedding = nn.Parameter(
            torch.zeros(1, 1, hidden_size)
        )

        self.timestep_embedder = TimestepEmbedder(
            hidden_size=hidden_size,
        )

        self.rgb_global_projection = nn.Sequential(
            nn.LayerNorm(rgb_feature_channels),
            nn.Linear(rgb_feature_channels, hidden_size),
        )

        self.blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    qk_norm=qk_norm,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            latent_channels=latent_channels,
        )

        nn.init.normal_(
            self.hsi_modality_embedding,
            std=0.02,
        )
        nn.init.normal_(
            self.rgb_modality_embedding,
            std=0.02,
        )

    @staticmethod
    def _format_latent_stat(
        value: TensorOrScalar,
        channels: int,
        name: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            value,
            dtype=torch.float32,
        )

        if tensor.numel() == 1:
            tensor = tensor.reshape(1).repeat(channels)
        elif tensor.numel() == channels:
            tensor = tensor.reshape(channels)
        else:
            raise ValueError(
                f"{name} must be a scalar or contain exactly "
                f"{channels} values."
            )

        return tensor.reshape(1, channels, 1, 1)

    def train(self, mode: bool = True) -> "RGBConditionedHSIMMDiT":
        super().train(mode)

        # Keep the frozen VAE in evaluation mode even when the
        # MM-DiT model is switched to training mode.
        if self.vae is not None and self.freeze_vae:
            self.vae.eval()

        return self

    def normalize_latent(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        return (
            latent - self.latent_mean.to(latent.dtype)
        ) / self.latent_std.to(latent.dtype)

    def denormalize_latent(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        return (
            latent * self.latent_std.to(latent.dtype)
            + self.latent_mean.to(latent.dtype)
        )

    def encode_hsi(
        self,
        hsi: torch.Tensor,
        sample: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Encode an HSI cube with the attached VAE.

        Returns:
            normalized_latent, mu, logvar
        """
        if self.vae is None:
            raise RuntimeError(
                "No VAE is attached to the model."
            )

        context = (
            torch.no_grad()
            if self.freeze_vae
            else contextlib.nullcontext()
        )

        with context:
            latent, mu, logvar = self.vae.encode(
                hsi,
                sample=sample,
            )

        normalized_latent = self.normalize_latent(latent)
        return normalized_latent, mu, logvar

    def decode_hsi_latent(
        self,
        normalized_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode a normalized HSI latent.

        This method intentionally does not use torch.no_grad(), so an
        auxiliary reconstruction loss can backpropagate through the
        frozen decoder into MM-DiT.
        """
        if self.vae is None:
            raise RuntimeError(
                "No VAE is attached to the model."
            )

        latent = self.denormalize_latent(
            normalized_latent
        )
        return self.vae.decode(latent)

    def encode_rgb_condition(
        self,
        rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            rgb_features:
                [B, rgb_feature_channels, H / 4, W / 4]
            rgb_global:
                [B, rgb_feature_channels]
        """
        rgb_features = self.rgb_encoder(rgb)
        rgb_global = rgb_features.mean(dim=(2, 3))
        return rgb_features, rgb_global

    def _prepare_timestep(
        self,
        timestep: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(
                timestep,
                device=device,
                dtype=torch.float32,
            )

        timestep = timestep.to(device=device)

        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        elif timestep.ndim == 2 and timestep.shape[-1] == 1:
            timestep = timestep.squeeze(-1)

        if timestep.ndim != 1:
            raise ValueError(
                "timestep must be a scalar or a tensor of shape [B]."
            )

        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)

        if timestep.shape[0] != batch_size:
            raise ValueError(
                "The timestep batch size does not match the input batch."
            )

        return timestep

    def _unpatchify(
        self,
        patches: torch.Tensor,
        grid_height: int,
        grid_width: int,
    ) -> torch.Tensor:
        batch_size = patches.shape[0]
        patch_size = self.patch_size

        expected_tokens = grid_height * grid_width
        if patches.shape[1] != expected_tokens:
            raise ValueError(
                "The token count does not match the patch grid."
            )

        patches = patches.reshape(
            batch_size,
            grid_height,
            grid_width,
            patch_size,
            patch_size,
            self.latent_channels,
        )

        latent = patches.permute(
            0,
            5,
            1,
            3,
            2,
            4,
        ).contiguous()

        latent = latent.reshape(
            batch_size,
            self.latent_channels,
            grid_height * patch_size,
            grid_width * patch_size,
        )

        return latent

    def forward_with_rgb_condition(
        self,
        noisy_hsi_latent: torch.Tensor,
        timestep: torch.Tensor,
        rgb_features: torch.Tensor,
        rgb_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass using precomputed RGB features.

        Useful during iterative sampling because the RGB encoder only
        has to run once.
        """
        if noisy_hsi_latent.ndim != 4:
            raise ValueError(
                "noisy_hsi_latent must have shape [B, C, H, W]."
            )

        if noisy_hsi_latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, "
                f"but received {noisy_hsi_latent.shape[1]}."
            )

        if rgb_features.shape[0] != noisy_hsi_latent.shape[0]:
            raise ValueError(
                "RGB and HSI latent batch sizes must match."
            )

        if rgb_features.shape[-2:] != noisy_hsi_latent.shape[-2:]:
            raise ValueError(
                "RGB feature and HSI latent spatial sizes must match. "
                "Use spatially aligned RGB/HSI pairs."
            )

        if (
            noisy_hsi_latent.shape[-2] % self.patch_size != 0
            or noisy_hsi_latent.shape[-1] % self.patch_size != 0
        ):
            raise ValueError(
                "The HSI latent height and width must be divisible "
                "by patch_size."
            )

        batch_size = noisy_hsi_latent.shape[0]
        timestep = self._prepare_timestep(
            timestep=timestep,
            batch_size=batch_size,
            device=noisy_hsi_latent.device,
        )

        (
            hsi_tokens,
            hsi_grid_height,
            hsi_grid_width,
        ) = self.hsi_patch_embed(noisy_hsi_latent)

        (
            rgb_tokens,
            rgb_grid_height,
            rgb_grid_width,
        ) = self.rgb_patch_embed(rgb_features)

        if (
            hsi_grid_height != rgb_grid_height
            or hsi_grid_width != rgb_grid_width
        ):
            raise ValueError(
                "HSI and RGB patch grids must have the same size."
            )

        position_embedding = (
            _build_2d_sincos_position_embedding(
                embedding_dim=self.hidden_size,
                height=hsi_grid_height,
                width=hsi_grid_width,
                device=hsi_tokens.device,
                dtype=hsi_tokens.dtype,
            )
        )

        hsi_tokens = (
            hsi_tokens
            + position_embedding
            + self.hsi_modality_embedding
        )
        rgb_tokens = (
            rgb_tokens
            + position_embedding
            + self.rgb_modality_embedding
        )

        time_condition = self.timestep_embedder(
            timestep
        )
        rgb_condition = self.rgb_global_projection(
            rgb_global
        )
        condition = time_condition + rgb_condition

        for block in self.blocks:
            hsi_tokens, rgb_tokens = block(
                hsi_tokens=hsi_tokens,
                rgb_tokens=rgb_tokens,
                condition=condition,
            )

        velocity_patches = self.final_layer(
            hsi_tokens=hsi_tokens,
            condition=condition,
        )

        velocity = self._unpatchify(
            patches=velocity_patches,
            grid_height=hsi_grid_height,
            grid_width=hsi_grid_width,
        )

        return velocity

    def forward(
        self,
        noisy_hsi_latent: torch.Tensor,
        timestep: torch.Tensor,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the rectified-flow velocity.

        Args:
            noisy_hsi_latent:
                Normalized noisy HSI latent [B, Cz, h, w].
            timestep:
                Scalar or [B] tensor with values in [0, 1].
            rgb:
                Spatially aligned RGB image [B, 3, H, W].

        Returns:
            Predicted velocity [B, Cz, h, w].
        """
        rgb_features, rgb_global = (
            self.encode_rgb_condition(rgb)
        )

        return self.forward_with_rgb_condition(
            noisy_hsi_latent=noisy_hsi_latent,
            timestep=timestep,
            rgb_features=rgb_features,
            rgb_global=rgb_global,
        )

    def predict_clean_latent(
        self,
        noisy_hsi_latent: torch.Tensor,
        timestep: torch.Tensor,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover an estimate of z0 from:
            zt = z0 + t * velocity
        """
        velocity = self(
            noisy_hsi_latent=noisy_hsi_latent,
            timestep=timestep,
            rgb=rgb,
        )

        timestep = self._prepare_timestep(
            timestep=timestep,
            batch_size=noisy_hsi_latent.shape[0],
            device=noisy_hsi_latent.device,
        )

        timestep = timestep.to(
            dtype=noisy_hsi_latent.dtype
        ).reshape(-1, 1, 1, 1)

        return noisy_hsi_latent - timestep * velocity

    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        num_steps: int = 30,
        decode: bool = True,
    ) -> torch.Tensor:
        """
        Simple Euler sampler from t=1 to t=0.

        Call model.eval() before sampling.

        Returns:
            Decoded HSI cube when decode=True.
            Otherwise, the normalized HSI latent.
        """
        if num_steps < 1:
            raise ValueError(
                "num_steps must be at least 1."
            )

        rgb_features, rgb_global = (
            self.encode_rgb_condition(rgb)
        )

        latent_height = rgb_features.shape[-2]
        latent_width = rgb_features.shape[-1]

        if (
            latent_height % self.patch_size != 0
            or latent_width % self.patch_size != 0
        ):
            raise ValueError(
                "The latent spatial size inferred from RGB must be "
                "divisible by patch_size."
            )

        latent = torch.randn(
            rgb.shape[0],
            self.latent_channels,
            latent_height,
            latent_width,
            device=rgb.device,
            dtype=rgb.dtype,
        )

        times = torch.linspace(
            1.0,
            0.0,
            num_steps + 1,
            device=rgb.device,
            dtype=torch.float32,
        )

        for step in range(num_steps):
            current_time = times[step]
            next_time = times[step + 1]

            timestep = current_time.expand(
                rgb.shape[0]
            )

            velocity = self.forward_with_rgb_condition(
                noisy_hsi_latent=latent,
                timestep=timestep,
                rgb_features=rgb_features,
                rgb_global=rgb_global,
            )

            delta_time = (
                next_time - current_time
            ).to(latent.dtype)

            latent = latent + delta_time * velocity

        if not decode:
            return latent

        return self.decode_hsi_latent(latent)


# Convenient short alias.
HSIMMDiT = RGBConditionedHSIMMDiT
