"""DDS2M-inspired RGB-to-HSI model.

This file keeps the original DDS2M design as closely as possible while
adapting the observation from a degraded HSI to an RGB image when only
paired RGB/HSI training images are available.

The file contains three main parts:

1. RGBSpectralResponse
   A learnable *linear* HSI -> RGB forward operator. It is trained first
   from paired HSI/RGB images and then frozen. Keeping it linear allows the
   original SVD-based DDS2M measurement conditioning to be retained.

2. VS2MGenerator
   The original-style untrained spatio-spectral generator:
       fixed spatial noise -> R independent attention skip U-Nets
       fixed spectral noise -> R independent FCNs
       abundance matrix @ endmember matrix -> HSI

3. RGBDDS2M
   A wrapper providing the original-style diffusion fitting loss,
   learned RGB data-consistency loss, measurement-informed initialization,
   and one SVD-conditioned reverse diffusion step.

Important:
- This is a per-image test-time optimization model, not a normal feed-forward
  RGB-to-HSI network.
- Batch size must be 1 for the VS2M generator, matching the original method.
- The HSI/RGB pair is used to train the forward operator; the HSI target is
  not required during test-time restoration.
- Images should be normalized to [0, 1] for the generator/operator losses.
  Diffusion states are represented in [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _center_crop_to(x: Tensor, height: int, width: int) -> Tensor:
    """Center-crop a BCHW tensor to the requested spatial size."""
    h, w = x.shape[-2:]
    if height > h or width > w:
        raise ValueError(
            f"Cannot crop tensor of spatial size {(h, w)} to {(height, width)}."
        )
    top = (h - height) // 2
    left = (w - width) // 2
    return x[..., top : top + height, left : left + width]


def _match_spatial(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Center-crop two BCHW tensors to their common minimum spatial size."""
    h = min(a.shape[-2], b.shape[-2])
    w = min(a.shape[-1], b.shape[-1])
    return _center_crop_to(a, h, w), _center_crop_to(b, h, w)


def _scalar_like(value: Tensor | float, ref: Tensor) -> Tensor:
    """Convert a scalar value to a scalar tensor on ref's device and dtype."""
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected a scalar tensor, got shape {tuple(value.shape)}.")
        return value.to(device=ref.device, dtype=ref.dtype).reshape(())
    return torch.tensor(float(value), device=ref.device, dtype=ref.dtype)


def hsi_total_variation(x: Tensor) -> Tensor:
    """Simple spatial-spectral total variation for a BxKxHxW HSI tensor."""
    if x.ndim != 4:
        raise ValueError(f"Expected BxKxHxW, got {tuple(x.shape)}.")
    tv_h = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    tv_w = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    if x.shape[1] > 1:
        tv_s = (x[:, 1:, ...] - x[:, :-1, ...]).abs().mean()
    else:
        tv_s = x.new_zeros(())
    return tv_h + tv_w + tv_s


# -----------------------------------------------------------------------------
# CBAM, matching the attention used in the released DDS2M implementation
# -----------------------------------------------------------------------------


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(self.shared(self.max_pool(x)) + self.shared(self.avg_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Spatial-attention kernel size must be odd.")
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        max_map = x.max(dim=1, keepdim=True).values
        avg_map = x.mean(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([max_map, avg_map], dim=1)))


class CBAMBlock(nn.Module):
    """Residual CBAM block used at selected decoder levels."""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 49,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x + residual


# -----------------------------------------------------------------------------
# Original-style skip/U-Net abundance generator
# -----------------------------------------------------------------------------


class ReflectionConv2d(nn.Module):
    """Reflection-padded convolution, matching pad='reflection' in DDS2M."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pad(x))


class _SkipStage(nn.Module):
    """One recursive stage of the original deep-image-prior skip network."""

    def __init__(
        self,
        level: int,
        in_channels: int,
        channels_down: Sequence[int],
        channels_up: Sequence[int],
        channels_skip: Sequence[int],
        kernels_down: Sequence[int],
        kernels_up: Sequence[int],
        use_attention: bool,
    ) -> None:
        super().__init__()
        self.level = level
        self.is_deepest = level == len(channels_down) - 1
        down_ch = channels_down[level]
        up_ch = channels_up[level]
        skip_ch = channels_skip[level]
        self.skip_ch = skip_ch

        if skip_ch > 0:
            self.skip_path = nn.Sequential(
                nn.Conv2d(in_channels, skip_ch, kernel_size=1, bias=True),
                nn.BatchNorm2d(skip_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.skip_path = None

        # Original code with downsample_mode='avg' performs a stride-1
        # convolution followed by average pooling by 2.
        self.down_first = nn.Sequential(
            ReflectionConv2d(in_channels, down_ch, kernels_down[level]),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(down_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down_second = nn.Sequential(
            ReflectionConv2d(down_ch, down_ch, kernels_down[level]),
            nn.BatchNorm2d(down_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if self.is_deepest:
            self.child = None
            deeper_out_ch = down_ch
        else:
            self.child = _SkipStage(
                level=level + 1,
                in_channels=down_ch,
                channels_down=channels_down,
                channels_up=channels_up,
                channels_skip=channels_skip,
                kernels_down=kernels_down,
                kernels_up=kernels_up,
                use_attention=use_attention,
            )
            deeper_out_ch = channels_up[level + 1]

        concat_ch = skip_ch + deeper_out_ch
        self.concat_bn = nn.BatchNorm2d(concat_ch)

        # Released code adds CBAM at levels 2 and 3 for the six-scale model:
        #     if i > len(channels)-5 and i < len(channels)-2
        attention_here = (
            use_attention
            and level > len(channels_down) - 5
            and level < len(channels_down) - 2
        )
        self.attention = CBAMBlock(concat_ch) if attention_here else nn.Identity()

        self.up_conv = nn.Sequential(
            ReflectionConv2d(concat_ch, up_ch, kernels_up[level]),
            nn.BatchNorm2d(up_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(up_ch, up_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(up_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        skip_features = self.skip_path(x) if self.skip_path is not None else None

        deeper = self.down_second(self.down_first(x))
        if self.child is not None:
            deeper = self.child(deeper)

        deeper = F.interpolate(
            deeper,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )

        if skip_features is not None:
            skip_features, deeper = _match_spatial(skip_features, deeper)
            merged = torch.cat([skip_features, deeper], dim=1)
        else:
            merged = deeper

        merged = self.concat_bn(merged)
        merged = self.attention(merged)
        return self.up_conv(merged)


class AttentionSkipUNet(nn.Module):
    """DDS2M-style U-Net used to generate one abundance map."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels_down: Sequence[int] = (16, 32, 64, 128, 128, 128),
        channels_up: Sequence[int] = (16, 32, 64, 128, 128, 128),
        channels_skip: Sequence[int] = (0, 0, 4, 4, 4, 4),
        kernels_down: Sequence[int] = (7, 7, 5, 5, 3, 3),
        kernels_up: Sequence[int] = (7, 7, 5, 5, 3, 3),
        use_attention: bool = True,
        output_activation: Literal["none", "sigmoid"] = "none",
    ) -> None:
        super().__init__()
        lengths = {
            len(channels_down),
            len(channels_up),
            len(channels_skip),
            len(kernels_down),
            len(kernels_up),
        }
        if len(lengths) != 1:
            raise ValueError("All per-level configuration sequences must have equal length.")
        self.body = _SkipStage(
            level=0,
            in_channels=in_channels,
            channels_down=channels_down,
            channels_up=channels_up,
            channels_skip=channels_skip,
            kernels_down=kernels_down,
            kernels_up=kernels_up,
            use_attention=use_attention,
        )
        self.head = nn.Conv2d(channels_up[0], out_channels, kernel_size=1, bias=True)
        if output_activation == "none":
            self.output_activation = nn.Identity()
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported output activation: {output_activation}")

    def forward(self, x: Tensor) -> Tensor:
        return self.output_activation(self.head(self.body(x)))


# -----------------------------------------------------------------------------
# Original-style spectral FCN
# -----------------------------------------------------------------------------


class SpectralFCN(nn.Module):
    """DDS2M spectral network: K -> 128 -> 256 -> 256 -> 128 -> K."""

    def __init__(
        self,
        bands: int,
        hidden: Sequence[int] = (128, 256, 256, 128),
    ) -> None:
        super().__init__()
        if not hidden:
            raise ValueError("hidden must contain at least one width.")
        layers: list[nn.Module] = [nn.Linear(bands, hidden[0]), nn.ReLU6()]
        for in_dim, out_dim in zip(hidden[:-1], hidden[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden[-1], bands))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# Learnable HSI -> RGB operator from paired images
# -----------------------------------------------------------------------------


class RGBSpectralResponse(nn.Module):
    """Learnable linear per-pixel HSI -> RGB operator.

    The weight matrix has shape [3, K]. No bias is used because the original
    DDS2M reverse equations assume y = Hx + noise with a linear H.

    Train this module first using paired HSI/RGB images, then freeze it before
    per-image DDS2M reconstruction.
    """

    def __init__(
        self,
        bands: int,
        constrain_nonnegative: bool = False,
    ) -> None:
        super().__init__()
        self.bands = bands
        self.constrain_nonnegative = constrain_nonnegative

        if constrain_nonnegative:
            # Softmax gives smooth nonnegative row-normalized responses.
            self.raw_weight = nn.Parameter(torch.zeros(3, bands))
            self.log_gain = nn.Parameter(torch.zeros(3))
        else:
            weight = torch.empty(3, bands)
            nn.init.normal_(weight, mean=0.0, std=1.0 / max(1, bands) ** 0.5)
            self.raw_weight = nn.Parameter(weight)
            self.register_parameter("log_gain", None)

    @property
    def weight(self) -> Tensor:
        if not self.constrain_nonnegative:
            return self.raw_weight
        response = F.softmax(self.raw_weight, dim=1)
        gain = F.softplus(self.log_gain).unsqueeze(1)
        return response * gain

    def forward(self, hsi: Tensor) -> Tensor:
        if hsi.ndim != 4 or hsi.shape[1] != self.bands:
            raise ValueError(
                f"Expected HSI shape Bx{self.bands}xHxW, got {tuple(hsi.shape)}."
            )
        return torch.einsum("ck,bkhw->bchw", self.weight, hsi)

    def pseudoinverse(self, rgb: Tensor, rcond: float = 1e-6) -> Tensor:
        """Minimum-norm HSI estimate R^+ RGB, applied independently per pixel."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB shape Bx3xHxW, got {tuple(rgb.shape)}.")
        pinv = torch.linalg.pinv(self.weight, rcond=rcond)  # [K, 3]
        return torch.einsum("kc,bchw->bkhw", pinv, rgb)

    def svd(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Return U, singular values, and Vh for the 3xK response matrix."""
        return torch.linalg.svd(self.weight, full_matrices=True)

    def smoothness_regularizer(self) -> Tensor:
        """Second-order spectral smoothness of each RGB response curve."""
        w = self.weight
        if w.shape[1] < 3:
            return w.new_zeros(())
        second = w[:, 2:] - 2.0 * w[:, 1:-1] + w[:, :-2]
        return second.square().mean()


# -----------------------------------------------------------------------------
# Original-style VS2M generator
# -----------------------------------------------------------------------------


class VS2MGenerator(nn.Module):
    """Untrained DDS2M spatio-spectral HSI generator.

    This module is intentionally image-specific. Instantiate a fresh model
    for each RGB image and optimize its parameters during reconstruction.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        bands: int,
        rank: int = 10,
        use_attention: bool = True,
        spatial_output_activation: Literal["none", "sigmoid"] = "none",
        final_output_activation: Literal["none", "sigmoid"] = "none",
        noise_seed: int = 0,
    ) -> None:
        super().__init__()
        height, width = image_size
        if height < 128 or width < 128:
            raise ValueError(
                "The faithful six-scale BatchNorm architecture expects spatial "
                "dimensions of at least 128. Use 128x128 or larger patches."
            )
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError(
                "For the faithful six-scale implementation, H and W must be "
                "divisible by 64."
            )
        if bands < 4:
            raise ValueError("RGB-to-HSI requires more than three spectral bands.")
        if rank < 1:
            raise ValueError("rank must be positive.")

        self.height = height
        self.width = width
        self.bands = bands
        self.rank = rank
        self.final_output_activation = final_output_activation

        # As in the released code, all R independent networks receive the same
        # fixed input, but each has independent parameters.
        self.spatial_nets = nn.ModuleList(
            [
                AttentionSkipUNet(
                    in_channels=1,
                    out_channels=1,
                    use_attention=use_attention,
                    output_activation=spatial_output_activation,
                )
                for _ in range(rank)
            ]
        )
        self.spectral_nets = nn.ModuleList(
            [SpectralFCN(bands=bands) for _ in range(rank)]
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_seed)
        spatial_noise = torch.rand(1, 1, height, width, generator=generator)
        spectral_noise = torch.rand(1, bands, generator=generator)
        self.register_buffer("fixed_spatial_noise", spatial_noise)
        self.register_buffer("fixed_spectral_noise", spectral_noise)

    def regenerate_fixed_noise(self, seed: int) -> None:
        """Regenerate the fixed seed tensors before starting a new image."""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        spatial = torch.rand(
            1, 1, self.height, self.width, generator=generator
        ).to(device=self.fixed_spatial_noise.device, dtype=self.fixed_spatial_noise.dtype)
        spectral = torch.rand(1, self.bands, generator=generator).to(
            device=self.fixed_spectral_noise.device,
            dtype=self.fixed_spectral_noise.dtype,
        )
        self.fixed_spatial_noise.copy_(spatial)
        self.fixed_spectral_noise.copy_(spectral)

    def forward(
        self,
        return_components: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        # Batch size is intentionally one, as in the original test-time method.
        spatial_maps = [net(self.fixed_spatial_noise) for net in self.spatial_nets]
        abundance = torch.cat(spatial_maps, dim=1)  # [1, R, H, W]

        spectra = [net(self.fixed_spectral_noise) for net in self.spectral_nets]
        endmembers = torch.stack(spectra, dim=1)  # [1, R, K]

        hsi = torch.einsum("brhw,brk->bkhw", abundance, endmembers)
        if self.final_output_activation == "sigmoid":
            hsi = torch.sigmoid(hsi)
        elif self.final_output_activation != "none":
            raise RuntimeError(
                f"Unsupported final activation {self.final_output_activation}."
            )

        if return_components:
            return hsi, abundance, endmembers
        return hsi


# -----------------------------------------------------------------------------
# Full RGB adaptation wrapper
# -----------------------------------------------------------------------------


@dataclass
class LossBundle:
    total: Tensor
    diffusion: Tensor
    rgb: Tensor
    tv: Tensor
    supervised: Tensor


class RGBDDS2M(nn.Module):
    """DDS2M-style RGB-to-HSI wrapper using only paired RGB/HSI data.

    Usage is deliberately two-stage:

    Stage A: learn the HSI->RGB operator from all paired images.
        rgb_hat = model.operator(hsi_gt)
        optimize MSE(rgb_hat, rgb) + response smoothness

    Stage B: freeze model.operator. For each RGB test image, create a fresh
        RGBDDS2M (or fresh generator), initialize x_t from RGB, and optimize
        only model.generator at each reverse diffusion timestep.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        bands: int,
        rank: int = 10,
        operator: Optional[RGBSpectralResponse] = None,
        constrain_operator_nonnegative: bool = False,
        use_attention: bool = True,
        spatial_output_activation: Literal["none", "sigmoid"] = "none",
        final_output_activation: Literal["none", "sigmoid"] = "none",
        noise_seed: int = 0,
    ) -> None:
        super().__init__()
        self.bands = bands
        self.image_size = image_size
        self.generator = VS2MGenerator(
            image_size=image_size,
            bands=bands,
            rank=rank,
            use_attention=use_attention,
            spatial_output_activation=spatial_output_activation,
            final_output_activation=final_output_activation,
            noise_seed=noise_seed,
        )
        self.operator = operator or RGBSpectralResponse(
            bands=bands,
            constrain_nonnegative=constrain_operator_nonnegative,
        )

    def forward(
        self,
        return_components: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        return self.generator(return_components=return_components)

    def freeze_operator(self) -> None:
        self.operator.eval()
        for parameter in self.operator.parameters():
            parameter.requires_grad_(False)

    def unfreeze_operator(self) -> None:
        for parameter in self.operator.parameters():
            parameter.requires_grad_(True)

    def operator_training_loss(
        self,
        hsi_gt_01: Tensor,
        rgb_01: Tensor,
        smoothness_weight: float = 0.0,
    ) -> Dict[str, Tensor]:
        """Loss for learning the missing camera response from paired images."""
        rgb_hat = self.operator(hsi_gt_01)
        reconstruction = F.mse_loss(rgb_hat, rgb_01)
        smoothness = self.operator.smoothness_regularizer()
        total = reconstruction + smoothness_weight * smoothness
        return {
            "total": total,
            "rgb_reconstruction": reconstruction,
            "operator_smoothness": smoothness,
            "rgb_hat": rgb_hat,
        }

    def restoration_losses(
        self,
        rgb_01: Tensor,
        x_t_m11: Optional[Tensor] = None,
        alpha_bar_t: Optional[Tensor | float] = None,
        hsi_gt_01: Optional[Tensor] = None,
        diffusion_weight: float = 1.0,
        rgb_weight: float = 1.0,
        tv_weight: float = 0.0,
        supervised_weight: float = 0.0,
    ) -> LossBundle:
        """Compute losses for one inner VS2M optimization step.

        The faithful test-time setup uses:
            diffusion_weight > 0
            rgb_weight >= 0
            tv_weight >= 0
            supervised_weight = 0

        `hsi_gt_01` is optional and should only be used for supervised ablations,
        not for the faithful test-time reconstruction.
        """
        hsi_01 = self.generator()
        rgb_hat = self.operator(hsi_01)
        rgb_loss = F.mse_loss(rgb_hat, rgb_01)
        tv_loss = hsi_total_variation(hsi_01)

        if x_t_m11 is not None:
            if alpha_bar_t is None:
                raise ValueError("alpha_bar_t is required when x_t_m11 is provided.")
            alpha = _scalar_like(alpha_bar_t, hsi_01)
            x0_m11 = hsi_01 * 2.0 - 1.0
            diffusion_loss = F.mse_loss(alpha.sqrt() * x0_m11, x_t_m11)
        else:
            diffusion_loss = hsi_01.new_zeros(())

        if hsi_gt_01 is not None:
            supervised_loss = F.l1_loss(hsi_01, hsi_gt_01)
        else:
            supervised_loss = hsi_01.new_zeros(())

        total = (
            diffusion_weight * diffusion_loss
            + rgb_weight * rgb_loss
            + tv_weight * tv_loss
            + supervised_weight * supervised_loss
        )
        return LossBundle(
            total=total,
            diffusion=diffusion_loss,
            rgb=rgb_loss,
            tv=tv_loss,
            supervised=supervised_loss,
        )

    @torch.no_grad()
    def initialize_xt_from_rgb(
        self,
        rgb_m11: Tensor,
        alpha_bar_t0: Tensor | float,
        rgb_noise_std_m11: float = 0.0,
        generator: Optional[torch.Generator] = None,
        eps: float = 1e-8,
    ) -> Tensor:
        """Measurement-informed DDS2M initialization in the spectral SVD basis.

        This is the per-pixel counterpart of the original initialization for
        H = I_(HW) kron R, where R is the learned 3xK spectral response.
        """
        if rgb_m11.ndim != 4 or rgb_m11.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW RGB, got {tuple(rgb_m11.shape)}.")
        alpha = _scalar_like(alpha_bar_t0, rgb_m11)
        sigma = torch.sqrt((1.0 - alpha).clamp_min(0.0)) / alpha.sqrt().clamp_min(eps)
        if sigma <= eps:
            raise ValueError("alpha_bar_t0 must correspond to a nonzero diffusion level.")

        u, singulars, vh = self.operator.svd()
        ut_y = torch.einsum("dc,bchw->bdhw", u.transpose(0, 1), rgb_m11)
        noise_v = torch.randn(
            rgb_m11.shape[0],
            self.bands,
            rgb_m11.shape[2],
            rgb_m11.shape[3],
            device=rgb_m11.device,
            dtype=rgb_m11.dtype,
            generator=generator,
        )
        init_v = sigma * noise_v

        sigma_y = torch.tensor(
            float(rgb_noise_std_m11),
            device=rgb_m11.device,
            dtype=rgb_m11.dtype,
        )
        for index in range(min(3, singulars.numel())):
            s = singulars[index].clamp_min(eps)
            if s * sigma > sigma_y:
                measurement = ut_y[:, index] / s
                remaining = torch.sqrt(
                    (sigma.square() - (sigma_y / s).square()).clamp_min(0.0)
                )
                init_v[:, index] = measurement + remaining * noise_v[:, index]

        init_v = init_v / sigma
        v = vh.transpose(0, 1)
        return torch.einsum("kl,blhw->bkhw", v, init_v)

    @torch.no_grad()
    def reverse_step(
        self,
        x_t_m11: Tensor,
        x0_hat_m11: Tensor,
        rgb_m11: Tensor,
        alpha_bar_t: Tensor | float,
        alpha_bar_next: Tensor | float,
        rgb_noise_std_m11: float = 0.0,
        eta_a: float = 0.95,
        eta_b: float = 1.0,
        eta_c: float = 0.95,
        generator: Optional[torch.Generator] = None,
        eps: float = 1e-8,
    ) -> Tensor:
        """One original-style SVD-conditioned reverse diffusion step.

        `alpha_bar_next` is the cumulative alpha of the next, cleaner step.
        For the final x_0 step use alpha_bar_next = 1.
        """
        if x_t_m11.shape != x0_hat_m11.shape:
            raise ValueError("x_t_m11 and x0_hat_m11 must have identical shapes.")
        if x_t_m11.shape[1] != self.bands:
            raise ValueError(
                f"Expected {self.bands} HSI bands, got {x_t_m11.shape[1]}."
            )
        if rgb_m11.shape[0] != x_t_m11.shape[0] or rgb_m11.shape[-2:] != x_t_m11.shape[-2:]:
            raise ValueError("RGB and HSI batch/spatial shapes must match.")

        alpha_t = _scalar_like(alpha_bar_t, x_t_m11)
        alpha_next = _scalar_like(alpha_bar_next, x_t_m11)

        epsilon_hat = (
            x_t_m11 - alpha_t.sqrt() * x0_hat_m11
        ) / torch.sqrt((1.0 - alpha_t).clamp_min(eps))

        sigma_next = torch.sqrt((1.0 - alpha_next).clamp_min(0.0)) / alpha_next.sqrt().clamp_min(eps)

        u, singulars, vh = self.operator.svd()
        v = vh.transpose(0, 1)
        vt_x0 = torch.einsum("lk,bkhw->blhw", vh, x0_hat_m11)
        vt_eps = torch.einsum("lk,bkhw->blhw", vh, epsilon_hat)
        ut_y = torch.einsum("dc,bchw->bdhw", u.transpose(0, 1), rgb_m11)

        noise_v = torch.randn(
            vt_x0.shape,
            device=vt_x0.device,
            dtype=vt_x0.dtype,
            generator=generator,
        )

        std_c = sigma_next * float(eta_c)
        deterministic_c = torch.sqrt(
            (sigma_next.square() - std_c.square()).clamp_min(0.0)
        )
        next_v = vt_x0 + deterministic_c * vt_eps + std_c * noise_v

        sigma_y = torch.tensor(
            float(rgb_noise_std_m11),
            device=x_t_m11.device,
            dtype=x_t_m11.dtype,
        )

        measured_dims = min(3, singulars.numel())
        if sigma_y <= eps:
            # Exact noiseless RGB consistency. The first measured spectral SVD
            # coordinates are set directly from the RGB observation.
            for index in range(measured_dims):
                s = singulars[index].clamp_min(eps)
                next_v[:, index] = ut_y[:, index] / s
        else:
            for index in range(measured_dims):
                s = singulars[index].clamp_min(eps)
                if s * sigma_next < sigma_y:
                    # Current diffusion state is cleaner than the measurement.
                    std_a = sigma_next * float(eta_a)
                    deterministic_a = torch.sqrt(
                        (sigma_next.square() - std_a.square()).clamp_min(0.0)
                    )
                    residual = (ut_y[:, index] - s * vt_x0[:, index]) / sigma_y
                    next_v[:, index] = (
                        vt_x0[:, index]
                        + deterministic_a * residual
                        + std_a * noise_v[:, index]
                    )
                else:
                    # Measurement is more reliable at this diffusion level.
                    variance = sigma_next.square() - (
                        sigma_y / s
                    ).square() * float(eta_b) ** 2
                    std_b = torch.sqrt(variance.clamp_min(0.0))
                    next_v[:, index] = (
                        float(eta_b) * (ut_y[:, index] / s)
                        + (1.0 - float(eta_b)) * vt_x0[:, index]
                        + std_b * noise_v[:, index]
                    )

        x_mod_next = torch.einsum("kl,blhw->bkhw", v, next_v)
        return alpha_next.sqrt() * x_mod_next


__all__ = [
    "AttentionSkipUNet",
    "CBAMBlock",
    "LossBundle",
    "RGBDDS2M",
    "RGBSpectralResponse",
    "SpectralFCN",
    "VS2MGenerator",
    "hsi_total_variation",
]
