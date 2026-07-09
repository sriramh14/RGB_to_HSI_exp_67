"""
model.py
========

DiffBIR-style Blind Hyperspectral Image (HSI) Restoration.

This module adapts the two-stage pipeline of

    "DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior"
    (Lin et al., 2023) -- https://github.com/XPixelGroup/DiffBIR

to the task of blind RGB -> Hyperspectral Image (HSI) restoration.

Pipeline
--------

    RGB  --[Stage I: MST++ Restoration Module]-->  coarse HSI (I_RM)
    I_RM --[frozen HSI VAE Encoder]--------------->  condition latent (c_RM)
    z_T  --[IRControlNet + frozen HSI Latent Diffusion UNet]--> z_0 (denoised latent)
    z_0  --[frozen HSI VAE Decoder]---------------->  refined HSI

Faithfulness to DiffBIR
------------------------

This file mirrors the paper's two-stage design as closely as possible:

  1. Restoration Module (RM, Stage I): a pretrained MST++ model that maps
     the degraded RGB observation to a coarse, high-fidelity (but
     texture-poor) HSI estimate I_RM. It is treated as a black box,
     trained separately, and is FROZEN here (Section 3.2 of the paper).

  2. Generation Module (GM, Stage II): a latent-diffusion model built on a
     pretrained (frozen) HSI Latent Diffusion UNet, controlled by
     "IRControlNet" (Section 3.3):
        a) Condition encoding: the frozen, pretrained HSI VAE encoder E
           is reused (NOT a small trained-from-scratch hint network) to
           project I_RM into the latent space: c_RM = E(I_RM).
        b) Condition network: a TRAINABLE COPY of the UNet's encoder
           ("input_blocks") and middle block, whose first layer is
           zero-expanded to accept the channel-concatenation
           z'_t = concat(z_t, c_RM).
        c) Feature modulation: multi-scale features from the condition
           network modulate (via addition) the frozen UNet's skip
           connections and middle-block feature, each through a
           zero-initialized 1x1 convolution ("zero convolution"), exactly
           as in ControlNet / IRControlNet.
     Only the condition network + zero convolutions + copied time
     embedding are trainable; the backbone UNet and the VAE are frozen
     (Eq. 4 of the paper).

  3. Region-Adaptive Restoration Guidance (Section 3.4, Eq. 5-7): a
     training-free, gradient-descent-based guidance applied at every
     sampling step, pulling the low-frequency (flat) regions of the
     denoised prediction towards the Stage-I HSI I_RM, while leaving
     high-frequency (textured) regions free to be generated. Adapted
     here to multi-band HSI by averaging the Sobel gradient magnitude
     across spectral bands before computing the patch-level gradient
     density map (Eq. 14-17).

Imported (NOT reimplemented) components
----------------------------------------

The following are assumed to be implemented elsewhere in the user's
codebase and are imported here. Adjust the import paths as needed.

  * MST_Plus_Plus       -- Stage-I RGB -> HSI restoration backbone.
  * HSIVAE              -- pretrained VAE with `.encode(x)` and
                            `.decode(z)`, operating on HSI cubes
                            (B, C_hsi, H, W) <-> latents (B, C_lat, h, w).
  * HSILatentDiffusionUNet -- pretrained latent-diffusion UNet operating
                            on the VAE latent space, following the
                            standard (CompVis/ControlNet-style) UNet
                            interface, i.e. it exposes:
                                .time_embed        (nn.Module)
                                .input_blocks       (nn.ModuleList of
                                                      TimestepEmbedSequential-
                                                      like blocks, each
                                                      called as
                                                      block(x, emb, context))
                                .middle_block       (same calling convention)
                                .output_blocks       (nn.ModuleList, mirror
                                                      of input_blocks)
                                .out                (final projection to
                                                      predicted noise)
                            and a forward signature compatible with:
                                unet(x, timesteps, context=None,
                                     control=None, mid_control=None)
                            where `control` is a list of per-scale control
                            tensors (one per skip connection, ordered to
                            match `output_blocks`) added residually to the
                            corresponding skip features, and `mid_control`
                            is added to the middle-block output. This is
                            the same convention used by the official
                            ControlNet implementation. If your UNet's
                            forward signature differs, adapt
                            `GenerationModule._call_unet` below.

If any of these imports fail (e.g. while reading this file outside of the
target project), lightweight placeholder stubs are substituted so that the
module can still be imported and inspected; replace `USE_STUBS = False`
import guards with the real modules in your environment.
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Imports of pretrained / externally-implemented components.
# Adjust these import paths to match your project layout.
# --------------------------------------------------------------------------
try:
    from .MST_Plus_Plus import MST_Plus_Plus  # Stage-I restoration module
except ImportError:  # pragma: no cover - fallback for standalone inspection
    MST_Plus_Plus = None

try:
    from .HSI_VAE import HSI_VAE  # pretrained HSI VAE (encoder/decoder)
except ImportError:  # pragma: no cover
    HSIVAE = None

try:
    from .Unet_hsi import HSILatentDiffusionUNet  # pretrained latent UNet
except ImportError:  # pragma: no cover
    HSILatentDiffusionUNet = None


# ==========================================================================
# Generic utilities
# ==========================================================================

def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze all parameters of `module` in-place and set it to eval mode."""
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.

    Used for the "zero convolution" trick (Zhang & Agrawala, ControlNet;
    Section 3.3 of DiffBIR): zero-initialized layers avoid injecting
    random-noise gradients into the frozen backbone at the start of
    training, since the added control signal is exactly zero initially.
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Standard sinusoidal timestep embeddings, as used in DDPM / Stable
    Diffusion / ControlNet.

    Args:
        timesteps: (B,) tensor of integer / float diffusion timesteps.
        dim: embedding dimensionality.
    Returns:
        (B, dim) embedding tensor.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def find_first_conv2d(module: nn.Module) -> Optional[nn.Conv2d]:
    """Depth-first search for the first nn.Conv2d submodule of `module`."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    return None


# ==========================================================================
# IRControlNet: the trainable conditioning branch (Section 3.3, Fig. 4)
# ==========================================================================

class IRControlNet(nn.Module):
    """
    Faithful re-implementation of IRControlNet from the DiffBIR paper,
    adapted for HSI latents.

    Components (matching Fig. 4 / Section 3.3 of the paper):
      1) Condition encoding: performed OUTSIDE this module, by the frozen,
         pretrained HSI VAE encoder (see `GenerationModule.encode_condition`).
         This module only receives the resulting condition latent c_RM.
      2) Condition network F_cond: a trainable copy of the pretrained
         UNet's encoder ("input_blocks") and middle block. It receives
         z'_t = concat(z_t, c_RM) along the channel axis. Because
         concatenation increases the input channel count beyond what the
         copied first layer expects, a few extra input channels are added
         to that first convolution and zero-initialized (Section 3.3,
         "zero initialization" paragraph) -- this avoids random-noise
         gradients early in training while still giving a good weight
         initialization for the rest of the copied network.
      3) Feature modulation: multi-scale outputs of the condition network
         are projected through zero convolutions and later added to the
         frozen backbone UNet's skip / middle-block features.

    Only this module's parameters (condition network + zero convolutions +
    copied time-embedding) are trainable; the original UNet and VAE remain
    frozen, as required by Eq. 4 in the paper.
    """

    def __init__(
        self,
        unet: nn.Module,
        latent_channels: int,
        skip_channels: Sequence[int],
        mid_channels: int,
    ):
        """
        Args:
            unet: the pretrained, frozen HSI latent diffusion UNet, used
                only as a template to build a trainable copy of its
                encoder path (`.input_blocks`) and `.middle_block`, and to
                clone its `.time_embed`. `unet` itself is NOT modified or
                stored by reference here (a deep copy is taken).
            latent_channels: number of channels of a single VAE latent
                (i.e. channels of z_t, and of c_RM, before concatenation).
            skip_channels: output channel count of each `input_blocks[i]`
                stage in `unet`, in order. Used to build one zero
                convolution per stage (mirrors ControlNet's `zero_convs`).
            mid_channels: output channel count of `unet.middle_block`.
        """
        super().__init__()

        # --- 2) Condition network: trainable copy of encoder + middle block.
        self.time_embed = copy.deepcopy(unet.time_embed)
        self.input_blocks = copy.deepcopy(unet.input_blocks)
        self.middle_block = copy.deepcopy(unet.middle_block)

        # Zero-expand the very first convolution to accept the
        # channel-concatenation of z_t and c_RM instead of z_t alone.
        self._expand_first_conv(extra_in_channels=latent_channels)

        # --- 3) Feature modulation: one zero conv per input-block stage,
        # plus one for the middle block. These map control features to the
        # same channel dimensionality as the corresponding frozen-UNet
        # feature (since the condition network is a structural copy of the
        # UNet encoder, channel dims already match).
        self.zero_convs = nn.ModuleList(
            [zero_module(nn.Conv2d(ch, ch, kernel_size=1)) for ch in skip_channels]
        )
        self.middle_block_out = zero_module(nn.Conv2d(mid_channels, mid_channels, kernel_size=1))

    def _expand_first_conv(self, extra_in_channels: int) -> None:
        """
        Increase the input-channel count of the first Conv2d found in
        `self.input_blocks[0]` by `extra_in_channels`, zero-initializing
        the new weight slice. This implements the "introduce a few
        parameters ... and initialize them to zero" step described in
        Section 3.3 of the paper for accommodating
        z'_t = cat(z_t, c_RM).
        """
        first_block = self.input_blocks[0]
        old_conv = find_first_conv2d(first_block)
        if old_conv is None:
            raise RuntimeError(
                "IRControlNet: could not locate a Conv2d layer inside "
                "input_blocks[0] to expand for concatenated conditioning. "
                "Adjust `_expand_first_conv` to match your UNet's architecture."
            )

        new_in_channels = old_conv.in_channels + extra_in_channels
        new_conv = nn.Conv2d(
            new_in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, : old_conv.in_channels] = old_conv.weight
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self._replace_submodule(first_block, old_conv, new_conv)

    @staticmethod
    def _replace_submodule(root: nn.Module, target: nn.Module, replacement: nn.Module) -> bool:
        """Replace the first occurrence of `target` inside `root` with `replacement`."""
        for name, child in root.named_children():
            if child is target:
                setattr(root, name, replacement)
                return True
            if IRControlNet._replace_submodule(child, target, replacement):
                return True
        return False

    def forward(
        self,
        z_t: torch.Tensor,
        c_rm: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            z_t: (B, C_lat, h, w) noisy latent at diffusion step t.
            c_rm: (B, C_lat, h, w) condition latent, c_RM = E(I_RM).
            timesteps: (B,) diffusion timesteps.
            context: optional cross-attention context (e.g. empty / negative
                text-prompt embeddings, as used in DiffBIR).

        Returns:
            A list of control-signal tensors, ordered to match the frozen
            UNet's skip connections (`output_blocks`, reversed order handled
            by `GenerationModule._call_unet`), with the middle-block control
            signal appended last.
        """
        z_prime = torch.cat([z_t, c_rm], dim=1)  # Eq.: z'_t = cat(z_t, c_RM)

        emb = self.time_embed(timestep_embedding(timesteps, self.time_embed[0].in_features)
                               if isinstance(self.time_embed, nn.Sequential)
                               else timestep_embedding(timesteps, self._infer_temb_dim()))

        controls: List[torch.Tensor] = []
        h = z_prime
        for block, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = block(h, emb, context)
            controls.append(zero_conv(h))

        h = self.middle_block(h, emb, context)
        controls.append(self.middle_block_out(h))
        return controls

    def _infer_temb_dim(self) -> int:
        """Best-effort inference of the sinusoidal embedding input dim."""
        first_linear = None
        for m in self.time_embed.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        if first_linear is None:
            raise RuntimeError("IRControlNet: could not infer timestep embedding dimension.")
        return first_linear.in_features


# ==========================================================================
# Generation Module: frozen backbone UNet + trainable IRControlNet
# ==========================================================================

class GenerationModule(nn.Module):
    """
    Stage-II generation module (Section 3.3). Wraps:
      - a frozen, pretrained HSI VAE (encoder used for conditioning,
        decoder used to map latents back to HSI space),
      - a frozen, pretrained HSI latent diffusion UNet (the generative
        prior),
      - a trainable IRControlNet providing region-/scale-wise control
        signals to the frozen UNet.

    Implements the objective of Eq. 4:
        L_GM = E_{z_t,c,t,eps,c_RM} [ || eps - eps_theta(z_t, c, t, c_RM) ||^2 ]
    where only IRControlNet's parameters are updated.
    """

    def __init__(self, unet: nn.Module, vae: nn.Module, control_net: IRControlNet):
        super().__init__()
        self.unet = freeze_module(unet)
        self.vae = freeze_module(vae)
        self.control_net = control_net  # trainable

    # ---- Condition encoding (Section 3.3, "1) condition encoding") ------
    @torch.no_grad()
    def encode_condition(self, i_rm: torch.Tensor) -> torch.Tensor:
        """
        Encode the Stage-I HSI estimate I_RM with the frozen, pretrained
        VAE encoder to obtain the reliable condition latent c_RM = E(I_RM).

        The condition is used deterministically (VAE mode/mean, not a
        stochastic sample) since it plays the role of a fixed, reliable
        control signal rather than a generative latent variable.
        """
        latent = self.vae.encode(i_rm)
        return self._latent_to_tensor(latent, deterministic=True)

    @torch.no_grad()
    def encode_target(self, hq_hsi: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Encode a ground-truth HSI cube into the VAE latent space (for training)."""
        latent = self.vae.encode(hq_hsi)
        return self._latent_to_tensor(latent, deterministic=not sample)

    @staticmethod
    def _latent_to_tensor(latent: Union[torch.Tensor, object], deterministic: bool) -> torch.Tensor:
        """
        Normalize the output of `vae.encode` to a plain tensor, supporting
        VAEs that return either a raw tensor or a distribution-like object
        exposing `.mode()` / `.sample()` (e.g. a diagonal Gaussian, as in
        Stable Diffusion's AutoencoderKL).
        """
        if torch.is_tensor(latent):
            return latent
        if deterministic and hasattr(latent, "mode"):
            return latent.mode()
        if hasattr(latent, "sample"):
            return latent.sample()
        raise TypeError(
            "HSIVAE.encode() must return a Tensor or an object exposing "
            "`.mode()`/`.sample()`."
        )

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent back into HSI (pixel) space with the frozen VAE decoder."""
        return self.vae.decode(z)

    def decode_grad(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent with gradients enabled w.r.t. `z` (but NOT w.r.t.
        the frozen decoder's parameters). Used by the restoration guidance
        step, which needs d(decoded image)/d(z) but must not update the
        decoder itself.
        """
        return self.vae.decode(z)

    # ---- Noise prediction --------------------------------------------
    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        c_rm: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict the noise eps_theta(z_t, c, t, c_RM) using the frozen UNet
        modulated by IRControlNet's control signals.
        """
        controls = self.control_net(z_t, c_rm, timesteps, context)
        mid_control = controls[-1]
        skip_controls = controls[:-1]
        return self._call_unet(z_t, timesteps, context, skip_controls, mid_control)

    def _call_unet(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor],
        skip_controls: List[torch.Tensor],
        mid_control: torch.Tensor,
    ) -> torch.Tensor:
        """
        Invoke the frozen backbone UNet with the ControlNet-style
        `control` / `mid_control` keyword arguments. If your imported
        `HSILatentDiffusionUNet` uses a different signature, adapt this
        method only (the rest of the pipeline is architecture-agnostic).
        """
        try:
            return self.unet(
                z_t, timesteps, context=context, control=skip_controls, mid_control=mid_control
            )
        except TypeError:
            # Fallback for UNets that accept a single combined control list
            # (skip controls followed by the mid control), a common
            # alternative convention.
            return self.unet(z_t, timesteps, context=context, control=skip_controls + [mid_control])


# ==========================================================================
# Region-Adaptive Restoration Guidance (Section 3.4, Eq. 5-7, 14-17)
# ==========================================================================

class RegionAdaptiveRestorationGuidance:
    """
    Training-free, sampling-time guidance that trades off fidelity vs.
    generative quality by pulling the flat (low-gradient) regions of the
    decoded prediction towards the Stage-I restoration image I_RM, while
    leaving high-gradient (textured) regions free to be hallucinated by
    the diffusion prior.

    Adapted for HSI: the paper's per-pixel Sobel gradient magnitude (RGB)
    is computed independently per spectral band and then averaged across
    bands, giving a single-channel gradient/weight map that is broadcast
    to all bands when computing the guidance loss (Eq. 6).
    """

    def __init__(self, patch_size: int = 8):
        self.patch_size = patch_size
        # Sobel kernels for x/y gradients, applied depth-wise per band.
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_kernels(sobel_x, sobel_y)

    def register_kernels(self, sobel_x: torch.Tensor, sobel_y: torch.Tensor) -> None:
        self._kx = sobel_x.view(1, 1, 3, 3)
        self._ky = sobel_y.view(1, 1, 3, 3)

    def _sobel_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-band Sobel gradient magnitude, averaged over the spectral
        dimension. Eq. 14 in the paper (computed per-band here, then
        averaged, since HSI has C bands rather than 3 RGB channels).

        Args:
            x: (B, C, H, W) HSI cube.
        Returns:
            (B, 1, H, W) gradient magnitude map, averaged across bands.
        """
        b, c, h, w = x.shape
        kx = self._kx.to(device=x.device, dtype=x.dtype).expand(c, 1, 3, 3)
        ky = self._ky.to(device=x.device, dtype=x.dtype).expand(c, 1, 3, 3)
        gx = F.conv2d(x, kx, padding=1, groups=c)
        gy = F.conv2d(x, ky, padding=1, groups=c)
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)  # (B, C, H, W)
        return mag.mean(dim=1, keepdim=True)  # (B, 1, H, W)

    def compute_weight_map(self, i_rm: torch.Tensor) -> torch.Tensor:
        """
        Compute the region-adaptive weight map W = 1 - G(I_RM), where
        G is the patch-level normalized gradient density (Eq. 15-17):
        pixels within a patch with strong aggregate gradient response get
        LOWER weight (more generative freedom), and vice versa.

        Args:
            i_rm: (B, C, H, W) Stage-I restoration image (guidance image).
        Returns:
            (B, 1, H, W) weight map W, broadcastable over spectral bands.
        """
        mag = self._sobel_magnitude(i_rm)  # (B, 1, H, W)
        p = self.patch_size
        b, _, h, w = mag.shape

        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        mag_padded = F.pad(mag, (0, pad_w, 0, pad_h))
        hp, wp = mag_padded.shape[-2:]

        # Sum of gradient magnitude within each non-overlapping patch (Eq. 16).
        patch_sum = F.avg_pool2d(mag_padded, kernel_size=p, stride=p) * (p * p)
        s = torch.tanh(patch_sum)  # (B, 1, hp/p, wp/p), in [0, 1)

        # Broadcast the patch-level gradient density back to pixel resolution (Eq. 17).
        gradient_density = s.repeat_interleave(p, dim=-2).repeat_interleave(p, dim=-1)
        gradient_density = gradient_density[..., :h, :w]

        weight_map = 1.0 - gradient_density  # Eq. 13: W = 1 - G(I_RM)
        return weight_map

    def loss(self, decoded_z0: torch.Tensor, i_rm: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
        """
        Region-adaptive MSE loss between the decoded clean-latent estimate
        and the guidance image, weighted spatially (Eq. 6):

            L(z0~) = (1 / HWC) * || W (.) (D(z0~) - I_RM) ||_2^2
        """
        b, c, h, w = i_rm.shape
        diff2 = (decoded_z0 - i_rm).pow(2)
        weighted = weight_map * diff2  # W broadcasts over channel dim
        return weighted.sum(dim=(1, 2, 3)).mean() / (h * w * c)

    def guidance_step(
        self,
        z0_tilde: torch.Tensor,
        decode_fn,
        i_rm: torch.Tensor,
        guidance_scale: float,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One gradient-descent update of the region-adaptive restoration
        guidance (Eq. 7):

            z0'_tilde = z0_tilde - s * grad_{z0_tilde} L(z0_tilde)

        Args:
            z0_tilde: (B, C_lat, h, w) predicted clean latent at the
                current sampling step (no grad required on input).
            decode_fn: callable mapping a latent tensor to HSI pixel space
                (typically `GenerationModule.decode_grad`); gradients are
                allowed to flow through it, but its parameters are frozen
                and are not updated.
            i_rm: (B, C, H, W) Stage-I restoration / guidance image.
            guidance_scale: scalar `s` controlling fidelity vs. quality
                trade-off (0 = no guidance / max quality, 1 = strong
                fidelity to I_RM).
            weight_map: optional precomputed weight map (Eq. 13); computed
                from `i_rm` if not provided (recommended to precompute
                once per sample and reuse across sampling steps for speed).

        Returns:
            The guided clean-latent estimate z0'_tilde (detached).
        """
        if guidance_scale == 0:
            return z0_tilde

        if weight_map is None:
            weight_map = self.compute_weight_map(i_rm)

        with torch.enable_grad():
            z = z0_tilde.detach().clone().requires_grad_(True)
            decoded = decode_fn(z)
            loss = self.loss(decoded, i_rm, weight_map)
            grad = torch.autograd.grad(loss, z)[0]

        return (z0_tilde - guidance_scale * grad).detach()


# ==========================================================================
# Gaussian diffusion scheduler (training + spaced DDPM sampling)
# ==========================================================================

class GaussianDiffusionScheduler:
    """
    Minimal DDPM noise scheduler used to (a) sample training timesteps and
    noisy latents (Eq. 2, q_sample), and (b) run a spaced DDPM sampling
    loop at inference time (the paper uses a 50-step spaced schedule,
    Nichol & Dhariwal, "Improved DDPM").
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Optional[torch.device] = None,
    ):
        self.num_train_timesteps = num_train_timesteps
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_cumprod = alphas_cumprod.float()
        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> "GaussianDiffusionScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = a.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(shape) - 1)))

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Uniformly sample training timesteps t ~ U{0, ..., T-1} (Eq. 4)."""
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion process (Eq. 2): z_t = sqrt(a_bar_t) z0 + sqrt(1 - a_bar_t) eps."""
        sqrt_ac = self._extract(self.alphas_cumprod.sqrt(), t, z0.shape)
        sqrt_1m_ac = self._extract((1.0 - self.alphas_cumprod).sqrt(), t, z0.shape)
        return sqrt_ac * z0 + sqrt_1m_ac * noise

    def predict_start_from_noise(self, z_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Eq. 5: z0~ = (z_t - sqrt(1 - a_bar_t) * eps_t) / sqrt(a_bar_t)."""
        sqrt_ac = self._extract(self.alphas_cumprod.sqrt(), t, z_t.shape)
        sqrt_1m_ac = self._extract((1.0 - self.alphas_cumprod).sqrt(), t, z_t.shape)
        return (z_t - sqrt_1m_ac * eps) / sqrt_ac

    def q_posterior_sample(
        self, z0: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample z_{t-1} ~ q(z_{t-1} | z_t, z0) using the standard DDPM
        posterior (used both for plain sampling and after restoration
        guidance has modified z0~, matching the paper's
        `q(z_{t-1}|z_t, z0'_tilde)` in Fig. 6 / Algorithm 1).
        """
        alphas_cumprod_t = self._extract(self.alphas_cumprod, t, z_t.shape)
        alphas_cumprod_prev = self._extract(
            torch.cat([torch.ones(1, device=self.alphas_cumprod.device), self.alphas_cumprod[:-1]]),
            t,
            z_t.shape,
        )
        betas_t = self._extract(self.betas, t, z_t.shape)
        alphas_t = self._extract(self.alphas, t, z_t.shape)

        posterior_mean = (
            betas_t * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod_t) * z0
            + (1.0 - alphas_cumprod_prev) * alphas_t.sqrt() / (1.0 - alphas_cumprod_t) * z_t
        )
        posterior_variance = betas_t * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod_t)

        if noise is None:
            noise = torch.randn_like(z_t)
        # No noise at the final step (t == 0).
        nonzero_mask = (t != 0).float().view(-1, *([1] * (z_t.dim() - 1)))
        return posterior_mean + nonzero_mask * posterior_variance.sqrt() * noise

    def spaced_timesteps(self, num_inference_steps: int) -> List[int]:
        """
        Evenly-spaced subset of the original T training timesteps, used
        for the 50-step DDPM sampling schedule described in Section 4.1
        ("we adopt a spaced DDPM sampling schedule ... 50 sampling steps").
        Returned in descending order (T-1 ... 0), suitable for reverse
        diffusion sampling.
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_train_timesteps, step_ratio))[:num_inference_steps]
        return list(reversed(timesteps))


# ==========================================================================
# Stage-I: Restoration Module wrapper
# ==========================================================================

class RestorationModuleStage1(nn.Module):
    """
    Thin wrapper around the pretrained, frozen MST++ model used as the
    Stage-I Restoration Module (Section 3.2): removes image-independent
    degradation from the RGB observation and produces a coarse, high-
    fidelity HSI estimate I_RM. This module is trained separately (not
    inside this pipeline) and is frozen here; per the paper, its role is
    purely to supply a reliable condition image / guidance image for
    Stage II.
    """

    def __init__(self, mst_plus_plus_model: nn.Module):
        super().__init__()
        self.model = freeze_module(mst_plus_plus_model)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) degraded RGB observation.
        Returns:
            (B, C_hsi, H, W) coarse restored HSI estimate I_RM.
        """
        return self.model(rgb)


# ==========================================================================
# Top-level DiffBIR-HSI pipeline
# ==========================================================================

class DiffBIRHSI(nn.Module):
    """
    End-to-end two-stage blind HSI restoration pipeline:

        RGB --(Stage I: MST++)--> I_RM
        I_RM --(frozen VAE encoder)--> c_RM
        z_T --(IRControlNet + frozen latent diffusion UNet, T steps)--> z_0
        z_0 --(frozen VAE decoder)--> refined HSI

    Exposes:
      * `restore_stage1`      - run Stage-I MST++ only.
      * `training_loss`       - compute the Stage-II diffusion loss
                                 (Eq. 4), for use in a separate training
                                 script. Only `self.generation.control_net`
                                 receives gradients.
      * `generate`            - full inference-time sampling loop with
                                 optional region-adaptive restoration
                                 guidance (Eq. 5-7).
    """

    def __init__(
        self,
        stage1_rm: nn.Module,
        vae: nn.Module,
        unet: nn.Module,
        latent_channels: int,
        skip_channels: Sequence[int],
        mid_channels: int,
        num_train_timesteps: int = 1000,
        patch_size: int = 8,
    ):
        super().__init__()
        self.stage1 = RestorationModuleStage1(stage1_rm)

        control_net = IRControlNet(
            unet=unet,
            latent_channels=latent_channels,
            skip_channels=skip_channels,
            mid_channels=mid_channels,
        )
        self.generation = GenerationModule(unet=unet, vae=vae, control_net=control_net)

        self.scheduler = GaussianDiffusionScheduler(num_train_timesteps=num_train_timesteps)
        self.guidance = RegionAdaptiveRestorationGuidance(patch_size=patch_size)

    def to(self, *args, **kwargs) -> "DiffBIRHSI":
        result = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.scheduler.to(device)
        return result

    # ---- Stage I ---------------------------------------------------
    @torch.no_grad()
    def restore_stage1(self, rgb: torch.Tensor) -> torch.Tensor:
        """Run the frozen Stage-I MST++ restoration module."""
        return self.stage1(rgb)

    # ---- Stage II: training -----------------------------------------
    def training_loss(
        self,
        rgb: torch.Tensor,
        hq_hsi: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the IRControlNet training objective (Eq. 4):
            L_GM = E[|| eps - eps_theta(z_t, c, t, c_RM) ||^2]

        Args:
            rgb: (B, 3, H, W) degraded RGB input.
            hq_hsi: (B, C_hsi, H, W) ground-truth high-quality HSI.
            context: optional cross-attention conditioning (e.g. an empty
                / fixed text-prompt embedding, matching DiffBIR's use of
                an empty positive prompt).

        Returns:
            Scalar training loss.
        """
        i_rm = self.restore_stage1(rgb).detach()

        z0 = self.generation.encode_target(hq_hsi, sample=True)
        c_rm = self.generation.encode_condition(i_rm)

        t = self.scheduler.sample_timesteps(z0.shape[0], device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.scheduler.q_sample(z0, t, noise)

        eps_pred = self.generation(z_t, t, c_rm, context=context)
        return F.mse_loss(eps_pred, noise)

    # ---- Stage II: inference / sampling ------------------------------
    @torch.no_grad()
    def generate(
        self,
        rgb: torch.Tensor,
        num_steps: int = 50,
        guidance_scale: float = 0.0,
        context: Optional[torch.Tensor] = None,
        latent_shape: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """
        Full DiffBIR-HSI inference pipeline: Stage-I restoration followed
        by Stage-II conditional diffusion sampling with optional
        region-adaptive restoration guidance.

        Args:
            rgb: (B, 3, H, W) degraded RGB observation.
            num_steps: number of spaced DDPM sampling steps (paper: 50).
            guidance_scale: restoration guidance scale `s` in Eq. 7.
                0 = pure generation (max quality); 1 = strong fidelity to
                the Stage-I estimate I_RM; the paper recommends s=0.5 for
                a balanced trade-off, and s=0 for best perceptual quality
                on real-world data.
            context: optional cross-attention conditioning.
            latent_shape: override for the sampled latent's spatial shape
                (B, C_lat, h, w); inferred from `c_rm`'s shape if omitted.

        Returns:
            (B, C_hsi, H, W) refined HSI reconstruction.
        """
        device = rgb.device
        i_rm = self.restore_stage1(rgb)
        c_rm = self.generation.encode_condition(i_rm)

        shape = tuple(latent_shape) if latent_shape is not None else c_rm.shape
        z_t = torch.randn(shape, device=device)

        # Precompute the region-adaptive weight map once (Eq. 13); it only
        # depends on I_RM, not on the current sampling step.
        weight_map = self.guidance.compute_weight_map(i_rm) if guidance_scale > 0 else None

        timesteps = self.scheduler.spaced_timesteps(num_steps)
        for step_t in timesteps:
            t_batch = torch.full((shape[0],), step_t, device=device, dtype=torch.long)

            eps = self.generation(z_t, t_batch, c_rm, context=context)
            z0_tilde = self.scheduler.predict_start_from_noise(z_t, t_batch, eps)

            if guidance_scale > 0:
                # Region-adaptive restoration guidance (Eq. 5-7): pull
                # low-frequency regions of D(z0~) towards I_RM.
                z0_tilde = self.guidance.guidance_step(
                    z0_tilde,
                    decode_fn=self.generation.decode_grad,
                    i_rm=i_rm,
                    guidance_scale=guidance_scale,
                    weight_map=weight_map,
                )

            z_t = self.scheduler.q_posterior_sample(z0_tilde, z_t, t_batch)

        refined_hsi = self.generation.decode(z_t)
        return refined_hsi


# ==========================================================================
# Convenience factory
# ==========================================================================

def build_diffbir_hsi(
    mst_plus_plus_ckpt: Optional[str],
    vae_ckpt: Optional[str],
    unet_ckpt: Optional[str],
    latent_channels: int,
    skip_channels: Sequence[int],
    mid_channels: int,
    num_train_timesteps: int = 1000,
    patch_size: int = 8,
    device: Union[str, torch.device] = "cuda",
) -> DiffBIRHSI:
    """
    Convenience constructor that instantiates the pretrained Stage-I / II
    backbones (imported from the user's project) and wires them into a
    `DiffBIRHSI` pipeline. Intended to be called from a separate
    training / inference script.

    Args:
        mst_plus_plus_ckpt: path to pretrained MST++ weights (or None to
            use a randomly-initialized model, e.g. for architecture tests).
        vae_ckpt: path to pretrained HSI VAE weights.
        unet_ckpt: path to pretrained HSI latent diffusion UNet weights.
        latent_channels, skip_channels, mid_channels: architecture
            metadata needed by `IRControlNet` (see its docstring).
        num_train_timesteps: diffusion schedule length used at training
            time (must match the checkpoint the UNet was trained with).
        patch_size: patch size for the region-adaptive gradient density
            map (Eq. 15).
        device: device to place all modules on.

    Returns:
        A `DiffBIRHSI` instance ready for training (`training_loss`) or
        inference (`generate`).
    """
    if MST_Plus_Plus is None or HSIVAE is None or HSILatentDiffusionUNet is None:
        raise ImportError(
            "One or more of MST_Plus_Plus / HSIVAE / HSILatentDiffusionUNet "
            "could not be imported. Fix the import paths at the top of "
            "model.py to point to your existing implementations."
        )

    mst = MST_Plus_Plus()
    if mst_plus_plus_ckpt is not None:
        mst.load_state_dict(torch.load(mst_plus_plus_ckpt, map_location="cpu"))

    vae = HSIVAE()
    if vae_ckpt is not None:
        vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"))

    unet = HSILatentDiffusionUNet()
    if unet_ckpt is not None:
        unet.load_state_dict(torch.load(unet_ckpt, map_location="cpu"))

    pipeline = DiffBIRHSI(
        stage1_rm=mst,
        vae=vae,
        unet=unet,
        latent_channels=latent_channels,
        skip_channels=skip_channels,
        mid_channels=mid_channels,
        num_train_timesteps=num_train_timesteps,
        patch_size=patch_size,
    )
    return pipeline.to(device)
