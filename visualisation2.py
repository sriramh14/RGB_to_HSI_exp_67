"""
Full-resolution RGB-to-HSI visualization using overlap-tiled latent diffusion.

This script:
1. Loads one full-resolution RGB image without resizing or center cropping.
2. Pads only on the right/bottom when required.
3. Runs the trained fixed-size RGB-conditioned latent diffusion model on
   overlapping tiles.
4. Blends predicted latent tiles.
5. Decodes the complete blended latent canvas once.
6. Saves:
      - predicted_hsi.npy       [C, H, W]
      - predicted_hsi.mat       variable: cube, shape [H, W, C]
      - visualization.png
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from PIL import Image

from models.DiT_adaptive_conditioning import RGB_to_HSI_w_diffusion


def load_rgb_image(path: str) -> Tuple[torch.Tensor, np.ndarray]:
    image = Image.open(path).convert("RGB")
    rgb_hwc = np.asarray(image, dtype=np.float32) / 255.0
    rgb_chw = np.transpose(rgb_hwc, (2, 0, 1))
    tensor = torch.from_numpy(rgb_chw.copy()).float().unsqueeze(0)
    return tensor, rgb_hwc


def extract_vae_state_dict(checkpoint: Dict) -> Dict:
    for key in ("vae_state_dict", "model_state_dict", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]

    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    raise KeyError(
        "Could not find a VAE state dictionary. Expected 'vae_state_dict', "
        "'model_state_dict', or 'state_dict'."
    )


def load_model(
    checkpoint_path: str,
    vae_checkpoint_path: str | None,
    device: torch.device,
) -> Tuple[RGB_to_HSI_w_diffusion, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("model_config", {})

    num_train_timesteps = int(config.get("num_train_timesteps", config.get("T", 1000)))

    model = RGB_to_HSI_w_diffusion(
        hsi_channels=int(config.get("hsi_channels", 31)),
        base_channels=int(config.get("base_channels", 64)),
        latent_channels=int(config.get("latent_channels", 16)),
        num_res_blocks=int(config.get("num_res_blocks", 2)),
        hidden_size=int(config.get("hidden_size", 128)),
        depth=int(config.get("depth", 12)),
        num_heads=int(config.get("num_heads", 4)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        learn_sigma=bool(config.get("learn_sigma", False)),
        patch_size=int(config.get("patch_size", 4)),
        input_size=int(config.get("input_size", 64)),
        T=num_train_timesteps,
    )

    if "dit_state_dict" not in checkpoint:
        raise KeyError("The diffusion checkpoint does not contain 'dit_state_dict'.")

    model.dit.load_state_dict(checkpoint["dit_state_dict"], strict=True)

    if "vae_state_dict" in checkpoint:
        model.vae.load_state_dict(checkpoint["vae_state_dict"], strict=True)
    else:
        if vae_checkpoint_path is None:
            raise ValueError(
                "The diffusion checkpoint does not contain VAE weights. "
                "Pass --vae-checkpoint."
            )
        vae_checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
        model.vae.load_state_dict(extract_vae_state_dict(vae_checkpoint), strict=True)

    # Some DiT implementations do not store input_size as an attribute.
    # The full-resolution tiler needs the trained latent tile size.
    if not hasattr(model.dit, "input_size"):
        model.dit.input_size = int(config.get("input_size", 64))

    model.vae.requires_grad_(False)
    model.eval()
    model.to(device)
    return model, config


def padded_length(length: int, tile_size: int, stride: int) -> int:
    if length <= tile_size:
        return tile_size
    number_of_strides = math.ceil((length - tile_size) / stride)
    return tile_size + number_of_strides * stride


def pad_full_rgb(
    rgb: torch.Tensor,
    tile_size: int,
    overlap: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB [1,3,H,W], got {tuple(rgb.shape)}.")

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size.")

    original_height, original_width = rgb.shape[-2:]
    padded_height = padded_length(original_height, tile_size, stride)
    padded_width = padded_length(original_width, tile_size, stride)

    rgb = F.pad(
        rgb,
        (0, padded_width - original_width, 0, padded_height - original_height),
        mode="replicate",
    )
    return rgb, (original_height, original_width)


def create_blending_window(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    window_y = torch.hann_window(
        height, periodic=False, device=device, dtype=dtype
    ).clamp_min(1e-3)
    window_x = torch.hann_window(
        width, periodic=False, device=device, dtype=dtype
    ).clamp_min(1e-3)
    return (window_y[:, None] * window_x[None, :]).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def sample_latent_tile(
    model: RGB_to_HSI_w_diffusion,
    rgb_tile: torch.Tensor,
    scheduler: DDPMScheduler,
    latent_channels: int,
    latent_size: int,
    num_inference_steps: int,
    generator: torch.Generator,
    use_amp: bool,
) -> torch.Tensor:
    device = rgb_tile.device

    latent = torch.randn(
        (1, latent_channels, latent_size, latent_size),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    scheduler.set_timesteps(num_inference_steps, device=device)

    for timestep in scheduler.timesteps:
        t_batch = torch.full(
            (1,), int(timestep.item()), device=device, dtype=torch.long
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            predicted_noise = model.dit(latent, t_batch, rgb_tile)

        if model.dit.learn_sigma:
            predicted_noise = predicted_noise[:, : model.dit.in_channels]

        latent = scheduler.step(
            model_output=predicted_noise.float(),
            timestep=timestep,
            sample=latent,
            generator=generator,
        ).prev_sample

    return latent


@torch.no_grad()
def predict_full_resolution(
    model: RGB_to_HSI_w_diffusion,
    rgb: torch.Tensor,
    scheduler: DDPMScheduler,
    tile_size: int,
    overlap: int,
    num_inference_steps: int,
    seed: int,
    use_amp: bool,
) -> torch.Tensor:
    device = rgb.device

    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size.")

    input_size = int(model.dit.input_size)
    latent_channels = int(model.dit.in_channels)

    if tile_size % input_size != 0:
        raise ValueError(
            f"tile_size={tile_size} must be divisible by latent input_size={input_size}."
        )

    vae_scale_factor = tile_size // input_size
    if overlap % vae_scale_factor != 0:
        raise ValueError(
            f"overlap={overlap} must be divisible by VAE scale factor={vae_scale_factor}."
        )

    stride = tile_size - overlap
    padded_rgb, original_shape = pad_full_rgb(rgb, tile_size, overlap)
    padded_height, padded_width = padded_rgb.shape[-2:]

    latent_height = padded_height // vae_scale_factor
    latent_width = padded_width // vae_scale_factor

    latent_sum = torch.zeros(
        (1, latent_channels, latent_height, latent_width),
        device=device,
        dtype=torch.float32,
    )
    weight_sum = torch.zeros(
        (1, 1, latent_height, latent_width),
        device=device,
        dtype=torch.float32,
    )

    latent_window = create_blending_window(
        input_size, input_size, device, torch.float32
    )

    y_positions = list(range(0, padded_height - tile_size + 1, stride))
    x_positions = list(range(0, padded_width - tile_size + 1, stride))
    total_tiles = len(y_positions) * len(x_positions)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    print(f"Original RGB: {original_shape[0]} x {original_shape[1]}")
    print(f"Padded RGB:   {padded_height} x {padded_width}")
    print(f"Tile size:    {tile_size}, overlap: {overlap}")
    print(f"Total tiles:  {total_tiles}")

    tile_index = 0
    for y in y_positions:
        for x in x_positions:
            tile_index += 1
            rgb_tile = padded_rgb[:, :, y : y + tile_size, x : x + tile_size]

            latent_tile = sample_latent_tile(
                model=model,
                rgb_tile=rgb_tile,
                scheduler=scheduler,
                latent_channels=latent_channels,
                latent_size=input_size,
                num_inference_steps=num_inference_steps,
                generator=generator,
                use_amp=use_amp,
            )

            latent_y = y // vae_scale_factor
            latent_x = x // vae_scale_factor

            latent_sum[
                :, :, latent_y : latent_y + input_size, latent_x : latent_x + input_size
            ] += latent_tile * latent_window

            weight_sum[
                :, :, latent_y : latent_y + input_size, latent_x : latent_x + input_size
            ] += latent_window

            print(f"\rSampling tile {tile_index}/{total_tiles}", end="", flush=True)

    print()

    full_latent = latent_sum / weight_sum.clamp_min(1e-8)

    if hasattr(model, "denormalize_latent"):
        full_latent = model.denormalize_latent(full_latent)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp and device.type == "cuda",
    ):
        predicted_hsi = model.vae.decode(full_latent)

    predicted_hsi = predicted_hsi.float().clamp(0.0, 1.0)
    original_height, original_width = original_shape
    return predicted_hsi[:, :, :original_height, :original_width]


def percentile_stretch(image: np.ndarray) -> np.ndarray:
    output = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        values = image[..., channel]
        low = np.percentile(values, 1.0)
        high = np.percentile(values, 99.0)
        output[..., channel] = np.clip(
            (values - low) / (high - low + 1e-8), 0.0, 1.0
        )
    return output


def make_pseudo_rgb(
    hsi_chw: np.ndarray,
    red_band: int,
    green_band: int,
    blue_band: int,
) -> np.ndarray:
    channels = hsi_chw.shape[0]
    for band in (red_band, green_band, blue_band):
        if band < 0 or band >= channels:
            raise ValueError(
                f"Band index {band} is invalid for a {channels}-band cube."
            )

    pseudo_rgb = np.stack(
        [hsi_chw[red_band], hsi_chw[green_band], hsi_chw[blue_band]],
        axis=-1,
    )
    return percentile_stretch(pseudo_rgb)


def save_visualization(
    rgb_hwc: np.ndarray,
    hsi_chw: np.ndarray,
    output_path: Path,
    red_band: int,
    green_band: int,
    blue_band: int,
) -> None:
    pseudo_rgb = make_pseudo_rgb(
        hsi_chw, red_band, green_band, blue_band
    )
    mean_spectrum = hsi_chw.mean(axis=(1, 2))

    figure = plt.figure(figsize=(16, 9))

    axis = figure.add_subplot(2, 3, 1)
    axis.imshow(np.clip(rgb_hwc, 0.0, 1.0))
    axis.set_title("Input full-resolution RGB")
    axis.axis("off")

    axis = figure.add_subplot(2, 3, 2)
    axis.imshow(pseudo_rgb)
    axis.set_title(
        f"Predicted HSI pseudo-RGB\nR/G/B bands: {red_band}/{green_band}/{blue_band}"
    )
    axis.axis("off")

    for plot_index, band_index in enumerate(
        [blue_band, green_band, red_band], start=3
    ):
        axis = figure.add_subplot(2, 3, plot_index)
        band_image = axis.imshow(hsi_chw[band_index], cmap="viridis")
        axis.set_title(f"Predicted band {band_index}")
        axis.axis("off")
        figure.colorbar(band_image, ax=axis, fraction=0.046, pad=0.04)

    axis = figure.add_subplot(2, 3, 6)
    axis.plot(np.arange(hsi_chw.shape[0]), mean_spectrum, marker="o", markersize=3)
    axis.set_title("Spatial mean spectrum")
    axis.set_xlabel("Band index")
    axis.set_ylabel("Mean reflectance")
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    # ============================================================
    # User-editable configuration
    # ============================================================
    RGB_PATH = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_RGB/Train_RGB/ARAD_1K_0003.jpg"
    CHECKPOINT_PATH = "./diffusion_checkpoints/best_diffusion.pth"
    VAE_CHECKPOINT_PATH = None  # Set only if needed
    OUTPUT_DIR = "./full_resolution_result"

    TILE_SIZE = 256
    OVERLAP = 64
    NUM_INFERENCE_STEPS = 100
    SEED = 42
    USE_AMP = True

    # Pseudo-RGB band selection for visualization
    RED_BAND = 20
    GREEN_BAND = 15
    BLUE_BAND = 5
    # ============================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = bool(
        USE_AMP and device.type == "cuda"
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Device: {device}")
    print(f"Mixed precision: {use_amp}")

    model, config = load_model(
        checkpoint_path=CHECKPOINT_PATH,
        vae_checkpoint_path=VAE_CHECKPOINT_PATH,
        device=device,
    )

    rgb, rgb_hwc = load_rgb_image(RGB_PATH)
    rgb = rgb.to(device)

    num_train_timesteps = int(
        config.get(
            "num_train_timesteps",
            config.get("T", 1000),
        )
    )

    beta_schedule = str(
        config.get(
            "beta_schedule",
            "squaredcos_cap_v2",
        )
    )

    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        prediction_type="epsilon",
        clip_sample=False,
    )

    predicted_hsi = predict_full_resolution(
        model=model,
        rgb=rgb,
        scheduler=scheduler,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
        use_amp=use_amp,
    )

    hsi_chw = (
        predicted_hsi[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    np.save(
        output_dir / "predicted_hsi.npy",
        hsi_chw,
    )

    sio.savemat(
        output_dir / "predicted_hsi.mat",
        {
            "cube": np.transpose(
                hsi_chw,
                (1, 2, 0),
            )
        },
    )

    save_visualization(
        rgb_hwc=rgb_hwc,
        hsi_chw=hsi_chw,
        output_path=output_dir / "visualization.png",
        red_band=RED_BAND,
        green_band=GREEN_BAND,
        blue_band=BLUE_BAND,
    )

    print(
        "Predicted HSI shape:",
        tuple(hsi_chw.shape),
    )
    print(
        "Saved:",
        output_dir / "predicted_hsi.npy",
    )
    print(
        "Saved:",
        output_dir / "predicted_hsi.mat",
    )
    print(
        "Saved:",
        output_dir / "visualization.png",
    )


if __name__ == "__main__":
    main()

