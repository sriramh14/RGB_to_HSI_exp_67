"""Visualization script for the full RGB-to-HSI diffusion model.

Runs the complete DDPM reverse denoising loop conditioned on an RGB image,
decodes the final latent with the frozen VAE decoder, and plots the result
alongside the ground-truth HSI and an error map.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

from models.diffusion_DiT import RGB_to_HSI_w_diffusion


# ============================================================
# Configuration
# ============================================================

HSI_DATA_DIR = (
    "/kaggle/input/datasets/sriramhari14/"
    "ntire-2022/Train_spectral/Train_spectral"
)

RGB_DATA_DIR = (
    "/kaggle/input/datasets/sriramhari14/"
    "ntire-2022/Train_RGB/Train_RGB"
)

CHECKPOINT_PATH = (
    "./diffusion_checkpoints/best_diffusion.pth"
)

OUTPUT_DIR = "./diffusion_visualizations"

# ── Model architecture (must match checkpoint) ────────────────────────────────
HSI_CHANNELS    = 31
BASE_CHANNELS   = 64
LATENT_CHANNELS = 16
NUM_RES_BLOCKS  = 2
HIDDEN_SIZE     = 128
DEPTH           = 10
NUM_HEADS       = 16
MLP_RATIO       = 4.0
PATCH_SIZE      = 4
INPUT_SIZE      = 16
LEARN_SIGMA     = True

# ── Diffusion scheduler ───────────────────────────────────────────────────────
NUM_TRAIN_TIMESTEPS = 1000
BETA_SCHEDULE       = "squaredcos_cap_v2"

# Number of denoising steps at inference.
# Fewer steps = faster but lower quality. 200 is a good balance.
NUM_INFERENCE_STEPS = 200

# ── Data ──────────────────────────────────────────────────────────────────────
HSI_KEY           = "cube"
PATCH_SIZE_PX     = 64
NUM_IMAGES        = 5
NORMALIZATION     = "none"

SUPPORTED_HSI_EXTENSIONS = {".mat", ".npy", ".npz", ".pt", ".pth"}
SUPPORTED_RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".npy", ".pt", ".pth"}

# ── Visualization ─────────────────────────────────────────────────────────────
# Approximate red, green, blue band indices (0-based) for 31-band 400–700 nm data.
RGB_BANDS            = (25, 15, 6)
RGB_LOW_PERCENTILE   = 1.0
RGB_HIGH_PERCENTILE  = 99.0

SPECTRAL_LOCATIONS = (
    (0.25, 0.25),
    (0.50, 0.50),
    (0.75, 0.75),
)

SEED         = 42
SAVE_FIGURES = True
SHOW_FIGURES = True


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# File discovery and pairing
# ============================================================

def find_hsi_files(data_dir: str) -> list[Path]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"HSI directory does not exist: {data_path}")
    files = sorted(
        p for p in data_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_HSI_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No HSI files found under {data_path}")
    return files


def pair_hsi_rgb_files(
    hsi_dir: str,
    rgb_dir: str,
) -> list[tuple[Path, Path]]:
    """Match HSI and RGB files by filename stem."""
    hsi_files = find_hsi_files(hsi_dir)

    rgb_path = Path(rgb_dir)
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB directory does not exist: {rgb_path}")

    rgb_by_stem = {
        p.stem: p
        for p in rgb_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_RGB_EXTENSIONS
    }

    pairs, missing = [], []
    for hsi_file in hsi_files:
        rgb_file = rgb_by_stem.get(hsi_file.stem)
        if rgb_file is not None:
            pairs.append((hsi_file, rgb_file))
        else:
            missing.append(hsi_file)

    if missing:
        print(f"Warning: {len(missing)} HSI files have no matching RGB file.")

    if not pairs:
        raise RuntimeError(
            "No HSI/RGB pairs found. "
            "Check that both directories use matching filename stems."
        )

    print(f"Found {len(pairs)} paired HSI/RGB files.")
    return pairs


# ============================================================
# HSI loading
# ============================================================

def extract_array_from_dictionary(
    data: dict,
    file_path: Path,
    preferred_key: Optional[str] = None,
) -> np.ndarray:
    if preferred_key and preferred_key in data:
        value = data[preferred_key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim == 3:
            return value

    candidates = [
        v for k, v in data.items()
        if not k.startswith("__")
        and isinstance(v, np.ndarray)
        and v.ndim == 3
        and np.issubdtype(v.dtype, np.number)
    ]
    if not candidates:
        raise ValueError(f"No numerical 3D HSI cube found in {file_path}")
    return max(candidates, key=lambda a: a.size)


def load_mat_v73(file_path: Path, preferred_key: Optional[str] = None) -> np.ndarray:
    candidates = []
    with h5py.File(str(file_path), "r") as h5_file:
        if preferred_key and preferred_key in h5_file and isinstance(
            h5_file[preferred_key], h5py.Dataset
        ):
            cube = np.asarray(h5_file[preferred_key])
        else:
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                    try:
                        array = np.asarray(obj)
                        if np.issubdtype(array.dtype, np.number):
                            candidates.append((name, array))
                    except Exception:
                        pass
            h5_file.visititems(visitor)
            if not candidates:
                raise ValueError(f"No numerical 3D dataset found in {file_path}")
            _, cube = max(candidates, key=lambda item: item[1].size)

    return np.transpose(cube, axes=tuple(range(cube.ndim - 1, -1, -1)))


def load_hsi_file(file_path: Path) -> np.ndarray:
    ext = file_path.suffix.lower()
    if ext == ".mat":
        try:
            loaded = sio.loadmat(file_path)
            cube = extract_array_from_dictionary(loaded, file_path, HSI_KEY)
        except (NotImplementedError, ValueError, OSError):
            cube = load_mat_v73(file_path, HSI_KEY)
    elif ext == ".npy":
        cube = np.load(file_path)
    elif ext == ".npz":
        loaded = np.load(file_path)
        candidates = [loaded[k] for k in loaded.files if loaded[k].ndim == 3]
        if not candidates:
            raise ValueError(f"No 3D array found in {file_path}")
        cube = max(candidates, key=lambda a: a.size)
    elif ext in {".pt", ".pth"}:
        loaded = torch.load(file_path, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            cube = loaded.detach().cpu().numpy()
        elif isinstance(loaded, np.ndarray):
            cube = loaded
        elif isinstance(loaded, dict):
            cube = extract_array_from_dictionary(loaded, file_path, HSI_KEY)
        else:
            raise TypeError(f"Unsupported object in {file_path}: {type(loaded)}")
    else:
        raise ValueError(f"Unsupported HSI extension: {ext}")

    cube = np.asarray(cube, dtype=np.float32)
    cube = np.squeeze(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube in {file_path}, got shape {cube.shape}")
    return cube


def convert_to_chw(cube: np.ndarray, hsi_channels: int, file_path: Path) -> np.ndarray:
    if cube.shape[0] == hsi_channels:
        return cube
    if cube.shape[-1] == hsi_channels:
        return np.transpose(cube, (2, 0, 1))
    if cube.shape[1] == hsi_channels:
        return np.transpose(cube, (1, 0, 2))
    raise ValueError(
        f"Cannot locate the spectral dimension in {file_path}. "
        f"Shape: {cube.shape}; expected {hsi_channels} bands."
    )


def load_rgb_file(file_path: Path) -> np.ndarray:
    """Load RGB as float32 [3, H, W] in [0, 1]."""
    ext = file_path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg"}:
        from PIL import Image
        img = Image.open(file_path).convert("RGB")
        array = np.asarray(img, dtype=np.float32) / 255.0
        return np.transpose(array, (2, 0, 1))
    if ext == ".npy":
        array = np.load(file_path).astype(np.float32)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=0)
        elif array.shape[-1] == 3:
            array = np.transpose(array, (2, 0, 1))
        return array
    if ext in {".pt", ".pth"}:
        loaded = torch.load(file_path, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            array = loaded.float().numpy()
        else:
            raise TypeError(f"Unsupported object in RGB file {file_path}")
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=0)
        elif array.shape[-1] == 3:
            array = np.transpose(array, (2, 0, 1))
        return array
    raise ValueError(f"Unsupported RGB extension: {ext}")


# ============================================================
# Normalization and cropping
# ============================================================

def normalize_cube(cube: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return cube
    if mode == "minmax":
        lo, hi = cube.min(), cube.max()
        return (cube - lo) / (hi - lo + 1e-8)
    if mode == "band_minmax":
        lo = cube.min(axis=(1, 2), keepdims=True)
        hi = cube.max(axis=(1, 2), keepdims=True)
        return (cube - lo) / (hi - lo + 1e-8)
    raise ValueError(f"Unknown normalization mode: {mode}")


def center_crop_pair(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center-crop both modalities to [C, patch_size, patch_size]."""
    def _crop(t: torch.Tensor) -> torch.Tensor:
        _, h, w = t.shape
        ph = max(0, patch_size - h)
        pw = max(0, patch_size - w)
        if ph > 0 or pw > 0:
            t = F.pad(t, (0, pw, 0, ph), mode="replicate")
        _, h, w = t.shape
        top  = (h - patch_size) // 2
        left = (w - patch_size) // 2
        return t[:, top:top + patch_size, left:left + patch_size]

    return _crop(hsi), _crop(rgb)


# ============================================================
# Model loading
# ============================================================

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> RGB_to_HSI_w_diffusion:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    if not isinstance(ckpt, dict):
        raise TypeError("Expected the checkpoint to be a dictionary.")

    cfg = ckpt.get("model_config", {})

    model = RGB_to_HSI_w_diffusion(
        hsi_channels    = cfg.get("hsi_channels",    HSI_CHANNELS),
        base_channels   = cfg.get("base_channels",   BASE_CHANNELS),
        latent_channels = cfg.get("latent_channels", LATENT_CHANNELS),
        num_res_blocks  = cfg.get("num_res_blocks",  NUM_RES_BLOCKS),
        hidden_size     = cfg.get("hidden_size",     HIDDEN_SIZE),
        depth           = cfg.get("depth",           DEPTH),
        num_heads       = cfg.get("num_heads",       NUM_HEADS),
        mlp_ratio       = cfg.get("mlp_ratio",       MLP_RATIO),
        patch_size      = cfg.get("patch_size",      PATCH_SIZE),
        input_size      = cfg.get("input_size",      INPUT_SIZE),
        learn_sigma     = cfg.get("learn_sigma",     LEARN_SIGMA),
    ).to(device)

    if "dit_state_dict" not in ckpt or "vae_state_dict" not in ckpt:
        raise KeyError(
            "Checkpoint must contain both 'dit_state_dict' and 'vae_state_dict'. "
            "Re-save using the updated save_checkpoint() function."
        )

    model.dit.load_state_dict(ckpt["dit_state_dict"])
    model.vae.load_state_dict(ckpt["vae_state_dict"])
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    print(f"Loaded diffusion checkpoint: {ckpt_path}")
    epoch = ckpt.get("epoch", "unknown")
    val_loss = ckpt.get("validation_loss", float("nan"))
    print(f"  Epoch: {epoch} | Val noise loss: {val_loss:.6f}")

    return model


# ============================================================
# Inference — full DDPM reverse loop
# ============================================================

@torch.inference_mode()
def denoise(
    model: RGB_to_HSI_w_diffusion,
    rgb: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    num_inference_steps: int,
) -> torch.Tensor:
    """
    Run the full DDPM reverse denoising loop conditioned on rgb.

    Args:
        rgb: (1, 3, H, W) RGB condition image.

    Returns:
        hsi_recon: (1, HSI_CHANNELS, H, W) reconstructed HSI patch.
    """
    noise_scheduler.set_timesteps(num_inference_steps)

    # Determine latent spatial size from the VAE encoder's downsampling.
    # Two stride-2 downsamples: 64px → 16px.
    latent_h = rgb.shape[2] // 4
    latent_w = rgb.shape[3] // 4

    # Start from pure Gaussian noise in latent space.
    zt = torch.randn(
        1, model.dit.in_channels, latent_h, latent_w,
        device=device,
    )

    for t_scalar in noise_scheduler.timesteps:
        # Broadcast scalar timestep to batch dimension.
        t_batch = t_scalar.unsqueeze(0).to(device)

        # DiT predicts noise (and optionally sigma) from noisy latent + RGB.
        pred = model.dit(zt, t_batch, rgb)

        if model.dit.learn_sigma:
            # Only the first half of channels is the noise prediction.
            pred_noise = pred[:, :model.dit.in_channels]
        else:
            pred_noise = pred

        # Scheduler computes z_{t-1} from z_t and the predicted noise.
        scheduler_out = noise_scheduler.step(pred_noise, t_scalar, zt)
        zt = scheduler_out.prev_sample

    # Decode the final clean latent → HSI pixel space.
    hsi_recon = model.vae.decode(zt)
    hsi_recon = torch.clamp(hsi_recon, 0.0, 1.0)

    return hsi_recon


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
) -> dict:
    """Input tensors: [C, H, W]."""
    target         = target.float()
    reconstruction = reconstruction.float()

    abs_error  = torch.abs(reconstruction - target)
    mrae_value = torch.mean(abs_error / (torch.abs(target) + 1e-6))

    mse_value  = torch.mean((reconstruction - target).pow(2))
    rmse_value = torch.sqrt(mse_value)
    psnr_value = 10.0 * torch.log10(1.0 / (mse_value + 1e-10))

    # SAM — spectral angle mapper
    t_flat = target.permute(1, 2, 0).reshape(-1, target.shape[0])
    r_flat = reconstruction.permute(1, 2, 0).reshape(-1, reconstruction.shape[0])

    numerator   = torch.sum(t_flat * r_flat, dim=1)
    denominator = (
        torch.linalg.vector_norm(t_flat, dim=1)
        * torch.linalg.vector_norm(r_flat, dim=1)
    )
    cosine      = (numerator / (denominator + 1e-8)).clamp(-1 + 1e-7, 1 - 1e-7)
    sam_degrees = torch.mean(torch.acos(cosine)) * 180.0 / torch.pi

    return {
        "mrae": float(mrae_value.item()),
        "rmse": float(rmse_value.item()),
        "psnr": float(psnr_value.item()),
        "sam":  float(sam_degrees.item()),
    }


# ============================================================
# Visualization helpers
# ============================================================

def create_pseudo_rgb(
    cube: np.ndarray,
    rgb_bands: tuple[int, int, int],
    low_values:  Optional[np.ndarray] = None,
    high_values: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cube: [C, H, W]. Returns contrast-stretched [H, W, 3] image."""
    r, g, b = rgb_bands
    rgb = np.stack([cube[r], cube[g], cube[b]], axis=-1)

    if low_values is None:
        low_values  = np.percentile(rgb, RGB_LOW_PERCENTILE,  axis=(0, 1), keepdims=True)
    if high_values is None:
        high_values = np.percentile(rgb, RGB_HIGH_PERCENTILE, axis=(0, 1), keepdims=True)

    rgb = (rgb - low_values) / (high_values - low_values + 1e-8)
    return np.clip(rgb, 0.0, 1.0), low_values, high_values


def plot_result(
    target:         torch.Tensor,
    reconstruction: torch.Tensor,
    rgb_input:      torch.Tensor,
    file_name:      str,
    output_dir:     Path,
) -> None:
    """
    Four-panel figure:
      [0,0] Input RGB   [0,1] Ground-truth HSI pseudo-RGB
      [1,0] Prediction  [1,1] Spectral signatures
    Plus an error-map inset.
    """
    target_np  = target.cpu().float().numpy()
    pred_np    = reconstruction.cpu().float().numpy()
    rgb_np     = rgb_input.cpu().float().numpy().transpose(1, 2, 0)  # (H,W,3)
    rgb_np     = np.clip(rgb_np, 0.0, 1.0)

    target_display = np.clip(target_np, 0.0, 1.0)
    pred_display   = np.clip(pred_np,   0.0, 1.0)

    gt_rgb,   lo, hi = create_pseudo_rgb(target_display, RGB_BANDS)
    pred_rgb, _,  _  = create_pseudo_rgb(pred_display,   RGB_BANDS, lo, hi)

    error_map = np.mean(np.abs(pred_np - target_np), axis=0)
    metrics   = calculate_metrics(target, reconstruction)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── Row 0 ─────────────────────────────────────────────────────────────────
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title("Input RGB condition")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_rgb)
    axes[0, 1].set_title("Ground-truth HSI pseudo-RGB")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_rgb)
    axes[0, 2].set_title("Predicted HSI pseudo-RGB")
    axes[0, 2].axis("off")

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    err_img = axes[1, 0].imshow(error_map, cmap="magma")
    axes[1, 0].set_title("Mean absolute error (all bands)")
    axes[1, 0].axis("off")
    fig.colorbar(err_img, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Band-wise mean error
    band_error = np.mean(np.abs(pred_np - target_np), axis=(1, 2))
    axes[1, 1].bar(np.arange(len(band_error)), band_error, color="steelblue", alpha=0.8)
    axes[1, 1].set_title("Per-band mean absolute error")
    axes[1, 1].set_xlabel("Spectral band index")
    axes[1, 1].set_ylabel("Mean |error|")
    axes[1, 1].grid(alpha=0.3)

    # Spectral signatures at selected pixel locations
    h, w = target_np.shape[1], target_np.shape[2]
    band_axis = np.arange(target_np.shape[0])
    for loc_idx, (ry, rx) in enumerate(SPECTRAL_LOCATIONS, start=1):
        y = int(ry * (h - 1))
        x = int(rx * (w - 1))
        line = axes[1, 2].plot(
            band_axis, target_np[:, y, x],
            linewidth=2, label=f"GT point {loc_idx} ({y},{x})"
        )[0]
        axes[1, 2].plot(
            band_axis, pred_np[:, y, x],
            linestyle="--", linewidth=2,
            color=line.get_color(), label=f"Pred point {loc_idx}"
        )
    axes[1, 2].set_title("Spectral signatures")
    axes[1, 2].set_xlabel("Spectral band index")
    axes[1, 2].set_ylabel("Intensity")
    axes[1, 2].legend(fontsize=8, ncol=2)
    axes[1, 2].grid(alpha=0.3)

    fig.suptitle(
        f"{file_name}\n"
        f"MRAE: {metrics['mrae']:.6f} | "
        f"RMSE: {metrics['rmse']:.6f} | "
        f"PSNR: {metrics['psnr']:.3f} dB | "
        f"SAM: {metrics['sam']:.3f}°",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    if SAVE_FIGURES:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{file_name}_diffusion_prediction.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)

    print(
        f"Metrics for {file_name}: "
        f"MRAE={metrics['mrae']:.6f}, "
        f"RMSE={metrics['rmse']:.6f}, "
        f"PSNR={metrics['psnr']:.3f} dB, "
        f"SAM={metrics['sam']:.3f}°"
    )


# ============================================================
# Main
# ============================================================

@torch.inference_mode()
def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(CHECKPOINT_PATH, device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        beta_schedule=BETA_SCHEDULE,
    )

    all_pairs = pair_hsi_rgb_files(HSI_DATA_DIR, RGB_DATA_DIR)

    rng = random.Random(SEED)
    rng.shuffle(all_pairs)

    output_dir = Path(OUTPUT_DIR)
    processed  = 0

    for hsi_path, rgb_path in all_pairs:
        if processed >= NUM_IMAGES:
            break

        try:
            # ── Load HSI (ground truth) ───────────────────────────────────────
            cube = load_hsi_file(hsi_path)
            cube = convert_to_chw(cube, HSI_CHANNELS, hsi_path)

            if not np.isfinite(cube).all():
                raise ValueError("HSI cube contains NaN or Inf values.")

            cube = normalize_cube(cube, NORMALIZATION)
            hsi  = torch.from_numpy(np.ascontiguousarray(cube)).float()

            # ── Load RGB ──────────────────────────────────────────────────────
            rgb_array = load_rgb_file(rgb_path)
            rgb       = torch.from_numpy(np.ascontiguousarray(rgb_array)).float()

            # ── Crop to patch size ────────────────────────────────────────────
            hsi, rgb = center_crop_pair(hsi, rgb, PATCH_SIZE_PX)

            # ── Add batch dimension: [C,H,W] → [1,C,H,W] ─────────────────────
            hsi_batch = hsi.unsqueeze(0).to(device)
            rgb_batch = rgb.unsqueeze(0).to(device)

            print(f"\nFile: {hsi_path.name}")
            print(f"  HSI input shape: {tuple(hsi_batch.shape)}")
            print(f"  RGB input shape: {tuple(rgb_batch.shape)}")
            print(f"  Running {NUM_INFERENCE_STEPS}-step denoising loop...")

            # ── Full reverse diffusion → HSI reconstruction ───────────────────
            hsi_recon_batch = denoise(
                model=model,
                rgb=rgb_batch,
                noise_scheduler=noise_scheduler,
                device=device,
                num_inference_steps=NUM_INFERENCE_STEPS,
            )

            print(f"  Output shape: {tuple(hsi_recon_batch.shape)}")

            # Remove batch dim for plotting: [1,C,H,W] → [C,H,W]
            hsi_recon = hsi_recon_batch[0]
            target    = hsi_batch[0]

            plot_result(
                target=target,
                reconstruction=hsi_recon,
                rgb_input=rgb_batch[0],
                file_name=hsi_path.stem,
                output_dir=output_dir,
            )

            processed += 1

        except Exception as error:
            print(
                f"\nSkipping {hsi_path.name}: "
                f"{type(error).__name__}: {error}"
            )

    if processed == 0:
        raise RuntimeError("No files were successfully visualized.")

    print(f"\nCompleted: {processed} visualization(s) saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
