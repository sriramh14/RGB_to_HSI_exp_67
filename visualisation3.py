"""Visualize reconstruction quality of a trained hyperspectral VAE.

Edit only the configuration section before running.

Expected model behavior
-----------------------
The VAE receives an HSI tensor [B, C, H, W]. Its forward method may return:
    reconstruction
    (reconstruction, mu, logvar)
    (reconstruction, mu, logvar, latent)
    or a dictionary containing equivalent values.

The script creates, for each selected HSI cube:
    1. Original pseudo-RGB image
    2. VAE reconstruction pseudo-RGB image
    3. Mean absolute spectral-error map
    4. Original/reconstructed spectral curves at three pixels

It also prints reconstruction metrics and latent statistics when available.
"""

from __future__ import annotations

import csv
import importlib
import inspect
import random
from pathlib import Path
from typing import Any, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

from models.HSI_VAE import HSIVAE


# ============================================================
# Configuration: edit this section
# ============================================================

DATA_DIR = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_spectral"
CHECKPOINT_PATH = (
    "/kaggle/working/RGB_to_HSI_exp_67/vae_checkpoints/best_vae.pth"
)
OUTPUT_DIR = "./vae_reconstruction_visualizations"

# Python module and class containing the exact VAE architecture used in training.
# Example: if the class is defined as models/vae.py -> class HSIVAE(...), use:
MODEL_MODULE = "models.vae"
MODEL_CLASS = "HSIVAE"

# These values must match training. Values from checkpoint["model_config"]
# override matching entries below when available.
MODEL_KWARGS = {
    "hsi_channels": 31,
    "base_channels": 64,
    "latent_channels": 16,
    "num_res_blocks": 2,
}

HSI_CHANNELS = 31
HSI_KEY = "cube"

# Use None to visualize the full image. A fixed crop is safer when the VAE was
# trained on patches or expects dimensions divisible by its downsampling factor.
PATCH_SIZE: Optional[int] = 64
NUM_IMAGES = 5

# Must be identical to VAE training preprocessing:
# "none", "minmax", or "band_minmax".
NORMALIZATION = "none"

# Zero-based pseudo-RGB bands for a 31-band 400-700 nm cube.
RGB_BANDS = (25, 15, 6)
RGB_LOW_PERCENTILE = 1.0
RGB_HIGH_PERCENTILE = 99.0

# Relative (row, column) locations used for spectral plots.
SPECTRAL_LOCATIONS = (
    (0.25, 0.25),
    (0.50, 0.50),
    (0.75, 0.75),
)

DATA_RANGE = 1.0
SEED = 42
SAVE_FIGURES = True
SHOW_FIGURES = True

SUPPORTED_EXTENSIONS = {".mat", ".npy", ".npz"}


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
# HSI loading
# ============================================================


def find_hsi_files(data_dir: str) -> list[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No supported HSI files found under {root}")
    return files


def extract_3d_array(
    data: dict[str, Any],
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

    candidates: list[np.ndarray] = []
    for key, value in data.items():
        if str(key).startswith("__"):
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim == 3 and np.issubdtype(value.dtype, np.number):
            candidates.append(value)

    if not candidates:
        raise ValueError(f"No numerical 3D HSI cube found in {file_path}")
    return max(candidates, key=lambda array: array.size)


def load_mat_v73(file_path: Path, preferred_key: Optional[str]) -> np.ndarray:
    candidates: list[np.ndarray] = []

    with h5py.File(file_path, "r") as h5_file:
        if (
            preferred_key
            and preferred_key in h5_file
            and isinstance(h5_file[preferred_key], h5py.Dataset)
        ):
            cube = np.asarray(h5_file[preferred_key])
        else:

            def visitor(_: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                    array = np.asarray(obj)
                    if np.issubdtype(array.dtype, np.number):
                        candidates.append(array)

            h5_file.visititems(visitor)
            if not candidates:
                raise ValueError(f"No numerical 3D dataset found in {file_path}")
            cube = max(candidates, key=lambda array: array.size)

    # MATLAB v7.3 arrays are commonly exposed in reversed dimension order.
    return np.transpose(cube, tuple(range(cube.ndim - 1, -1, -1)))


def load_hsi_file(file_path: Path) -> np.ndarray:
    suffix = file_path.suffix.lower()

    if suffix == ".mat":
        try:
            cube = extract_3d_array(
                sio.loadmat(file_path),
                file_path=file_path,
                preferred_key=HSI_KEY,
            )
        except (NotImplementedError, ValueError, OSError):
            cube = load_mat_v73(file_path, preferred_key=HSI_KEY)
    elif suffix == ".npy":
        cube = np.load(file_path)
    elif suffix == ".npz":
        with np.load(file_path) as loaded:
            candidates = [loaded[key] for key in loaded.files if loaded[key].ndim == 3]
        if not candidates:
            raise ValueError(f"No 3D array found in {file_path}")
        cube = max(candidates, key=lambda array: array.size)
    else:
        raise ValueError(f"Unsupported HSI file: {file_path}")

    cube = np.squeeze(np.asarray(cube, dtype=np.float32))
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, found {cube.shape} in {file_path}")
    return cube


def convert_to_chw(cube: np.ndarray, channels: int, file_path: Path) -> np.ndarray:
    if cube.shape[0] == channels:
        return cube
    if cube.shape[-1] == channels:
        return np.transpose(cube, (2, 0, 1))
    if cube.shape[1] == channels:
        return np.transpose(cube, (1, 0, 2))
    raise ValueError(
        f"Cannot locate the {channels}-band spectral axis in {file_path}; "
        f"cube shape is {cube.shape}."
    )


def normalize_cube(cube: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return cube
    if mode == "minmax":
        minimum = float(cube.min())
        maximum = float(cube.max())
        return (cube - minimum) / (maximum - minimum + 1e-8)
    if mode == "band_minmax":
        minimum = cube.min(axis=(1, 2), keepdims=True)
        maximum = cube.max(axis=(1, 2), keepdims=True)
        return (cube - minimum) / (maximum - minimum + 1e-8)
    raise ValueError(f"Unknown normalization mode: {mode}")


def center_crop_or_pad(cube: torch.Tensor, patch_size: Optional[int]) -> torch.Tensor:
    if patch_size is None:
        return cube

    _, height, width = cube.shape
    pad_h = max(0, patch_size - height)
    pad_w = max(0, patch_size - width)
    if pad_h or pad_w:
        cube = F.pad(cube, (0, pad_w, 0, pad_h), mode="replicate")

    _, height, width = cube.shape
    top = (height - patch_size) // 2
    left = (width - patch_size) // 2
    return cube[:, top : top + patch_size, left : left + patch_size]


# ============================================================
# VAE loading and inference
# ============================================================


def import_model_class() -> type[torch.nn.Module]:
    module = importlib.import_module(MODEL_MODULE)
    model_class = getattr(module, MODEL_CLASS, None)
    if model_class is None:
        raise AttributeError(
            f"Class {MODEL_CLASS!r} was not found in module {MODEL_MODULE!r}."
        )
    return model_class


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint must contain a state-dict dictionary.")

    for key in ("model_state_dict", "state_dict", "model", "vae_state_dict", "vae"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            checkpoint = value
            break

    if not checkpoint or not all(isinstance(key, str) for key in checkpoint):
        raise ValueError("Could not locate a model state dictionary in the checkpoint.")

    cleaned: dict[str, torch.Tensor] = {}
    prefixes = ("module.", "model.", "vae.")
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value

    if not cleaned:
        raise ValueError("The extracted state dictionary contains no tensors.")
    return cleaned


def constructor_kwargs(
    model_class: type[torch.nn.Module], checkpoint: dict[str, Any]
) -> dict[str, Any]:
    kwargs = dict(MODEL_KWARGS)
    checkpoint_config = checkpoint.get("model_config", {})
    if isinstance(checkpoint_config, dict):
        kwargs.update(checkpoint_config)

    signature = inspect.signature(model_class.__init__)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs

    valid_names = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in valid_names}


def load_vae(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a dictionary checkpoint.")

    model_class = import_model_class()
    kwargs = constructor_kwargs(model_class, checkpoint)
    model = model_class(**kwargs)

    state_dict = extract_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The checkpoint does not exactly match the configured VAE architecture.\n"
            f"Module: {MODEL_MODULE}\nClass: {MODEL_CLASS}\n"
            f"Constructor arguments: {kwargs}\n\nOriginal error:\n{error}"
        ) from error

    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(f"Loaded checkpoint: {path}")
    print(f"VAE class: {MODEL_MODULE}.{MODEL_CLASS}")
    print(f"Constructor arguments: {kwargs}")
    return model


def find_tensor(dictionary: dict[str, Any], names: tuple[str, ...]) -> Optional[torch.Tensor]:
    for name in names:
        value = dictionary.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def unpack_vae_output(output: Any) -> tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    reconstruction: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
    latent: Optional[torch.Tensor] = None

    if isinstance(output, torch.Tensor):
        reconstruction = output
    elif isinstance(output, dict):
        reconstruction = find_tensor(
            output, ("reconstruction", "recon", "x_recon", "decoded", "output", "pred")
        )
        mu = find_tensor(output, ("mu", "mean", "posterior_mean"))
        logvar = find_tensor(output, ("logvar", "log_var", "posterior_logvar"))
        latent = find_tensor(output, ("latent", "z", "sample"))
    elif isinstance(output, (tuple, list)):
        tensors = [value for value in output if isinstance(value, torch.Tensor)]
        if tensors:
            reconstruction = tensors[0]
        if len(tensors) > 1:
            mu = tensors[1]
        if len(tensors) > 2:
            logvar = tensors[2]
        if len(tensors) > 3:
            latent = tensors[3]
    else:
        raise TypeError(f"Unsupported VAE output type: {type(output)}")

    if reconstruction is None:
        raise ValueError("Could not identify the reconstruction tensor in the VAE output.")
    return reconstruction, mu, logvar, latent


def run_vae(
    model: torch.nn.Module, hsi_batch: torch.Tensor
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    forward_signature = inspect.signature(model.forward)
    if "sample" in forward_signature.parameters:
        output = model(hsi_batch, sample=False)
    else:
        output = model(hsi_batch)
    return unpack_vae_output(output)


# ============================================================
# Metrics and visualization
# ============================================================


def calculate_metrics(target: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    target = target.float()
    reconstruction = reconstruction.float()
    difference = reconstruction - target

    mae = difference.abs().mean()
    mrae = (difference.abs() / (target.abs() + 1e-6)).mean()
    mse = difference.square().mean()
    rmse = mse.sqrt()
    psnr = 10.0 * torch.log10(
        torch.tensor(DATA_RANGE**2, device=mse.device) / (mse + 1e-10)
    )

    target_vectors = target.permute(1, 2, 0).reshape(-1, target.shape[0])
    recon_vectors = reconstruction.permute(1, 2, 0).reshape(-1, reconstruction.shape[0])
    cosine = F.cosine_similarity(target_vectors, recon_vectors, dim=1, eps=1e-8)
    sam = torch.rad2deg(torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))).mean()

    return {
        "mae": float(mae.item()),
        "mrae": float(mrae.item()),
        "rmse": float(rmse.item()),
        "psnr": float(psnr.item()),
        "sam": float(sam.item()),
    }


def create_pseudo_rgb(
    cube: np.ndarray,
    low_values: Optional[np.ndarray] = None,
    high_values: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for band in RGB_BANDS:
        if band < 0 or band >= cube.shape[0]:
            raise IndexError(f"Pseudo-RGB band {band} is invalid for {cube.shape[0]} bands.")

    rgb = np.stack([cube[RGB_BANDS[0]], cube[RGB_BANDS[1]], cube[RGB_BANDS[2]]], axis=-1)
    if low_values is None:
        low_values = np.percentile(
            rgb, RGB_LOW_PERCENTILE, axis=(0, 1), keepdims=True
        )
    if high_values is None:
        high_values = np.percentile(
            rgb, RGB_HIGH_PERCENTILE, axis=(0, 1), keepdims=True
        )
    rgb = (rgb - low_values) / (high_values - low_values + 1e-8)
    return np.clip(rgb, 0.0, 1.0), low_values, high_values


def plot_reconstruction(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    file_name: str,
    output_dir: Path,
) -> dict[str, float]:
    target_np = target.detach().cpu().float().numpy()
    reconstruction_np = reconstruction.detach().cpu().float().numpy()

    target_rgb, low, high = create_pseudo_rgb(target_np)
    reconstruction_rgb, _, _ = create_pseudo_rgb(reconstruction_np, low, high)
    error_map = np.mean(np.abs(reconstruction_np - target_np), axis=0)
    metrics = calculate_metrics(target, reconstruction)

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].imshow(target_rgb)
    axes[0, 0].set_title("Original HSI pseudo-RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(reconstruction_rgb)
    axes[0, 1].set_title("VAE reconstruction pseudo-RGB")
    axes[0, 1].axis("off")

    error_image = axes[1, 0].imshow(error_map, cmap="magma")
    axes[1, 0].set_title("Mean absolute error over spectral bands")
    axes[1, 0].axis("off")
    figure.colorbar(error_image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    height, width = target_np.shape[1:]
    band_axis = np.arange(target_np.shape[0])
    for index, (relative_y, relative_x) in enumerate(SPECTRAL_LOCATIONS, start=1):
        y = int(relative_y * (height - 1))
        x = int(relative_x * (width - 1))
        line = axes[1, 1].plot(
            band_axis,
            target_np[:, y, x],
            linewidth=2,
            label=f"GT P{index} ({y},{x})",
        )[0]
        axes[1, 1].plot(
            band_axis,
            reconstruction_np[:, y, x],
            linestyle="--",
            linewidth=2,
            color=line.get_color(),
            label=f"Recon P{index}",
        )

    axes[1, 1].set_title("Spectral signatures")
    axes[1, 1].set_xlabel("Band index")
    axes[1, 1].set_ylabel("Intensity")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(fontsize=8, ncol=2)

    figure.suptitle(
        f"{file_name}\n"
        f"MRAE: {metrics['mrae']:.6f} | RMSE: {metrics['rmse']:.6f} | "
        f"PSNR: {metrics['psnr']:.3f} dB | SAM: {metrics['sam']:.3f}°",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    if SAVE_FIGURES:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{file_name}_vae_reconstruction.png"
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if SHOW_FIGURES:
        plt.show()
    plt.close(figure)
    return metrics


def print_latent_statistics(
    mu: Optional[torch.Tensor],
    logvar: Optional[torch.Tensor],
    latent: Optional[torch.Tensor],
) -> None:
    if mu is not None:
        print(
            f"mu shape={tuple(mu.shape)}, mean={mu.mean().item():.6f}, "
            f"std={mu.std(unbiased=False).item():.6f}"
        )
    if logvar is not None:
        print(
            f"logvar shape={tuple(logvar.shape)}, mean={logvar.mean().item():.6f}, "
            f"std={logvar.std(unbiased=False).item():.6f}"
        )
    if latent is not None:
        print(
            f"latent shape={tuple(latent.shape)}, mean={latent.mean().item():.6f}, "
            f"std={latent.std(unbiased=False).item():.6f}"
        )


# ============================================================
# Main
# ============================================================


@torch.inference_mode()
def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_vae(CHECKPOINT_PATH, device)
    all_files = find_hsi_files(DATA_DIR)
    random.Random(SEED).shuffle(all_files)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []

    for file_path in all_files:
        if len(metric_rows) >= min(NUM_IMAGES, len(all_files)):
            break

        try:
            cube = load_hsi_file(file_path)
            cube = convert_to_chw(cube, HSI_CHANNELS, file_path)
            if not np.isfinite(cube).all():
                raise ValueError("Cube contains NaN or infinite values.")

            cube = normalize_cube(cube, NORMALIZATION)
            hsi = torch.from_numpy(np.ascontiguousarray(cube)).float()
            hsi = center_crop_or_pad(hsi, PATCH_SIZE)
            hsi_batch = hsi.unsqueeze(0).to(device, non_blocking=True)

            reconstruction, mu, logvar, latent = run_vae(model, hsi_batch)
            if reconstruction.ndim != 4:
                raise ValueError(
                    f"Expected reconstruction [B,C,H,W], found {tuple(reconstruction.shape)}"
                )
            if reconstruction.shape != hsi_batch.shape:
                raise ValueError(
                    f"Input/reconstruction shapes differ: {tuple(hsi_batch.shape)} vs "
                    f"{tuple(reconstruction.shape)}"
                )

            target = hsi_batch[0]
            reconstruction = reconstruction[0]

            print(f"\nFile: {file_path.name}")
            print(f"Input shape: {tuple(target.shape)}")
            print(f"Reconstruction shape: {tuple(reconstruction.shape)}")
            print_latent_statistics(mu, logvar, latent)

            metrics = plot_reconstruction(
                target=target,
                reconstruction=reconstruction,
                file_name=file_path.stem,
                output_dir=output_dir,
            )
            metric_rows.append({"file": file_path.name, **metrics})
            print(
                f"MAE={metrics['mae']:.6f}, MRAE={metrics['mrae']:.6f}, "
                f"RMSE={metrics['rmse']:.6f}, PSNR={metrics['psnr']:.3f} dB, "
                f"SAM={metrics['sam']:.3f}°"
            )

        except Exception as error:
            print(
                f"\nSkipping {file_path}\n"
                f"Reason: {type(error).__name__}: {error}"
            )

    if not metric_rows:
        raise RuntimeError("No HSI files were successfully visualized.")

    csv_path = output_dir / "vae_visualization_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"\nCompleted visualizations: {len(metric_rows)}")
    print(f"Metrics CSV: {csv_path.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
