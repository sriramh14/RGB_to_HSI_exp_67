"""Visualize reconstructions produced by a trained hyperspectral VAE.

Before running:
    1. Replace the HSIVAE import below with the import used in your project.
    2. Set DATA_DIR and CHECKPOINT_PATH.
    3. Make sure MODEL_KWARGS, HSI_CHANNELS, PATCH_SIZE, and NORMALIZATION
       match the settings used during VAE training.

Expected VAE input:
    HSI tensor with shape [B, C, H, W].

Supported VAE forward outputs:
    reconstruction
    reconstruction, mu, logvar
    reconstruction, mu, logvar, latent
    or a dictionary containing equivalent tensors.

For every selected HSI file, the script creates:
    1. Ground-truth pseudo-RGB image
    2. VAE reconstruction pseudo-RGB image
    3. Mean absolute error map across spectral bands
    4. Ground-truth and reconstructed spectral signatures

It also saves MAE, MRAE, RMSE, PSNR, and SAM to a CSV file.
"""

from __future__ import annotations

import csv
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

# ============================================================
# Import your VAE class here
# ============================================================

# Replace this line with the actual import used in your project.
# The imported class must be named HSIVAE in this script.
from models.HSI_VAE import HSIVAE


# ============================================================
# Configuration
# ============================================================

DATA_DIR = (
    "/kaggle/input/datasets/sriramhari14/"
    "ntire-2022/Train_spectral"
)

CHECKPOINT_PATH = (
    "/kaggle/working/RGB_to_HSI_exp_67/"
    "vae_checkpoints/best_vae.pth"
)

OUTPUT_DIR = "./vae_reconstruction_visualizations"

# These arguments must match the VAE architecture used during training.
# When checkpoint["model_config"] exists, matching values stored there
# override the values below.
MODEL_KWARGS = {
    "hsi_channels": 31,
    "base_channels": 64,
    "latent_channels": 16,
    "num_res_blocks": 2,
}

HSI_CHANNELS = 31

# Name of the HSI variable inside MATLAB files.
HSI_KEY = "cube"

# Use None to process the complete HSI image.
# Use an integer when the VAE was trained on fixed-size patches.
PATCH_SIZE: Optional[int] = 64

NUM_IMAGES = 5

# Must match preprocessing used during VAE training:
# "none", "minmax", or "band_minmax".
NORMALIZATION = "none"

# Approximate pseudo-RGB bands for a 31-band 400-700 nm HSI cube.
# Values are zero-based indices in the order red, green, blue.
RGB_BANDS = (25, 15, 6)

RGB_LOW_PERCENTILE = 1.0
RGB_HIGH_PERCENTILE = 99.0

# Relative spatial coordinates used for spectral plots.
SPECTRAL_LOCATIONS = (
    (0.25, 0.25),
    (0.50, 0.50),
    (0.75, 0.75),
)

# Use 1.0 for data normalized to [0, 1].
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
# HSI file loading
# ============================================================


def find_hsi_files(data_dir: str) -> list[Path]:
    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root}"
        )

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise RuntimeError(
            f"No supported HSI files were found under {root}"
        )

    return files


def extract_3d_array(
    data: dict[str, Any],
    file_path: Path,
    preferred_key: Optional[str] = None,
) -> np.ndarray:
    """Extract a numerical 3D array from a loaded dictionary."""

    if preferred_key is not None and preferred_key in data:
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

        if (
            value.ndim == 3
            and np.issubdtype(value.dtype, np.number)
        ):
            candidates.append(value)

    if not candidates:
        raise ValueError(
            f"No numerical 3D HSI cube was found in {file_path}"
        )

    return max(candidates, key=lambda array: array.size)


def load_mat_v73(
    file_path: Path,
    preferred_key: Optional[str],
) -> np.ndarray:
    """Load a MATLAB v7.3 HSI file using h5py."""

    candidates: list[np.ndarray] = []

    with h5py.File(file_path, "r") as h5_file:
        if (
            preferred_key is not None
            and preferred_key in h5_file
            and isinstance(h5_file[preferred_key], h5py.Dataset)
        ):
            cube = np.asarray(h5_file[preferred_key])

        else:

            def visitor(_: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                    try:
                        array = np.asarray(obj)

                        if np.issubdtype(array.dtype, np.number):
                            candidates.append(array)

                    except (OSError, TypeError, ValueError):
                        pass

            h5_file.visititems(visitor)

            if not candidates:
                raise ValueError(
                    f"No numerical 3D dataset was found in {file_path}"
                )

            cube = max(candidates, key=lambda array: array.size)

    # h5py commonly exposes MATLAB dimensions in reverse order.
    return np.transpose(
        cube,
        tuple(range(cube.ndim - 1, -1, -1)),
    )


def load_hsi_file(file_path: Path) -> np.ndarray:
    suffix = file_path.suffix.lower()

    if suffix == ".mat":
        try:
            loaded = sio.loadmat(file_path)
            cube = extract_3d_array(
                loaded,
                file_path=file_path,
                preferred_key=HSI_KEY,
            )

        except (NotImplementedError, ValueError, OSError):
            cube = load_mat_v73(
                file_path=file_path,
                preferred_key=HSI_KEY,
            )

    elif suffix == ".npy":
        cube = np.load(file_path)

    elif suffix == ".npz":
        with np.load(file_path) as loaded:
            candidates = [
                loaded[key]
                for key in loaded.files
                if loaded[key].ndim == 3
            ]

        if not candidates:
            raise ValueError(
                f"No 3D array was found in {file_path}"
            )

        cube = max(candidates, key=lambda array: array.size)

    else:
        raise ValueError(
            f"Unsupported HSI file type: {file_path}"
        )

    cube = np.squeeze(
        np.asarray(cube, dtype=np.float32)
    )

    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D HSI cube in {file_path}, "
            f"but found shape {cube.shape}"
        )

    return cube


def convert_to_chw(
    cube: np.ndarray,
    channels: int,
    file_path: Path,
) -> np.ndarray:
    """Convert an HSI cube to [C, H, W]."""

    if cube.shape[0] == channels:
        return cube

    if cube.shape[-1] == channels:
        return np.transpose(cube, (2, 0, 1))

    if cube.shape[1] == channels:
        return np.transpose(cube, (1, 0, 2))

    raise ValueError(
        f"Cannot locate the {channels}-band spectral axis in "
        f"{file_path}. Cube shape: {cube.shape}"
    )


# ============================================================
# Preprocessing
# ============================================================


def normalize_cube(
    cube: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return cube

    if mode == "minmax":
        minimum = float(cube.min())
        maximum = float(cube.max())

        return (
            cube - minimum
        ) / (
            maximum - minimum + 1e-8
        )

    if mode == "band_minmax":
        minimum = cube.min(
            axis=(1, 2),
            keepdims=True,
        )

        maximum = cube.max(
            axis=(1, 2),
            keepdims=True,
        )

        return (
            cube - minimum
        ) / (
            maximum - minimum + 1e-8
        )

    raise ValueError(
        f"Unknown normalization mode: {mode}"
    )


def center_crop_or_pad(
    cube: torch.Tensor,
    patch_size: Optional[int],
) -> torch.Tensor:
    """Center-crop or replicate-pad a [C, H, W] tensor."""

    if patch_size is None:
        return cube

    _, height, width = cube.shape

    pad_height = max(0, patch_size - height)
    pad_width = max(0, patch_size - width)

    if pad_height > 0 or pad_width > 0:
        cube = F.pad(
            cube,
            (0, pad_width, 0, pad_height),
            mode="replicate",
        )

    _, height, width = cube.shape

    top = (height - patch_size) // 2
    left = (width - patch_size) // 2

    return cube[
        :,
        top : top + patch_size,
        left : left + patch_size,
    ]


# ============================================================
# VAE loading
# ============================================================


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    """Extract and clean a state dictionary from common checkpoint formats."""

    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The checkpoint must be a dictionary or torch.nn.Module."
        )

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "vae_state_dict",
        "vae",
    ):
        value = checkpoint.get(key)

        if isinstance(value, torch.nn.Module):
            checkpoint = value.state_dict()
            break

        if isinstance(value, dict) and value:
            checkpoint = value
            break

    cleaned: dict[str, torch.Tensor] = {}

    removable_prefixes = (
        "module.",
        "model.",
        "vae.",
    )

    for key, value in checkpoint.items():
        if not isinstance(key, str):
            continue

        if not isinstance(value, torch.Tensor):
            continue

        cleaned_key = key

        # Remove wrappers introduced by DataParallel or training containers.
        prefix_removed = True

        while prefix_removed:
            prefix_removed = False

            for prefix in removable_prefixes:
                if cleaned_key.startswith(prefix):
                    cleaned_key = cleaned_key[len(prefix) :]
                    prefix_removed = True

        cleaned[cleaned_key] = value

    if not cleaned:
        raise ValueError(
            "No tensor parameters were found in the checkpoint."
        )

    return cleaned


def get_constructor_kwargs(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Build valid constructor arguments for the imported HSIVAE class."""

    kwargs = dict(MODEL_KWARGS)

    checkpoint_config = checkpoint.get("model_config", {})

    if isinstance(checkpoint_config, dict):
        kwargs.update(checkpoint_config)

    signature = inspect.signature(HSIVAE.__init__)

    accepts_arbitrary_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_arbitrary_kwargs:
        return kwargs

    valid_argument_names = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    return {
        key: value
        for key, value in kwargs.items()
        if key in valid_argument_names
    }


def load_vae(
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if isinstance(checkpoint, dict):
        constructor_kwargs = get_constructor_kwargs(checkpoint)
    else:
        constructor_kwargs = dict(MODEL_KWARGS)

    model = HSIVAE(**constructor_kwargs)

    state_dict = extract_state_dict(checkpoint)

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as error:
        raise RuntimeError(
            "The checkpoint does not exactly match HSIVAE.\n"
            f"HSIVAE constructor arguments: {constructor_kwargs}\n\n"
            f"Original load_state_dict error:\n{error}"
        ) from error

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(f"Loaded checkpoint: {path}")
    print(f"VAE class: {HSIVAE.__name__}")
    print(f"Constructor arguments: {constructor_kwargs}")

    return model


# ============================================================
# VAE output handling
# ============================================================


def find_tensor(
    dictionary: dict[str, Any],
    names: tuple[str, ...],
) -> Optional[torch.Tensor]:
    for name in names:
        value = dictionary.get(name)

        if isinstance(value, torch.Tensor):
            return value

    return None


def unpack_vae_output(
    output: Any,
) -> tuple[
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
            output,
            (
                "reconstruction",
                "recon",
                "x_recon",
                "decoded",
                "output",
                "pred",
            ),
        )

        mu = find_tensor(
            output,
            ("mu", "mean", "posterior_mean"),
        )

        logvar = find_tensor(
            output,
            ("logvar", "log_var", "posterior_logvar"),
        )

        latent = find_tensor(
            output,
            ("latent", "z", "sample"),
        )

    elif isinstance(output, (tuple, list)):
        tensors = [
            value
            for value in output
            if isinstance(value, torch.Tensor)
        ]

        if len(tensors) >= 1:
            reconstruction = tensors[0]

        if len(tensors) >= 2:
            mu = tensors[1]

        if len(tensors) >= 3:
            logvar = tensors[2]

        if len(tensors) >= 4:
            latent = tensors[3]

    else:
        raise TypeError(
            f"Unsupported VAE output type: {type(output)}"
        )

    if reconstruction is None:
        raise ValueError(
            "Could not identify a reconstruction tensor "
            "in the VAE output."
        )

    return reconstruction, mu, logvar, latent


def run_vae(
    model: torch.nn.Module,
    hsi_batch: torch.Tensor,
) -> tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Run deterministic VAE inference when sample=False is supported."""

    forward_signature = inspect.signature(model.forward)

    if "sample" in forward_signature.parameters:
        output = model(
            hsi_batch,
            sample=False,
        )
    else:
        output = model(hsi_batch)

    return unpack_vae_output(output)


# ============================================================
# Metrics
# ============================================================


def calculate_metrics(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
) -> dict[str, float]:
    """Calculate HSI reconstruction metrics for [C, H, W] tensors."""

    target = target.float()
    reconstruction = reconstruction.float()

    difference = reconstruction - target

    mae = difference.abs().mean()

    mrae = (
        difference.abs()
        / (target.abs() + 1e-6)
    ).mean()

    mse = difference.square().mean()
    rmse = torch.sqrt(mse)

    data_range_squared = torch.tensor(
        DATA_RANGE**2,
        dtype=mse.dtype,
        device=mse.device,
    )

    psnr = 10.0 * torch.log10(
        data_range_squared / (mse + 1e-10)
    )

    # Treat each spatial location as one spectral vector.
    target_vectors = target.permute(
        1,
        2,
        0,
    ).reshape(-1, target.shape[0])

    reconstruction_vectors = reconstruction.permute(
        1,
        2,
        0,
    ).reshape(-1, reconstruction.shape[0])

    cosine_similarity = F.cosine_similarity(
        target_vectors,
        reconstruction_vectors,
        dim=1,
        eps=1e-8,
    )

    cosine_similarity = cosine_similarity.clamp(
        -1.0 + 1e-7,
        1.0 - 1e-7,
    )

    sam_degrees = torch.rad2deg(
        torch.acos(cosine_similarity)
    ).mean()

    return {
        "mae": float(mae.item()),
        "mrae": float(mrae.item()),
        "rmse": float(rmse.item()),
        "psnr": float(psnr.item()),
        "sam": float(sam_degrees.item()),
    }


# ============================================================
# Visualization
# ============================================================


def create_pseudo_rgb(
    cube: np.ndarray,
    low_values: Optional[np.ndarray] = None,
    high_values: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a contrast-stretched pseudo-RGB image from [C, H, W]."""

    for band_index in RGB_BANDS:
        if not 0 <= band_index < cube.shape[0]:
            raise IndexError(
                f"Pseudo-RGB band {band_index} is invalid "
                f"for a {cube.shape[0]}-band cube."
            )

    rgb = np.stack(
        [
            cube[RGB_BANDS[0]],
            cube[RGB_BANDS[1]],
            cube[RGB_BANDS[2]],
        ],
        axis=-1,
    )

    if low_values is None:
        low_values = np.percentile(
            rgb,
            RGB_LOW_PERCENTILE,
            axis=(0, 1),
            keepdims=True,
        )

    if high_values is None:
        high_values = np.percentile(
            rgb,
            RGB_HIGH_PERCENTILE,
            axis=(0, 1),
            keepdims=True,
        )

    rgb = (
        rgb - low_values
    ) / (
        high_values - low_values + 1e-8
    )

    return (
        np.clip(rgb, 0.0, 1.0),
        low_values,
        high_values,
    )


def plot_reconstruction(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    file_name: str,
    output_dir: Path,
) -> dict[str, float]:
    """Create and optionally save one reconstruction figure."""

    target_np = (
        target.detach()
        .cpu()
        .float()
        .numpy()
    )

    reconstruction_np = (
        reconstruction.detach()
        .cpu()
        .float()
        .numpy()
    )

    target_rgb, low_values, high_values = create_pseudo_rgb(
        target_np
    )

    reconstruction_rgb, _, _ = create_pseudo_rgb(
        reconstruction_np,
        low_values=low_values,
        high_values=high_values,
    )

    error_map = np.mean(
        np.abs(reconstruction_np - target_np),
        axis=0,
    )

    metrics = calculate_metrics(
        target=target,
        reconstruction=reconstruction,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
    )

    axes[0, 0].imshow(target_rgb)
    axes[0, 0].set_title("Original HSI pseudo-RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(reconstruction_rgb)
    axes[0, 1].set_title("VAE reconstruction pseudo-RGB")
    axes[0, 1].axis("off")

    error_image = axes[1, 0].imshow(
        error_map,
        cmap="magma",
    )

    axes[1, 0].set_title(
        "Mean absolute error over spectral bands"
    )
    axes[1, 0].axis("off")

    figure.colorbar(
        error_image,
        ax=axes[1, 0],
        fraction=0.046,
        pad=0.04,
    )

    height = target_np.shape[1]
    width = target_np.shape[2]
    band_axis = np.arange(target_np.shape[0])

    for point_index, (
        relative_y,
        relative_x,
    ) in enumerate(SPECTRAL_LOCATIONS, start=1):
        y = int(relative_y * (height - 1))
        x = int(relative_x * (width - 1))

        target_spectrum = target_np[:, y, x]
        reconstruction_spectrum = reconstruction_np[:, y, x]

        target_line = axes[1, 1].plot(
            band_axis,
            target_spectrum,
            linewidth=2,
            label=f"GT P{point_index} ({y},{x})",
        )[0]

        axes[1, 1].plot(
            band_axis,
            reconstruction_spectrum,
            linestyle="--",
            linewidth=2,
            color=target_line.get_color(),
            label=f"Recon P{point_index}",
        )

    axes[1, 1].set_title("Spectral signatures")
    axes[1, 1].set_xlabel("Spectral band index")
    axes[1, 1].set_ylabel("Intensity")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(fontsize=8, ncol=2)

    figure.suptitle(
        f"{file_name}\n"
        f"MRAE: {metrics['mrae']:.6f} | "
        f"RMSE: {metrics['rmse']:.6f} | "
        f"PSNR: {metrics['psnr']:.3f} dB | "
        f"SAM: {metrics['sam']:.3f}°",
        fontsize=13,
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.94)
    )

    if SAVE_FIGURES:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / f"{file_name}_vae_reconstruction.png"
        )

        figure.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        print(f"Saved visualization: {output_path}")

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
            f"mu shape: {tuple(mu.shape)} | "
            f"mean: {mu.mean().item():.6f} | "
            f"std: {mu.std(unbiased=False).item():.6f}"
        )

    if logvar is not None:
        print(
            f"logvar shape: {tuple(logvar.shape)} | "
            f"mean: {logvar.mean().item():.6f} | "
            f"std: {logvar.std(unbiased=False).item():.6f}"
        )

    if latent is not None:
        print(
            f"latent shape: {tuple(latent.shape)} | "
            f"mean: {latent.mean().item():.6f} | "
            f"std: {latent.std(unbiased=False).item():.6f}"
        )


# ============================================================
# Main
# ============================================================


@torch.inference_mode()
def main() -> None:
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    model = load_vae(
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    all_files = find_hsi_files(DATA_DIR)

    random_generator = random.Random(SEED)
    random_generator.shuffle(all_files)

    number_to_process = min(
        NUM_IMAGES,
        len(all_files),
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_rows: list[dict[str, Any]] = []

    for file_path in all_files:
        if len(metric_rows) >= number_to_process:
            break

        try:
            cube = load_hsi_file(file_path)

            cube = convert_to_chw(
                cube=cube,
                channels=HSI_CHANNELS,
                file_path=file_path,
            )

            if not np.isfinite(cube).all():
                raise ValueError(
                    "Cube contains NaN or infinite values."
                )

            cube = normalize_cube(
                cube=cube,
                mode=NORMALIZATION,
            )

            hsi = torch.from_numpy(
                np.ascontiguousarray(cube)
            ).float()

            hsi = center_crop_or_pad(
                cube=hsi,
                patch_size=PATCH_SIZE,
            )

            hsi_batch = hsi.unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            reconstruction, mu, logvar, latent = run_vae(
                model=model,
                hsi_batch=hsi_batch,
            )

            if reconstruction.ndim != 4:
                raise ValueError(
                    "Expected reconstruction shape [B, C, H, W], "
                    f"but found {tuple(reconstruction.shape)}"
                )

            if reconstruction.shape != hsi_batch.shape:
                raise ValueError(
                    "Input and reconstruction shapes differ: "
                    f"input={tuple(hsi_batch.shape)}, "
                    f"reconstruction={tuple(reconstruction.shape)}"
                )

            target = hsi_batch[0]
            reconstruction = reconstruction[0]

            print(f"\nFile: {file_path.name}")
            print(f"Input shape: {tuple(target.shape)}")
            print(
                "Reconstruction shape: "
                f"{tuple(reconstruction.shape)}"
            )

            print_latent_statistics(
                mu=mu,
                logvar=logvar,
                latent=latent,
            )

            metrics = plot_reconstruction(
                target=target,
                reconstruction=reconstruction,
                file_name=file_path.stem,
                output_dir=output_dir,
            )

            metric_rows.append(
                {
                    "file": file_path.name,
                    **metrics,
                }
            )

            print(
                f"MAE={metrics['mae']:.6f}, "
                f"MRAE={metrics['mrae']:.6f}, "
                f"RMSE={metrics['rmse']:.6f}, "
                f"PSNR={metrics['psnr']:.3f} dB, "
                f"SAM={metrics['sam']:.3f}°"
            )

        except Exception as error:
            print(
                f"\nSkipping file: {file_path}\n"
                f"Reason: {type(error).__name__}: {error}"
            )

    if not metric_rows:
        raise RuntimeError(
            "No HSI files were successfully visualized."
        )

    csv_path = (
        output_dir
        / "vae_visualization_metrics.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(metric_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(metric_rows)

    print(
        f"\nCompleted visualizations: {len(metric_rows)}"
    )
    print(f"Metrics CSV: {csv_path.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
