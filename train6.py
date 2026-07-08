"""Train an unconditional latent diffusion prior for HSI-VAE latents.

The script intentionally uses argparse only for ``--mode``. Edit all paths and
hyperparameters in the configuration section below.

Modes
-----
    python train_hsi_latent_diffusion_prior.py --mode train
    python train_hsi_latent_diffusion_prior.py --mode visualize
    python train_hsi_latent_diffusion_prior.py --mode train_visualize

What is trained
---------------
A latent diffusion U-Net that predicts the Gaussian noise added to clean HSI-VAE
latents. The HSI-VAE is frozen and used only for deterministic encoding and
image-space decoding during validation/visualisation.

Validation
----------
Validation samples a random timestep exactly like training, predicts the noise,
computes the noise MSE loss, computes timestep-noise MRAE, estimates the clean
latent from that one timestep, decodes it, and reports image metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
# Adjust only these imports if you place the files somewhere else.

from models.Unet_hsi import HSILatentDiffusionUNet

from models.HSI_VAE import HSIVAE



# =============================================================================
# Configuration: edit values here; argparse is used only for --mode
# =============================================================================

TRAIN_HSI_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Train_spectral/Train_spectral"
)
VALIDATION_HSI_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Valid_spectral/Valid_spectral"
)

OUTPUT_DIR = Path("./hsi_latent_diffusion_prior_checkpoints")
BEST_CHECKPOINT = OUTPUT_DIR / "best_hsi_latent_diffusion_prior.pth"
LAST_CHECKPOINT = OUTPUT_DIR / "last_hsi_latent_diffusion_prior.pth"
RESUME_CHECKPOINT: Optional[str] = None

VAE_CHECKPOINT = "./hsi_vae_checkpoints/best_hsi_vae.pth"
STRICT_VAE_CHECKPOINT = True
VAE_STATE_KEYS = (
    "model_state_dict",
    "vae_state_dict",
    "state_dict",
    "model",
    "params",
)

VISUALIZATION_CHECKPOINT = BEST_CHECKPOINT
VISUALIZATION_DIR = Path("./hsi_latent_diffusion_prior_visualizations")
VISUALIZATION_FILE = VISUALIZATION_DIR / "random_validation_one_step_reconstructions.png"

HSI_KEY = "cube"
HSI_CHANNELS = 31
LATENT_CHANNELS = 16
SUPPORTED_HSI_EXTENSIONS = {".npy", ".npz", ".mat", ".pt", ".pth"}
HSI_NORMALIZATION = "none"  # "none", "minmax", or "band_minmax"

TRAIN_PAIR_VALIDATION_CACHE = OUTPUT_DIR / "training_hsi_validation_cache.pth"
VALIDATION_PAIR_VALIDATION_CACHE = OUTPUT_DIR / "validation_hsi_validation_cache.pth"
LATENT_STATS_CACHE = OUTPUT_DIR / "hsi_vae_latent_stats.pth"

# VAE architecture. These must match the checkpoint that you load.
VAE_KWARGS: Dict[str, Any] = {
    "hsi_channels": HSI_CHANNELS,
    "base_channels": 64,
    "latent_channels": LATENT_CHANNELS,
    "num_res_blocks": 2,
}

# Latent diffusion prior architecture.
DENOISER_KWARGS: Dict[str, Any] = {
    "latent_channels": LATENT_CHANNELS,
    "base_channels": 64,
    "channel_multipliers": (1, 2, 4, 4),
    "num_res_blocks": 2,
    "attention_levels": (2, 3),
    "num_heads": 8,
    "dropout": 0.0,
}

# Diffusion schedule.
NUM_DIFFUSION_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 2e-2

# Dataset and training settings.
TRAIN_CROP_SIZE = 256
VALIDATION_CROP_SIZE = 256
PATCHES_PER_IMAGE = 2
USE_AUGMENTATION = True

# Your VAE downsamples by 4. The U-Net downsamples 3 more times, so crops and
# visualisation padding should be divisible by 4 * 8 = 32.
MODEL_DOWNSAMPLE_FACTOR = 32

BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 2
NUM_EPOCHS = 75
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MIN_LEARNING_RATE = 1e-6
GRADIENT_CLIP_NORM = 1.0
NUM_WORKERS = 4
USE_AMP = True
PREFER_BFLOAT16 = True
FP16_INITIAL_SCALE = 1024.0
FP16_GROWTH_INTERVAL = 2000
PRINT_EVERY = 30
SEED = 42

# Visualization settings.
NUM_VISUALIZATION_IMAGES = 5
VISUALIZATION_BANDS = (20, 10, 2)
VISUALIZATION_TIMESTEP_MODE = "random"  # "random" or "fixed"
VISUALIZATION_FIXED_TIMESTEP = 500
FIGURE_DPI = 180
CLAMP_RECONSTRUCTION: Optional[Tuple[float, float]] = None  # e.g. (0.0, 1.0)


# =============================================================================
# Reproducibility and AMP helpers
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_amp_dtype(device: torch.device) -> torch.dtype:
    if (
        device.type == "cuda"
        and PREFER_BFLOAT16
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=get_amp_dtype(device),
        enabled=True,
    )


def make_grad_scaler(device: torch.device, use_amp: bool):
    enabled = use_amp and device.type == "cuda" and get_amp_dtype(device) == torch.float16
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
            init_scale=FP16_INITIAL_SCALE,
            growth_interval=FP16_GROWTH_INTERVAL,
        )
    except (AttributeError, TypeError):
        from torch.cuda.amp import GradScaler

        return GradScaler(
            enabled=enabled,
            init_scale=FP16_INITIAL_SCALE,
            growth_interval=FP16_GROWTH_INTERVAL,
        )


# =============================================================================
# HSI loading
# =============================================================================


def _extract_3d_array_from_mapping(
    data: dict,
    file_path: Path,
    preferred_key: Optional[str] = None,
) -> np.ndarray:
    if preferred_key is not None and preferred_key in data:
        value = data[preferred_key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray) and value.ndim == 3:
            return value

    candidates: List[np.ndarray] = []
    for key, value in data.items():
        if str(key).startswith("__"):
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 3
            and np.issubdtype(value.dtype, np.number)
        ):
            candidates.append(value)

    if not candidates:
        raise ValueError(f"No numeric three-dimensional array was found in {file_path}.")
    return max(candidates, key=lambda array: array.size)


def load_mat_v73(
    file_path: Path,
    preferred_key: Optional[str] = None,
) -> np.ndarray:
    candidates: List[Tuple[str, np.ndarray]] = []

    with h5py.File(str(file_path), "r") as h5_file:
        if (
            preferred_key is not None
            and preferred_key in h5_file
            and isinstance(h5_file[preferred_key], h5py.Dataset)
            and h5_file[preferred_key].ndim == 3
        ):
            candidates.append((preferred_key, np.asarray(h5_file[preferred_key])))

        if not candidates:
            def visitor(name, obj):
                if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                    return
                try:
                    if np.issubdtype(obj.dtype, np.number):
                        candidates.append((name, np.asarray(obj)))
                except TypeError:
                    return

            h5_file.visititems(visitor)

    if not candidates:
        raise ValueError(f"No numeric three-dimensional HSI dataset was found in {file_path}.")

    _, cube = max(candidates, key=lambda item: item[1].size)
    # MATLAB v7.3/HDF5 arrays are commonly stored with reversed dimensions.
    return np.transpose(cube, axes=tuple(range(cube.ndim - 1, -1, -1)))


def load_hsi_file(file_path: Path) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension == ".npy":
        cube = np.load(file_path)
    elif extension == ".npz":
        with np.load(file_path) as loaded:
            candidates = [loaded[key] for key in loaded.files if loaded[key].ndim == 3]
            if not candidates:
                raise ValueError(f"No three-dimensional array was found in {file_path}.")
            cube = max(candidates, key=lambda array: array.size)
    elif extension == ".mat":
        try:
            loaded = sio.loadmat(file_path)
            cube = _extract_3d_array_from_mapping(
                loaded,
                file_path=file_path,
                preferred_key=HSI_KEY,
            )
        except (NotImplementedError, ValueError):
            cube = load_mat_v73(file_path=file_path, preferred_key=HSI_KEY)
    elif extension in {".pt", ".pth"}:
        try:
            loaded = torch.load(file_path, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(file_path, map_location="cpu")

        if isinstance(loaded, torch.Tensor):
            cube = loaded.detach().cpu().numpy()
        elif isinstance(loaded, np.ndarray):
            cube = loaded
        elif isinstance(loaded, dict):
            cube = _extract_3d_array_from_mapping(
                loaded,
                file_path=file_path,
                preferred_key=HSI_KEY,
            )
        else:
            raise TypeError(f"Unsupported object type in {file_path}: {type(loaded)}")
    else:
        raise ValueError(f"Unsupported HSI extension: {extension}")

    cube = np.asarray(cube, dtype=np.float32)
    cube = np.squeeze(cube)
    if cube.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional HSI cube in {file_path}, "
            f"but found shape {cube.shape}."
        )
    return cube


def convert_hsi_to_chw(
    cube: np.ndarray,
    hsi_channels: int,
    file_path: Path,
) -> np.ndarray:
    if cube.shape[0] == hsi_channels:
        return np.ascontiguousarray(cube)
    if cube.shape[-1] == hsi_channels:
        return np.ascontiguousarray(np.transpose(cube, (2, 0, 1)))
    if cube.shape[1] == hsi_channels:
        return np.ascontiguousarray(np.transpose(cube, (1, 0, 2)))
    raise ValueError(
        f"Could not identify the spectral axis in {file_path}. "
        f"Found shape {cube.shape}; expected {hsi_channels} bands."
    )


def normalize_hsi_cube(cube: np.ndarray, mode: str) -> np.ndarray:
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
    raise ValueError(f"Unknown HSI normalization mode: {mode}")


# =============================================================================
# File discovery, metadata checking, and cache
# =============================================================================


def find_files(
    directory: str,
    extensions: Sequence[str],
    kind: str,
) -> List[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"{kind} directory does not exist: {root}")

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not files:
        raise RuntimeError(f"No supported {kind} files were found in {root}.")
    return files


def make_files_fingerprint(files: Sequence[Path]) -> str:
    records = []
    for file_path in files:
        stat = file_path.stat()
        records.append(f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def is_possible_hsi_shape(shape: Sequence[int], hsi_channels: int) -> bool:
    return (
        len(shape) == 3
        and hsi_channels in shape
        and all(int(size) > 0 for size in shape)
    )


def inspect_hdf5_mat_file(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    candidates: List[Tuple[str, Tuple[int, ...]]] = []

    with h5py.File(str(file_path), "r") as h5_file:
        if hsi_key in h5_file and isinstance(h5_file[hsi_key], h5py.Dataset):
            dataset = h5_file[hsi_key]
            candidates.append((hsi_key, tuple(int(value) for value in dataset.shape)))
        else:
            def visitor(name, obj):
                if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                    return
                try:
                    if np.issubdtype(obj.dtype, np.number):
                        candidates.append((name, tuple(int(value) for value in obj.shape)))
                except TypeError:
                    return

            h5_file.visititems(visitor)

    if not candidates:
        raise ValueError(f"No numerical three-dimensional dataset was found in {file_path}.")
    if not any(is_possible_hsi_shape(shape, hsi_channels) for _, shape in candidates):
        raise ValueError(
            f"No {hsi_channels}-band cube was found in {file_path}. "
            f"HDF5 datasets: {candidates}"
        )


def inspect_standard_mat_file(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    try:
        metadata = sio.whosmat(file_path)
    except (NotImplementedError, ValueError, OSError):
        inspect_hdf5_mat_file(
            file_path=file_path,
            hsi_channels=hsi_channels,
            hsi_key=hsi_key,
        )
        return

    candidates = [
        (name, tuple(int(value) for value in shape))
        for name, shape, _ in metadata
        if len(shape) == 3
    ]
    if not candidates:
        raise ValueError(f"No three-dimensional array was found in {file_path}.")

    preferred = [candidate for candidate in candidates if candidate[0] == hsi_key]
    arrays_to_check = preferred if preferred else candidates
    if not any(is_possible_hsi_shape(shape, hsi_channels) for _, shape in arrays_to_check):
        raise ValueError(
            f"No {hsi_channels}-band cube was found in {file_path}. "
            f"MATLAB arrays: {candidates}"
        )


def inspect_hsi_file_metadata(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    if file_path.suffix.lower() == ".mat":
        inspect_standard_mat_file(
            file_path=file_path,
            hsi_channels=hsi_channels,
            hsi_key=hsi_key,
        )
        return

    cube = load_hsi_file(file_path)
    if not is_possible_hsi_shape(cube.shape, hsi_channels):
        raise ValueError(f"Invalid HSI shape {cube.shape} in {file_path}.")


def filter_valid_hsi_files(
    files: Sequence[Path],
    hsi_channels: int,
    log_path: Path,
    cache_path: Path,
) -> List[Path]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(files)
    fingerprint = make_files_fingerprint(files)
    file_lookup = {str(path.resolve()): path for path in files}

    if cache_path.exists():
        try:
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            except TypeError:
                cached = torch.load(cache_path, map_location="cpu")

            if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
                valid_paths = cached.get("valid_hsi_paths", [])
                invalid_records = cached.get("invalid_records", [])
                valid_files = [file_lookup[path] for path in valid_paths if path in file_lookup]
                print(f"\nUsing cached HSI validation: {cache_path}")
                print(f"Valid files: {len(valid_files)} | Invalid: {len(invalid_records)}")
                if valid_files:
                    return valid_files
        except Exception as error:
            print(
                "\nCould not use the validation cache. "
                f"The dataset will be checked again. Reason: {error}"
            )

    print("\nChecking HSI file metadata before use...")
    valid_files: List[Path] = []
    invalid_records: List[dict] = []

    for index, hsi_path in enumerate(files, start=1):
        try:
            inspect_hsi_file_metadata(
                file_path=hsi_path,
                hsi_channels=hsi_channels,
                hsi_key=HSI_KEY,
            )
            valid_files.append(hsi_path)
        except Exception as error:
            invalid_records.append(
                {
                    "path": str(hsi_path.resolve()),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                "\nSkipping invalid HSI file:\n"
                f"  File: {hsi_path}\n  Error: {error}"
            )

        if index % 100 == 0 or index == len(files):
            print(
                f"Checked {index}/{len(files)} | "
                f"Valid: {len(valid_files)} | Invalid: {len(invalid_records)}"
            )

    if not valid_files:
        raise RuntimeError("No valid HSI files remain after metadata validation.")

    if invalid_records:
        with log_path.open("w", encoding="utf-8") as log_file:
            for record in invalid_records:
                log_file.write(f"{record['path']} | {record['error']}\n")
        print(f"Invalid-file log saved to: {log_path}")

    torch.save(
        {
            "fingerprint": fingerprint,
            "valid_hsi_paths": [str(path.resolve()) for path in valid_files],
            "invalid_records": invalid_records,
        },
        cache_path,
    )
    print(f"Validation cache saved to: {cache_path}")
    return valid_files


# =============================================================================
# Spatial transforms and dataset
# =============================================================================


def _pad_tensor_to_minimum_size(
    tensor: torch.Tensor,
    minimum_height: int,
    minimum_width: int,
) -> torch.Tensor:
    _, height, width = tensor.shape
    pad_height = max(0, minimum_height - height)
    pad_width = max(0, minimum_width - width)
    if pad_height == 0 and pad_width == 0:
        return tensor
    return F.pad(tensor, (0, pad_width, 0, pad_height), mode="replicate")


def random_crop_hsi(
    hsi: torch.Tensor,
    crop_size: int,
) -> torch.Tensor:
    hsi = _pad_tensor_to_minimum_size(hsi, crop_size, crop_size)
    _, height, width = hsi.shape
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    return hsi[:, top:top + crop_size, left:left + crop_size]


def center_crop_hsi(
    hsi: torch.Tensor,
    crop_size: int,
) -> torch.Tensor:
    hsi = _pad_tensor_to_minimum_size(hsi, crop_size, crop_size)
    _, height, width = hsi.shape
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return hsi[:, top:top + crop_size, left:left + crop_size]


def augment_hsi(hsi: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        hsi = torch.flip(hsi, dims=[1])
    if random.random() < 0.5:
        hsi = torch.flip(hsi, dims=[2])
    rotations = random.randint(0, 3)
    if rotations:
        hsi = torch.rot90(hsi, k=rotations, dims=(1, 2))
    return hsi.contiguous()


def pad_hsi_to_multiple(
    hsi: torch.Tensor,
    multiple: int,
) -> Tuple[torch.Tensor, int, int]:
    _, original_height, original_width = hsi.shape
    pad_height = (multiple - original_height % multiple) % multiple
    pad_width = (multiple - original_width % multiple) % multiple

    if pad_height == 0 and pad_width == 0:
        return hsi, original_height, original_width

    hsi = F.pad(hsi, (0, pad_width, 0, pad_height), mode="replicate")
    return hsi, original_height, original_width


class HSIDataset(Dataset):
    def __init__(
        self,
        hsi_files: Sequence[Path],
        hsi_channels: int,
        crop_size: Optional[int],
        patches_per_image: int,
        training: bool,
        normalization: str,
        augment: bool,
        return_paths: bool = False,
    ) -> None:
        self.hsi_files = list(hsi_files)
        self.hsi_channels = hsi_channels
        self.crop_size = crop_size
        self.patches_per_image = patches_per_image
        self.training = training
        self.normalization = normalization
        self.augment = augment
        self.return_paths = return_paths

        if training and crop_size is None:
            raise ValueError("Training requires a finite crop_size.")
        if patches_per_image < 1:
            raise ValueError("patches_per_image must be at least 1.")

    def __len__(self) -> int:
        multiplier = self.patches_per_image if self.training else 1
        return len(self.hsi_files) * multiplier

    def _load_hsi(self, file_index: int) -> Tuple[torch.Tensor, Path]:
        hsi_path = self.hsi_files[file_index]
        hsi_array = convert_hsi_to_chw(
            load_hsi_file(hsi_path),
            hsi_channels=self.hsi_channels,
            file_path=hsi_path,
        )
        hsi_array = normalize_hsi_cube(hsi_array, mode=self.normalization)

        if not np.isfinite(hsi_array).all():
            raise ValueError(f"HSI contains NaN/Inf: {hsi_path}")

        hsi = torch.from_numpy(hsi_array.copy()).float()
        return hsi, hsi_path

    def __getitem__(self, index: int):
        file_index = index // self.patches_per_image if self.training else index
        hsi, hsi_path = self._load_hsi(file_index)

        if self.crop_size is not None:
            if self.training:
                hsi = random_crop_hsi(hsi, self.crop_size)
            else:
                hsi = center_crop_hsi(hsi, self.crop_size)

        if self.training and self.augment:
            hsi = augment_hsi(hsi)

        if self.return_paths:
            return hsi, str(hsi_path)
        return hsi


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
        generator=generator,
    )


# =============================================================================
# Diffusion schedule
# =============================================================================


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()

        betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.num_timesteps = num_timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars),
        )

    @staticmethod
    def extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        target_shape: torch.Size | Tuple[int, ...],
    ) -> torch.Tensor:
        extracted = values.gather(0, timesteps)
        return extracted.reshape(
            timesteps.shape[0],
            *((1,) * (len(target_shape) - 1)),
        )

    def add_noise(
        self,
        z0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_bar = self.extract(
            self.sqrt_alpha_bars,
            timesteps,
            z0.shape,
        )
        sqrt_one_minus_alpha_bar = self.extract(
            self.sqrt_one_minus_alpha_bars,
            timesteps,
            z0.shape,
        )
        return sqrt_alpha_bar * z0 + sqrt_one_minus_alpha_bar * noise

    def predict_clean_from_noise(
        self,
        z_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_bar = self.extract(
            self.sqrt_alpha_bars,
            timesteps,
            z_t.shape,
        )
        sqrt_one_minus_alpha_bar = self.extract(
            self.sqrt_one_minus_alpha_bars,
            timesteps,
            z_t.shape,
        )
        return (
            z_t - sqrt_one_minus_alpha_bar * predicted_noise
        ) / sqrt_alpha_bar.clamp_min(1e-8)


# =============================================================================
# Model, VAE, latent helpers, and checkpoints
# =============================================================================


def load_torch_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def strip_prefix_if_present(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def extract_state_dict(
    checkpoint: object,
    candidate_keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in candidate_keys:
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(
                torch.is_tensor(tensor) for tensor in value.values()
            ):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise KeyError(f"Could not find a state_dict using keys: {tuple(candidate_keys)}")


def normalize_vae_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    state_dict = strip_prefix_if_present(state_dict, "module.")
    state_dict = strip_prefix_if_present(state_dict, "vae.")
    state_dict = strip_prefix_if_present(state_dict, "model.")
    return state_dict


def build_vae(device: torch.device) -> HSIVAE:
    vae = HSIVAE(**VAE_KWARGS)
    checkpoint = load_torch_checkpoint(VAE_CHECKPOINT, device="cpu")
    state_dict = extract_state_dict(checkpoint, VAE_STATE_KEYS)
    state_dict = normalize_vae_state_dict(state_dict)
    incompatible = vae.load_state_dict(state_dict, strict=STRICT_VAE_CHECKPOINT)
    if not STRICT_VAE_CHECKPOINT:
        print(
            "Loaded VAE non-strictly | "
            f"missing={len(incompatible.missing_keys)} | "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    print(f"Loaded frozen HSI-VAE checkpoint: {VAE_CHECKPOINT}")
    return vae


def build_denoiser(device: torch.device) -> HSILatentDiffusionUNet:
    model = HSILatentDiffusionUNet(**DENOISER_KWARGS)
    return model.to(device)


@torch.no_grad()
def encode_hsi_mean(
    vae: HSIVAE,
    hsi: torch.Tensor,
) -> torch.Tensor:
    # Your VAE returns z, mu, logvar. We use sample=False so z == mu.
    encoded = vae.encode(hsi, sample=False)
    if isinstance(encoded, tuple):
        if len(encoded) == 3:
            z, mu, logvar = encoded
            return mu
        return encoded[0]
    if hasattr(encoded, "latent_dist"):
        return encoded.latent_dist.mode()
    if hasattr(encoded, "mode"):
        return encoded.mode()
    return encoded


def normalize_latent(
    z: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return (z - latent_mean) / latent_std.clamp_min(eps)


def denormalize_latent(
    z: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> torch.Tensor:
    return z * latent_std + latent_mean


@torch.no_grad()
def decode_latent(
    vae: HSIVAE,
    z_norm: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> torch.Tensor:
    z = denormalize_latent(z_norm, latent_mean, latent_std)
    reconstruction = vae.decode(z)
    if hasattr(reconstruction, "sample"):
        reconstruction = reconstruction.sample
    if CLAMP_RECONSTRUCTION is not None:
        low, high = CLAMP_RECONSTRUCTION
        reconstruction = reconstruction.clamp(low, high)
    return reconstruction


@torch.no_grad()
def load_latent_stats_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    checkpoint = load_torch_checkpoint(checkpoint_path, device="cpu")
    if isinstance(checkpoint, dict) and "latent_mean" in checkpoint and "latent_std" in checkpoint:
        latent_mean = checkpoint["latent_mean"].float().to(device).view(1, -1, 1, 1)
        latent_std = checkpoint["latent_std"].float().to(device).view(1, -1, 1, 1)
        if latent_mean.shape[1] != LATENT_CHANNELS or latent_std.shape[1] != LATENT_CHANNELS:
            raise ValueError(
                "Checkpoint latent statistics do not match LATENT_CHANNELS: "
                f"mean={tuple(latent_mean.shape)}, std={tuple(latent_std.shape)}"
            )
        print("Loaded latent statistics from VAE checkpoint.")
        return latent_mean, latent_std
    return None


@torch.no_grad()
def compute_latent_statistics(
    vae: HSIVAE,
    files: Sequence[Path],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if LATENT_STATS_CACHE.exists():
        cached = load_torch_checkpoint(LATENT_STATS_CACHE, device="cpu")
        if isinstance(cached, dict) and "latent_mean" in cached and "latent_std" in cached:
            print(f"Loaded cached latent statistics: {LATENT_STATS_CACHE}")
            return (
                cached["latent_mean"].float().to(device).view(1, -1, 1, 1),
                cached["latent_std"].float().to(device).view(1, -1, 1, 1),
            )

    print("\nComputing HSI-VAE latent statistics from the training set...")
    stats_dataset = HSIDataset(
        hsi_files=files,
        hsi_channels=HSI_CHANNELS,
        crop_size=TRAIN_CROP_SIZE,
        patches_per_image=1,
        training=False,
        normalization=HSI_NORMALIZATION,
        augment=False,
    )
    stats_loader = make_loader(
        stats_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        device=device,
    )

    channel_sum = torch.zeros(LATENT_CHANNELS, device=device, dtype=torch.float64)
    channel_sq_sum = torch.zeros(LATENT_CHANNELS, device=device, dtype=torch.float64)
    total_count = 0

    vae.eval()
    for batch_index, hsi in enumerate(stats_loader, start=1):
        hsi = hsi.to(device, non_blocking=True).float()
        z = encode_hsi_mean(vae, hsi).double()
        channel_sum += z.sum(dim=(0, 2, 3))
        channel_sq_sum += (z ** 2).sum(dim=(0, 2, 3))
        total_count += z.shape[0] * z.shape[2] * z.shape[3]

        if batch_index % 20 == 0 or batch_index == len(stats_loader):
            print(f"  Stats batch {batch_index}/{len(stats_loader)}")

    mean = channel_sum / max(total_count, 1)
    variance = channel_sq_sum / max(total_count, 1) - mean ** 2
    std = torch.sqrt(variance.clamp_min(1e-12))

    latent_mean = mean.float().view(1, -1, 1, 1)
    latent_std = std.float().view(1, -1, 1, 1)

    LATENT_STATS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent_mean": latent_mean.cpu().view(-1),
            "latent_std": latent_std.cpu().view(-1),
            "source": "computed_from_training_set",
        },
        LATENT_STATS_CACHE,
    )
    print(f"Saved latent statistics to: {LATENT_STATS_CACHE}")
    return latent_mean.to(device), latent_std.to(device)


def prepare_latent_statistics(
    vae: HSIVAE,
    train_files: Sequence[Path],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    checkpoint_stats = load_latent_stats_from_checkpoint(VAE_CHECKPOINT, device)
    if checkpoint_stats is not None:
        return checkpoint_stats
    return compute_latent_statistics(vae, train_files, device)


def save_checkpoint(
    path: Path,
    denoiser: HSILatentDiffusionUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    epoch: int,
    best_validation_loss: float,
    training_metrics: dict,
    validation_metrics: dict,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "denoiser_state_dict": denoiser.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "latent_mean": latent_mean.detach().cpu().view(-1),
            "latent_std": latent_std.detach().cpu().view(-1),
            "num_diffusion_timesteps": NUM_DIFFUSION_TIMESTEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "denoiser_kwargs": DENOISER_KWARGS,
            "vae_checkpoint": VAE_CHECKPOINT,
            "vae_kwargs": VAE_KWARGS,
        },
        path,
    )


def load_denoiser_checkpoint(
    denoiser: HSILatentDiffusionUNet,
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    checkpoint = load_torch_checkpoint(checkpoint_path, device="cpu")
    state_dict = extract_state_dict(
        checkpoint,
        candidate_keys=("denoiser_state_dict", "model_state_dict", "state_dict"),
    )
    state_dict = strip_prefix_if_present(state_dict, "module.")
    denoiser.load_state_dict(state_dict, strict=True)

    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint to be a dictionary containing latent stats.")
    if "latent_mean" not in checkpoint or "latent_std" not in checkpoint:
        raise KeyError("The denoiser checkpoint must contain latent_mean and latent_std.")

    latent_mean = checkpoint["latent_mean"].float().to(device).view(1, -1, 1, 1)
    latent_std = checkpoint["latent_std"].float().to(device).view(1, -1, 1, 1)
    print(f"Loaded denoiser checkpoint: {checkpoint_path}")
    return latent_mean, latent_std, checkpoint


# =============================================================================
# Metrics
# =============================================================================


def _metric_to_float(value: torch.Tensor | float) -> float:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.detach().float()
    if value.numel() != 1:
        value = value.mean()
    return float(value.item())


def mrae_metric(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target) / (torch.abs(target) + eps))


def rmse_metric(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((prediction - target) ** 2).clamp_min(1e-12))


def psnr_metric(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((prediction - target) ** 2).clamp_min(1e-12)
    data_range = (target.max() - target.min()).clamp_min(1e-8)
    return 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse)


def sam_metric(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Spectral angle mapper, averaged over pixels and batch.
    dot = torch.sum(prediction * target, dim=1)
    pred_norm = torch.linalg.vector_norm(prediction, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1)
    cosine = dot / (pred_norm * target_norm + eps)
    cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.mean(torch.acos(cosine))


def ssim_metric(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Average per-band local SSIM using an 11x11 uniform window.
    prediction = prediction.float()
    target = target.float()
    data_range = (target.max() - target.min()).clamp_min(1e-8)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    kernel_size = 11
    padding = kernel_size // 2
    mu_x = F.avg_pool2d(prediction, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)

    sigma_x = F.avg_pool2d(prediction * prediction, kernel_size, stride=1, padding=padding) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, kernel_size, stride=1, padding=padding) - mu_y ** 2
    sigma_xy = F.avg_pool2d(prediction * target, kernel_size, stride=1, padding=padding) - mu_x * mu_y

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return torch.mean(numerator / denominator.clamp_min(1e-12))


@torch.no_grad()
def calculate_image_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    prediction = prediction.detach().float()
    target = target.detach().float()
    if prediction.shape != target.shape:
        raise ValueError(
            f"Metric shape mismatch: prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}"
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Prediction contains NaN or Inf.")
    if not torch.isfinite(target).all():
        raise FloatingPointError("Target contains NaN or Inf.")

    metric_sums = {
        "mrae": 0.0,
        "psnr": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "ssim": 0.0,
    }

    for index in range(prediction.shape[0]):
        pred_i = prediction[index:index + 1]
        target_i = target[index:index + 1]
        metric_sums["mrae"] += _metric_to_float(mrae_metric(pred_i, target_i))
        metric_sums["psnr"] += _metric_to_float(psnr_metric(pred_i, target_i))
        metric_sums["rmse"] += _metric_to_float(rmse_metric(pred_i, target_i))
        metric_sums["sam"] += _metric_to_float(sam_metric(pred_i, target_i))
        metric_sums["ssim"] += _metric_to_float(ssim_metric(pred_i, target_i))

    batch_size = prediction.shape[0]
    return {key: value / batch_size for key, value in metric_sums.items()}


# =============================================================================
# Pair preparation
# =============================================================================


def prepare_training_and_validation_files() -> Tuple[List[Path], List[Path]]:
    train_files = find_files(TRAIN_HSI_DIR, SUPPORTED_HSI_EXTENSIONS, "training HSI")
    validation_files = find_files(VALIDATION_HSI_DIR, SUPPORTED_HSI_EXTENSIONS, "validation HSI")

    train_files = filter_valid_hsi_files(
        files=train_files,
        hsi_channels=HSI_CHANNELS,
        log_path=OUTPUT_DIR / "invalid_training_hsi_files.txt",
        cache_path=TRAIN_PAIR_VALIDATION_CACHE,
    )
    validation_files = filter_valid_hsi_files(
        files=validation_files,
        hsi_channels=HSI_CHANNELS,
        log_path=OUTPUT_DIR / "invalid_validation_hsi_files.txt",
        cache_path=VALIDATION_PAIR_VALIDATION_CACHE,
    )
    return train_files, validation_files


def prepare_validation_files() -> List[Path]:
    validation_files = find_files(VALIDATION_HSI_DIR, SUPPORTED_HSI_EXTENSIONS, "validation HSI")
    return filter_valid_hsi_files(
        files=validation_files,
        hsi_channels=HSI_CHANNELS,
        log_path=OUTPUT_DIR / "invalid_validation_hsi_files.txt",
        cache_path=VALIDATION_PAIR_VALIDATION_CACHE,
    )


# =============================================================================
# Training and validation
# =============================================================================


def make_random_timesteps(
    batch_size: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=NUM_DIFFUSION_TIMESTEPS,
        size=(batch_size,),
        device=device,
        dtype=torch.long,
        generator=generator,
    )


def train_one_epoch(
    denoiser: HSILatentDiffusionUNet,
    vae: HSIVAE,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> dict:
    denoiser.train()
    vae.eval()

    sums = {
        "loss": 0.0,
        "noise_mrae": 0.0,
    }
    sample_count = 0

    for batch_index, hsi in enumerate(loader, start=1):
        hsi = hsi.to(device, non_blocking=True).float()
        batch_size = hsi.shape[0]

        with torch.no_grad():
            z0 = encode_hsi_mean(vae, hsi)
            z0 = normalize_latent(z0, latent_mean, latent_std)

        timesteps = make_random_timesteps(batch_size, device)
        noise = torch.randn_like(z0)
        z_t = schedule.add_noise(z0, noise, timesteps)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            predicted_noise = denoiser(
                noisy_latent=z_t,
                timesteps=timesteps,
            )

        loss = F.mse_loss(predicted_noise.float(), noise.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch {batch_index}: {float(loss.detach())}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            denoiser.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            noise_mrae = mrae_metric(predicted_noise.float(), noise.float())

        sums["loss"] += float(loss.detach()) * batch_size
        sums["noise_mrae"] += float(noise_mrae.detach()) * batch_size
        sample_count += batch_size

        if batch_index % PRINT_EVERY == 0 or batch_index == len(loader):
            print(
                f"  Batch {batch_index:04d}/{len(loader):04d} | "
                f"loss={sums['loss'] / sample_count:.6f} | "
                f"noise MRAE@t={sums['noise_mrae'] / sample_count:.6f} | "
                f"grad={float(gradient_norm):.4f}"
            )

    return {key: value / sample_count for key, value in sums.items()}


@torch.no_grad()
def validate_one_epoch(
    denoiser: HSILatentDiffusionUNet,
    vae: HSIVAE,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """Validate with a randomly sampled timestep, matching the training loss.

    The validation loss is MSE(predicted_noise, actual_noise) at a randomly
    sampled timestep. ``noise_mrae`` is the same random-timestep comparison with
    MRAE. Image metrics are computed after estimating z0 from that one timestep
    and decoding it through the frozen HSI-VAE.
    """
    denoiser.eval()
    vae.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + 100_000)

    sums = {
        "loss": 0.0,
        "noise_mrae": 0.0,
        "latent_x0_mrae": 0.0,
        "decoded_mrae": 0.0,
        "decoded_psnr": 0.0,
        "decoded_rmse": 0.0,
        "decoded_sam": 0.0,
        "decoded_ssim": 0.0,
    }
    timestep_sum = 0.0
    sample_count = 0

    for hsi in loader:
        hsi = hsi.to(device, non_blocking=True).float()
        batch_size = hsi.shape[0]

        z0 = encode_hsi_mean(vae, hsi)
        z0 = normalize_latent(z0, latent_mean, latent_std)

        timesteps = make_random_timesteps(batch_size, device, generator=generator)
        noise = torch.randn(
            z0.shape,
            generator=generator,
            device=device,
            dtype=z0.dtype,
        )
        z_t = schedule.add_noise(z0, noise, timesteps)

        with autocast_context(device, use_amp):
            predicted_noise = denoiser(
                noisy_latent=z_t,
                timesteps=timesteps,
            )

        loss = F.mse_loss(predicted_noise.float(), noise.float())
        noise_mrae = mrae_metric(predicted_noise.float(), noise.float())

        predicted_z0 = schedule.predict_clean_from_noise(
            z_t=z_t.float(),
            predicted_noise=predicted_noise.float(),
            timesteps=timesteps,
        )
        latent_x0_mrae = mrae_metric(predicted_z0, z0.float())

        decoded_prediction = decode_latent(
            vae=vae,
            z_norm=predicted_z0,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )
        image_metrics = calculate_image_metrics(decoded_prediction, hsi)

        sums["loss"] += float(loss.detach()) * batch_size
        sums["noise_mrae"] += float(noise_mrae.detach()) * batch_size
        sums["latent_x0_mrae"] += float(latent_x0_mrae.detach()) * batch_size
        sums["decoded_mrae"] += image_metrics["mrae"] * batch_size
        sums["decoded_psnr"] += image_metrics["psnr"] * batch_size
        sums["decoded_rmse"] += image_metrics["rmse"] * batch_size
        sums["decoded_sam"] += image_metrics["sam"] * batch_size
        sums["decoded_ssim"] += image_metrics["ssim"] * batch_size
        timestep_sum += float(timesteps.detach().float().sum())
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("The validation DataLoader produced no samples.")

    metrics = {key: value / sample_count for key, value in sums.items()}
    metrics["mean_timestep"] = timestep_sum / sample_count
    metrics["evaluated_images"] = sample_count
    return metrics


def run_training(
    train_files: Sequence[Path],
    validation_files: Sequence[Path],
    device: torch.device,
    use_amp: bool,
) -> None:
    if TRAIN_CROP_SIZE % MODEL_DOWNSAMPLE_FACTOR != 0:
        raise ValueError(
            f"TRAIN_CROP_SIZE={TRAIN_CROP_SIZE} must be divisible by "
            f"MODEL_DOWNSAMPLE_FACTOR={MODEL_DOWNSAMPLE_FACTOR}."
        )
    if VALIDATION_CROP_SIZE % MODEL_DOWNSAMPLE_FACTOR != 0:
        raise ValueError(
            f"VALIDATION_CROP_SIZE={VALIDATION_CROP_SIZE} must be divisible by "
            f"MODEL_DOWNSAMPLE_FACTOR={MODEL_DOWNSAMPLE_FACTOR}."
        )

    train_dataset = HSIDataset(
        hsi_files=train_files,
        hsi_channels=HSI_CHANNELS,
        crop_size=TRAIN_CROP_SIZE,
        patches_per_image=PATCHES_PER_IMAGE,
        training=True,
        normalization=HSI_NORMALIZATION,
        augment=USE_AUGMENTATION,
    )
    validation_dataset = HSIDataset(
        hsi_files=validation_files,
        hsi_channels=HSI_CHANNELS,
        crop_size=VALIDATION_CROP_SIZE,
        patches_per_image=1,
        training=False,
        normalization=HSI_NORMALIZATION,
        augment=False,
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=(len(train_dataset) >= BATCH_SIZE),
        device=device,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        device=device,
    )

    vae = build_vae(device)
    latent_mean, latent_std = prepare_latent_statistics(vae, train_files, device)
    denoiser = build_denoiser(device)
    schedule = DiffusionSchedule(
        num_timesteps=NUM_DIFFUSION_TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
    ).to(device)

    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )
    scaler = make_grad_scaler(device, use_amp)

    start_epoch = 1
    best_validation_loss = float("inf")

    if RESUME_CHECKPOINT is not None:
        checkpoint = load_torch_checkpoint(RESUME_CHECKPOINT, device="cpu")
        state_dict = extract_state_dict(
            checkpoint,
            candidate_keys=("denoiser_state_dict", "model_state_dict", "state_dict"),
        )
        denoiser.load_state_dict(strip_prefix_if_present(state_dict, "module."), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "latent_mean" in checkpoint and "latent_std" in checkpoint:
            latent_mean = checkpoint["latent_mean"].float().to(device).view(1, -1, 1, 1)
            latent_std = checkpoint["latent_std"].float().to(device).view(1, -1, 1, 1)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        print(f"Resumed training from {RESUME_CHECKPOINT} at epoch {start_epoch}.")

    trainable_count = sum(parameter.numel() for parameter in denoiser.parameters())
    frozen_vae_count = sum(parameter.numel() for parameter in vae.parameters())
    amp_dtype = get_amp_dtype(device)
    print(
        f"\nDevice: {device}\n"
        f"AMP: {use_amp} ({amp_dtype if use_amp else 'float32'})\n"
        f"Training HSI files: {len(train_files)}\n"
        f"Validation HSI files: {len(validation_files)}\n"
        f"Frozen VAE parameters: {frozen_vae_count:,}\n"
        f"Trainable prior U-Net parameters: {trainable_count:,}\n"
        f"Latent channels: {LATENT_CHANNELS}\n"
        f"Diffusion timesteps: {NUM_DIFFUSION_TIMESTEPS}"
    )

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n{'=' * 80}\nEpoch {epoch}/{NUM_EPOCHS}\n{'=' * 80}")

        training_metrics = train_one_epoch(
            denoiser=denoiser,
            vae=vae,
            schedule=schedule,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            latent_mean=latent_mean,
            latent_std=latent_std,
            device=device,
            use_amp=use_amp,
        )
        validation_metrics = validate_one_epoch(
            denoiser=denoiser,
            vae=vae,
            schedule=schedule,
            loader=validation_loader,
            latent_mean=latent_mean,
            latent_std=latent_std,
            device=device,
            use_amp=use_amp,
        )
        lr_scheduler.step()

        print(
            f"Epoch {epoch:03d} | LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"train loss={training_metrics['loss']:.6f} | "
            f"train noise MRAE@t={training_metrics['noise_mrae']:.6f} | "
            f"val loss={validation_metrics['loss']:.6f} | "
            f"val noise MRAE@t={validation_metrics['noise_mrae']:.6f} | "
            f"val latent-z0 MRAE@t={validation_metrics['latent_x0_mrae']:.6f} | "
            f"mean t={validation_metrics['mean_timestep']:.1f}"
        )
        print(
            "Validation one-step decoded reconstruction metrics "
            f"({validation_metrics['evaluated_images']} images) | "
            f"MRAE={validation_metrics['decoded_mrae']:.6f} | "
            f"PSNR={validation_metrics['decoded_psnr']:.4f} | "
            f"RMSE={validation_metrics['decoded_rmse']:.6f} | "
            f"SAM={validation_metrics['decoded_sam']:.6f} | "
            f"SSIM={validation_metrics['decoded_ssim']:.4f}"
        )

        save_checkpoint(
            path=LAST_CHECKPOINT,
            denoiser=denoiser,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            scaler=scaler,
            epoch=epoch,
            best_validation_loss=best_validation_loss,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            save_checkpoint(
                path=BEST_CHECKPOINT,
                denoiser=denoiser,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                scaler=scaler,
                epoch=epoch,
                best_validation_loss=best_validation_loss,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                latent_mean=latent_mean,
                latent_std=latent_std,
            )
            print(
                f"Saved new best checkpoint: {BEST_CHECKPOINT} | "
                f"validation loss={best_validation_loss:.6f}"
            )


# =============================================================================
# Visualisation
# =============================================================================


def hsi_triplet_to_display(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    error_map: torch.Tensor,
    bands: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_np = target.detach().float().cpu().numpy()
    reconstruction_np = reconstruction.detach().float().cpu().numpy()
    error_np = error_map.detach().float().cpu().numpy()

    for band in bands:
        if not 0 <= band < target_np.shape[0]:
            raise ValueError(
                f"Visualization band {band} is outside [0, {target_np.shape[0] - 1}]."
            )

    def select(cube: np.ndarray) -> np.ndarray:
        return np.stack([cube[band] for band in bands], axis=-1)

    target_rgb = select(target_np)
    reconstruction_rgb = select(reconstruction_np)

    minimum = target_rgb.min(axis=(0, 1), keepdims=True)
    maximum = target_rgb.max(axis=(0, 1), keepdims=True)
    scale = maximum - minimum + 1e-8

    target_display = np.clip((target_rgb - minimum) / scale, 0.0, 1.0)
    reconstruction_display = np.clip((reconstruction_rgb - minimum) / scale, 0.0, 1.0)

    error_display = error_np
    if error_display.ndim == 3:
        error_display = error_display.mean(axis=0)
    error_display = error_display - error_display.min()
    error_display = error_display / (error_display.max() + 1e-8)

    return target_display, reconstruction_display, error_display


@torch.no_grad()
def reconstruct_from_random_timestep(
    hsi_batch: torch.Tensor,
    vae: HSIVAE,
    denoiser: HSILatentDiffusionUNet,
    schedule: DiffusionSchedule,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hsi_batch = hsi_batch.to(device, non_blocking=True).float()
    z0 = encode_hsi_mean(vae, hsi_batch)
    z0 = normalize_latent(z0, latent_mean, latent_std)

    if VISUALIZATION_TIMESTEP_MODE == "fixed":
        timesteps = torch.full(
            (hsi_batch.shape[0],),
            int(VISUALIZATION_FIXED_TIMESTEP),
            device=device,
            dtype=torch.long,
        )
        timesteps = timesteps.clamp(0, NUM_DIFFUSION_TIMESTEPS - 1)
    else:
        timesteps = make_random_timesteps(hsi_batch.shape[0], device, generator=generator)

    noise = torch.randn(
        z0.shape,
        generator=generator,
        device=device,
        dtype=z0.dtype,
    )
    z_t = schedule.add_noise(z0, noise, timesteps)

    with autocast_context(device, use_amp):
        predicted_noise = denoiser(
            noisy_latent=z_t,
            timesteps=timesteps,
        )

    predicted_z0 = schedule.predict_clean_from_noise(
        z_t=z_t.float(),
        predicted_noise=predicted_noise.float(),
        timesteps=timesteps,
    )
    reconstruction = decode_latent(
        vae=vae,
        z_norm=predicted_z0,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )
    return reconstruction, timesteps, predicted_noise


@torch.no_grad()
def run_visualization(
    denoiser: HSILatentDiffusionUNet,
    vae: HSIVAE,
    schedule: DiffusionSchedule,
    validation_files: Sequence[Path],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> Path:
    denoiser.eval()
    vae.eval()
    if not validation_files:
        raise RuntimeError("The validation file list is empty.")

    number_to_select = min(NUM_VISUALIZATION_IMAGES, len(validation_files))
    selected_indices = random.Random(SEED).sample(
        range(len(validation_files)),
        k=number_to_select,
    )

    dataset = HSIDataset(
        hsi_files=validation_files,
        hsi_channels=HSI_CHANNELS,
        crop_size=None,
        patches_per_image=1,
        training=False,
        normalization=HSI_NORMALIZATION,
        augment=False,
        return_paths=True,
    )

    column_titles = (
        "Ground-truth HSI\n(pseudo-RGB)",
        "One-step reconstruction\nfrom random t",
        "Mean absolute error",
        "Mean spectral curve",
    )
    figure, axes = plt.subplots(
        number_to_select,
        4,
        figsize=(18, 4.2 * number_to_select),
        squeeze=False,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + 300_000)

    for row, dataset_index in enumerate(selected_indices):
        hsi, hsi_path_string = dataset[dataset_index]
        padded_hsi, original_height, original_width = pad_hsi_to_multiple(
            hsi=hsi,
            multiple=MODEL_DOWNSAMPLE_FACTOR,
        )

        reconstruction, timesteps, _ = reconstruct_from_random_timestep(
            hsi_batch=padded_hsi.unsqueeze(0),
            vae=vae,
            denoiser=denoiser,
            schedule=schedule,
            latent_mean=latent_mean,
            latent_std=latent_std,
            device=device,
            use_amp=use_amp,
            generator=generator,
        )

        reconstruction = reconstruction[0, :, :original_height, :original_width].float().cpu()
        target = hsi[:, :original_height, :original_width].float().cpu()
        error = torch.abs(reconstruction - target)
        metrics = calculate_image_metrics(
            prediction=reconstruction.unsqueeze(0),
            target=target.unsqueeze(0),
        )
        target_display, reconstruction_display, error_display = hsi_triplet_to_display(
            target=target,
            reconstruction=reconstruction,
            error_map=error,
            bands=VISUALIZATION_BANDS,
        )

        stem = Path(hsi_path_string).stem
        panels = (target_display, reconstruction_display, error_display)
        for column, panel in enumerate(panels):
            axis = axes[row, column]
            if column == 2:
                axis.imshow(panel, cmap="magma")
            else:
                axis.imshow(panel)
            axis.axis("off")
            if row == 0:
                axis.set_title(column_titles[column], fontsize=12, fontweight="bold")

        # Mean spectral curve plot.
        axis = axes[row, 3]
        target_curve = target.mean(dim=(1, 2)).numpy()
        reconstruction_curve = reconstruction.mean(dim=(1, 2)).numpy()
        axis.plot(target_curve, label="GT")
        axis.plot(reconstruction_curve, label="Recon")
        axis.set_xlabel("Band")
        axis.set_ylabel("Mean intensity")
        axis.grid(True, alpha=0.3)
        if row == 0:
            axis.set_title(column_titles[3], fontsize=12, fontweight="bold")
            axis.legend(fontsize=8)

        axes[row, 0].set_ylabel(
            f"{stem}\n"
            f"t={int(timesteps[0].item())}\n"
            f"MRAE {metrics['mrae']:.4f} | PSNR {metrics['psnr']:.2f} dB\n"
            f"RMSE {metrics['rmse']:.4f} | SAM {metrics['sam']:.4f}\n"
            f"SSIM {metrics['ssim']:.4f}",
            fontsize=9,
            rotation=0,
            labelpad=110,
            va="center",
        )

    figure.suptitle(
        "Random validation examples: HSI latent diffusion prior one-step reconstruction",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.08, 0.01, 1.0, 0.985))

    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(VISUALIZATION_FILE, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved labelled visualization to: {VISUALIZATION_FILE}")
    return VISUALIZATION_FILE


def load_model_for_visualization(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[HSILatentDiffusionUNet, HSIVAE, DiffusionSchedule, torch.Tensor, torch.Tensor]:
    vae = build_vae(device)
    denoiser = build_denoiser(device)
    latent_mean, latent_std, _ = load_denoiser_checkpoint(denoiser, checkpoint_path, device)
    denoiser.to(device)
    denoiser.eval()
    schedule = DiffusionSchedule(
        num_timesteps=NUM_DIFFUSION_TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
    ).to(device)
    return denoiser, vae, schedule, latent_mean, latent_std


# =============================================================================
# Mode parser and main
# =============================================================================


def parse_mode() -> str:
    parser = argparse.ArgumentParser(
        description="Train or visualize an HSI latent diffusion prior."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("train", "visualize", "train_visualize"),
        help=(
            "train: train only; visualize: load VISUALIZATION_CHECKPOINT and "
            "render five random validation images; train_visualize: train and "
            "then visualize the best checkpoint."
        ),
    )
    return parser.parse_args().mode


def main() -> None:
    mode = parse_mode()
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    if mode in {"train", "train_visualize"}:
        train_files, validation_files = prepare_training_and_validation_files()
        run_training(
            train_files=train_files,
            validation_files=validation_files,
            device=device,
            use_amp=use_amp,
        )
    else:
        validation_files = prepare_validation_files()

    if mode in {"visualize", "train_visualize"}:
        checkpoint_path = BEST_CHECKPOINT if mode == "train_visualize" else VISUALIZATION_CHECKPOINT
        denoiser, vae, schedule, latent_mean, latent_std = load_model_for_visualization(
            checkpoint_path,
            device,
        )
        run_visualization(
            denoiser=denoiser,
            vae=vae,
            schedule=schedule,
            validation_files=validation_files,
            latent_mean=latent_mean,
            latent_std=latent_std,
            device=device,
            use_amp=use_amp,
        )


if __name__ == "__main__":
    main()
