"""Train an HSI VAE from native-resolution hyperspectral images.

Key behaviour
-------------
1. Every HSI cube is loaded at its original spatial resolution.
2. Training samples are random spatial crops taken from the native-resolution cube.
3. The same spatial augmentation is applied to every spectral band.
4. Validation uses the complete native-resolution image, not a center crop.
5. Validation batch size is one so differently sized images are supported.
6. Images are padded only immediately before the VAE forward pass so their height
   and width are compatible with the VAE downsampling factor. The reconstruction
   is cropped back to the original size before losses and metrics are calculated.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from models.HSI_VAE import HSIVAE

from loss.mrae import mrae
from loss.psnr import psnr
from loss.rmse import rmse
from loss.sam import sam
from loss.ssim import ssim


# =============================================================================
# Configuration
# =============================================================================

HSI_DATA_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Train_spectral/Train_spectral"
)
OUTPUT_DIR = "./vae_checkpoints"

HSI_KEY = "cube"
HSI_CHANNELS = 31
SUPPORTED_HSI_EXTENSIONS = {".mat", ".npy", ".npz", ".pt", ".pth"}

# VAE architecture. These values must match models.HSI_VAE.HSIVAE.
BASE_CHANNELS = 64
LATENT_CHANNELS = 16
NUM_RES_BLOCKS = 2

# Set this to the total spatial reduction performed by the encoder.
# For two stride-2 downsampling layers, the factor is 4.
VAE_DOWNSAMPLE_FACTOR = 4

# Training uses native-resolution cubes as the source, then samples random crops.
TRAIN_CROP_SIZE = 256
CROPS_PER_IMAGE = 1

# Spatial augmentation. These transforms preserve spectral values and apply the
# exact same spatial operation to every HSI band.
USE_AUGMENTATION = True
HORIZONTAL_FLIP_PROBABILITY = 0.5
VERTICAL_FLIP_PROBABILITY = 0.5
USE_RANDOM_90_DEGREE_ROTATION = True

# "none", "minmax", or "band_minmax".
NORMALIZATION = "none"

VALIDATION_FRACTION = 0.10
SEED = 42

TRAIN_BATCH_SIZE = 4
# Keep this at one because validation images can have different H x W shapes.
VALIDATION_BATCH_SIZE = 1
NUM_WORKERS = 4

NUM_EPOCHS = 75
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
USE_AMP = True
PRINT_EVERY = 30

# Reconstruction and KL settings.
RECONSTRUCTION_LOSS = "mse"  # "mse", "l1", or "smooth_l1"
KL_BETA_START = 0.0
KL_BETA_END = 1e-3
KL_WARMUP_EPOCHS = 30

# The best checkpoint is selected using validation total loss.
VALIDATION_CACHE = Path(OUTPUT_DIR) / "vae_validation_cache.pth"
HSI_CHECKER_VERSION = "previous-script-metadata-cache-v1"


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker a reproducible independent RNG state."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# HSI loading
# =============================================================================


def _select_largest_3d_array(
    arrays: Sequence[Tuple[str, np.ndarray]],
    file_path: Path,
) -> np.ndarray:
    candidates = [
        (name, value)
        for name, value in arrays
        if isinstance(value, np.ndarray)
        and value.ndim == 3
        and np.issubdtype(value.dtype, np.number)
    ]
    if not candidates:
        raise ValueError(f"No numerical three-dimensional HSI array found in {file_path}")
    return max(candidates, key=lambda item: item[1].size)[1]


def load_mat_v73(file_path: Path, hsi_key: str) -> np.ndarray:
    """Load a MATLAB v7.3 HDF5 file."""
    candidates: List[Tuple[str, np.ndarray]] = []

    with h5py.File(str(file_path), "r") as h5_file:
        if hsi_key in h5_file and isinstance(h5_file[hsi_key], h5py.Dataset):
            dataset = h5_file[hsi_key]
            if dataset.ndim == 3:
                candidates.append((hsi_key, np.asarray(dataset)))

        if not candidates:
            def visitor(name: str, obj: Any) -> None:
                if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                    return
                try:
                    array = np.asarray(obj)
                    if np.issubdtype(array.dtype, np.number):
                        candidates.append((name, array))
                except Exception:
                    return

            h5_file.visititems(visitor)

    cube = _select_largest_3d_array(candidates, file_path)

    # MATLAB v7.3 arrays commonly appear with reversed axis order in h5py.
    # convert_to_chw() below still verifies the location of the 31-band axis.
    return np.transpose(cube, axes=tuple(range(cube.ndim - 1, -1, -1)))


def extract_array_from_dictionary(
    data: Dict[str, Any],
    file_path: Path,
    hsi_key: str,
) -> np.ndarray:
    if hsi_key in data:
        preferred = data[hsi_key]
        if isinstance(preferred, torch.Tensor):
            preferred = preferred.detach().cpu().numpy()
        if isinstance(preferred, np.ndarray) and preferred.ndim == 3:
            return preferred

    arrays: List[Tuple[str, np.ndarray]] = []
    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            arrays.append((key, value))

    return _select_largest_3d_array(arrays, file_path)


def load_hsi_file(file_path: Path, hsi_key: str = HSI_KEY) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension == ".npy":
        cube = np.load(file_path)

    elif extension == ".npz":
        with np.load(file_path) as loaded:
            if hsi_key in loaded.files and loaded[hsi_key].ndim == 3:
                cube = loaded[hsi_key]
            else:
                candidates = [
                    (key, loaded[key])
                    for key in loaded.files
                    if loaded[key].ndim == 3
                ]
                cube = _select_largest_3d_array(candidates, file_path)

    elif extension == ".mat":
        try:
            loaded = sio.loadmat(file_path)
            cube = extract_array_from_dictionary(loaded, file_path, hsi_key)
        except (NotImplementedError, ValueError):
            cube = load_mat_v73(file_path, hsi_key)

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
            cube = extract_array_from_dictionary(loaded, file_path, hsi_key)
        else:
            raise TypeError(
                f"Unsupported object type {type(loaded).__name__} in {file_path}"
            )

    else:
        raise ValueError(f"Unsupported HSI extension: {extension}")

    cube = np.asarray(cube, dtype=np.float32)
    cube = np.squeeze(cube)

    if cube.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional HSI cube in {file_path}, "
            f"but found shape {cube.shape}"
        )

    return cube


def convert_to_chw(
    cube: np.ndarray,
    hsi_channels: int,
    file_path: Path,
) -> np.ndarray:
    """Convert [H,W,C] or [C,H,W] to [C,H,W]."""
    if cube.shape[0] == hsi_channels:
        return cube
    if cube.shape[-1] == hsi_channels:
        return np.transpose(cube, (2, 0, 1))

    raise ValueError(
        f"Cannot locate the {hsi_channels}-band spectral axis in {file_path}. "
        f"Found shape {cube.shape}."
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


# =============================================================================
# File discovery and validation
# EXACT METADATA/CACHE CHECKER ADAPTED FROM THE PREVIOUS RGB-HSI SCRIPT
# =============================================================================


def find_hsi_files(data_dir: str) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"HSI directory does not exist: {root}")

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_HSI_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No supported HSI files found in {root}")

    return files


def make_files_fingerprint(files: Sequence[Path]) -> str:
    """Create a cache fingerprint from file paths, sizes, and modification times."""
    records = []
    for file_path in files:
        stat = file_path.stat()
        records.append(
            f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        )
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def is_possible_hsi_shape(shape: Sequence[int], hsi_channels: int) -> bool:
    """Return True when shape can represent a non-empty 3D HSI cube."""
    return (
        len(shape) == 3
        and hsi_channels in shape
        and all(int(dimension) > 0 for dimension in shape)
    )


def inspect_hdf5_mat_file(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    """Inspect a MATLAB v7.3/HDF5 file without loading its full HSI cube."""
    candidates: List[Tuple[str, Tuple[int, ...]]] = []

    try:
        with h5py.File(str(file_path), "r") as h5_file:
            if hsi_key in h5_file and isinstance(h5_file[hsi_key], h5py.Dataset):
                dataset = h5_file[hsi_key]
                candidates.append(
                    (hsi_key, tuple(int(value) for value in dataset.shape))
                )
            else:
                def visitor(name: str, obj: Any) -> None:
                    if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                        return
                    try:
                        if np.issubdtype(obj.dtype, np.number):
                            candidates.append(
                                (name, tuple(int(value) for value in obj.shape))
                            )
                    except TypeError:
                        pass

                h5_file.visititems(visitor)
    except OSError as error:
        raise OSError(
            f"Could not inspect MATLAB v7.3 file:\n{file_path}\n"
            f"Reason: {error}"
        ) from error

    if not candidates:
        raise ValueError(f"No numerical 3D dataset found in {file_path}")

    if not any(
        is_possible_hsi_shape(shape, hsi_channels)
        for _, shape in candidates
    ):
        raise ValueError(
            f"No {hsi_channels}-band cube found in {file_path}. "
            f"HDF5 datasets: {candidates}"
        )


def inspect_standard_mat_file(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    """Inspect a pre-v7.3 MATLAB file using metadata only."""
    try:
        metadata = sio.whosmat(file_path)
    except (NotImplementedError, ValueError, OSError):
        # MATLAB v7.3 files are HDF5 containers.
        inspect_hdf5_mat_file(file_path, hsi_channels, hsi_key)
        return

    candidates = [
        (name, tuple(int(value) for value in shape))
        for name, shape, _ in metadata
        if len(shape) == 3
    ]

    if not candidates:
        raise ValueError(f"No 3D array found in {file_path}")

    # Prefer HSI_KEY when it exists. Otherwise accept any valid 3D cube.
    preferred = [candidate for candidate in candidates if candidate[0] == hsi_key]
    to_check = preferred if preferred else candidates

    if not any(
        is_possible_hsi_shape(shape, hsi_channels)
        for _, shape in to_check
    ):
        raise ValueError(
            f"No {hsi_channels}-band cube found in {file_path}. "
            f"MAT arrays: {candidates}"
        )


def inspect_hsi_file_metadata(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    """Validate an HSI file before training.

    MATLAB files are checked from their headers/metadata, so their complete cubes
    are not loaded during the initial validation pass. Other supported formats
    reuse the normal loader because they do not have the same MAT metadata path.
    """
    if file_path.suffix.lower() == ".mat":
        inspect_standard_mat_file(file_path, hsi_channels, hsi_key)
        return

    cube = load_hsi_file(file_path, hsi_key)
    if not is_possible_hsi_shape(cube.shape, hsi_channels):
        raise ValueError(
            f"Invalid HSI shape {cube.shape} in {file_path}. "
            f"Expected a non-empty 3D cube containing {hsi_channels} bands."
        )


def filter_valid_hsi_files(
    files: List[Path],
    hsi_channels: int,
    hsi_key: str,
    log_path: Path,
) -> List[Path]:
    """
    Validate HSI files exactly like the earlier script:

    * .mat files are checked from metadata only.
    * MATLAB v7.3 files are checked through HDF5 headers.
    * other supported formats are loaded and shape-checked.
    * valid/invalid results are cached using path, size, and mtime.
    * invalid files are written to a text log.
    """
    VALIDATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = make_files_fingerprint(files)
    file_lookup = {str(file_path.resolve()): file_path for file_path in files}

    # Try to reuse the previous validation result.
    if VALIDATION_CACHE.exists():
        try:
            try:
                cached = torch.load(
                    VALIDATION_CACHE,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                cached = torch.load(VALIDATION_CACHE, map_location="cpu")

            if (
                isinstance(cached, dict)
                and cached.get("fingerprint") == fingerprint
                and cached.get("checker_version") == HSI_CHECKER_VERSION
            ):
                valid_paths = cached.get("valid_hsi_paths", [])
                invalid_records = cached.get("invalid_records", [])
                valid_files = [
                    file_lookup[path]
                    for path in valid_paths
                    if path in file_lookup
                ]

                print("\nUsing cached HSI-file validation.")
                print(f"Valid files:   {len(valid_files)}")
                print(f"Invalid files: {len(invalid_records)}")

                for record in invalid_records:
                    print(
                        "\nCached invalid file:\n"
                        f"  File:  {record['path']}\n"
                        f"  Error: {record['error']}"
                    )

                if valid_files:
                    return valid_files

        except Exception as error:
            print(
                "Could not use validation cache. Re-scanning.\n"
                f"Reason: {error}"
            )

    # Full scan. MAT files are inspected by metadata only.
    print("\nChecking HSI file metadata before training...")

    valid_files: List[Path] = []
    invalid_records: List[Dict[str, str]] = []

    for index, file_path in enumerate(files, start=1):
        try:
            inspect_hsi_file_metadata(
                file_path=file_path,
                hsi_channels=hsi_channels,
                hsi_key=hsi_key,
            )
            valid_files.append(file_path)

        except Exception as error:
            invalid_records.append(
                {
                    "path": str(file_path.resolve()),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                "\nSkipping invalid file:\n"
                f"  File:  {file_path}\n"
                f"  Error: {error}"
            )

        if index % 100 == 0 or index == len(files):
            print(
                f"Checked {index}/{len(files)} | "
                f"Valid: {len(valid_files)} | "
                f"Invalid: {len(invalid_records)}"
            )

    if not valid_files:
        raise RuntimeError("No valid HSI files remain after validation.")

    if invalid_records:
        with log_path.open("w", encoding="utf-8") as handle:
            for record in invalid_records:
                handle.write(f"{record['path']} | {record['error']}\n")
        print(f"\nInvalid-file log saved to: {log_path}")
    elif log_path.exists():
        # Prevent a stale invalid-file log from an older dataset scan.
        log_path.unlink()

    torch.save(
        {
            "checker_version": HSI_CHECKER_VERSION,
            "fingerprint": fingerprint,
            "valid_hsi_paths": [
                str(file_path.resolve()) for file_path in valid_files
            ],
            "invalid_records": invalid_records,
        },
        VALIDATION_CACHE,
    )
    print(f"Validation cache saved to: {VALIDATION_CACHE}")

    return valid_files


# =============================================================================
# Native-resolution crops and augmentation
# =============================================================================


def pad_to_minimum_size(cube: torch.Tensor, minimum_size: int) -> torch.Tensor:
    """Replicate-pad [C,H,W] only when an image is smaller than a train crop."""
    _, height, width = cube.shape
    pad_bottom = max(0, minimum_size - height)
    pad_right = max(0, minimum_size - width)

    if pad_bottom == 0 and pad_right == 0:
        return cube

    return F.pad(
        cube,
        (0, pad_right, 0, pad_bottom),
        mode="replicate",
    )


def random_crop(cube: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Take a random crop from a cube already loaded at native resolution."""
    cube = pad_to_minimum_size(cube, crop_size)
    _, height, width = cube.shape

    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)

    return cube[:, top : top + crop_size, left : left + crop_size]


def augment_hsi(cube: torch.Tensor) -> torch.Tensor:
    """Apply spatial transforms identically to all spectral bands."""
    if random.random() < HORIZONTAL_FLIP_PROBABILITY:
        cube = torch.flip(cube, dims=(2,))

    if random.random() < VERTICAL_FLIP_PROBABILITY:
        cube = torch.flip(cube, dims=(1,))

    if USE_RANDOM_90_DEGREE_ROTATION:
        rotations = random.randint(0, 3)
        if rotations:
            cube = torch.rot90(cube, k=rotations, dims=(1, 2))

    return cube.contiguous()


# =============================================================================
# Dataset and split
# =============================================================================


class HSIDataset(Dataset):
    """HSI-only dataset.

    Training:
        Load the complete native-resolution cube -> random crop -> augmentation.

    Validation:
        Load and return the complete native-resolution cube without any crop.
    """

    def __init__(
        self,
        files: List[Path],
        hsi_channels: int,
        training: bool,
        normalization: str,
        crop_size: int | None = None,
        crops_per_image: int = 1,
        augment: bool = False,
    ) -> None:
        self.files = files
        self.hsi_channels = hsi_channels
        self.training = training
        self.normalization = normalization
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.augment = augment

        if self.training and (self.crop_size is None or self.crop_size <= 0):
            raise ValueError("A positive crop_size is required for training.")
        if self.crops_per_image <= 0:
            raise ValueError("crops_per_image must be positive.")

    def __len__(self) -> int:
        if self.training:
            return len(self.files) * self.crops_per_image
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_index = index // self.crops_per_image if self.training else index
        file_path = self.files[file_index]

        cube = load_hsi_file(file_path, HSI_KEY)
        cube = convert_to_chw(cube, self.hsi_channels, file_path)

        if not np.isfinite(cube).all():
            raise ValueError(f"NaN or Inf values found in {file_path}")

        cube = normalize_cube(cube, self.normalization)
        tensor = torch.from_numpy(np.ascontiguousarray(cube)).float()

        if self.training:
            tensor = random_crop(tensor, int(self.crop_size))
            if self.augment:
                tensor = augment_hsi(tensor)

        return tensor


def split_files(
    files: List[Path],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("VALIDATION_FRACTION must be between zero and one.")

    shuffled = files.copy()
    random.Random(seed).shuffle(shuffled)

    validation_size = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_files = shuffled[:validation_size]
    training_files = shuffled[validation_size:]

    if not training_files:
        raise RuntimeError("No files remain for training after the split.")

    return training_files, validation_files


# =============================================================================
# VAE forward helpers
# =============================================================================


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad [B,C,H,W] on the bottom/right and return the original H,W."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], found shape {tuple(tensor.shape)}")
    if multiple <= 0:
        raise ValueError("multiple must be positive")

    original_height, original_width = tensor.shape[-2:]
    padded_height = ((original_height + multiple - 1) // multiple) * multiple
    padded_width = ((original_width + multiple - 1) // multiple) * multiple

    pad_bottom = padded_height - original_height
    pad_right = padded_width - original_width

    if pad_bottom == 0 and pad_right == 0:
        return tensor, (original_height, original_width)

    padded = F.pad(
        tensor,
        (0, pad_right, 0, pad_bottom),
        mode="replicate",
    )
    return padded, (original_height, original_width)


def unpack_vae_output(output: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Support common HSIVAE output formats."""
    if isinstance(output, (tuple, list)) and len(output) >= 3:
        reconstruction, mu, logvar = output[:3]
        return reconstruction, mu, logvar

    if isinstance(output, dict):
        reconstruction = output.get("reconstruction", output.get("recon"))
        mu = output.get("mu")
        logvar = output.get("logvar")
        if reconstruction is not None and mu is not None and logvar is not None:
            return reconstruction, mu, logvar

    if all(hasattr(output, name) for name in ("reconstruction", "mu", "logvar")):
        return output.reconstruction, output.mu, output.logvar

    raise TypeError(
        "HSIVAE.forward() must return (reconstruction, mu, logvar), a dictionary "
        "with those values, or an object exposing those attributes."
    )


def forward_vae(
    model: HSIVAE,
    hsi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the VAE with size-compatible padding, then remove output padding."""
    padded_hsi, (height, width) = pad_to_multiple(
        hsi,
        VAE_DOWNSAMPLE_FACTOR,
    )
    reconstruction, mu, logvar = unpack_vae_output(model(padded_hsi))

    reconstruction = reconstruction[..., :height, :width]
    return reconstruction, mu, logvar


# =============================================================================
# Losses and metrics
# =============================================================================


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if RECONSTRUCTION_LOSS == "mse":
        return F.mse_loss(reconstruction, target)
    if RECONSTRUCTION_LOSS == "l1":
        return F.l1_loss(reconstruction, target)
    if RECONSTRUCTION_LOSS == "smooth_l1":
        return F.smooth_l1_loss(reconstruction, target)
    raise ValueError(f"Unknown reconstruction loss: {RECONSTRUCTION_LOSS}")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL divergence from q(z|x) to a standard normal prior."""
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def get_kl_beta(epoch: int) -> float:
    if KL_WARMUP_EPOCHS <= 0:
        return KL_BETA_END

    progress = min(max((epoch - 1) / KL_WARMUP_EPOCHS, 0.0), 1.0)
    return KL_BETA_START + progress * (KL_BETA_END - KL_BETA_START)


@torch.no_grad()
def calculate_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    reconstruction = reconstruction.float()
    target = target.float()

    return {
        "mrae": float(mrae(target, reconstruction).item()),
        "rmse": float(rmse(target, reconstruction).item()),
        "sam": float(sam(target, reconstruction).item()),
        "psnr": float(psnr(target, reconstruction).item()),
        "ssim": float(ssim(target, reconstruction).item()),
    }


# =============================================================================
# Training and validation
# =============================================================================


def train_one_epoch(
    model: HSIVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    beta: float,
) -> Dict[str, float]:
    model.train()

    totals = {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "kl_loss": 0.0,
        "mrae": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    sample_count = 0

    for batch_index, hsi in enumerate(loader, start=1):
        hsi = hsi.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            reconstruction, mu, logvar = forward_vae(model, hsi)
            recon_loss = reconstruction_loss(reconstruction, hsi)
            kl_loss = kl_divergence(mu, logvar)
            total_loss = recon_loss + beta * kl_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite VAE loss: "
                f"total={total_loss.item()}, "
                f"reconstruction={recon_loss.item()}, KL={kl_loss.item()}"
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        metrics = calculate_metrics(reconstruction.detach(), hsi.detach())
        batch_size = hsi.size(0)
        sample_count += batch_size

        totals["total_loss"] += float(total_loss.detach().item()) * batch_size
        totals["reconstruction_loss"] += float(recon_loss.detach().item()) * batch_size
        totals["kl_loss"] += float(kl_loss.detach().item()) * batch_size
        for name, value in metrics.items():
            totals[name] += value * batch_size

        if batch_index % PRINT_EVERY == 0 or batch_index == len(loader):
            denominator = max(sample_count, 1)
            print(
                f"  Batch {batch_index:04d}/{len(loader):04d} | "
                f"Total: {totals['total_loss'] / denominator:.6f} | "
                f"Recon: {totals['reconstruction_loss'] / denominator:.6f} | "
                f"KL: {totals['kl_loss'] / denominator:.6f} | "
                f"MRAE: {totals['mrae'] / denominator:.6f} | "
                f"PSNR: {totals['psnr'] / denominator:.4f}"
            )

    return {
        name: value / max(sample_count, 1)
        for name, value in totals.items()
    }


@torch.no_grad()
def validate(
    model: HSIVAE,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    beta: float,
) -> Dict[str, float]:
    """Validate directly on every complete native-resolution HSI image."""
    model.eval()

    totals = {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "kl_loss": 0.0,
        "mrae": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    image_count = 0

    for hsi in loader:
        # VALIDATION_BATCH_SIZE is one, so each iteration can have a different
        # full-resolution H x W shape.
        hsi = hsi.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            reconstruction, mu, logvar = forward_vae(model, hsi)
            recon_loss = reconstruction_loss(reconstruction, hsi)
            kl_loss = kl_divergence(mu, logvar)
            total_loss = recon_loss + beta * kl_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite validation loss for shape {tuple(hsi.shape)}"
            )

        metrics = calculate_metrics(reconstruction, hsi)
        image_count += 1

        totals["total_loss"] += float(total_loss.item())
        totals["reconstruction_loss"] += float(recon_loss.item())
        totals["kl_loss"] += float(kl_loss.item())
        for name, value in metrics.items():
            totals[name] += value

    return {
        name: value / max(image_count, 1)
        for name, value in totals.items()
    }


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(
    output_path: Path,
    model: HSIVAE,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    beta: float,
    validation_metrics: Dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "kl_beta": beta,
            "validation_metrics": validation_metrics,
            "model_config": {
                "hsi_channels": HSI_CHANNELS,
                "base_channels": BASE_CHANNELS,
                "latent_channels": LATENT_CHANNELS,
                "num_res_blocks": NUM_RES_BLOCKS,
            },
            "training_config": {
                "normalization": NORMALIZATION,
                "train_crop_size": TRAIN_CROP_SIZE,
                "vae_downsample_factor": VAE_DOWNSAMPLE_FACTOR,
                "reconstruction_loss": RECONSTRUCTION_LOSS,
                "kl_beta_start": KL_BETA_START,
                "kl_beta_end": KL_BETA_END,
                "kl_warmup_epochs": KL_WARMUP_EPOCHS,
                "use_augmentation": USE_AUGMENTATION,
            },
        },
        output_path,
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = find_hsi_files(HSI_DATA_DIR)
    print("\nHSI validation mode: metadata inspection + fingerprint cache")
    all_files = filter_valid_hsi_files(
        files=all_files,
        hsi_channels=HSI_CHANNELS,
        hsi_key=HSI_KEY,
        log_path=output_dir / "invalid_hsi_files.txt",
    )
    training_files, validation_files = split_files(
        all_files,
        VALIDATION_FRACTION,
        SEED,
    )

    print(f"\nDevice: {device}")
    print(f"Mixed precision: {use_amp}")
    print(f"Total HSI files: {len(all_files)}")
    print(f"Validation cache: {VALIDATION_CACHE}")
    print(f"Invalid-file log: {output_dir / 'invalid_hsi_files.txt'}")
    print(f"Training files: {len(training_files)}")
    print(f"Validation files: {len(validation_files)}")
    print(f"Training crop size: {TRAIN_CROP_SIZE} x {TRAIN_CROP_SIZE}")
    print("Validation mode: complete native-resolution images")

    training_dataset = HSIDataset(
        files=training_files,
        hsi_channels=HSI_CHANNELS,
        training=True,
        normalization=NORMALIZATION,
        crop_size=TRAIN_CROP_SIZE,
        crops_per_image=CROPS_PER_IMAGE,
        augment=USE_AUGMENTATION,
    )
    validation_dataset = HSIDataset(
        files=validation_files,
        hsi_channels=HSI_CHANNELS,
        training=False,
        normalization=NORMALIZATION,
        crop_size=None,
        crops_per_image=1,
        augment=False,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    training_loader = DataLoader(
        training_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
    )

    model = HSIVAE(
        hsi_channels=HSI_CHANNELS,
        base_channels=BASE_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        num_res_blocks=NUM_RES_BLOCKS,
    ).to(device)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    print(
        f"Trainable parameters: "
        f"{sum(parameter.numel() for parameter in trainable_parameters):,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-7,
    )
    scaler = GradScaler(enabled=use_amp)

    best_validation_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        beta = get_kl_beta(epoch)

        print(
            f"\nEpoch {epoch:03d}/{NUM_EPOCHS:03d} | "
            f"KL beta: {beta:.8f}"
        )

        training_metrics = train_one_epoch(
            model=model,
            loader=training_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            beta=beta,
        )

        validation_metrics = validate(
            model=model,
            loader=validation_loader,
            device=device,
            use_amp=use_amp,
            beta=beta,
        )

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:03d}/{NUM_EPOCHS:03d} | "
            f"LR: {current_lr:.2e} | Beta: {beta:.8f}\n"
            f"  Train total: {training_metrics['total_loss']:.6f} | "
            f"Train reconstruction: "
            f"{training_metrics['reconstruction_loss']:.6f} | "
            f"Train KL: {training_metrics['kl_loss']:.6f}\n"
            f"  Val total: {validation_metrics['total_loss']:.6f} | "
            f"Val reconstruction: "
            f"{validation_metrics['reconstruction_loss']:.6f} | "
            f"Val KL: {validation_metrics['kl_loss']:.6f}\n"
            f"  Val MRAE: {validation_metrics['mrae']:.6f} | "
            f"Val RMSE: {validation_metrics['rmse']:.6f} | "
            f"Val SAM: {validation_metrics['sam']:.6f} | "
            f"Val PSNR: {validation_metrics['psnr']:.4f} | "
            f"Val SSIM: {validation_metrics['ssim']:.4f}"
        )

        save_checkpoint(
            output_path=output_dir / "last_vae.pth",
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            epoch=epoch,
            beta=beta,
            validation_metrics=validation_metrics,
        )

        if validation_metrics["total_loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["total_loss"]
            save_checkpoint(
                output_path=output_dir / "best_vae.pth",
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                epoch=epoch,
                beta=beta,
                validation_metrics=validation_metrics,
            )
            print(
                "  New best checkpoint: "
                f"validation total loss = {best_validation_loss:.6f}"
            )


if __name__ == "__main__":
    main()
