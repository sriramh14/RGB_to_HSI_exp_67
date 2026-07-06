
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Project imports
# ============================================================================
# Adjust only these import paths if your project layout is different.
from models.HSI_VAE import HSIVAE
from models.hsi_mmdit import RGBConditionedHSIMMDiT

from loss.mrae import mrae
from loss.psnr import psnr
from loss.rmse import rmse
from loss.sam import sam
from loss.ssim import ssim


# ============================================================================
# Configuration
# ============================================================================

# "train", "infer", or "train_and_infer"
RUN_MODE = "train"

# Training data.
TRAIN_HSI_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Train_spectral/Train_spectral"
)
TRAIN_RGB_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Train_RGB/Train_RGB"
)

# Validation data is intentionally in a separate pair of folders.
VALIDATION_HSI_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/Valid_spectral/Valid_spectral"
)
VALIDATION_RGB_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/Valid_RGB/Valid_RGB"
)

# Used to initialise training. New MM-DiT checkpoints also contain the
# complete frozen VAE, so inference does not require this file.
# It is retained as a fallback for older MM-DiT checkpoints.
VAE_CHECKPOINT = "./vae_checkpoints/best_vae.pth"
OUTPUT_DIR = "./mmdit_checkpoints"

# Dataset-validation caches. They are reused while the HSI file paths,
# sizes, and modification times remain unchanged.
TRAIN_PAIR_VALIDATION_CACHE = (
    Path(OUTPUT_DIR) / "training_pair_validation_cache.pth"
)
VALIDATION_PAIR_VALIDATION_CACHE = (
    Path(OUTPUT_DIR) / "validation_pair_validation_cache.pth"
)

# Used by RUN_MODE="infer". The best checkpoint is normally selected here.
INFERENCE_CHECKPOINT = "./mmdit_checkpoints/best_mmdit.pth"
RESUME_CHECKPOINT: Optional[str] = None

# Number of randomly selected full-resolution validation images in inference mode.
NUM_RANDOM_INFERENCE_IMAGES = 5
INFERENCE_OUTPUT_DIR = "./mmdit_inference"
INFERENCE_STEPS = 30

# HSI/RGB file layout.
HSI_KEY = "cube"
SUPPORTED_HSI_EXTENSIONS = {".npy", ".npz", ".mat", ".pt", ".pth"}
SUPPORTED_RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".npy", ".pt", ".pth"}

# This must match the normalization used to train the HSI VAE.
# "none", "minmax", or "band_minmax".
HSI_NORMALIZATION = "none"

# Range used only when CLAMP_PREDICTION_FOR_METRICS=True.
# The imported PSNR/SSIM functions retain their own definitions.
METRIC_DATA_RANGE = 1.0
CLAMP_PREDICTION_FOR_METRICS = False

# VAE architecture.
HSI_CHANNELS = 31
BASE_CHANNELS = 64
LATENT_CHANNELS = 16
NUM_RES_BLOCKS = 2

# The supplied VAE has two stride-2 downsamples.
VAE_DOWNSAMPLE_FACTOR = 4

# MM-DiT architecture.
HIDDEN_SIZE = 512
DEPTH = 8
NUM_HEADS = 8
MLP_RATIO = 4.0
MMDIT_PATCH_SIZE = 2
RGB_BASE_CHANNELS = 64
RGB_FEATURE_CHANNELS = 256
QK_NORM = True

# The image size must be divisible by:
# VAE_DOWNSAMPLE_FACTOR * MMDIT_PATCH_SIZE = 8 by default.
TRAIN_CROP_SIZE = 256
PATCHES_PER_IMAGE = 1

# Training.
BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 2
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MIN_LEARNING_RATE = 1e-7
GRADIENT_CLIP_NORM = 1.0

NUM_WORKERS = 4
USE_AMP = True
USE_AUGMENTATION = True
SEED = 42
PRINT_EVERY = 30

# Rectified-flow timestep distribution:
# u ~ Normal(LOGIT_NORMAL_MEAN, LOGIT_NORMAL_STD^2)
# t = sigmoid(u)
LOGIT_NORMAL_MEAN = 0.0
LOGIT_NORMAL_STD = 1.0

# Optional auxiliary latent reconstruction loss.
# Set to zero for pure rectified-flow velocity training.
LATENT_L1_WEIGHT = 0.0

# Validation flow loss is evaluated on a deterministic uniform timestep grid.
VALIDATION_TIMESTEP_GRID_SIZE = 100

# Actual RGB -> HSI reconstruction metrics require iterative sampling.
# None evaluates every validation image. A positive integer limits cost.
VALIDATION_METRIC_MAX_IMAGES: Optional[int] = 20
VALIDATION_INFERENCE_STEPS = 30

# Training pixel metrics use the one-step x0 estimate at the sampled t.
# These are useful diagnostics but are not the same as full inference metrics.
COMPUTE_TRAIN_ONE_STEP_METRICS = True
TRAIN_METRIC_EVERY = 1

# Latent statistics.
# The script first looks for mean/std in the VAE checkpoint.
# If absent, it computes per-channel statistics on the training set.
COMPUTE_LATENT_STATS_IF_MISSING = True
LATENT_STATS_MAX_BATCHES: Optional[int] = None
LATENT_STD_EPS = 1e-6

# Pseudo-RGB HSI bands used only for saved inference previews.
# Change these indices for your wavelength ordering.
VISUALIZATION_BANDS = (20, 10, 2)


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ============================================================================
# AMP helpers
# ============================================================================

def autocast_context(
    device: torch.device,
    enabled: bool,
):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled,
    )


# ============================================================================
# HSI/RGB loading
# ============================================================================

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
        raise ValueError(
            f"No numeric three-dimensional array was found in {file_path}."
        )

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
            array = np.asarray(h5_file[preferred_key])
            candidates.append((preferred_key, array))

        if not candidates:
            def visitor(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                if obj.ndim != 3:
                    return
                try:
                    if np.issubdtype(obj.dtype, np.number):
                        candidates.append((name, np.asarray(obj)))
                except TypeError:
                    return

            h5_file.visititems(visitor)

    if not candidates:
        raise ValueError(
            f"No numeric three-dimensional HSI dataset was found in {file_path}."
        )

    _, cube = max(candidates, key=lambda item: item[1].size)

    # MATLAB v7.3/HDF5 arrays are commonly stored with reversed dimensions.
    # convert_to_chw() below performs the final spectral-axis identification.
    cube = np.transpose(
        cube,
        axes=tuple(range(cube.ndim - 1, -1, -1)),
    )
    return cube


def load_hsi_file(file_path: Path) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension == ".npy":
        cube = np.load(file_path)

    elif extension == ".npz":
        loaded = np.load(file_path)
        candidates = [
            loaded[key]
            for key in loaded.files
            if loaded[key].ndim == 3
        ]
        if not candidates:
            raise ValueError(
                f"No three-dimensional array was found in {file_path}."
            )
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
            cube = load_mat_v73(
                file_path=file_path,
                preferred_key=HSI_KEY,
            )

    elif extension in {".pt", ".pth"}:
        try:
            loaded = torch.load(
                file_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            loaded = torch.load(
                file_path,
                map_location="cpu",
            )

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
            raise TypeError(
                f"Unsupported object type in {file_path}: {type(loaded)}"
            )

    else:
        raise ValueError(
            f"Unsupported HSI extension: {extension}"
        )

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
        return np.ascontiguousarray(
            np.transpose(cube, (2, 0, 1))
        )

    if cube.shape[1] == hsi_channels:
        return np.ascontiguousarray(
            np.transpose(cube, (1, 0, 2))
        )

    raise ValueError(
        f"Could not identify the spectral axis in {file_path}. "
        f"Found shape {cube.shape}; expected {hsi_channels} bands."
    )


def load_rgb_file(file_path: Path) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension in {".png", ".jpg", ".jpeg"}:
        image = Image.open(file_path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.ascontiguousarray(
            np.transpose(array, (2, 0, 1))
        )

    if extension == ".npy":
        array = np.load(file_path).astype(np.float32)

    elif extension in {".pt", ".pth"}:
        try:
            loaded = torch.load(
                file_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            loaded = torch.load(
                file_path,
                map_location="cpu",
            )

        if isinstance(loaded, torch.Tensor):
            array = loaded.detach().cpu().float().numpy()
        elif isinstance(loaded, np.ndarray):
            array = loaded.astype(np.float32)
        else:
            raise TypeError(
                f"Unsupported RGB object in {file_path}: {type(loaded)}"
            )

    else:
        raise ValueError(
            f"Unsupported RGB extension: {extension}"
        )

    array = np.squeeze(array)

    if array.ndim == 2:
        array = np.stack([array, array, array], axis=0)
    elif array.ndim == 3 and array.shape[0] == 3:
        pass
    elif array.ndim == 3 and array.shape[-1] == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(
            f"Could not convert RGB file {file_path} to CHW. "
            f"Found shape {array.shape}."
        )

    array = np.asarray(array, dtype=np.float32)

    # Normalize common uint-like NPY/PT representations.
    if np.nanmax(array) > 1.5:
        array = array / 255.0

    return np.ascontiguousarray(array)


def normalize_hsi_cube(
    cube: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return cube

    if mode == "minmax":
        minimum = float(cube.min())
        maximum = float(cube.max())
        return (
            (cube - minimum)
            / (maximum - minimum + 1e-8)
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
            (cube - minimum)
            / (maximum - minimum + 1e-8)
        )

    raise ValueError(
        f"Unknown HSI normalization mode: {mode}"
    )


# ============================================================================
# File discovery, pairing, and validation
# ============================================================================

def find_files(
    directory: str,
    extensions: Sequence[str],
    kind: str,
) -> List[Path]:
    root = Path(directory)

    if not root.exists():
        raise FileNotFoundError(
            f"{kind} directory does not exist: {root}"
        )

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    )

    if not files:
        raise RuntimeError(
            f"No supported {kind} files were found in {root}."
        )

    return files


def _index_unique_stems(
    files: Sequence[Path],
    kind: str,
) -> Dict[str, Path]:
    index: Dict[str, Path] = {}

    for path in files:
        if path.stem in index:
            raise RuntimeError(
                f"Duplicate {kind} filename stem '{path.stem}'.\n"
                f"First:  {index[path.stem]}\n"
                f"Second: {path}"
            )
        index[path.stem] = path

    return index


def pair_hsi_rgb_files(
    hsi_directory: str,
    rgb_directory: str,
) -> List[Tuple[Path, Path]]:
    hsi_files = find_files(
        hsi_directory,
        SUPPORTED_HSI_EXTENSIONS,
        "HSI",
    )
    rgb_files = find_files(
        rgb_directory,
        SUPPORTED_RGB_EXTENSIONS,
        "RGB",
    )

    hsi_by_stem = _index_unique_stems(
        hsi_files,
        "HSI",
    )
    rgb_by_stem = _index_unique_stems(
        rgb_files,
        "RGB",
    )

    shared_stems = sorted(
        set(hsi_by_stem) & set(rgb_by_stem)
    )

    missing_rgb = sorted(
        set(hsi_by_stem) - set(rgb_by_stem)
    )
    missing_hsi = sorted(
        set(rgb_by_stem) - set(hsi_by_stem)
    )

    if missing_rgb:
        print(
            f"Warning: {len(missing_rgb)} HSI files have no matching RGB file."
        )
    if missing_hsi:
        print(
            f"Warning: {len(missing_hsi)} RGB files have no matching HSI file."
        )

    if not shared_stems:
        raise RuntimeError(
            "No paired HSI/RGB files were found. "
            "The paired files must have identical filename stems."
        )

    pairs = [
        (
            hsi_by_stem[stem],
            rgb_by_stem[stem],
        )
        for stem in shared_stems
    ]

    print(
        f"Found {len(pairs)} paired files in:\n"
        f"  HSI: {hsi_directory}\n"
        f"  RGB: {rgb_directory}"
    )
    return pairs


def make_files_fingerprint(
    files: Sequence[Path],
) -> str:
    """
    Create a cache fingerprint from file path, size, and modification time.

    The cache is invalidated automatically if an HSI file is added, removed,
    replaced, or modified.
    """
    records = []

    for file_path in files:
        stat = file_path.stat()
        records.append(
            f"{file_path.resolve()}|"
            f"{stat.st_size}|"
            f"{stat.st_mtime_ns}"
        )

    return hashlib.sha256(
        "\n".join(records).encode("utf-8")
    ).hexdigest()


def is_possible_hsi_shape(
    shape: Sequence[int],
    hsi_channels: int,
) -> bool:
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
    """
    Inspect a MATLAB v7.3/HDF5 file without loading its full HSI cube.
    """
    candidates: List[
        Tuple[str, Tuple[int, ...]]
    ] = []

    with h5py.File(
        str(file_path),
        "r",
    ) as h5_file:
        if (
            hsi_key in h5_file
            and isinstance(
                h5_file[hsi_key],
                h5py.Dataset,
            )
        ):
            dataset = h5_file[hsi_key]
            candidates.append(
                (
                    hsi_key,
                    tuple(
                        int(value)
                        for value in dataset.shape
                    ),
                )
            )
        else:
            def visitor(name, obj):
                if (
                    not isinstance(obj, h5py.Dataset)
                    or obj.ndim != 3
                ):
                    return

                try:
                    if np.issubdtype(
                        obj.dtype,
                        np.number,
                    ):
                        candidates.append(
                            (
                                name,
                                tuple(
                                    int(value)
                                    for value in obj.shape
                                ),
                            )
                        )
                except TypeError:
                    return

            h5_file.visititems(visitor)

    if not candidates:
        raise ValueError(
            f"No numerical three-dimensional dataset "
            f"was found in {file_path}."
        )

    if not any(
        is_possible_hsi_shape(
            shape,
            hsi_channels,
        )
        for _, shape in candidates
    ):
        raise ValueError(
            f"No {hsi_channels}-band cube was found in "
            f"{file_path}. HDF5 datasets: {candidates}"
        )


def inspect_standard_mat_file(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    """
    Inspect a standard MATLAB file using scipy.io.whosmat(), which reads
    array metadata rather than loading every array.
    """
    try:
        metadata = sio.whosmat(file_path)
    except (
        NotImplementedError,
        ValueError,
        OSError,
    ):
        # MATLAB v7.3 files require HDF5 inspection.
        inspect_hdf5_mat_file(
            file_path=file_path,
            hsi_channels=hsi_channels,
            hsi_key=hsi_key,
        )
        return

    candidates = [
        (
            name,
            tuple(
                int(value)
                for value in shape
            ),
        )
        for name, shape, _ in metadata
        if len(shape) == 3
    ]

    if not candidates:
        raise ValueError(
            f"No three-dimensional array was found in "
            f"{file_path}."
        )

    preferred = [
        candidate
        for candidate in candidates
        if candidate[0] == hsi_key
    ]
    arrays_to_check = (
        preferred
        if preferred
        else candidates
    )

    if not any(
        is_possible_hsi_shape(
            shape,
            hsi_channels,
        )
        for _, shape in arrays_to_check
    ):
        raise ValueError(
            f"No {hsi_channels}-band cube was found in "
            f"{file_path}. MATLAB arrays: {candidates}"
        )


def inspect_hsi_file_metadata(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    """
    Validate an HSI file in the same way as the earlier training script.

    .mat:
        Inspect metadata only with whosmat() or an HDF5 header.

    .npy/.npz/.pt/.pth:
        Reuse the normal loader because these formats are comparatively
        inexpensive to inspect.
    """
    if file_path.suffix.lower() == ".mat":
        inspect_standard_mat_file(
            file_path=file_path,
            hsi_channels=hsi_channels,
            hsi_key=hsi_key,
        )
        return

    cube = load_hsi_file(file_path)

    if not is_possible_hsi_shape(
        cube.shape,
        hsi_channels,
    ):
        raise ValueError(
            f"Invalid HSI shape {cube.shape} in "
            f"{file_path}."
        )


def filter_valid_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    hsi_channels: int,
    log_path: Path,
    cache_path: Path,
) -> List[Tuple[Path, Path]]:
    """
    Metadata-first HSI validation with a persistent cache.

    This mirrors the checking approach in the earlier script:
      1. Pair files by filename stem.
      2. Check MATLAB files through metadata rather than loading full cubes.
      3. Cache valid/invalid results using a file fingerprint.
      4. Skip invalid HSI files and write their errors to a log.

    RGB files have already been checked for a supported extension during
    pairing. Their full pixel data is loaded only by the Dataset.
    """
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = list(pairs)
    hsi_files = [
        hsi_path
        for hsi_path, _ in pairs
    ]

    fingerprint = make_files_fingerprint(
        hsi_files
    )

    pair_lookup = {
        str(hsi_path.resolve()): (
            hsi_path,
            rgb_path,
        )
        for hsi_path, rgb_path in pairs
    }

    # ------------------------------------------------------------------
    # Reuse an unchanged validation cache.
    # ------------------------------------------------------------------
    if cache_path.exists():
        try:
            try:
                cached = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                cached = torch.load(
                    cache_path,
                    map_location="cpu",
                )

            if (
                isinstance(cached, dict)
                and cached.get("fingerprint")
                == fingerprint
            ):
                valid_paths = cached.get(
                    "valid_hsi_paths",
                    [],
                )
                invalid_records = cached.get(
                    "invalid_records",
                    [],
                )

                valid_pairs = [
                    pair_lookup[path]
                    for path in valid_paths
                    if path in pair_lookup
                ]

                print(
                    f"\nUsing cached pair validation: "
                    f"{cache_path}"
                )
                print(
                    f"Valid pairs:   {len(valid_pairs)}"
                )
                print(
                    f"Invalid files: "
                    f"{len(invalid_records)}"
                )

                for record in invalid_records:
                    print(
                        "\nCached invalid file:\n"
                        f"  File:  {record['path']}\n"
                        f"  Error: {record['error']}"
                    )

                if valid_pairs:
                    return valid_pairs

        except Exception as error:
            print(
                "\nCould not use the validation cache. "
                "The dataset will be checked again.\n"
                f"Reason: {error}"
            )

    # ------------------------------------------------------------------
    # Perform a fresh metadata-first scan.
    # ------------------------------------------------------------------
    print(
        "\nChecking HSI file metadata before use..."
    )

    valid_pairs: List[
        Tuple[Path, Path]
    ] = []
    invalid_records: List[dict] = []

    for index, (
        hsi_path,
        rgb_path,
    ) in enumerate(
        pairs,
        start=1,
    ):
        try:
            inspect_hsi_file_metadata(
                file_path=hsi_path,
                hsi_channels=hsi_channels,
                hsi_key=HSI_KEY,
            )
            valid_pairs.append(
                (hsi_path, rgb_path)
            )

        except Exception as error:
            invalid_records.append(
                {
                    "path": str(
                        hsi_path.resolve()
                    ),
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

            print(
                "\nSkipping invalid HSI file:\n"
                f"  File:  {hsi_path}\n"
                f"  Error: {error}"
            )

        if (
            index % 100 == 0
            or index == len(pairs)
        ):
            print(
                f"Checked {index}/{len(pairs)} | "
                f"Valid: {len(valid_pairs)} | "
                f"Invalid: {len(invalid_records)}"
            )

    if not valid_pairs:
        raise RuntimeError(
            "No valid HSI/RGB pairs remain after "
            "metadata validation."
        )

    if invalid_records:
        with log_path.open(
            "w",
            encoding="utf-8",
        ) as log_file:
            for record in invalid_records:
                log_file.write(
                    f"{record['path']} | "
                    f"{record['error']}\n"
                )

        print(
            f"\nInvalid-file log saved to: "
            f"{log_path}"
        )

    torch.save(
        {
            "fingerprint": fingerprint,
            "valid_hsi_paths": [
                str(hsi_path.resolve())
                for hsi_path, _ in valid_pairs
            ],
            "invalid_records": invalid_records,
        },
        cache_path,
    )

    print(
        f"Validation cache saved to: "
        f"{cache_path}"
    )

    return valid_pairs


# ============================================================================
# Paired spatial transforms
# ============================================================================

def _pad_tensor_to_minimum_size(
    tensor: torch.Tensor,
    minimum_height: int,
    minimum_width: int,
) -> torch.Tensor:
    _, height, width = tensor.shape

    pad_height = max(
        0,
        minimum_height - height,
    )
    pad_width = max(
        0,
        minimum_width - width,
    )

    if pad_height == 0 and pad_width == 0:
        return tensor

    return F.pad(
        tensor,
        (0, pad_width, 0, pad_height),
        mode="replicate",
    )


def random_crop_pair(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
    crop_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hsi = _pad_tensor_to_minimum_size(
        hsi,
        crop_size,
        crop_size,
    )
    rgb = _pad_tensor_to_minimum_size(
        rgb,
        crop_size,
        crop_size,
    )

    _, height, width = hsi.shape

    top = random.randint(
        0,
        height - crop_size,
    )
    left = random.randint(
        0,
        width - crop_size,
    )

    return (
        hsi[
            :,
            top:top + crop_size,
            left:left + crop_size,
        ],
        rgb[
            :,
            top:top + crop_size,
            left:left + crop_size,
        ],
    )


def center_crop_pair(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
    crop_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hsi = _pad_tensor_to_minimum_size(
        hsi,
        crop_size,
        crop_size,
    )
    rgb = _pad_tensor_to_minimum_size(
        rgb,
        crop_size,
        crop_size,
    )

    _, height, width = hsi.shape

    top = (height - crop_size) // 2
    left = (width - crop_size) // 2

    return (
        hsi[
            :,
            top:top + crop_size,
            left:left + crop_size,
        ],
        rgb[
            :,
            top:top + crop_size,
            left:left + crop_size,
        ],
    )


def augment_pair(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        hsi = torch.flip(hsi, dims=[1])
        rgb = torch.flip(rgb, dims=[1])

    if random.random() < 0.5:
        hsi = torch.flip(hsi, dims=[2])
        rgb = torch.flip(rgb, dims=[2])

    rotations = random.randint(0, 3)
    if rotations:
        hsi = torch.rot90(
            hsi,
            k=rotations,
            dims=(1, 2),
        )
        rgb = torch.rot90(
            rgb,
            k=rotations,
            dims=(1, 2),
        )

    return hsi.contiguous(), rgb.contiguous()


def pad_pair_to_multiple(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
    multiple: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:
    _, original_height, original_width = hsi.shape

    pad_height = (
        multiple - original_height % multiple
    ) % multiple
    pad_width = (
        multiple - original_width % multiple
    ) % multiple

    if pad_height == 0 and pad_width == 0:
        return (
            hsi,
            rgb,
            original_height,
            original_width,
        )

    hsi = F.pad(
        hsi,
        (0, pad_width, 0, pad_height),
        mode="replicate",
    )
    rgb = F.pad(
        rgb,
        (0, pad_width, 0, pad_height),
        mode="replicate",
    )

    return (
        hsi,
        rgb,
        original_height,
        original_width,
    )


# ============================================================================
# Dataset
# ============================================================================

class HSIRGBPairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        hsi_channels: int,
        crop_size: Optional[int],
        patches_per_image: int,
        training: bool,
        normalization: str,
        augment: bool,
        return_paths: bool = False,
    ):
        self.pairs = list(pairs)
        self.hsi_channels = hsi_channels
        self.crop_size = crop_size
        self.patches_per_image = patches_per_image
        self.training = training
        self.normalization = normalization
        self.augment = augment
        self.return_paths = return_paths

        if training and crop_size is None:
            raise ValueError(
                "Training requires a finite crop_size."
            )

    def __len__(self) -> int:
        multiplier = (
            self.patches_per_image
            if self.training
            else 1
        )
        return len(self.pairs) * multiplier

    def _load_pair(
        self,
        pair_index: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Path,
        Path,
    ]:
        hsi_path, rgb_path = self.pairs[pair_index]

        hsi_array = convert_hsi_to_chw(
            load_hsi_file(hsi_path),
            hsi_channels=self.hsi_channels,
            file_path=hsi_path,
        )
        hsi_array = normalize_hsi_cube(
            hsi_array,
            mode=self.normalization,
        )

        rgb_array = load_rgb_file(rgb_path)

        if hsi_array.shape[1:] != rgb_array.shape[1:]:
            raise ValueError(
                f"Spatial mismatch for pair {hsi_path.stem}: "
                f"HSI={hsi_array.shape[1:]}, "
                f"RGB={rgb_array.shape[1:]}."
            )

        if not np.isfinite(hsi_array).all():
            raise ValueError(
                f"HSI contains NaN/Inf: {hsi_path}"
            )
        if not np.isfinite(rgb_array).all():
            raise ValueError(
                f"RGB contains NaN/Inf: {rgb_path}"
            )

        hsi = torch.from_numpy(
            hsi_array.copy()
        ).float()
        rgb = torch.from_numpy(
            rgb_array.copy()
        ).float()

        return hsi, rgb, hsi_path, rgb_path

    def __getitem__(self, index: int):
        if self.training:
            pair_index = (
                index // self.patches_per_image
            )
        else:
            pair_index = index

        hsi, rgb, hsi_path, rgb_path = (
            self._load_pair(pair_index)
        )

        if self.crop_size is not None:
            if self.training:
                hsi, rgb = random_crop_pair(
                    hsi,
                    rgb,
                    crop_size=self.crop_size,
                )
            else:
                hsi, rgb = center_crop_pair(
                    hsi,
                    rgb,
                    crop_size=self.crop_size,
                )

        if self.training and self.augment:
            hsi, rgb = augment_pair(
                hsi,
                rgb,
            )

        if self.return_paths:
            return (
                hsi,
                rgb,
                str(hsi_path),
                str(rgb_path),
            )

        return hsi, rgb


# ============================================================================
# VAE loading and latent statistics
# ============================================================================

def _load_torch_checkpoint(
    path: str | Path,
    device: torch.device | str,
):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def _strip_module_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    if all(
        key.startswith("module.")
        for key in state_dict
    ):
        return {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    return state_dict


def _extract_state_dict(
    checkpoint,
    candidate_keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in candidate_keys:
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return _strip_module_prefix(state)

        if checkpoint and all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        ):
            return _strip_module_prefix(checkpoint)

    raise KeyError(
        "Could not locate a model state_dict in the checkpoint."
    )


def _extract_latent_stats(
    checkpoint,
    latent_channels: int,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if not isinstance(checkpoint, dict):
        return None, None

    candidate_containers = [checkpoint]

    for key in (
        "latent_stats",
        "statistics",
        "model_config",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            candidate_containers.append(value)

    mean_keys = (
        "latent_mean",
        "z_mean",
        "mean",
    )
    std_keys = (
        "latent_std",
        "z_std",
        "std",
    )

    mean = None
    std = None

    for container in candidate_containers:
        if mean is None:
            for key in mean_keys:
                if key in container:
                    mean = torch.as_tensor(
                        container[key],
                        dtype=torch.float32,
                    )
                    break

        if std is None:
            for key in std_keys:
                if key in container:
                    std = torch.as_tensor(
                        container[key],
                        dtype=torch.float32,
                    )
                    break

    if mean is None or std is None:
        return None, None

    if mean.numel() not in (1, latent_channels):
        return None, None
    if std.numel() not in (1, latent_channels):
        return None, None

    if mean.numel() == 1:
        mean = mean.repeat(latent_channels)
    if std.numel() == 1:
        std = std.repeat(latent_channels)

    return (
        mean.reshape(latent_channels),
        std.reshape(latent_channels),
    )


def load_pretrained_vae(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[
    HSIVAE,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    checkpoint = _load_torch_checkpoint(
        checkpoint_path,
        device="cpu",
    )

    config = (
        checkpoint.get("model_config", {})
        if isinstance(checkpoint, dict)
        else {}
    )

    vae = HSIVAE(
        hsi_channels=config.get(
            "hsi_channels",
            HSI_CHANNELS,
        ),
        base_channels=config.get(
            "base_channels",
            BASE_CHANNELS,
        ),
        latent_channels=config.get(
            "latent_channels",
            LATENT_CHANNELS,
        ),
        num_res_blocks=config.get(
            "num_res_blocks",
            NUM_RES_BLOCKS,
        ),
    )

    state_dict = _extract_state_dict(
        checkpoint,
        candidate_keys=(
            "model_state_dict",
            "vae_state_dict",
            "state_dict",
        ),
    )

    vae.load_state_dict(
        state_dict,
        strict=True,
    )
    vae.eval()

    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    latent_mean, latent_std = (
        _extract_latent_stats(
            checkpoint,
            latent_channels=LATENT_CHANNELS,
        )
    )

    return (
        vae.to(device),
        latent_mean,
        latent_std,
    )


@torch.no_grad()
def compute_latent_statistics(
    vae: HSIVAE,
    loader: DataLoader,
    latent_channels: int,
    device: torch.device,
    max_batches: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    print(
        "\nComputing per-channel statistics of the VAE posterior mean..."
    )

    channel_sum = torch.zeros(
        latent_channels,
        dtype=torch.float64,
        device=device,
    )
    channel_square_sum = torch.zeros(
        latent_channels,
        dtype=torch.float64,
        device=device,
    )
    value_count = 0

    vae.eval()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        hsi = batch[0].to(
            device,
            non_blocking=True,
        )

        _, mu, _ = vae.encode(
            hsi,
            sample=False,
        )

        mu64 = mu.double()

        channel_sum += mu64.sum(
            dim=(0, 2, 3)
        )
        channel_square_sum += (
            mu64.square().sum(
                dim=(0, 2, 3)
            )
        )

        value_count += (
            mu.shape[0]
            * mu.shape[2]
            * mu.shape[3]
        )

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        if batch_index % 100 == 0:
            print(
                f"  Latent-stat batches: {batch_index}"
            )

    if value_count == 0:
        raise RuntimeError(
            "No values were processed while computing latent statistics."
        )

    mean = channel_sum / value_count
    variance = (
        channel_square_sum / value_count
        - mean.square()
    )
    std = torch.sqrt(
        variance.clamp_min(
            LATENT_STD_EPS**2
        )
    )

    mean = mean.float().cpu()
    std = std.float().cpu()

    print(
        f"Latent mean: {mean.tolist()}\n"
        f"Latent std:  {std.tolist()}"
    )

    return mean, std


# ============================================================================
# MM-DiT construction and checkpoint handling
# ============================================================================

def default_model_config() -> dict:
    return {
        "latent_channels": LATENT_CHANNELS,
        "hidden_size": HIDDEN_SIZE,
        "depth": DEPTH,
        "num_heads": NUM_HEADS,
        "patch_size": MMDIT_PATCH_SIZE,
        "rgb_base_channels": RGB_BASE_CHANNELS,
        "rgb_feature_channels": RGB_FEATURE_CHANNELS,
        "mlp_ratio": MLP_RATIO,
        "qk_norm": QK_NORM,
    }


def default_vae_config() -> dict:
    return {
        "hsi_channels": HSI_CHANNELS,
        "base_channels": BASE_CHANNELS,
        "latent_channels": LATENT_CHANNELS,
        "num_res_blocks": NUM_RES_BLOCKS,
    }


def vae_config_from_model(
    vae: HSIVAE,
) -> dict:
    """
    Read the actual VAE architecture from the instantiated frozen model.

    This prevents a checkpoint mismatch when the VAE checkpoint architecture
    differs from the constants near the top of this training script.
    """
    return {
        "hsi_channels": int(
            vae.encoder.input_conv.in_channels
        ),
        "base_channels": int(
            vae.encoder.input_conv.out_channels
        ),
        "latent_channels": int(
            vae.encoder.output_conv.out_channels // 2
        ),
        "num_res_blocks": int(
            len(vae.encoder.level1)
        ),
    }


def build_vae_from_config(
    vae_config: Optional[dict],
    device: torch.device,
) -> HSIVAE:
    config = default_vae_config()

    if vae_config is not None:
        for key in config:
            if key in vae_config:
                config[key] = vae_config[key]

    vae = HSIVAE(
        hsi_channels=config["hsi_channels"],
        base_channels=config["base_channels"],
        latent_channels=config["latent_channels"],
        num_res_blocks=config["num_res_blocks"],
    )

    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    return vae.to(device)


def build_mmdit(
    vae: HSIVAE,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    device: torch.device,
    model_config: Optional[dict] = None,
) -> RGBConditionedHSIMMDiT:
    config = default_model_config()

    if model_config is not None:
        for key in config:
            if key in model_config:
                config[key] = model_config[key]

    model = RGBConditionedHSIMMDiT(
        latent_channels=config["latent_channels"],
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        patch_size=config["patch_size"],
        rgb_base_channels=config["rgb_base_channels"],
        rgb_feature_channels=config["rgb_feature_channels"],
        mlp_ratio=config["mlp_ratio"],
        qk_norm=config["qk_norm"],
        vae=vae,
        freeze_vae=True,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )

    return model.to(device)


def save_checkpoint(
    output_path: Path,
    model: RGBConditionedHSIMMDiT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_validation_flow_loss: float,
    validation_metrics: dict,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_flow_loss": (
                best_validation_flow_loss
            ),
            "validation_metrics": validation_metrics,
            "latent_mean": (
                model.latent_mean.detach()
                .float()
                .cpu()
                .reshape(-1)
            ),
            "latent_std": (
                model.latent_std.detach()
                .float()
                .cpu()
                .reshape(-1)
            ),
            "model_config": default_model_config(),
            "vae_config": vae_config_from_model(
                model.vae
            ),
            "contains_embedded_vae": True,
        },
        output_path,
    )


def load_mmdit_weights(
    model: RGBConditionedHSIMMDiT,
    checkpoint,
) -> None:
    state_dict = _extract_state_dict(
        checkpoint,
        candidate_keys=(
            "model_state_dict",
            "mmdit_state_dict",
            "state_dict",
        ),
    )

    contains_vae_weights = any(
        key.startswith("vae.")
        for key in state_dict
    )

    incompatible = model.load_state_dict(
        state_dict,
        strict=contains_vae_weights,
    )

    if contains_vae_weights:
        # strict=True already verifies every MM-DiT, RGB encoder, VAE, and
        # latent-statistics key.
        return

    # Backward compatibility for checkpoints created before the VAE was saved.
    unexpected = list(incompatible.unexpected_keys)
    missing_non_vae = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("vae.")
    ]

    if unexpected:
        raise RuntimeError(
            "Unexpected MM-DiT checkpoint keys:\n"
            + "\n".join(unexpected)
        )

    if missing_non_vae:
        raise RuntimeError(
            "Missing MM-DiT checkpoint keys:\n"
            + "\n".join(missing_non_vae)
        )


# ============================================================================
# Rectified flow
# ============================================================================

def sample_logit_normal_timesteps(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    normal_sample = (
        torch.randn(
            batch_size,
            device=device,
        )
        * std
        + mean
    )
    return torch.sigmoid(normal_sample)


def construct_rectified_flow_batch(
    clean_latent: torch.Tensor,
    timestep: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    noise = torch.randn_like(clean_latent)

    timestep_view = timestep.to(
        clean_latent.dtype
    ).reshape(-1, 1, 1, 1)

    noisy_latent = (
        (1.0 - timestep_view) * clean_latent
        + timestep_view * noise
    )

    target_velocity = noise - clean_latent

    return (
        noisy_latent,
        target_velocity,
        noise,
    )


def recover_clean_latent(
    noisy_latent: torch.Tensor,
    predicted_velocity: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    timestep_view = timestep.to(
        noisy_latent.dtype
    ).reshape(-1, 1, 1, 1)

    return (
        noisy_latent
        - timestep_view * predicted_velocity
    )


# ============================================================================
# HSI metric aggregation using the project's existing metric functions
# ============================================================================

def _metric_output_to_scalar(
    value,
    metric_name: str,
) -> float:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    value = value.detach().float()

    # The imported functions are expected to return a scalar. Taking the mean
    # also supports implementations that return a one-element/per-image tensor.
    if value.numel() != 1:
        value = value.mean()

    scalar = float(value.item())

    if not math_is_finite_or_positive_infinity(scalar):
        raise FloatingPointError(
            f"{metric_name} returned a non-finite value: {scalar}"
        )

    return scalar


def math_is_finite_or_positive_infinity(value: float) -> bool:
    # Positive infinity is valid for PSNR when prediction exactly equals target.
    return bool(
        np.isfinite(value)
        or value == float("inf")
    )


@torch.no_grad()
def calculate_project_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """
    Calculate metrics with the functions imported from the project's loss folder.

    The argument order matches the user's existing code:
        metric(target, reconstruction)
    """
    prediction = prediction.detach().float()
    target = target.detach().float()

    if prediction.shape != target.shape:
        raise ValueError(
            f"Metric shape mismatch: prediction={prediction.shape}, "
            f"target={target.shape}"
        )

    if not torch.isfinite(prediction).all():
        raise FloatingPointError(
            "Prediction contains NaN or Inf during metric calculation."
        )
    if not torch.isfinite(target).all():
        raise FloatingPointError(
            "Target contains NaN or Inf during metric calculation."
        )

    return {
        "mrae": _metric_output_to_scalar(
            mrae(target, prediction),
            "MRAE",
        ),
        "rmse": _metric_output_to_scalar(
            rmse(target, prediction),
            "RMSE",
        ),
        "sam": _metric_output_to_scalar(
            sam(target, prediction),
            "SAM",
        ),
        "psnr": _metric_output_to_scalar(
            psnr(target, prediction),
            "PSNR",
        ),
        "ssim": _metric_output_to_scalar(
            ssim(target, prediction),
            "SSIM",
        ),
    }


@dataclass
class HSIMetricAccumulator:
    """
    Average the project's existing metrics equally over validation images.

    Metrics are called separately for each image. This avoids depending on the
    internal batch reduction used by each imported function and correctly
    handles a smaller final batch.
    """

    data_range: float = 1.0
    clamp_prediction: bool = False

    mrae_sum: float = 0.0
    rmse_sum: float = 0.0
    sam_sum: float = 0.0
    psnr_sum: float = 0.0
    ssim_sum: float = 0.0
    image_count: int = 0

    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        prediction = prediction.detach().float()
        target = target.detach().float()

        if prediction.shape != target.shape:
            raise ValueError(
                f"Metric shape mismatch: prediction={prediction.shape}, "
                f"target={target.shape}"
            )

        if self.clamp_prediction:
            prediction = prediction.clamp(
                0.0,
                self.data_range,
            )

        for sample_index in range(prediction.shape[0]):
            sample_metrics = calculate_project_metrics(
                prediction=prediction[
                    sample_index:sample_index + 1
                ],
                target=target[
                    sample_index:sample_index + 1
                ],
            )

            self.mrae_sum += sample_metrics["mrae"]
            self.rmse_sum += sample_metrics["rmse"]
            self.sam_sum += sample_metrics["sam"]
            self.psnr_sum += sample_metrics["psnr"]
            self.ssim_sum += sample_metrics["ssim"]
            self.image_count += 1

    def compute(self) -> dict:
        if self.image_count == 0:
            return {
                "mrae": float("nan"),
                "rmse": float("nan"),
                "sam": float("nan"),
                "psnr": float("nan"),
                "ssim": float("nan"),
            }

        denominator = float(self.image_count)

        return {
            "mrae": self.mrae_sum / denominator,
            "rmse": self.rmse_sum / denominator,
            "sam": self.sam_sum / denominator,
            "psnr": self.psnr_sum / denominator,
            "ssim": self.ssim_sum / denominator,
        }


# ============================================================================
# Continuous timestep-bin flow-loss tracking
# ============================================================================

TIMESTEP_BINS = (
    (0.00, 0.10),
    (0.10, 0.25),
    (0.25, 0.50),
    (0.50, 0.75),
    (0.75, 0.90),
    (0.90, 1.00),
)


def create_timestep_tracker() -> dict:
    return {
        (lower, upper): {
            "loss_sum": 0.0,
            "count": 0,
        }
        for lower, upper in TIMESTEP_BINS
    }


@torch.no_grad()
def update_timestep_tracker(
    tracker: dict,
    timestep: torch.Tensor,
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
) -> None:
    per_sample_loss = F.mse_loss(
        predicted_velocity.detach().float(),
        target_velocity.detach().float(),
        reduction="none",
    ).mean(dim=(1, 2, 3))

    for lower, upper in TIMESTEP_BINS:
        if upper == 1.0:
            mask = (
                (timestep >= lower)
                & (timestep <= upper)
            )
        else:
            mask = (
                (timestep >= lower)
                & (timestep < upper)
            )

        if not mask.any():
            continue

        tracker[(lower, upper)][
            "loss_sum"
        ] += per_sample_loss[mask].sum().item()

        tracker[(lower, upper)][
            "count"
        ] += int(mask.sum().item())


def finalize_timestep_tracker(
    tracker: dict,
) -> dict:
    result = {}

    for (lower, upper), values in tracker.items():
        count = values["count"]
        name = f"{lower:.2f}-{upper:.2f}"

        result[name] = {
            "flow_loss": (
                values["loss_sum"] / count
                if count > 0
                else float("nan")
            ),
            "count": count,
        }

    return result


def print_timestep_tracker(
    title: str,
    result: dict,
) -> None:
    print(f"\n{title}")

    for name, values in result.items():
        count = values["count"]
        loss = values["flow_loss"]

        if count == 0:
            print(
                f"  t={name} | no samples"
            )
        else:
            print(
                f"  t={name} | "
                f"flow loss={loss:.6f} | "
                f"samples={count}"
            )


# ============================================================================
# Training and validation
# ============================================================================

def train_one_epoch(
    model: RGBConditionedHSIMMDiT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.train()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    flow_loss_sum = 0.0
    latent_l1_sum = 0.0
    total_samples = 0

    metric_accumulator = HSIMetricAccumulator(
        data_range=METRIC_DATA_RANGE,
        clamp_prediction=(
            CLAMP_PREDICTION_FOR_METRICS
        ),
    )

    metric_batches = 0
    timestep_tracker = create_timestep_tracker()

    for batch_index, (hsi, rgb) in enumerate(
        loader,
        start=1,
    ):
        hsi = hsi.to(
            device,
            non_blocking=True,
        )
        rgb = rgb.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # The VAE is frozen and encode_hsi() uses no_grad internally.
        clean_latent, _, _ = model.encode_hsi(
            hsi,
            sample=False,
        )

        timestep = sample_logit_normal_timesteps(
            batch_size=hsi.shape[0],
            device=device,
            mean=LOGIT_NORMAL_MEAN,
            std=LOGIT_NORMAL_STD,
        )

        (
            noisy_latent,
            target_velocity,
            _,
        ) = construct_rectified_flow_batch(
            clean_latent=clean_latent,
            timestep=timestep,
        )

        with autocast_context(
            device=device,
            enabled=use_amp,
        ):
            predicted_velocity = model(
                noisy_hsi_latent=noisy_latent,
                timestep=timestep,
                rgb=rgb,
            )

            flow_loss = F.mse_loss(
                predicted_velocity.float(),
                target_velocity.float(),
            )

            predicted_clean_latent = (
                recover_clean_latent(
                    noisy_latent=noisy_latent,
                    predicted_velocity=predicted_velocity,
                    timestep=timestep,
                )
            )

            latent_l1_loss = F.l1_loss(
                predicted_clean_latent.float(),
                clean_latent.float(),
            )

            loss = (
                flow_loss
                + LATENT_L1_WEIGHT
                * latent_l1_loss
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch "
                f"{batch_index}: {loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=GRADIENT_CLIP_NORM,
            )
        )

        if not torch.isfinite(gradient_norm):
            optimizer.zero_grad(
                set_to_none=True
            )
            raise FloatingPointError(
                f"Non-finite gradient norm at batch "
                f"{batch_index}: {gradient_norm.item()}"
            )

        scaler.step(optimizer)
        scaler.update()

        update_timestep_tracker(
            tracker=timestep_tracker,
            timestep=timestep.detach(),
            predicted_velocity=predicted_velocity,
            target_velocity=target_velocity,
        )

        batch_size = hsi.shape[0]
        flow_loss_sum += (
            flow_loss.detach().item()
            * batch_size
        )
        latent_l1_sum += (
            latent_l1_loss.detach().item()
            * batch_size
        )
        total_samples += batch_size

        if (
            COMPUTE_TRAIN_ONE_STEP_METRICS
            and batch_index % TRAIN_METRIC_EVERY == 0
        ):
            with torch.no_grad():
                hsi_reconstruction = (
                    model.decode_hsi_latent(
                        predicted_clean_latent.detach()
                    )
                )

                metric_accumulator.update(
                    prediction=hsi_reconstruction,
                    target=hsi,
                )
                metric_batches += 1

        if (
            batch_index % PRINT_EVERY == 0
            or batch_index == len(loader)
        ):
            message = (
                f"  Batch {batch_index:04d}/"
                f"{len(loader):04d} | "
                f"flow={flow_loss_sum / total_samples:.6f} | "
                f"latent L1={latent_l1_sum / total_samples:.6f} | "
                f"grad={float(gradient_norm):.4f}"
            )

            if metric_batches > 0:
                current_metrics = (
                    metric_accumulator.compute()
                )
                message += (
                    f" | one-step MRAE="
                    f"{current_metrics['mrae']:.6f}"
                    f" | one-step SAM="
                    f"{current_metrics['sam']:.6f}"
                )

            print(message)

    result = {
        "flow_loss": (
            flow_loss_sum / total_samples
        ),
        "latent_l1": (
            latent_l1_sum / total_samples
        ),
        "timestep_losses": (
            finalize_timestep_tracker(
                timestep_tracker
            )
        ),
    }

    if metric_batches > 0:
        one_step_metrics = (
            metric_accumulator.compute()
        )
        result.update(
            {
                f"one_step_{key}": value
                for key, value
                in one_step_metrics.items()
            }
        )

    return result


@torch.no_grad()
def validate_flow_loss(
    model: RGBConditionedHSIMMDiT,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()

    flow_loss_sum = 0.0
    total_samples = 0
    sample_offset = 0
    timestep_tracker = create_timestep_tracker()

    # Use a fixed RNG so validation noise is repeatable across epochs.
    generator = torch.Generator(
        device=device
    )
    generator.manual_seed(
        SEED + 10_000
    )

    for hsi, rgb in loader:
        hsi = hsi.to(
            device,
            non_blocking=True,
        )
        rgb = rgb.to(
            device,
            non_blocking=True,
        )

        clean_latent, _, _ = model.encode_hsi(
            hsi,
            sample=False,
        )

        batch_size = hsi.shape[0]

        grid_indices = (
            torch.arange(
                sample_offset,
                sample_offset + batch_size,
                device=device,
            )
            % VALIDATION_TIMESTEP_GRID_SIZE
        )

        timestep = (
            grid_indices.float() + 0.5
        ) / VALIDATION_TIMESTEP_GRID_SIZE

        sample_offset += batch_size

        noise = torch.randn(
            clean_latent.shape,
            generator=generator,
            device=device,
            dtype=clean_latent.dtype,
        )

        timestep_view = timestep.to(
            clean_latent.dtype
        ).reshape(-1, 1, 1, 1)

        noisy_latent = (
            (1.0 - timestep_view) * clean_latent
            + timestep_view * noise
        )
        target_velocity = noise - clean_latent

        with autocast_context(
            device=device,
            enabled=use_amp,
        ):
            predicted_velocity = model(
                noisy_hsi_latent=noisy_latent,
                timestep=timestep,
                rgb=rgb,
            )

        per_sample_loss = F.mse_loss(
            predicted_velocity.float(),
            target_velocity.float(),
            reduction="none",
        ).mean(dim=(1, 2, 3))

        flow_loss_sum += (
            per_sample_loss.sum().item()
        )
        total_samples += batch_size

        update_timestep_tracker(
            tracker=timestep_tracker,
            timestep=timestep,
            predicted_velocity=predicted_velocity,
            target_velocity=target_velocity,
        )

    return {
        "flow_loss": (
            flow_loss_sum / total_samples
        ),
        "timestep_losses": (
            finalize_timestep_tracker(
                timestep_tracker
            )
        ),
    }


@torch.no_grad()
def validate_reconstruction_metrics(
    model: RGBConditionedHSIMMDiT,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    num_steps: int,
    max_images: Optional[int],
) -> dict:
    """
    Compute real RGB -> HSI reconstruction metrics using the full Euler sampler.

    This is intentionally separate from validation flow loss. Calculating metrics
    from a random-t one-step estimate is not the same as evaluating inference.
    """
    model.eval()

    accumulator = HSIMetricAccumulator(
        data_range=METRIC_DATA_RANGE,
        clamp_prediction=(
            CLAMP_PREDICTION_FOR_METRICS
        ),
    )

    evaluated_images = 0

    rng_devices = []
    if device.type == "cuda":
        rng_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]

    with torch.random.fork_rng(
        devices=rng_devices,
        enabled=True,
    ):
        torch.manual_seed(SEED + 20_000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(
                SEED + 20_000
            )

        for hsi, rgb in loader:
            if (
                max_images is not None
                and evaluated_images >= max_images
            ):
                break

            if max_images is not None:
                remaining = (
                    max_images - evaluated_images
                )
                hsi = hsi[:remaining]
                rgb = rgb[:remaining]

            hsi = hsi.to(
                device,
                non_blocking=True,
            )
            rgb = rgb.to(
                device,
                non_blocking=True,
            )

            with autocast_context(
                device=device,
                enabled=use_amp,
            ):
                reconstruction = model.sample(
                    rgb=rgb,
                    num_steps=num_steps,
                    decode=True,
                )

            accumulator.update(
                prediction=reconstruction,
                target=hsi,
            )

            evaluated_images += hsi.shape[0]

    metrics = accumulator.compute()
    metrics["evaluated_images"] = (
        evaluated_images
    )
    metrics["sampling_steps"] = num_steps

    return metrics


# ============================================================================
# Inference preview saving
# ============================================================================

def _normalize_rgb_for_display(
    rgb: np.ndarray,
) -> np.ndarray:
    rgb = np.transpose(
        rgb,
        (1, 2, 0),
    )
    return np.clip(
        rgb,
        0.0,
        1.0,
    )


def _hsi_to_pseudo_rgb_pair(
    target: np.ndarray,
    prediction: np.ndarray,
    bands: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    channel_count = target.shape[0]

    for band in bands:
        if not 0 <= band < channel_count:
            raise ValueError(
                f"Visualization band {band} is outside "
                f"the valid range [0, {channel_count - 1}]."
            )

    target_rgb = np.stack(
        [target[band] for band in bands],
        axis=-1,
    )
    prediction_rgb = np.stack(
        [prediction[band] for band in bands],
        axis=-1,
    )

    # Use target-derived scaling for both target and prediction, so the
    # prediction preview is not independently contrast-normalized.
    minimum = target_rgb.min(
        axis=(0, 1),
        keepdims=True,
    )
    maximum = target_rgb.max(
        axis=(0, 1),
        keepdims=True,
    )

    scale = maximum - minimum + 1e-8

    target_rgb = (
        target_rgb - minimum
    ) / scale
    prediction_rgb = (
        prediction_rgb - minimum
    ) / scale

    return (
        np.clip(target_rgb, 0.0, 1.0),
        np.clip(prediction_rgb, 0.0, 1.0),
    )


def save_inference_preview(
    output_path: Path,
    rgb: np.ndarray,
    target_hsi: np.ndarray,
    prediction_hsi: np.ndarray,
) -> None:
    rgb_display = _normalize_rgb_for_display(
        rgb
    )
    target_display, prediction_display = (
        _hsi_to_pseudo_rgb_pair(
            target=target_hsi,
            prediction=prediction_hsi,
            bands=VISUALIZATION_BANDS,
        )
    )

    panels = [
        rgb_display,
        target_display,
        prediction_display,
    ]

    panel_images = [
        Image.fromarray(
            (panel * 255.0)
            .round()
            .astype(np.uint8)
        )
        for panel in panels
    ]

    width = sum(
        image.width
        for image in panel_images
    )
    height = max(
        image.height
        for image in panel_images
    )

    canvas = Image.new(
        "RGB",
        (width, height),
    )

    x_offset = 0
    for image in panel_images:
        canvas.paste(
            image,
            (x_offset, 0),
        )
        x_offset += image.width

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    canvas.save(output_path)


@torch.no_grad()
def run_random_validation_inference(
    model: RGBConditionedHSIMMDiT,
    validation_pairs: Sequence[
        Tuple[Path, Path]
    ],
    device: torch.device,
    use_amp: bool,
    output_directory: Path,
    number_of_images: int,
    num_steps: int,
) -> dict:
    model.eval()

    if not validation_pairs:
        raise RuntimeError(
            "The validation pair list is empty."
        )

    number_to_select = min(
        number_of_images,
        len(validation_pairs),
    )

    selected_indices = random.Random(
        SEED
    ).sample(
        range(len(validation_pairs)),
        k=number_to_select,
    )

    dataset = HSIRGBPairDataset(
        pairs=validation_pairs,
        hsi_channels=HSI_CHANNELS,
        crop_size=None,
        patches_per_image=1,
        training=False,
        normalization=HSI_NORMALIZATION,
        augment=False,
        return_paths=True,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_accumulator = HSIMetricAccumulator(
        data_range=METRIC_DATA_RANGE,
        clamp_prediction=(
            CLAMP_PREDICTION_FOR_METRICS
        ),
    )

    required_multiple = (
        VAE_DOWNSAMPLE_FACTOR
        * MMDIT_PATCH_SIZE
    )

    print(
        f"\nRunning full-resolution inference on "
        f"{number_to_select} random validation images..."
    )

    for output_index, dataset_index in enumerate(
        selected_indices,
        start=1,
    ):
        (
            hsi,
            rgb,
            hsi_path_string,
            rgb_path_string,
        ) = dataset[dataset_index]

        (
            padded_hsi,
            padded_rgb,
            original_height,
            original_width,
        ) = pad_pair_to_multiple(
            hsi=hsi,
            rgb=rgb,
            multiple=required_multiple,
        )

        rgb_batch = (
            padded_rgb.unsqueeze(0)
            .to(device)
        )

        with autocast_context(
            device=device,
            enabled=use_amp,
        ):
            prediction = model.sample(
                rgb=rgb_batch,
                num_steps=num_steps,
                decode=True,
            )

        prediction = prediction[
            :,
            :,
            :original_height,
            :original_width,
        ].float().cpu()

        target_batch = (
            hsi.unsqueeze(0).float()
        )

        sample_accumulator = HSIMetricAccumulator(
            data_range=METRIC_DATA_RANGE,
            clamp_prediction=(
                CLAMP_PREDICTION_FOR_METRICS
            ),
        )
        sample_accumulator.update(
            prediction=prediction,
            target=target_batch,
        )
        sample_metrics = (
            sample_accumulator.compute()
        )

        overall_accumulator.update(
            prediction=prediction,
            target=target_batch,
        )

        stem = Path(hsi_path_string).stem
        prefix = (
            output_directory
            / f"{output_index:02d}_{stem}"
        )

        prediction_numpy = (
            prediction[0].numpy()
        )
        target_numpy = hsi.numpy()
        rgb_numpy = rgb.numpy()

        np.savez_compressed(
            str(prefix) + ".npz",
            prediction=prediction_numpy,
            target=target_numpy,
            rgb=rgb_numpy,
            hsi_path=hsi_path_string,
            rgb_path=rgb_path_string,
            metrics=np.asarray(
                [
                    sample_metrics["mrae"],
                    sample_metrics["rmse"],
                    sample_metrics["sam"],
                    sample_metrics["psnr"],
                    sample_metrics["ssim"],
                ],
                dtype=np.float64,
            ),
            metric_names=np.asarray(
                [
                    "mrae",
                    "rmse",
                    "sam_radians",
                    "psnr",
                    "ssim",
                ]
            ),
        )

        save_inference_preview(
            output_path=Path(
                str(prefix) + "_preview.png"
            ),
            rgb=rgb_numpy,
            target_hsi=target_numpy,
            prediction_hsi=prediction_numpy,
        )

        print(
            f"  [{output_index}/{number_to_select}] {stem} | "
            f"MRAE={sample_metrics['mrae']:.6f} | "
            f"RMSE={sample_metrics['rmse']:.6f} | "
            f"SAM={sample_metrics['sam']:.6f} rad | "
            f"PSNR={sample_metrics['psnr']:.4f} | "
            f"SSIM={sample_metrics['ssim']:.4f}"
        )

    overall_metrics = (
        overall_accumulator.compute()
    )

    metrics_path = (
        output_directory
        / "random_inference_metrics.txt"
    )

    metrics_path.write_text(
        "\n".join(
            [
                f"images={number_to_select}",
                f"sampling_steps={num_steps}",
                f"mrae={overall_metrics['mrae']}",
                f"rmse={overall_metrics['rmse']}",
                f"sam_radians={overall_metrics['sam']}",
                f"psnr={overall_metrics['psnr']}",
                f"ssim={overall_metrics['ssim']}",
            ]
        ),
        encoding="utf-8",
    )

    print(
        "\nRandom inference mean metrics | "
        f"MRAE={overall_metrics['mrae']:.6f} | "
        f"RMSE={overall_metrics['rmse']:.6f} | "
        f"SAM={overall_metrics['sam']:.6f} rad | "
        f"PSNR={overall_metrics['psnr']:.4f} | "
        f"SSIM={overall_metrics['ssim']:.4f}"
    )

    return overall_metrics


# ============================================================================
# DataLoader construction
# ============================================================================

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
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        worker_init_fn=seed_worker,
        generator=generator,
    )


# ============================================================================
# Main workflows
# ============================================================================

def prepare_pairs(
    output_directory: Path,
) -> Tuple[
    List[Tuple[Path, Path]],
    List[Tuple[Path, Path]],
]:
    train_pairs = pair_hsi_rgb_files(
        hsi_directory=TRAIN_HSI_DIR,
        rgb_directory=TRAIN_RGB_DIR,
    )
    validation_pairs = pair_hsi_rgb_files(
        hsi_directory=VALIDATION_HSI_DIR,
        rgb_directory=VALIDATION_RGB_DIR,
    )

    train_pairs = filter_valid_pairs(
        pairs=train_pairs,
        hsi_channels=HSI_CHANNELS,
        log_path=(
            output_directory
            / "invalid_training_pairs.txt"
        ),
        cache_path=TRAIN_PAIR_VALIDATION_CACHE,
    )
    validation_pairs = filter_valid_pairs(
        pairs=validation_pairs,
        hsi_channels=HSI_CHANNELS,
        log_path=(
            output_directory
            / "invalid_validation_pairs.txt"
        ),
        cache_path=VALIDATION_PAIR_VALIDATION_CACHE,
    )

    return train_pairs, validation_pairs


def prepare_vae_and_stats(
    train_dataset: HSIRGBPairDataset,
    device: torch.device,
) -> Tuple[
    HSIVAE,
    torch.Tensor,
    torch.Tensor,
]:
    (
        vae,
        checkpoint_latent_mean,
        checkpoint_latent_std,
    ) = load_pretrained_vae(
        checkpoint_path=VAE_CHECKPOINT,
        device=device,
    )

    if (
        checkpoint_latent_mean is not None
        and checkpoint_latent_std is not None
    ):
        print(
            "\nUsing latent mean/std stored in the VAE checkpoint."
        )
        return (
            vae,
            checkpoint_latent_mean,
            checkpoint_latent_std,
        )

    if not COMPUTE_LATENT_STATS_IF_MISSING:
        print(
            "\nWarning: the VAE checkpoint has no latent statistics. "
            "Using mean=0 and std=1."
        )
        return (
            vae,
            torch.zeros(LATENT_CHANNELS),
            torch.ones(LATENT_CHANNELS),
        )

    statistics_loader = make_loader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        device=device,
    )

    latent_mean, latent_std = (
        compute_latent_statistics(
            vae=vae,
            loader=statistics_loader,
            latent_channels=LATENT_CHANNELS,
            device=device,
            max_batches=(
                LATENT_STATS_MAX_BATCHES
            ),
        )
    )

    return vae, latent_mean, latent_std


def train_workflow(
    device: torch.device,
    use_amp: bool,
    train_pairs: Sequence[Tuple[Path, Path]],
    validation_pairs: Sequence[
        Tuple[Path, Path]
    ],
    output_directory: Path,
) -> RGBConditionedHSIMMDiT:
    required_multiple = (
        VAE_DOWNSAMPLE_FACTOR
        * MMDIT_PATCH_SIZE
    )

    if TRAIN_CROP_SIZE % required_multiple != 0:
        raise ValueError(
            f"TRAIN_CROP_SIZE={TRAIN_CROP_SIZE} must be divisible by "
            f"{required_multiple}."
        )

    train_dataset = HSIRGBPairDataset(
        pairs=train_pairs,
        hsi_channels=HSI_CHANNELS,
        crop_size=TRAIN_CROP_SIZE,
        patches_per_image=PATCHES_PER_IMAGE,
        training=True,
        normalization=HSI_NORMALIZATION,
        augment=USE_AUGMENTATION,
    )
    validation_dataset = HSIRGBPairDataset(
        pairs=validation_pairs,
        hsi_channels=HSI_CHANNELS,
        crop_size=TRAIN_CROP_SIZE,
        patches_per_image=1,
        training=False,
        normalization=HSI_NORMALIZATION,
        augment=False,
    )

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=(
            len(train_dataset) >= BATCH_SIZE
        ),
        device=device,
    )
    validation_loader = make_loader(
        dataset=validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        device=device,
    )

    (
        vae,
        latent_mean,
        latent_std,
    ) = prepare_vae_and_stats(
        train_dataset=train_dataset,
        device=device,
    )

    model = build_mmdit(
        vae=vae,
        latent_mean=latent_mean,
        latent_std=latent_std,
        device=device,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    print(
        f"\nDevice: {device}\n"
        f"Mixed precision: {use_amp}\n"
        f"Training pairs: {len(train_pairs)}\n"
        f"Validation pairs: {len(validation_pairs)}\n"
        f"Trainable parameters: "
        f"{sum(p.numel() for p in trainable_parameters):,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
            eta_min=MIN_LEARNING_RATE,
        )
    )

    scaler = GradScaler(
        enabled=use_amp
    )

    start_epoch = 1
    best_validation_flow_loss = float("inf")

    if RESUME_CHECKPOINT is not None:
        resume_checkpoint = (
            _load_torch_checkpoint(
                RESUME_CHECKPOINT,
                device="cpu",
            )
        )

        load_mmdit_weights(
            model,
            resume_checkpoint,
        )

        optimizer.load_state_dict(
            resume_checkpoint[
                "optimizer_state_dict"
            ]
        )
        scheduler.load_state_dict(
            resume_checkpoint[
                "scheduler_state_dict"
            ]
        )

        if "scaler_state_dict" in resume_checkpoint:
            scaler.load_state_dict(
                resume_checkpoint[
                    "scaler_state_dict"
                ]
            )

        start_epoch = int(
            resume_checkpoint.get(
                "epoch",
                0,
            )
        ) + 1

        best_validation_flow_loss = float(
            resume_checkpoint.get(
                "best_validation_flow_loss",
                float("inf"),
            )
        )

        print(
            f"\nResumed from {RESUME_CHECKPOINT} "
            f"at epoch {start_epoch}."
        )

    for epoch in range(
        start_epoch,
        NUM_EPOCHS + 1,
    ):
        print(
            f"\n{'=' * 80}\n"
            f"Epoch {epoch}/{NUM_EPOCHS}\n"
            f"{'=' * 80}"
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        validation_flow = validate_flow_loss(
            model=model,
            loader=validation_loader,
            device=device,
            use_amp=use_amp,
        )

        validation_reconstruction = (
            validate_reconstruction_metrics(
                model=model,
                loader=validation_loader,
                device=device,
                use_amp=use_amp,
                num_steps=(
                    VALIDATION_INFERENCE_STEPS
                ),
                max_images=(
                    VALIDATION_METRIC_MAX_IMAGES
                ),
            )
        )

        scheduler.step()

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"\nEpoch {epoch:03d}/{NUM_EPOCHS:03d} | "
            f"LR={current_learning_rate:.2e} | "
            f"train flow={train_metrics['flow_loss']:.6f} | "
            f"validation flow="
            f"{validation_flow['flow_loss']:.6f}"
        )

        print(
            "Validation full-sampling metrics | "
            f"images={validation_reconstruction['evaluated_images']} | "
            f"steps={validation_reconstruction['sampling_steps']} | "
            f"MRAE={validation_reconstruction['mrae']:.6f} | "
            f"RMSE={validation_reconstruction['rmse']:.6f} | "
            f"SAM={validation_reconstruction['sam']:.6f} rad | "
            f"PSNR={validation_reconstruction['psnr']:.4f} | "
            f"SSIM={validation_reconstruction['ssim']:.4f}"
        )

        print_timestep_tracker(
            title="Training flow loss by timestep",
            result=train_metrics[
                "timestep_losses"
            ],
        )
        print_timestep_tracker(
            title="Validation flow loss by timestep",
            result=validation_flow[
                "timestep_losses"
            ],
        )

        combined_validation_metrics = {
            "flow": validation_flow,
            "reconstruction": (
                validation_reconstruction
            ),
        }

        save_checkpoint(
            output_path=(
                output_directory
                / "last_mmdit.pth"
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_validation_flow_loss=(
                best_validation_flow_loss
            ),
            validation_metrics=(
                combined_validation_metrics
            ),
        )

        if (
            validation_flow["flow_loss"]
            < best_validation_flow_loss
        ):
            best_validation_flow_loss = (
                validation_flow["flow_loss"]
            )

            save_checkpoint(
                output_path=(
                    output_directory
                    / "best_mmdit.pth"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_validation_flow_loss=(
                    best_validation_flow_loss
                ),
                validation_metrics=(
                    combined_validation_metrics
                ),
            )

            print(
                "New best checkpoint | "
                f"validation flow="
                f"{best_validation_flow_loss:.6f}"
            )

    return model


def load_model_for_inference(
    checkpoint_path: str,
    device: torch.device,
) -> RGBConditionedHSIMMDiT:
    checkpoint = _load_torch_checkpoint(
        checkpoint_path,
        device="cpu",
    )

    state_dict = _extract_state_dict(
        checkpoint,
        candidate_keys=(
            "model_state_dict",
            "mmdit_state_dict",
            "state_dict",
        ),
    )
    has_embedded_vae = any(
        key.startswith("vae.")
        for key in state_dict
    )

    checkpoint_mean, checkpoint_std = _extract_latent_stats(
        checkpoint,
        latent_channels=LATENT_CHANNELS,
    )

    if has_embedded_vae:
        # New checkpoint format: the checkpoint is self-contained.
        vae_config = (
            checkpoint.get("vae_config", {})
            if isinstance(checkpoint, dict)
            else {}
        )
        vae = build_vae_from_config(
            vae_config=vae_config,
            device=device,
        )
        vae_latent_mean = None
        vae_latent_std = None
        print(
            "\nLoading the frozen HSI VAE directly from the MM-DiT checkpoint."
        )
    else:
        # Backward compatibility with older MM-DiT checkpoints.
        (
            vae,
            vae_latent_mean,
            vae_latent_std,
        ) = load_pretrained_vae(
            checkpoint_path=VAE_CHECKPOINT,
            device=device,
        )
        print(
            "\nThis is an older MM-DiT checkpoint without embedded VAE "
            "weights; loading VAE_CHECKPOINT separately."
        )

    latent_mean = (
        checkpoint_mean
        if checkpoint_mean is not None
        else vae_latent_mean
    )
    latent_std = (
        checkpoint_std
        if checkpoint_std is not None
        else vae_latent_std
    )

    if latent_mean is None or latent_std is None:
        raise RuntimeError(
            "No latent mean/std was found in the MM-DiT checkpoint "
            "or the fallback VAE checkpoint."
        )

    model_config = (
        checkpoint.get("model_config", {})
        if isinstance(checkpoint, dict)
        else {}
    )

    model = build_mmdit(
        vae=vae,
        latent_mean=latent_mean,
        latent_std=latent_std,
        device=device,
        model_config=model_config,
    )

    load_mmdit_weights(
        model,
        checkpoint,
    )

    model.eval()
    return model


def main() -> None:
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    use_amp = (
        USE_AMP
        and device.type == "cuda"
    )

    output_directory = Path(OUTPUT_DIR)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RUN_MODE not in {
        "train",
        "infer",
        "train_and_infer",
    }:
        raise ValueError(
            "RUN_MODE must be 'train', 'infer', "
            "or 'train_and_infer'."
        )

    if RUN_MODE in {
        "train",
        "train_and_infer",
    }:
        train_pairs, validation_pairs = (
            prepare_pairs(output_directory)
        )
        model = train_workflow(
            device=device,
            use_amp=use_amp,
            train_pairs=train_pairs,
            validation_pairs=validation_pairs,
            output_directory=output_directory,
        )
    else:
        validation_pairs = pair_hsi_rgb_files(
            hsi_directory=VALIDATION_HSI_DIR,
            rgb_directory=VALIDATION_RGB_DIR,
        )
        validation_pairs = filter_valid_pairs(
            pairs=validation_pairs,
            hsi_channels=HSI_CHANNELS,
            log_path=(
                output_directory
                / "invalid_validation_pairs.txt"
            ),
            cache_path=VALIDATION_PAIR_VALIDATION_CACHE,
        )

        model = load_model_for_inference(
            checkpoint_path=INFERENCE_CHECKPOINT,
            device=device,
        )

    if RUN_MODE in {
        "infer",
        "train_and_infer",
    }:
        # For train_and_infer, load the saved best checkpoint rather than
        # silently using the final epoch's in-memory weights.
        if RUN_MODE == "train_and_infer":
            model = load_model_for_inference(
                checkpoint_path=str(
                    output_directory
                    / "best_mmdit.pth"
                ),
                device=device,
            )

        run_random_validation_inference(
            model=model,
            validation_pairs=validation_pairs,
            device=device,
            use_amp=use_amp,
            output_directory=Path(
                INFERENCE_OUTPUT_DIR
            ),
            number_of_images=(
                NUM_RANDOM_INFERENCE_IMAGES
            ),
            num_steps=INFERENCE_STEPS,
        )


if __name__ == "__main__":
    main()
