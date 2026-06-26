"""Train the HSI -> RGB operator used by the RGB adaptation of DDS2M.

Why this script trains only the operator
----------------------------------------
The original DDS2M spatial and spectral networks are *untrained* networks:
they are initialized from scratch and optimized separately for each RGB test
image during reverse diffusion. They are therefore not trained across the
paired dataset.

The only dataset-level quantity missing in the RGB-to-HSI setting is the
camera/forward operator R in

    RGB = R @ HSI.

Because only paired RGB/HSI images are available, this script learns that
3 x K linear operator from the pairs and saves it. The saved operator is then
loaded and frozen when running per-image DDS2M restoration.

The data-loading structure follows the supplied HSI VAE training script. The
main changes are intentionally limited to:

1. locating the paired RGB file for every HSI file;
2. loading RGB alongside HSI;
3. applying the same crop and augmentation to both;
4. returning ``rgb, hsi`` from the dataset.

Edit ``HSI_DATA_DIR``, ``RGB_DATA_DIR`` and ``resolve_rgb_path`` to match the
local paired-image layout.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from models.DDS2M_rgb_to_hsi import RGBSpectralResponse

# ============================================================
# Configuration
# ============================================================

# Set these paths manually.
HSI_DATA_DIR = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_spectral"
RGB_DATA_DIR = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_RGB"
OUTPUT_DIR = "./dds2m_rgb2hsi_checkpoints"

HSI_KEY = "cube"
VALIDATION_CACHE = Path(OUTPUT_DIR) / "hsi_validation_cache.pth"

HSI_CHANNELS = 31
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 4

BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

# The original reverse equations require a linear operator. Keeping this False
# learns an unconstrained 3 x K matrix, which is usually a better first test
# when the RGB files may contain an unknown processing pipeline.
CONSTRAIN_OPERATOR_NONNEGATIVE = False
OPERATOR_SMOOTHNESS_WEIGHT = 1e-4

VALIDATION_FRACTION = 0.1

# HSI normalization options retained from the supplied loader:
#   "none"        : dataset is already in the desired range
#   "minmax"      : normalize each complete cube independently
#   "band_minmax" : normalize every spectral band independently
# For a physically meaningful shared RGB operator, "none" is preferable when
# all HSI cubes are already consistently normalized to [0, 1].
HSI_NORMALIZATION = "none"

# RGB normalization:
#   "auto"   : integer images are divided by the dtype maximum; floating
#              arrays with values above 1 are divided by 255 when possible
#   "none"   : retain the loaded values
#   "minmax" : per-image min-max normalization
RGB_NORMALIZATION = "auto"

NUM_WORKERS = 4
USE_AMP = True
USE_AUGMENTATION = True

SEED = 42
GRADIENT_CLIP_NORM = 1.0
PRINT_EVERY = 30

SUPPORTED_HSI_EXTENSIONS = {
    ".npy",
    ".npz",
    ".mat",
    ".pt",
    ".pth",
}

SUPPORTED_RGB_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".npy",
    ".npz",
    ".mat",
    ".pt",
    ".pth",
}


# ============================================================
# Reproducibility
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# File loading: HSI
# ============================================================


def load_mat_v73(file_path: Path) -> np.ndarray:
    """Load the largest numerical 3D dataset from a MATLAB v7.3 file."""
    candidates = []

    def visit_dataset(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        try:
            if obj.ndim != 3:
                return
            array = np.asarray(obj)
            if np.issubdtype(array.dtype, np.number):
                candidates.append((name, array))
        except Exception:
            return

    try:
        with h5py.File(str(file_path), "r") as h5_file:
            h5_file.visititems(visit_dataset)
    except OSError as error:
        raise OSError(
            f"Could not read MATLAB v7.3 file:\n{file_path}\nReason: {error}"
        ) from error

    if not candidates:
        raise ValueError(f"No numerical 3D array found in: {file_path}")

    _, array = max(candidates, key=lambda item: item[1].size)

    # MATLAB v7.3 arrays are commonly exposed with reversed dimensions.
    array = np.transpose(
        array,
        axes=tuple(range(array.ndim - 1, -1, -1)),
    )
    return array


def extract_array_from_dictionary(
    data: dict,
    file_path: Path,
    preferred_channels: int | None = None,
) -> np.ndarray:
    candidates = []

    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if not isinstance(value, np.ndarray):
            continue
        value = np.squeeze(value)
        if value.ndim != 3:
            continue
        candidates.append(value)

    if not candidates:
        raise ValueError(f"No three-dimensional array found in {file_path}")

    if preferred_channels is not None:
        preferred = [
            array
            for array in candidates
            if preferred_channels in array.shape
        ]
        if preferred:
            return max(preferred, key=lambda array: array.size)

    return max(candidates, key=lambda array: array.size)


def load_hsi_file(file_path: Path) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension == ".npy":
        cube = np.load(file_path)

    elif extension == ".npz":
        loaded = np.load(file_path)
        candidates = [
            loaded[key]
            for key in loaded.files
            if np.squeeze(loaded[key]).ndim == 3
        ]
        if not candidates:
            raise ValueError(f"No three-dimensional array found in {file_path}")
        cube = max(candidates, key=lambda array: array.size)

    elif extension == ".mat":
        try:
            loaded = sio.loadmat(file_path)
            cube = extract_array_from_dictionary(
                loaded,
                file_path,
                preferred_channels=HSI_CHANNELS,
            )
        except (NotImplementedError, ValueError):
            cube = load_mat_v73(file_path)

    elif extension in {".pt", ".pth"}:
        loaded = torch.load(file_path, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            cube = loaded.detach().cpu().numpy()
        elif isinstance(loaded, np.ndarray):
            cube = loaded
        elif isinstance(loaded, dict):
            cube = extract_array_from_dictionary(
                loaded,
                file_path,
                preferred_channels=HSI_CHANNELS,
            )
        else:
            raise TypeError(f"Unsupported object in {file_path}: {type(loaded)}")

    else:
        raise ValueError(f"Unsupported HSI extension: {extension}")

    cube = np.asarray(cube, dtype=np.float32)
    cube = np.squeeze(cube)

    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D HSI cube in {file_path}, found shape {cube.shape}"
        )

    return cube


def convert_hsi_to_chw(
    cube: np.ndarray,
    hsi_channels: int,
    file_path: Path,
) -> np.ndarray:
    """Convert [C,H,W] or [H,W,C] to [C,H,W]."""
    if cube.shape[0] == hsi_channels:
        return cube
    if cube.shape[-1] == hsi_channels:
        return np.transpose(cube, (2, 0, 1))
    raise ValueError(
        f"Cannot identify the spectral dimension in {file_path}. "
        f"Shape: {cube.shape}, expected bands: {hsi_channels}"
    )


# ============================================================
# File loading: RGB
# ============================================================


def load_rgb_file(file_path: Path) -> np.ndarray:
    """Load RGB and return a float array before final normalization."""
    extension = file_path.suffix.lower()

    if extension in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        with Image.open(file_path) as image:
            image = image.convert("RGB")
            array = np.asarray(image)

    elif extension == ".npy":
        array = np.load(file_path)

    elif extension == ".npz":
        loaded = np.load(file_path)
        candidates = [
            loaded[key]
            for key in loaded.files
            if np.squeeze(loaded[key]).ndim == 3
            and 3 in np.squeeze(loaded[key]).shape
        ]
        if not candidates:
            raise ValueError(f"No RGB array found in {file_path}")
        array = max(candidates, key=lambda candidate: candidate.size)

    elif extension == ".mat":
        try:
            loaded = sio.loadmat(file_path)
            array = extract_array_from_dictionary(
                loaded,
                file_path,
                preferred_channels=3,
            )
        except (NotImplementedError, ValueError):
            array = load_mat_v73(file_path)

    elif extension in {".pt", ".pth"}:
        loaded = torch.load(file_path, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            array = loaded.detach().cpu().numpy()
        elif isinstance(loaded, np.ndarray):
            array = loaded
        elif isinstance(loaded, dict):
            array = extract_array_from_dictionary(
                loaded,
                file_path,
                preferred_channels=3,
            )
        else:
            raise TypeError(f"Unsupported object in {file_path}: {type(loaded)}")

    else:
        raise ValueError(f"Unsupported RGB extension: {extension}")

    array = np.asarray(array)
    array = np.squeeze(array)

    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3D RGB image in {file_path}, found shape {array.shape}"
        )

    return array


def convert_rgb_to_chw(array: np.ndarray, file_path: Path) -> np.ndarray:
    """Convert [3,H,W] or [H,W,3] to [3,H,W]."""
    if array.shape[0] == 3:
        return array
    if array.shape[-1] == 3:
        return np.transpose(array, (2, 0, 1))
    raise ValueError(
        f"Cannot identify the RGB channel dimension in {file_path}. "
        f"Shape: {array.shape}"
    )


# ============================================================
# Pairing
# ============================================================


def find_hsi_files(data_dir: str) -> List[Path]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"HSI data directory does not exist: {data_path}")

    files = sorted(
        path
        for path in data_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_HSI_EXTENSIONS
    )

    if not files:
        raise RuntimeError(
            f"No supported HSI files found in {data_path}. "
            f"Supported extensions: {sorted(SUPPORTED_HSI_EXTENSIONS)}"
        )
    return files


def resolve_rgb_path(hsi_path: Path) -> Path:
    """Map one HSI path to its paired RGB path.

    This is the only pairing function that normally needs editing.

    The default implementation assumes mirrored directory structures and
    identical stems, and searches the supported RGB extensions. Example:

        HSI_DATA_DIR/scene_01.mat
        RGB_DATA_DIR/scene_01.jpg

    For filenames such as ``scene_01_clean.mat`` and ``scene_01_rgb.jpg``, edit
    ``rgb_stem`` below.
    """
    hsi_root = Path(HSI_DATA_DIR)
    rgb_root = Path(RGB_DATA_DIR)

    relative_parent = hsi_path.relative_to(hsi_root).parent
    rgb_stem = hsi_path.stem

    for extension in sorted(SUPPORTED_RGB_EXTENSIONS):
        candidate = rgb_root / relative_parent / f"{rgb_stem}{extension}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No paired RGB file found for HSI:\n  {hsi_path}\n"
        f"Expected stem: {rgb_stem}\n"
        f"Searched under: {rgb_root / relative_parent}"
    )


def build_paired_files(hsi_files: Sequence[Path]) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    missing = []

    for hsi_path in hsi_files:
        try:
            rgb_path = resolve_rgb_path(hsi_path)
            pairs.append((hsi_path, rgb_path))
        except FileNotFoundError as error:
            missing.append(str(error))

    if missing:
        missing_log = Path(OUTPUT_DIR) / "missing_rgb_pairs.txt"
        missing_log.parent.mkdir(parents=True, exist_ok=True)
        missing_log.write_text("\n\n".join(missing), encoding="utf-8")
        print(
            f"Skipped {len(missing)} HSI files without a paired RGB image. "
            f"Details: {missing_log}"
        )

    if not pairs:
        raise RuntimeError("No valid RGB-HSI pairs were found.")

    return pairs


# ============================================================
# Optional HSI metadata validation retained from supplied script
# ============================================================


def make_files_fingerprint(files: List[Path]) -> str:
    records = []
    for file_path in files:
        stat = file_path.stat()
        records.append(
            f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        )
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def is_possible_hsi_shape(shape, hsi_channels: int) -> bool:
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
    candidates = []

    with h5py.File(str(file_path), "r") as h5_file:
        if hsi_key in h5_file and isinstance(h5_file[hsi_key], h5py.Dataset):
            dataset = h5_file[hsi_key]
            candidates.append((hsi_key, tuple(int(v) for v in dataset.shape)))
        else:
            def visitor(name, obj):
                if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                    return
                try:
                    is_numeric = np.issubdtype(obj.dtype, np.number)
                except TypeError:
                    is_numeric = False
                if is_numeric:
                    candidates.append((name, tuple(int(v) for v in obj.shape)))

            h5_file.visititems(visitor)

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
    try:
        metadata = sio.whosmat(file_path)
    except (NotImplementedError, ValueError, OSError):
        inspect_hdf5_mat_file(file_path, hsi_channels, hsi_key)
        return

    candidates = [
        (name, tuple(int(value) for value in shape))
        for name, shape, _ in metadata
        if len(shape) == 3
    ]

    if not candidates:
        raise ValueError(f"No 3D array found in {file_path}")

    preferred = [candidate for candidate in candidates if candidate[0] == hsi_key]
    candidates_to_check = preferred if preferred else candidates

    if not any(
        is_possible_hsi_shape(shape, hsi_channels)
        for _, shape in candidates_to_check
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
    if file_path.suffix.lower() == ".mat":
        inspect_standard_mat_file(file_path, hsi_channels, hsi_key)
        return

    cube = load_hsi_file(file_path)
    if not is_possible_hsi_shape(cube.shape, hsi_channels):
        raise ValueError(f"Invalid HSI shape {cube.shape} in {file_path}")


def filter_valid_hsi_files(
    files: List[Path],
    hsi_channels: int,
    log_path: Path,
) -> List[Path]:
    valid_files = []
    invalid_records = []

    log_path.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_CACHE.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = make_files_fingerprint(files)
    path_lookup = {str(path.resolve()): path for path in files}

    if VALIDATION_CACHE.exists():
        try:
            try:
                cached_data = torch.load(
                    VALIDATION_CACHE,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                cached_data = torch.load(VALIDATION_CACHE, map_location="cpu")

            if (
                isinstance(cached_data, dict)
                and cached_data.get("fingerprint") == fingerprint
            ):
                valid_files = [
                    path_lookup[path]
                    for path in cached_data.get("valid_paths", [])
                    if path in path_lookup
                ]
                invalid_records = cached_data.get("invalid_records", [])
                print("\nUsing cached HSI file validation.")
                print(f"Valid files: {len(valid_files)}")
                print(f"Invalid files: {len(invalid_records)}")
                if valid_files:
                    return valid_files
        except Exception as error:
            print(
                "Could not use validation cache. Running a new scan. "
                f"Reason: {error}"
            )

    print("\nChecking HSI file metadata before training...")

    for file_index, file_path in enumerate(files, start=1):
        try:
            inspect_hsi_file_metadata(file_path, hsi_channels, HSI_KEY)
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
                f"  File: {file_path}\n"
                f"  Error: {error}"
            )

        if file_index % 100 == 0 or file_index == len(files):
            print(
                f"Checked {file_index}/{len(files)} files | "
                f"Valid: {len(valid_files)} | "
                f"Invalid: {len(invalid_records)}"
            )

    if not valid_files:
        raise RuntimeError("No valid HSI files remain after validation.")

    if invalid_records:
        with open(log_path, "w", encoding="utf-8") as log_file:
            for record in invalid_records:
                log_file.write(f"{record['path']} | {record['error']}\n")

    torch.save(
        {
            "fingerprint": fingerprint,
            "valid_paths": [str(path.resolve()) for path in valid_files],
            "invalid_records": invalid_records,
        },
        VALIDATION_CACHE,
    )

    return valid_files


# ============================================================
# Normalization and synchronized patch extraction
# ============================================================


def normalize_cube(cube: np.ndarray, mode: str) -> np.ndarray:
    cube = cube.astype(np.float32, copy=False)

    if mode == "none":
        return cube

    if mode == "minmax":
        minimum = cube.min()
        maximum = cube.max()
        return (cube - minimum) / (maximum - minimum + 1e-8)

    if mode == "band_minmax":
        minimum = cube.min(axis=(1, 2), keepdims=True)
        maximum = cube.max(axis=(1, 2), keepdims=True)
        return (cube - minimum) / (maximum - minimum + 1e-8)

    raise ValueError(f"Unknown HSI normalization mode: {mode}")


def normalize_rgb(array: np.ndarray, mode: str) -> np.ndarray:
    original_dtype = array.dtype
    array = array.astype(np.float32, copy=False)

    if mode == "none":
        return array

    if mode == "minmax":
        minimum = float(array.min())
        maximum = float(array.max())
        return (array - minimum) / (maximum - minimum + 1e-8)

    if mode == "auto":
        if np.issubdtype(original_dtype, np.integer):
            maximum = float(np.iinfo(original_dtype).max)
            return array / maximum

        finite_max = float(np.nanmax(array))
        finite_min = float(np.nanmin(array))

        if finite_min >= 0.0 and finite_max <= 1.0 + 1e-6:
            return array

        if finite_min >= 0.0 and finite_max <= 255.0 + 1e-6:
            return array / 255.0

        raise ValueError(
            "Floating RGB values are outside [0,1] and [0,255]. "
            "Set RGB_NORMALIZATION explicitly for this dataset."
        )

    raise ValueError(f"Unknown RGB normalization mode: {mode}")


def pad_to_patch_size(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Input shape: [C,H,W]."""
    _, height, width = tensor.shape
    pad_height = max(0, patch_size - height)
    pad_width = max(0, patch_size - width)

    if pad_height == 0 and pad_width == 0:
        return tensor

    return F.pad(
        tensor,
        pad=(0, pad_width, 0, pad_height),
        mode="replicate",
    )


def random_crop(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    tensor = pad_to_patch_size(tensor, patch_size)
    _, height, width = tensor.shape

    top = random.randint(0, height - patch_size)
    left = random.randint(0, width - patch_size)

    return tensor[:, top : top + patch_size, left : left + patch_size]


def center_crop(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    tensor = pad_to_patch_size(tensor, patch_size)
    _, height, width = tensor.shape

    top = (height - patch_size) // 2
    left = (width - patch_size) // 2

    return tensor[:, top : top + patch_size, left : left + patch_size]


def spatial_augmentation(tensor: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        tensor = torch.flip(tensor, dims=[1])
    if random.random() < 0.5:
        tensor = torch.flip(tensor, dims=[2])

    number_of_rotations = random.randint(0, 3)
    if number_of_rotations > 0:
        tensor = torch.rot90(
            tensor,
            k=number_of_rotations,
            dims=[1, 2],
        )

    return tensor.contiguous()


# ============================================================
# Dataset
# ============================================================


class HSIPatchDataset(Dataset):
    """The supplied HSI dataset with paired RGB added.

    The crop and augmentation code remains unchanged. RGB and HSI are
    concatenated temporarily so they receive exactly the same spatial crop,
    flips and rotations, then split back into separate tensors.
    """

    def __init__(
        self,
        paired_files: Sequence[Tuple[Path, Path]],
        hsi_channels: int,
        patch_size: int,
        patches_per_image: int,
        training: bool,
        hsi_normalization: str,
        rgb_normalization: str,
        augment: bool,
    ) -> None:
        self.paired_files = list(paired_files)
        self.hsi_channels = hsi_channels
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.training = training
        self.hsi_normalization = hsi_normalization
        self.rgb_normalization = rgb_normalization
        self.augment = augment

    def __len__(self) -> int:
        if self.training:
            return len(self.paired_files) * self.patches_per_image
        return len(self.paired_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            file_index = index // self.patches_per_image
        else:
            file_index = index

        hsi_path, rgb_path = self.paired_files[file_index]

        hsi = load_hsi_file(hsi_path)
        hsi = convert_hsi_to_chw(hsi, self.hsi_channels, hsi_path)

        rgb = load_rgb_file(rgb_path)
        rgb = convert_rgb_to_chw(rgb, rgb_path)

        if not np.isfinite(hsi).all():
            raise ValueError(f"NaN or infinite values found in {hsi_path}")
        if not np.isfinite(rgb).all():
            raise ValueError(f"NaN or infinite values found in {rgb_path}")

        hsi = normalize_cube(hsi, self.hsi_normalization)
        rgb = normalize_rgb(rgb, self.rgb_normalization)

        hsi_tensor = torch.from_numpy(hsi.copy()).float()
        rgb_tensor = torch.from_numpy(rgb.copy()).float()

        if hsi_tensor.shape[-2:] != rgb_tensor.shape[-2:]:
            raise ValueError(
                "Paired RGB and HSI spatial sizes differ:\n"
                f"  HSI: {hsi_path} -> {tuple(hsi_tensor.shape)}\n"
                f"  RGB: {rgb_path} -> {tuple(rgb_tensor.shape)}\n"
                "Align the pairs before training rather than silently resizing."
            )

        # Concatenation is the only change needed to reuse exactly the same
        # crop and augmentation functions for the pair.
        combined = torch.cat([hsi_tensor, rgb_tensor], dim=0)

        if self.training:
            combined = random_crop(combined, self.patch_size)
            if self.augment:
                combined = spatial_augmentation(combined)
        else:
            combined = center_crop(combined, self.patch_size)

        hsi_tensor = combined[: self.hsi_channels]
        rgb_tensor = combined[self.hsi_channels : self.hsi_channels + 3]

        return rgb_tensor, hsi_tensor


# ============================================================
# Train-validation split
# ============================================================


def split_pairs(
    pairs: List[Tuple[Path, Path]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("VALIDATION_FRACTION must be between 0 and 1.")

    shuffled = pairs.copy()
    random_generator = random.Random(seed)
    random_generator.shuffle(shuffled)

    validation_size = max(1, int(len(shuffled) * validation_fraction))
    validation_pairs = shuffled[:validation_size]
    training_pairs = shuffled[validation_size:]

    if not training_pairs:
        raise RuntimeError("No pairs remain for training after splitting.")

    return training_pairs, validation_pairs


# ============================================================
# Loss and metrics
# ============================================================


def calculate_operator_loss(
    model: RGBSpectralResponse,
    hsi: torch.Tensor,
    rgb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb_hat = model(hsi)
    rgb_mse = F.mse_loss(rgb_hat, rgb)
    smoothness = model.smoothness_regularizer()
    total = rgb_mse + OPERATOR_SMOOTHNESS_WEIGHT * smoothness
    return total, rgb_mse, smoothness, rgb_hat


@torch.no_grad()
def calculate_rgb_metrics(
    rgb_hat: torch.Tensor,
    rgb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(rgb_hat, rgb)
    mae = F.l1_loss(rgb_hat, rgb)
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
    return mse, mae, psnr


# ============================================================
# Training
# ============================================================


def train_one_epoch(
    model: RGBSpectralResponse,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.train()

    totals = {
        "loss": 0.0,
        "rgb_mse": 0.0,
        "rgb_mae": 0.0,
        "rgb_psnr": 0.0,
        "smoothness": 0.0,
    }
    total_samples = 0

    for batch_index, (rgb, hsi) in enumerate(loader, start=1):
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            loss, rgb_mse_loss, smoothness, rgb_hat = calculate_operator_loss(
                model=model,
                hsi=hsi,
                rgb=rgb,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            _, rgb_mae, rgb_psnr = calculate_rgb_metrics(
                rgb_hat.detach(),
                rgb.detach(),
            )

        batch_size = hsi.size(0)
        totals["loss"] += loss.detach().item() * batch_size
        totals["rgb_mse"] += rgb_mse_loss.detach().item() * batch_size
        totals["rgb_mae"] += rgb_mae.detach().item() * batch_size
        totals["rgb_psnr"] += rgb_psnr.detach().item() * batch_size
        totals["smoothness"] += smoothness.detach().item() * batch_size
        total_samples += batch_size

        if batch_index % PRINT_EVERY == 0 or batch_index == len(loader):
            running = {key: value / total_samples for key, value in totals.items()}
            print(
                f"  Batch {batch_index:04d}/{len(loader):04d} | "
                f"Total: {running['loss']:.8f} | "
                f"RGB MSE: {running['rgb_mse']:.8f} | "
                f"RGB MAE: {running['rgb_mae']:.6f} | "
                f"RGB PSNR: {running['rgb_psnr']:.3f} dB | "
                f"Smooth: {running['smoothness']:.8f}"
            )

    return {key: value / total_samples for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: RGBSpectralResponse,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()

    totals = {
        "loss": 0.0,
        "rgb_mse": 0.0,
        "rgb_mae": 0.0,
        "rgb_psnr": 0.0,
        "smoothness": 0.0,
    }
    total_samples = 0

    for rgb, hsi in loader:
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            loss, rgb_mse_loss, smoothness, rgb_hat = calculate_operator_loss(
                model=model,
                hsi=hsi,
                rgb=rgb,
            )
            _, rgb_mae, rgb_psnr = calculate_rgb_metrics(rgb_hat, rgb)

        batch_size = hsi.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["rgb_mse"] += rgb_mse_loss.item() * batch_size
        totals["rgb_mae"] += rgb_mae.item() * batch_size
        totals["rgb_psnr"] += rgb_psnr.item() * batch_size
        totals["smoothness"] += smoothness.item() * batch_size
        total_samples += batch_size

    return {key: value / total_samples for key, value in totals.items()}


# ============================================================
# Checkpoint saving
# ============================================================


def save_checkpoint(
    output_path: Path,
    model: RGBSpectralResponse,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    validation_metrics: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        weight = model.weight.detach().cpu()
        singular_values = torch.linalg.svdvals(weight)

    torch.save(
        {
            "epoch": epoch,
            "operator_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation_metrics": validation_metrics,
            "operator_weight": weight,
            "operator_singular_values": singular_values,
            "model_config": {
                "bands": HSI_CHANNELS,
                "constrain_nonnegative": CONSTRAIN_OPERATOR_NONNEGATIVE,
            },
            "data_config": {
                "hsi_normalization": HSI_NORMALIZATION,
                "rgb_normalization": RGB_NORMALIZATION,
                "patch_size": PATCH_SIZE,
            },
        },
        output_path,
    )

    # Also save the actual 3 x K response matrix in a directly inspectable form.
    np.save(
        output_path.with_suffix(".npy"),
        weight.numpy(),
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_hsi_files = find_hsi_files(HSI_DATA_DIR)
    all_hsi_files = filter_valid_hsi_files(
        files=all_hsi_files,
        hsi_channels=HSI_CHANNELS,
        log_path=output_dir / "invalid_hsi_files.txt",
    )

    all_pairs = build_paired_files(all_hsi_files)
    training_pairs, validation_pairs = split_pairs(
        pairs=all_pairs,
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )

    print(f"Device: {device}")
    print(f"Mixed precision: {use_amp}")
    print(f"Total paired images: {len(all_pairs)}")
    print(f"Training pairs: {len(training_pairs)}")
    print(f"Validation pairs: {len(validation_pairs)}")

    training_dataset = HSIPatchDataset(
        paired_files=training_pairs,
        hsi_channels=HSI_CHANNELS,
        patch_size=PATCH_SIZE,
        patches_per_image=PATCHES_PER_IMAGE,
        training=True,
        hsi_normalization=HSI_NORMALIZATION,
        rgb_normalization=RGB_NORMALIZATION,
        augment=USE_AUGMENTATION,
    )

    validation_dataset = HSIPatchDataset(
        paired_files=validation_pairs,
        hsi_channels=HSI_CHANNELS,
        patch_size=PATCH_SIZE,
        patches_per_image=1,
        training=False,
        hsi_normalization=HSI_NORMALIZATION,
        rgb_normalization=RGB_NORMALIZATION,
        augment=False,
    )

    # DataLoader construction is unchanged from the supplied script. The
    # dataset now simply returns (rgb, hsi) rather than hsi alone.
    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
    )

    model = RGBSpectralResponse(
        bands=HSI_CHANNELS,
        constrain_nonnegative=CONSTRAIN_OPERATOR_NONNEGATIVE,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-7,
    )

    scaler = GradScaler(enabled=use_amp)

    best_validation_mse = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        training_metrics = train_one_epoch(
            model=model,
            loader=training_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        validation_metrics = validate(
            model=model,
            loader=validation_loader,
            device=device,
            use_amp=use_amp,
        )

        # CosineAnnealingLR does not take a validation loss argument.
        scheduler.step()

        current_learning_rate = optimizer.param_groups[0]["lr"]

        with torch.no_grad():
            singular_values = torch.linalg.svdvals(model.weight.detach()).cpu()

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS:03d} | "
            f"LR: {current_learning_rate:.2e} | "
            f"Train total: {training_metrics['loss']:.8f} | "
            f"Train RGB MSE: {training_metrics['rgb_mse']:.8f} | "
            f"Train RGB PSNR: {training_metrics['rgb_psnr']:.3f} dB | "
            f"Val total: {validation_metrics['loss']:.8f} | "
            f"Val RGB MSE: {validation_metrics['rgb_mse']:.8f} | "
            f"Val RGB MAE: {validation_metrics['rgb_mae']:.6f} | "
            f"Val RGB PSNR: {validation_metrics['rgb_psnr']:.3f} dB"
        )
        print(f"  Operator singular values: {singular_values.tolist()}")

        save_checkpoint(
            output_path=output_dir / "last_rgb_operator.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            validation_metrics=validation_metrics,
        )

        if validation_metrics["rgb_mse"] < best_validation_mse:
            best_validation_mse = validation_metrics["rgb_mse"]

            save_checkpoint(
                output_path=output_dir / "best_rgb_operator.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                validation_metrics=validation_metrics,
            )

            print(
                "Saved new best RGB operator checkpoint: "
                f"MSE={best_validation_mse:.8f}"
            )

    print("\nTraining complete.")
    print(f"Best operator: {output_dir / 'best_rgb_operator.pth'}")
    print(
        "Next, load this operator into RGBDDS2M, freeze it, and optimize a "
        "fresh VS2M generator separately for each RGB image during reverse "
        "diffusion."
    )


if __name__ == "__main__":
    main()
