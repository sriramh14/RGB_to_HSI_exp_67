import random
import hashlib
from pathlib import Path
from typing import List, Tuple
import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler

from models.HSI_VAE import HSIVAE
from models.diffusion_DiT import RGB_to_HSI_w_diffusion  # adjust to your actual module path

from loss.mrae import mrae
from loss.psnr import psnr
from loss.rmse import rmse
from loss.sam import sam
from loss.ssim import ssim


# ============================================================
# Configuration
# ============================================================

# Directories containing paired HSI and RGB files.
# HSI files and RGB files must share the same filename stem,
# e.g. "scene_001.mat" (HSI) and "scene_001.png" (RGB).
HSI_DATA_DIR  = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_spectral/Train_spectral"
RGB_DATA_DIR  = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_RGB/Train_RGB"

# Path to a pre-trained VAE checkpoint produced by the VAE training script.
VAE_CHECKPOINT = "./vae_checkpoints/best_vae.pth"

OUTPUT_DIR = "./diffusion_checkpoints"

HSI_KEY = "cube"
VALIDATION_CACHE = Path(OUTPUT_DIR) / "diffusion_validation_cache.pth"

# ── Model architecture ────────────────────────────────────────────────────────
HSI_CHANNELS    = 31
BASE_CHANNELS   = 64
LATENT_CHANNELS = 16
NUM_RES_BLOCKS  = 2

# DiT hyper-parameters
HIDDEN_SIZE  = 128
DEPTH        = 10
NUM_HEADS    = 16
MLP_RATIO    = 4.0
PATCH_SIZE   = 4       # DiT patch size (tokens)
INPUT_SIZE   = 16      # spatial size of the latent (64 px image → 2 downsamples → 16)
LEARN_SIGMA  = True

# ── Diffusion scheduler ───────────────────────────────────────────────────────
NUM_TRAIN_TIMESTEPS = 1000
BETA_SCHEDULE       = "squaredcos_cap_v2"   # cosine schedule; change to "linear" if preferred

# ── Training ──────────────────────────────────────────────────────────────────
PATCH_SIZE_PX      = 64    # spatial crop size fed to the model (pixels)
PATCHES_PER_IMAGE  = 4

BATCH_SIZE     = 4
NUM_EPOCHS     = 100
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4

VALIDATION_FRACTION = 0.1
NUM_WORKERS         = 4
USE_AMP             = True
USE_AUGMENTATION    = True

SEED                = 42
GRADIENT_CLIP_NORM  = 1.0
PRINT_EVERY         = 30

# ── File formats ──────────────────────────────────────────────────────────────
SUPPORTED_HSI_EXTENSIONS = {".npy", ".npz", ".mat", ".pt", ".pth"}
SUPPORTED_RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".npy", ".pt", ".pth"}


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# File loading  (HSI — identical helpers to the VAE script)
# ============================================================

def load_mat_v73(file_path: Path) -> np.ndarray:
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
        raise ValueError(f"No numerical 3D HSI array found in: {file_path}")

    _, cube = max(candidates, key=lambda item: item[1].size)
    cube = np.transpose(cube, axes=tuple(range(cube.ndim - 1, -1, -1)))
    return cube


def extract_array_from_dictionary(data: dict, file_path: Path) -> np.ndarray:
    candidates = []
    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray) and value.ndim == 3:
            candidates.append(value)
    if not candidates:
        raise ValueError(f"No three-dimensional HSI array found in {file_path}")
    return max(candidates, key=lambda array: array.size)


def load_hsi_file(file_path: Path) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension == ".npy":
        cube = np.load(file_path)

    elif extension == ".npz":
        loaded = np.load(file_path)
        candidates = [loaded[k] for k in loaded.files if loaded[k].ndim == 3]
        if not candidates:
            raise ValueError(f"No three-dimensional array found in {file_path}")
        cube = max(candidates, key=lambda a: a.size)

    elif extension == ".mat":
        try:
            loaded = sio.loadmat(file_path)
            cube = extract_array_from_dictionary(loaded, file_path)
        except NotImplementedError:
            cube = load_mat_v73(file_path)

    elif extension in {".pt", ".pth"}:
        loaded = torch.load(file_path, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            cube = loaded.detach().cpu().numpy()
        elif isinstance(loaded, np.ndarray):
            cube = loaded
        elif isinstance(loaded, dict):
            cube = extract_array_from_dictionary(loaded, file_path)
        else:
            raise TypeError(f"Unsupported object in {file_path}: {type(loaded)}")

    else:
        raise ValueError(f"Unsupported HSI extension: {extension}")

    cube = np.asarray(cube, dtype=np.float32)
    cube = np.squeeze(cube)

    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D cube in {file_path}, but found shape {cube.shape}"
        )
    return cube


def convert_to_chw(
    cube: np.ndarray,
    hsi_channels: int,
    file_path: Path,
) -> np.ndarray:
    if cube.shape[0] == hsi_channels:
        return cube
    if cube.shape[-1] == hsi_channels:
        return np.transpose(cube, (2, 0, 1))
    raise ValueError(
        f"Cannot identify the spectral dimension in {file_path}. "
        f"Shape: {cube.shape}, expected bands: {hsi_channels}"
    )


def load_rgb_file(file_path: Path) -> np.ndarray:
    """
    Load an RGB image as a float32 array in [C, H, W] format, range [0, 1].
    Supports: .png/.jpg/.jpeg (via PIL), .npy, .pt/.pth.
    """
    extension = file_path.suffix.lower()

    if extension in {".png", ".jpg", ".jpeg"}:
        from PIL import Image
        img = Image.open(file_path).convert("RGB")
        array = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
        return np.transpose(array, (2, 0, 1))                # (3, H, W)

    if extension == ".npy":
        array = np.load(file_path).astype(np.float32)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=0)
        elif array.shape[-1] == 3:
            array = np.transpose(array, (2, 0, 1))
        return array

    if extension in {".pt", ".pth"}:
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

    raise ValueError(f"Unsupported RGB extension: {extension}")


# ============================================================
# File discovery and pairing
# ============================================================

def find_hsi_files(data_dir: str) -> List[Path]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"HSI directory does not exist: {data_path}")
    files = sorted(
        p for p in data_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_HSI_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No HSI files found in {data_path}")
    return files


def pair_hsi_rgb_files(
    hsi_dir: str,
    rgb_dir: str,
) -> List[Tuple[Path, Path]]:
    """
    Match HSI files to RGB files by filename stem.
    Raises if no pairs are found.
    """
    hsi_files = find_hsi_files(hsi_dir)

    rgb_path = Path(rgb_dir)
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB directory does not exist: {rgb_path}")

    rgb_by_stem = {}
    for p in rgb_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_RGB_EXTENSIONS:
            rgb_by_stem[p.stem] = p

    pairs = []
    missing = []
    for hsi_file in hsi_files:
        rgb_file = rgb_by_stem.get(hsi_file.stem)
        if rgb_file is not None:
            pairs.append((hsi_file, rgb_file))
        else:
            missing.append(hsi_file)

    if missing:
        print(f"\nWarning: {len(missing)} HSI files have no matching RGB file:")
        for f in missing[:10]:
            print(f"  {f}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more.")

    if not pairs:
        raise RuntimeError(
            "No HSI/RGB pairs found. "
            "Check that HSI and RGB files share the same filename stem."
        )

    print(f"Found {len(pairs)} paired HSI/RGB files.")
    return pairs


# ============================================================
# Validation cache
# ============================================================

def make_files_fingerprint(files: List[Path]) -> str:
    records = []
    for file_path in files:
        stat = file_path.stat()
        records.append(
            f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        )
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()

# ============================================================
# HSI metadata inspection  (fast — no full cube load)
# ============================================================

def is_possible_hsi_shape(shape, hsi_channels: int) -> bool:
    return (
        len(shape) == 3
        and hsi_channels in shape
        and all(int(s) > 0 for s in shape)
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
                    if np.issubdtype(obj.dtype, np.number):
                        candidates.append((name, tuple(int(v) for v in obj.shape)))
                except TypeError:
                    pass
            h5_file.visititems(visitor)

    if not candidates:
        raise ValueError(f"No numerical 3D dataset found in {file_path}")

    if not any(is_possible_hsi_shape(shape, hsi_channels) for _, shape in candidates):
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
        # Falls back to HDF5 header inspection for v7.3 files.
        inspect_hdf5_mat_file(file_path, hsi_channels, hsi_key)
        return

    candidates = [
        (name, tuple(int(v) for v in shape))
        for name, shape, _ in metadata
        if len(shape) == 3
    ]

    if not candidates:
        raise ValueError(f"No 3D array found in {file_path}")

    preferred = [c for c in candidates if c[0] == hsi_key]
    to_check  = preferred if preferred else candidates

    if not any(is_possible_hsi_shape(shape, hsi_channels) for _, shape in to_check):
        raise ValueError(
            f"No {hsi_channels}-band cube found in {file_path}. "
            f"MAT arrays: {candidates}"
        )


def inspect_hsi_file_metadata(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    """
    Validate without loading the full cube.
    .mat  → whosmat() or HDF5 header only.
    other → full load only as a last resort (npy/npz/pt are fast anyway).
    """
    if file_path.suffix.lower() == ".mat":
        inspect_standard_mat_file(file_path, hsi_channels, hsi_key)
        return

    # For .npy / .npz / .pt / .pth the load is already cheap,
    # so we reuse the existing loader and just check the shape.
    cube = load_hsi_file(file_path)
    if not is_possible_hsi_shape(cube.shape, hsi_channels):
        raise ValueError(
            f"Invalid HSI shape {cube.shape} in {file_path}"
        )


# ============================================================
# Pair validation with cache  (replaces the old filter_valid_pairs)
# ============================================================

def filter_valid_pairs(
    pairs: List[Tuple[Path, Path]],
    hsi_channels: int,
    log_path: Path,
) -> List[Tuple[Path, Path]]:
    """
    Fast-validate HSI files by metadata only; RGB files are checked
    by extension. Results are cached and reused when the file set
    is unchanged (same paths, sizes, and mtimes).
    """
    VALIDATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    hsi_files   = [h for h, _ in pairs]
    fingerprint = make_files_fingerprint(hsi_files)
    pair_lookup = {str(h.resolve()): (h, r) for h, r in pairs}

    # ── Try cache ─────────────────────────────────────────────────────────────
    if VALIDATION_CACHE.exists():
        try:
            try:
                cached = torch.load(
                    VALIDATION_CACHE, map_location="cpu", weights_only=False
                )
            except TypeError:
                cached = torch.load(VALIDATION_CACHE, map_location="cpu")

            if (
                isinstance(cached, dict)
                and cached.get("fingerprint") == fingerprint
            ):
                valid_paths   = cached.get("valid_hsi_paths", [])
                invalid_records = cached.get("invalid_records", [])
                valid_pairs   = [
                    pair_lookup[p] for p in valid_paths if p in pair_lookup
                ]

                print("\nUsing cached pair validation.")
                print(f"Valid pairs:   {len(valid_pairs)}")
                print(f"Invalid files: {len(invalid_records)}")

                for record in invalid_records:
                    print(
                        f"\nCached invalid file:\n"
                        f"  File:  {record['path']}\n"
                        f"  Error: {record['error']}"
                    )

                if valid_pairs:
                    return valid_pairs

        except Exception as error:
            print(f"Could not use validation cache. Re-scanning.\nReason: {error}")

    # ── Full scan (metadata only for .mat, cheap load for others) ─────────────
    print("\nChecking HSI file metadata before training...")

    valid_pairs     = []
    invalid_records = []

    for idx, (hsi_file, rgb_file) in enumerate(pairs, start=1):
        try:
            inspect_hsi_file_metadata(
                file_path=hsi_file,
                hsi_channels=hsi_channels,
                hsi_key=HSI_KEY,
            )
            valid_pairs.append((hsi_file, rgb_file))

        except Exception as error:
            invalid_records.append({
                "path":  str(hsi_file.resolve()),
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"\nSkipping invalid file:\n  File: {hsi_file}\n  Error: {error}")

        if idx % 100 == 0 or idx == len(pairs):
            print(
                f"Checked {idx}/{len(pairs)} | "
                f"Valid: {len(valid_pairs)} | "
                f"Invalid: {len(invalid_records)}"
            )

    if not valid_pairs:
        raise RuntimeError("No valid HSI/RGB pairs remain after validation.")

    if invalid_records:
        with open(log_path, "w", encoding="utf-8") as f:
            for r in invalid_records:
                f.write(f"{r['path']} | {r['error']}\n")
        print(f"\nInvalid-file log saved to: {log_path}")

    torch.save(
        {
            "fingerprint":    fingerprint,
            "valid_hsi_paths": [str(h.resolve()) for h, _ in valid_pairs],
            "invalid_records": invalid_records,
        },
        VALIDATION_CACHE,
    )
    print(f"Validation cache saved to: {VALIDATION_CACHE}")
    return valid_pairs


# ============================================================
# Normalization and patch extraction  (unchanged from VAE script)
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


def pad_to_patch_size(cube: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, h, w = cube.shape
    ph = max(0, patch_size - h)
    pw = max(0, patch_size - w)
    if ph == 0 and pw == 0:
        return cube
    return F.pad(cube, (0, pw, 0, ph), mode="replicate")


def random_crop_pair(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop HSI and RGB at the same spatial location."""
    hsi = pad_to_patch_size(hsi, patch_size)
    rgb = pad_to_patch_size(rgb, patch_size)

    _, h, w = hsi.shape
    top  = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    hsi = hsi[:, top:top + patch_size, left:left + patch_size]
    rgb = rgb[:, top:top + patch_size, left:left + patch_size]
    return hsi, rgb


def center_crop_pair(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hsi = pad_to_patch_size(hsi, patch_size)
    rgb = pad_to_patch_size(rgb, patch_size)

    _, h, w = hsi.shape
    top  = (h - patch_size) // 2
    left = (w - patch_size) // 2

    hsi = hsi[:, top:top + patch_size, left:left + patch_size]
    rgb = rgb[:, top:top + patch_size, left:left + patch_size]
    return hsi, rgb


def spatial_augmentation_pair(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the same random flip / rotation to both modalities."""
    if random.random() < 0.5:
        hsi = torch.flip(hsi, dims=[1])
        rgb = torch.flip(rgb, dims=[1])
    if random.random() < 0.5:
        hsi = torch.flip(hsi, dims=[2])
        rgb = torch.flip(rgb, dims=[2])
    k = random.randint(0, 3)
    if k > 0:
        hsi = torch.rot90(hsi, k=k, dims=[1, 2])
        rgb = torch.rot90(rgb, k=k, dims=[1, 2])
    return hsi.contiguous(), rgb.contiguous()


# ============================================================
# Dataset
# ============================================================

class HSIRGBPairDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        hsi_channels: int,
        patch_size: int,
        patches_per_image: int,
        training: bool,
        normalization: str,
        augment: bool,
    ):
        self.pairs             = pairs
        self.hsi_channels      = hsi_channels
        self.patch_size        = patch_size
        self.patches_per_image = patches_per_image
        self.training          = training
        self.normalization     = normalization
        self.augment           = augment

    def __len__(self) -> int:
        if self.training:
            return len(self.pairs) * self.patches_per_image
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_index = index // self.patches_per_image if self.training else index
        hsi_path, rgb_path = self.pairs[file_index]

        # ── Load HSI ──────────────────────────────────────────────────────────
        cube = load_hsi_file(hsi_path)
        cube = convert_to_chw(cube, self.hsi_channels, hsi_path)

        if not np.isfinite(cube).all():
            raise ValueError(f"NaN or Inf values in {hsi_path}")

        cube = normalize_cube(cube, self.normalization)
        hsi  = torch.from_numpy(cube.copy()).float()

        # ── Load RGB ──────────────────────────────────────────────────────────
        rgb_array = load_rgb_file(rgb_path)              # (3, H, W) float32 [0,1]
        rgb       = torch.from_numpy(rgb_array.copy()).float()

        # ── Crop + augment (same transform for both) ──────────────────────────
        if self.training:
            hsi, rgb = random_crop_pair(hsi, rgb, self.patch_size)
            if self.augment:
                hsi, rgb = spatial_augmentation_pair(hsi, rgb)
        else:
            hsi, rgb = center_crop_pair(hsi, rgb, self.patch_size)

        return hsi, rgb


# ============================================================
# Train / validation split
# ============================================================

def split_pairs(
    pairs: List[Tuple[Path, Path]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("VALIDATION_FRACTION must be between 0 and 1.")

    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)

    val_size  = max(1, int(len(shuffled) * validation_fraction))
    val_pairs = shuffled[:val_size]
    trn_pairs = shuffled[val_size:]

    if not trn_pairs:
        raise RuntimeError("No pairs remain for training after splitting.")

    return trn_pairs, val_pairs


# ============================================================
# VAE loading
# ============================================================

def load_pretrained_vae(
    checkpoint_path: str,
    device: torch.device,
) -> HSIVAE:
    """Load a frozen, pre-trained HSIVAE from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = ckpt.get("model_config", {})
    vae = HSIVAE(
        hsi_channels  = cfg.get("hsi_channels",  HSI_CHANNELS),
        base_channels = cfg.get("base_channels",  BASE_CHANNELS),
        latent_channels = cfg.get("latent_channels", LATENT_CHANNELS),
        num_res_blocks = cfg.get("num_res_blocks",  NUM_RES_BLOCKS),
    )

    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()

    for param in vae.parameters():
        param.requires_grad_(False)

    return vae.to(device)


# ============================================================
# Metrics (pixel-space, computed from decoded HSI)
# ============================================================

def calculate_aux_losses(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mrae_value = mrae(target, reconstruction)
    rmse_value = rmse(target, reconstruction)
    sam_value  = sam(target, reconstruction)
    psnr_value = psnr(target, reconstruction)
    ssim_value = ssim(target, reconstruction)
    return mrae_value, rmse_value, sam_value, psnr_value, ssim_value


# ============================================================
# Training
# ============================================================
def sample_logit_normal_timesteps(
    batch_size: int,
    device: torch.device,
    m: float = 0.0,
    s: float = 1.0,
) -> torch.Tensor:
    """
    Sample timesteps from a logit-normal distribution.

    π_ln(t; m, s) = 1 / (s√(2π) · t(1-t)) · exp(-(logit(t) - m)² / 2s²)

    This is equivalent to:
        u ~ N(m, s²)
        t = sigmoid(u) = 1 / (1 + e^{-u})
    which keeps t strictly in (0, 1).

    Args:
        batch_size: number of timesteps to sample (one per batch item).
        device:     target device.
        m:          mean of the underlying normal (0.0 = symmetric around t=0.5).
        s:          std of the underlying normal (higher = wider spread).

    Returns:
        t: (batch_size,) float tensor with values in (0, 1).
    """
    u = torch.randn(batch_size, device=device) * s + m
    t = torch.sigmoid(u)   # logit-normal sample: t = sigmoid(N(m, s²))
    return t


def train_one_epoch(
    model: RGB_to_HSI_w_diffusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.train()

    total_noise_loss = 0.0
    total_mrae = total_rmse = total_sam = total_psnr = total_ssim = 0.0
    total_samples = 0
    trainable_params = list(model.dit.parameters())
    
    for batch_index, (hsi, rgb) in enumerate(loader, start=1):
        hsi = hsi.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)

        # Sample random timesteps for each item in the batch.
        t = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (hsi.size(0),),
            device=device,
        )

        # After
        #t = sample_logit_normal_timesteps(
            #batch_size=hsi.size(0),
            #device=device,
            #m=0.0,   # adjust to bias toward noisier (m < 0) or cleaner (m > 0) timesteps
            #s=1.0,
        #)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            loss, hsi_recon, _ = model(hsi, rgb, t, noise_scheduler)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        scaler.scale(loss).backward()
        #scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            trainable_params, max_norm=GRADIENT_CLIP_NORM
        )
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            mrae_v, rmse_v, sam_v, psnr_v, ssim_v = calculate_aux_losses(
                hsi_recon.detach(), hsi.detach()
            )

        batch_size = hsi.size(0)
        total_noise_loss += loss.detach().item() * batch_size
        total_mrae  += mrae_v.item() * batch_size
        total_rmse  += rmse_v.item() * batch_size
        total_sam   += sam_v.item()  * batch_size
        total_psnr  += psnr_v.item() * batch_size
        total_ssim  += ssim_v.item() * batch_size
        total_samples += batch_size

        if batch_index % PRINT_EVERY == 0 or batch_index == len(loader):
            n = total_samples
            print(
                f"  Batch {batch_index:04d}/{len(loader):04d} | "
                f"Noise loss: {total_noise_loss / n:.6f} | "
                f"MRAE: {total_mrae / n:.6f} | "
                f"RMSE: {total_rmse / n:.6f} | "
                f"SAM: {total_sam / n:.6f} | "
                f"PSNR: {total_psnr / n:.4f} | "
                f"SSIM: {total_ssim / n:.4f}"
            )

    n = total_samples
    return {
        "noise_loss": total_noise_loss / n,
        "mrae":  total_mrae  / n,
        "rmse":  total_rmse  / n,
        "sam":   total_sam   / n,
        "psnr":  total_psnr  / n,
        "ssim":  total_ssim  / n,
    }


@torch.no_grad()
def validate(
    model: RGB_to_HSI_w_diffusion,
    loader: DataLoader,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()

    total_noise_loss = 0.0
    total_mrae = total_rmse = total_sam = total_psnr = total_ssim = 0.0
    total_samples = 0

    for hsi, rgb in loader:
        hsi = hsi.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)

        t = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (hsi.size(0),),
            device=device,
        )

        with autocast(enabled=use_amp):
            loss, hsi_recon, _ = model(hsi, rgb, t, noise_scheduler)

        mrae_v, rmse_v, sam_v, psnr_v, ssim_v = calculate_aux_losses(
            hsi_recon, hsi
        )

        batch_size = hsi.size(0)
        total_noise_loss += loss.item() * batch_size
        total_mrae  += mrae_v.item() * batch_size
        total_rmse  += rmse_v.item() * batch_size
        total_sam   += sam_v.item()  * batch_size
        total_psnr  += psnr_v.item() * batch_size
        total_ssim  += ssim_v.item() * batch_size
        total_samples += batch_size

    n = total_samples
    return {
        "noise_loss": total_noise_loss / n,
        "mrae":  total_mrae  / n,
        "rmse":  total_rmse  / n,
        "sam":   total_sam   / n,
        "psnr":  total_psnr  / n,
        "ssim":  total_ssim  / n,
    }


# ============================================================
# Checkpoint saving
# ============================================================

def save_checkpoint(
    output_path: Path,
    model: RGB_to_HSI_w_diffusion,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "dit_state_dict": model.dit.state_dict(),
            "vae_state_dict": model.vae.state_dict(),  # ← added
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "model_config": {
                "hsi_channels":    HSI_CHANNELS,
                "base_channels":   BASE_CHANNELS,
                "latent_channels": LATENT_CHANNELS,
                "num_res_blocks":  NUM_RES_BLOCKS,
                "hidden_size":     HIDDEN_SIZE,
                "depth":           DEPTH,
                "num_heads":       NUM_HEADS,
                "mlp_ratio":       MLP_RATIO,
                "patch_size":      PATCH_SIZE,
                "input_size":      INPUT_SIZE,
                "learn_sigma":     LEARN_SIGMA,
            },
        },
        output_path,
    )

# ============================================================
# Main
# ============================================================

def main() -> None:
    set_seed(SEED)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    all_pairs = pair_hsi_rgb_files(HSI_DATA_DIR, RGB_DATA_DIR)
    all_pairs = filter_valid_pairs(
        pairs=all_pairs,
        hsi_channels=HSI_CHANNELS,
        log_path=output_dir / "invalid_hsi_files.txt",
    )
    train_pairs, val_pairs = split_pairs(all_pairs, VALIDATION_FRACTION, SEED)

    print(f"\nDevice: {device}")
    print(f"Mixed precision: {use_amp}")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")

    train_dataset = HSIRGBPairDataset(
        pairs=train_pairs,
        hsi_channels=HSI_CHANNELS,
        patch_size=PATCH_SIZE_PX,
        patches_per_image=PATCHES_PER_IMAGE,
        training=True,
        normalization="none",
        augment=USE_AUGMENTATION,
    )
    val_dataset = HSIRGBPairDataset(
        pairs=val_pairs,
        hsi_channels=HSI_CHANNELS,
        patch_size=PATCH_SIZE_PX,
        patches_per_image=1,
        training=False,
        normalization="none",
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    pretrained_vae = load_pretrained_vae(VAE_CHECKPOINT, device)

    model = RGB_to_HSI_w_diffusion(
        hsi_channels=HSI_CHANNELS,
        base_channels=BASE_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        num_res_blocks=NUM_RES_BLOCKS,
        hidden_size=HIDDEN_SIZE,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        learn_sigma=LEARN_SIGMA,
        patch_size=PATCH_SIZE,
        input_size=INPUT_SIZE,
        T = NUM_TRAIN_TIMESTEPS
    ).to(device)

    # Replace the randomly-initialised VAE inside the model with the
    # pre-trained one we just loaded (already frozen).
    model.vae = pretrained_vae

    # Only the DiT is trainable.
    trainable_params = list(model.dit.parameters())
    print(
        f"\nTrainable parameters: "
        f"{sum(p.numel() for p in trainable_params):,}"
    )

    # ── Noise scheduler ───────────────────────────────────────────────────────
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        beta_schedule=BETA_SCHEDULE,
    )

    # ── Optimiser and LR scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-7,
    )
    scaler = GradScaler(enabled=use_amp)

    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            noise_scheduler=noise_scheduler,
            device=device,
            use_amp=use_amp,
        )
        val_metrics = validate(
            model=model,
            loader=val_loader,
            noise_scheduler=noise_scheduler,
            device=device,
            use_amp=use_amp,
        )
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"\nEpoch {epoch:03d}/{NUM_EPOCHS:03d} | "
            f"LR: {current_lr:.2e} | "
            f"Train noise loss: {train_metrics['noise_loss']:.6f} | "
            f"Val noise loss:   {val_metrics['noise_loss']:.6f} | "
            f"Val MRAE: {val_metrics['mrae']:.6f} | "
            f"Val RMSE: {val_metrics['rmse']:.6f} | "
            f"Val SAM:  {val_metrics['sam']:.6f} | "
            f"Val PSNR: {val_metrics['psnr']:.4f} | "
            f"Val SSIM: {val_metrics['ssim']:.4f}"
        )

        # Always save latest.
        save_checkpoint(
            output_path=output_dir / "last_diffusion.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_loss=val_metrics["noise_loss"],
        )

        # Save best by validation noise loss.
        if val_metrics["noise_loss"] < best_val_loss:
            best_val_loss = val_metrics["noise_loss"]
            save_checkpoint(
                output_path=output_dir / "best_diffusion.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=best_val_loss,
            )
            print(f"  ✓ New best checkpoint: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
