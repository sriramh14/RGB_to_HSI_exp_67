import random
from pathlib import Path
from typing import List
import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from models.HSI_VAE import HSIVAE

# Change these imports to match your loss files.
from loss.vae_losses import reconstruction_loss,kl_divergence_loss
from loss.mrae import mrae
from loss.psnr import psnr
from loss.rmse import rmse
from loss.sam import sam
from loss.ssim import ssim


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_spectral"
OUTPUT_DIR = "./vae_checkpoints"

HSI_CHANNELS = 31
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 8

BASE_CHANNELS = 64
LATENT_CHANNELS = 16
NUM_RES_BLOCKS = 2

BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

KL_WEIGHT = 1e-6
VALIDATION_FRACTION = 0.1

# Options:
#   "none"        : dataset is already normalized
#   "minmax"      : normalize each full cube
#   "band_minmax" : normalize every spectral band separately
NORMALIZATION = "none"

NUM_WORKERS = 4
USE_AMP = True
USE_AUGMENTATION = True

SEED = 42
GRADIENT_CLIP_NORM = 1.0


SUPPORTED_EXTENSIONS = {
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
# File loading
# ============================================================

def load_mat_v73(file_path: Path) -> np.ndarray:
    """
    Loads the largest 3D numerical dataset from a MATLAB v7.3 file.
    """

    candidates = []

    def visit_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            try:
                array = np.asarray(obj)

                if (
                    array.ndim == 3
                    and np.issubdtype(array.dtype, np.number)
                ):
                    candidates.append((name, array))

            except Exception:
                pass

    with h5py.File(str(file_path), "r") as file:
        file.visititems(visit_dataset)

    if not candidates:
        raise ValueError(
            f"No numerical 3D HSI array found in MATLAB v7.3 file: "
            f"{file_path}"
        )

    _, cube = max(
        candidates,
        key=lambda item: item[1].size,
    )

    # MATLAB v7.3 dimensions are usually reversed when read with h5py.
    # Example:
    # MATLAB [H, W, C] -> h5py [C, W, H]
    cube = np.transpose(
        cube,
        axes=tuple(range(cube.ndim - 1, -1, -1)),
    )

    return cube


def find_hsi_files(data_dir: str) -> List[Path]:
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_path}"
        )

    files = sorted(
        path
        for path in data_path.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )

    if not files:
        raise RuntimeError(
            f"No supported HSI files found in {data_path}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return files


def extract_array_from_dictionary(
    data: dict,
    file_path: Path,
) -> np.ndarray:
    candidates = []

    for key, value in data.items():
        if key.startswith("__"):
            continue

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        if isinstance(value, np.ndarray) and value.ndim == 3:
            candidates.append(value)

    if not candidates:
        raise ValueError(
            f"No three-dimensional HSI array found in {file_path}"
        )

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
            if loaded[key].ndim == 3
        ]

        if not candidates:
            raise ValueError(
                f"No three-dimensional array found in {file_path}"
            )

        cube = max(candidates, key=lambda array: array.size)

    elif extension == ".mat":
        try:
            # MATLAB files older than v7.3
            loaded = sio.loadmat(file_path)
    
            cube = extract_array_from_dictionary(
                loaded,
                file_path,
            )

        except NotImplementedError:
            # MATLAB v7.3 HDF5 files
            cube = load_mat_v73(file_path)

    elif extension in {".pt", ".pth"}:
        loaded = torch.load(
            file_path,
            map_location="cpu",
        )

        if isinstance(loaded, torch.Tensor):
            cube = loaded.detach().cpu().numpy()

        elif isinstance(loaded, np.ndarray):
            cube = loaded

        elif isinstance(loaded, dict):
            cube = extract_array_from_dictionary(
                loaded,
                file_path,
            )

        else:
            raise TypeError(
                f"Unsupported object in {file_path}: "
                f"{type(loaded)}"
            )

    else:
        raise ValueError(
            f"Unsupported extension: {extension}"
        )

    cube = np.asarray(cube, dtype=np.float32)
    cube = np.squeeze(cube)

    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D cube in {file_path}, "
            f"but found shape {cube.shape}"
        )

    return cube


def convert_to_chw(
    cube: np.ndarray,
    hsi_channels: int,
    file_path: Path,
) -> np.ndarray:
    """
    Converts [C,H,W] or [H,W,C] into [C,H,W].
    """

    if cube.shape[0] == hsi_channels:
        return cube

    if cube.shape[-1] == hsi_channels:
        return np.transpose(
            cube,
            (2, 0, 1),
        )

    raise ValueError(
        f"Cannot identify the spectral dimension in {file_path}. "
        f"Shape: {cube.shape}, expected bands: {hsi_channels}"
    )


# ============================================================
# Normalization and patch extraction
# ============================================================

def normalize_cube(
    cube: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return cube

    if mode == "minmax":
        minimum = cube.min()
        maximum = cube.max()

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


def pad_to_patch_size(
    cube: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """
    Input shape: [C,H,W]
    """

    _, height, width = cube.shape

    pad_height = max(
        0,
        patch_size - height,
    )

    pad_width = max(
        0,
        patch_size - width,
    )

    if pad_height == 0 and pad_width == 0:
        return cube

    cube = F.pad(
        cube,
        pad=(
            0,
            pad_width,
            0,
            pad_height,
        ),
        mode="replicate",
    )

    return cube


def random_crop(
    cube: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    cube = pad_to_patch_size(
        cube,
        patch_size,
    )

    _, height, width = cube.shape

    top = random.randint(
        0,
        height - patch_size,
    )

    left = random.randint(
        0,
        width - patch_size,
    )

    return cube[
        :,
        top:top + patch_size,
        left:left + patch_size,
    ]


def center_crop(
    cube: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    cube = pad_to_patch_size(
        cube,
        patch_size,
    )

    _, height, width = cube.shape

    top = (
        height - patch_size
    ) // 2

    left = (
        width - patch_size
    ) // 2

    return cube[
        :,
        top:top + patch_size,
        left:left + patch_size,
    ]


def spatial_augmentation(
    cube: torch.Tensor,
) -> torch.Tensor:
    if random.random() < 0.5:
        cube = torch.flip(
            cube,
            dims=[1],
        )

    if random.random() < 0.5:
        cube = torch.flip(
            cube,
            dims=[2],
        )

    number_of_rotations = random.randint(
        0,
        3,
    )

    if number_of_rotations > 0:
        cube = torch.rot90(
            cube,
            k=number_of_rotations,
            dims=[1, 2],
        )

    return cube.contiguous()


# ============================================================
# Dataset
# ============================================================

class HSIPatchDataset(Dataset):
    def __init__(
        self,
        files: List[Path],
        hsi_channels: int,
        patch_size: int,
        patches_per_image: int,
        training: bool,
        normalization: str,
        augment: bool,
    ):
        self.files = files
        self.hsi_channels = hsi_channels
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.training = training
        self.normalization = normalization
        self.augment = augment

    def __len__(self) -> int:
        if self.training:
            return (
                len(self.files)
                * self.patches_per_image
            )

        return len(self.files)

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor:
        if self.training:
            file_index = (
                index
                // self.patches_per_image
            )
        else:
            file_index = index

        file_path = self.files[file_index]

        cube = load_hsi_file(file_path)

        cube = convert_to_chw(
            cube=cube,
            hsi_channels=self.hsi_channels,
            file_path=file_path,
        )

        if not np.isfinite(cube).all():
            raise ValueError(
                f"NaN or infinite values found in {file_path}"
            )

        cube = normalize_cube(
            cube,
            self.normalization,
        )

        cube = torch.from_numpy(
            cube.copy()
        ).float()

        if self.training:
            cube = random_crop(
                cube,
                self.patch_size,
            )

            if self.augment:
                cube = spatial_augmentation(cube)

        else:
            cube = center_crop(
                cube,
                self.patch_size,
            )

        return cube


# ============================================================
# Train-validation split
# ============================================================

def split_files(
    files: List[Path],
    validation_fraction: float,
    seed: int,
) -> tuple[List[Path], List[Path]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "VALIDATION_FRACTION must be between 0 and 1."
        )

    shuffled_files = files.copy()

    random_generator = random.Random(seed)
    random_generator.shuffle(shuffled_files)

    validation_size = max(
        1,
        int(
            len(shuffled_files)
            * validation_fraction
        ),
    )

    validation_files = shuffled_files[
        :validation_size
    ]

    training_files = shuffled_files[
        validation_size:
    ]

    if not training_files:
        raise RuntimeError(
            "No files remain for training after splitting."
        )

    return training_files, validation_files


# ============================================================
# Loss
# ============================================================

def calculate_vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Change this function if your loss signatures differ.
    """

    reconstruction_value = reconstruction_loss(
        reconstruction,
        target,
    )

    kl_value = kl_divergence_loss(
        mu,
        logvar,
    )

    total_loss = (
        reconstruction_value
        + KL_WEIGHT * kl_value
    )

    return (
        total_loss,
        reconstruction_value,
        kl_value,
    ) 


def calculate_aux_losses (
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    mrae_value = mrae(target,reconstruction)
    rmse_value = rmse(target,reconstruction)
    sam_value = sam(target,reconstruction)
    psnr_value = psnr(target,reconstruction)
    ssim_value = ssim(target,reconstruction)
    
    return (mrae_value,rmse_value,sam_value,psnr_value,ssim_value)

# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model: HSIVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.train()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    total_samples = 0

    for hsi in loader:
        hsi = hsi.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(enabled=use_amp):
            reconstruction, mu, logvar, _ = model(
                hsi,
                sample=True,
            )

            (
                loss,
                reconstruction_value,
                kl_value,
            ) = calculate_vae_loss(
                reconstruction=reconstruction,
                target=hsi,
                mu=mu,
                logvar=logvar,
            )
            (mrae_value,rmse_value,sam_value,psnr_value,ssim_value) = calculate_aux_losses(reconstruction = reconstruction,target = hsi,mu = mu,logvar = logvar)
            

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss: {loss.item()}"
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        batch_size = hsi.size(0)

        total_loss += (
            loss.detach().item()
            * batch_size
        )

        total_reconstruction += (
            reconstruction_value.detach().item()
            * batch_size
        )

        total_kl += (
            kl_value.detach().item()
            * batch_size
        )
        total_mrae += (
            mrae_value
            * batch_size
        )
        total_rmse += (
            rmse_value
            * batch_size
        )
        total_sam += (
            sam_value
            * batch_size
        )
        total_psnr += (
            psnr_value
            * batch_size
        )
        total_ssim += (
            ssim_value
            * batch_size
        )

        total_samples += batch_size

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "reconstruction": (
            total_reconstruction
            / total_samples
        ),
        "kl": (
            total_kl
            / total_samples
        ),
        "mrae": (
            total_mrae
            / total_samples
        ),
        "rmse": (
            total_rmse
            / total_samples
        ),
        "sam": (
            total_sam
            / total_samples
        ),
        "psnr": (
            total_psnr
            / total_samples
        ),
        "ssim": (
            total_ssim
            / total_samples
        ),
    }


@torch.no_grad()
def validate(
    model: HSIVAE,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    total_samples = 0

    for hsi in loader:
        hsi = hsi.to(
            device,
            non_blocking=True,
        )

        with autocast(enabled=use_amp):
            # Use the posterior mean during validation.
            reconstruction, mu, logvar, _ = model(
                hsi,
                sample=False,
            )

            (
                loss,
                reconstruction_value,
                kl_value,
            ) = calculate_vae_loss(
                reconstruction=reconstruction,
                target=hsi,
                mu=mu,
                logvar=logvar,
            )
            mrae_value,rmse_value,sam_value,psnr_value,ssim_value = calculate_aux_losses (reconstruction = reconstruction,target = hsi,mu = mu,logvar = logvar)

        batch_size = hsi.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_reconstruction += (
            reconstruction_value.item()
            * batch_size
        )

        total_kl += (
            kl_value.item()
            * batch_size
        )
        total_mrae += (
            mrae_value.detach().item()
            * batch_size
        )
        total_rmse += (
            rmse_value.detach().item()
            * batch_size
        )
        total_sam += (
            sam_value.detach().item()
            * batch_size
        )
        total_psnr += (
            psnr_value.detach().item()
            * batch_size
        )
        total_ssim += (
            ssim_value.detach().item()
            * batch_size
        )

        total_samples += batch_size

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "reconstruction": (
            total_reconstruction
            / total_samples
        ),
        "kl": (
            total_kl
            / total_samples
        ),
        "mrae": (
            total_mrae
            / total_samples
        ),
        "rmse": (
            total_rmse
            / total_samples
        ),
        "sam": (
            total_sam
            / total_samples
        ),
        "psnr": (
            total_psnr
            / total_samples
        ),
        "ssim": (
            total_ssim
            / total_samples
        ),
    }


# ============================================================
# Checkpoint saving
# ============================================================

def save_checkpoint(
    output_path: Path,
    model: HSIVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "validation_loss": validation_loss,
            "model_config": {
                "hsi_channels": HSI_CHANNELS,
                "base_channels": BASE_CHANNELS,
                "latent_channels": LATENT_CHANNELS,
                "num_res_blocks": NUM_RES_BLOCKS,
            },
        },
        output_path,
    )


# ============================================================
# Main
# ============================================================

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

    all_files = find_hsi_files(DATA_DIR)

    (
        training_files,
        validation_files,
    ) = split_files(
        files=all_files,
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )

    print(f"Device: {device}")
    print(f"Mixed precision: {use_amp}")
    print(f"Total images: {len(all_files)}")
    print(
        f"Training images: "
        f"{len(training_files)}"
    )
    print(
        f"Validation images: "
        f"{len(validation_files)}"
    )

    training_dataset = HSIPatchDataset(
        files=training_files,
        hsi_channels=HSI_CHANNELS,
        patch_size=PATCH_SIZE,
        patches_per_image=PATCHES_PER_IMAGE,
        training=True,
        normalization=NORMALIZATION,
        augment=USE_AUGMENTATION,
    )

    validation_dataset = HSIPatchDataset(
        files=validation_files,
        hsi_channels=HSI_CHANNELS,
        patch_size=PATCH_SIZE,
        patches_per_image=1,
        training=False,
        normalization=NORMALIZATION,
        augment=False,
    )

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

    model = HSIVAE(
        hsi_channels=HSI_CHANNELS,
        base_channels=BASE_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        num_res_blocks=NUM_RES_BLOCKS,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
    )

    scaler = GradScaler(
        enabled=use_amp
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_loss = float("inf")

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
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

        scheduler.step(
            validation_metrics["loss"]
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS:03d} | "
            f"LR: {current_learning_rate:.2e} | "
            f"Train total: "
            f"{training_metrics['loss']:.6f} | "
            f"Train reconstruction: "
            f"{training_metrics['reconstruction']:.6f} | "
            f"Train KL: "
            f"{training_metrics['kl']:.6f} | "
            f"Val total: "
            f"{validation_metrics['loss']:.6f} | "
            f"Val reconstruction: "
            f"{validation_metrics['reconstruction']:.6f} | "
            f"Val KL: "
            f"{validation_metrics['kl']:.6f}"
        )

        save_checkpoint(
            output_path=(
                output_dir
                / "last_vae.pth"
            ),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_loss=(
                validation_metrics["loss"]
            ),
        )

        if (
            validation_metrics["loss"]
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_metrics["loss"]
            )

            save_checkpoint(
                output_path=(
                    output_dir
                    / "best_vae.pth"
                ),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=(
                    best_validation_loss
                ),
            )

            print(
                "Saved new best checkpoint: "
                f"{best_validation_loss:.6f}"
            )


if __name__ == "__main__":
    main()
