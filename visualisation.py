import random
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

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

HSI_CHANNELS = 31
PATCH_SIZE = 64

# Number of HSI files to visualize.
NUM_IMAGES = 5

# The variable containing the HSI cube inside the MAT file.
HSI_KEY = "cube"

# Must match the normalization used during VAE training:
# "none", "minmax", or "band_minmax".
NORMALIZATION = "none"

# Approximate red, green and blue bands.
#
# For 31 bands ordered approximately from 400 nm to 700 nm:
# red   ≈ band 25
# green ≈ band 15
# blue  ≈ band 6
#
# These are zero-based indices.
RGB_BANDS = (25, 15, 6)

# Percentiles used for visual contrast stretching.
RGB_LOW_PERCENTILE = 1.0
RGB_HIGH_PERCENTILE = 99.0

# Spectral pixels displayed in the spectral-curve plot.
# Coordinates are fractions of the patch height and width.
SPECTRAL_LOCATIONS = (
    (0.25, 0.25),
    (0.50, 0.50),
    (0.75, 0.75),
)

SEED = 42

# Save every figure to OUTPUT_DIR.
SAVE_FIGURES = True

# Attempt to display figures during execution.
SHOW_FIGURES = True

SUPPORTED_EXTENSIONS = {
    ".mat",
    ".npy",
    ".npz",
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

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# HSI loading
# ============================================================

def find_hsi_files(data_dir: str) -> list[Path]:
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_path}"
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
            f"No HSI files found under {data_path}"
        )

    return files


def extract_array_from_dictionary(
    data: dict,
    file_path: Path,
    preferred_key: Optional[str] = None,
) -> np.ndarray:
    if (
        preferred_key is not None
        and preferred_key in data
    ):
        value = data[preferred_key]

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        value = np.asarray(value)

        if value.ndim == 3:
            return value

    candidates = []

    for key, value in data.items():
        if key.startswith("__"):
            continue

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        if (
            isinstance(value, np.ndarray)
            and value.ndim == 3
            and np.issubdtype(value.dtype, np.number)
        ):
            candidates.append(
                (key, value)
            )

    if not candidates:
        raise ValueError(
            f"No numerical 3D HSI cube found in {file_path}"
        )

    _, array = max(
        candidates,
        key=lambda item: item[1].size,
    )

    return array


def load_mat_v73(
    file_path: Path,
    preferred_key: Optional[str] = None,
) -> np.ndarray:
    candidates = []

    with h5py.File(
        str(file_path),
        "r",
    ) as h5_file:

        if (
            preferred_key is not None
            and preferred_key in h5_file
            and isinstance(
                h5_file[preferred_key],
                h5py.Dataset,
            )
        ):
            cube = np.asarray(
                h5_file[preferred_key]
            )

        else:
            def visitor(name, obj):
                if (
                    isinstance(obj, h5py.Dataset)
                    and obj.ndim == 3
                ):
                    try:
                        array = np.asarray(obj)

                        if np.issubdtype(
                            array.dtype,
                            np.number,
                        ):
                            candidates.append(
                                (name, array)
                            )

                    except Exception:
                        pass

            h5_file.visititems(visitor)

            if not candidates:
                raise ValueError(
                    f"No numerical 3D dataset found in "
                    f"{file_path}"
                )

            _, cube = max(
                candidates,
                key=lambda item: item[1].size,
            )

    # MATLAB v7.3 dimensions are commonly exposed in
    # reverse order through h5py.
    cube = np.transpose(
        cube,
        axes=tuple(
            range(cube.ndim - 1, -1, -1)
        ),
    )

    return cube


def load_hsi_file(file_path: Path) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension == ".mat":
        try:
            loaded = sio.loadmat(file_path)

            cube = extract_array_from_dictionary(
                loaded,
                file_path=file_path,
                preferred_key=HSI_KEY,
            )

        except (
            NotImplementedError,
            ValueError,
            OSError,
        ):
            cube = load_mat_v73(
                file_path=file_path,
                preferred_key=HSI_KEY,
            )

    elif extension == ".npy":
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
                f"No 3D array found in {file_path}"
            )

        cube = max(
            candidates,
            key=lambda array: array.size,
        )

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
                file_path=file_path,
                preferred_key=HSI_KEY,
            )

        else:
            raise TypeError(
                f"Unsupported object stored in "
                f"{file_path}: {type(loaded)}"
            )

    else:
        raise ValueError(
            f"Unsupported file extension: {extension}"
        )

    cube = np.asarray(
        cube,
        dtype=np.float32,
    )

    cube = np.squeeze(cube)

    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D HSI cube in {file_path}, "
            f"but found shape {cube.shape}"
        )

    return cube


def convert_to_chw(
    cube: np.ndarray,
    hsi_channels: int,
    file_path: Path,
) -> np.ndarray:
    """
    Convert an HSI cube to [channels, height, width].
    """

    if cube.shape[0] == hsi_channels:
        return cube

    if cube.shape[-1] == hsi_channels:
        return np.transpose(
            cube,
            (2, 0, 1),
        )

    if cube.shape[1] == hsi_channels:
        return np.transpose(
            cube,
            (1, 0, 2),
        )

    raise ValueError(
        f"Cannot locate the spectral dimension in "
        f"{file_path}. Shape: {cube.shape}; "
        f"expected {hsi_channels} bands."
    )


# ============================================================
# Normalization and cropping
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


def center_crop(
    cube: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """
    Input:
        cube: [C,H,W]

    Output:
        patch: [C,patch_size,patch_size]
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

    if pad_height > 0 or pad_width > 0:
        cube = F.pad(
            cube,
            (
                0,
                pad_width,
                0,
                pad_height,
            ),
            mode="replicate",
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


# ============================================================
# Model loading
# ============================================================

def load_vae(
    checkpoint_path: str,
    device: torch.device,
) -> HSIVAE:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected the checkpoint to be a dictionary."
        )

    model_config = checkpoint.get(
        "model_config",
        {},
    )

    hsi_channels = model_config.get(
        "hsi_channels",
        HSI_CHANNELS,
    )

    base_channels = model_config.get(
        "base_channels",
        64,
    )

    latent_channels = model_config.get(
        "latent_channels",
        16,
    )

    num_res_blocks = model_config.get(
        "num_res_blocks",
        2,
    )

    model = HSIVAE(
        hsi_channels=hsi_channels,
        base_channels=base_channels,
        latent_channels=latent_channels,
        num_res_blocks=num_res_blocks,
    ).to(device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "model" in checkpoint:
        state_dict = checkpoint["model"]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(
        f"Loaded VAE checkpoint: {checkpoint_path}"
    )

    print(
        "Model configuration: "
        f"bands={hsi_channels}, "
        f"base_channels={base_channels}, "
        f"latent_channels={latent_channels}, "
        f"residual_blocks={num_res_blocks}"
    )

    return model


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
) -> dict:
    """
    Input tensors have shape [C,H,W].
    """

    target = target.float()
    reconstruction = reconstruction.float()

    absolute_error = torch.abs(
        reconstruction - target
    )

    mrae_value = torch.mean(
        absolute_error
        / (
            torch.abs(target) + 1e-6
        )
    )

    mse_value = torch.mean(
        (
            reconstruction - target
        ).pow(2)
    )

    rmse_value = torch.sqrt(
        mse_value
    )

    psnr_value = 10.0 * torch.log10(
        1.0 / (
            mse_value + 1e-10
        )
    )

    # Each spatial pixel is treated as one spectral vector.
    target_pixels = target.permute(
        1,
        2,
        0,
    ).reshape(-1, target.shape[0])

    reconstruction_pixels = reconstruction.permute(
        1,
        2,
        0,
    ).reshape(-1, reconstruction.shape[0])

    numerator = torch.sum(
        target_pixels * reconstruction_pixels,
        dim=1,
    )

    denominator = (
        torch.linalg.vector_norm(
            target_pixels,
            dim=1,
        )
        * torch.linalg.vector_norm(
            reconstruction_pixels,
            dim=1,
        )
    )

    cosine = numerator / (
        denominator + 1e-8
    )

    cosine = torch.clamp(
        cosine,
        min=-1.0 + 1e-7,
        max=1.0 - 1e-7,
    )

    sam_radians = torch.mean(
        torch.acos(cosine)
    )

    sam_degrees = (
        sam_radians
        * 180.0
        / torch.pi
    )

    return {
        "mrae": float(mrae_value.item()),
        "rmse": float(rmse_value.item()),
        "psnr": float(psnr_value.item()),
        "sam": float(sam_degrees.item()),
    }


# ============================================================
# Visualization helpers
# ============================================================

def create_pseudo_rgb(
    cube: np.ndarray,
    rgb_bands: tuple[int, int, int],
    low_values: Optional[np.ndarray] = None,
    high_values: Optional[np.ndarray] = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    cube shape: [C,H,W]

    The same contrast-stretch values can be applied to the
    target and reconstruction to make their comparison fair.
    """

    red_index, green_index, blue_index = rgb_bands

    maximum_band = cube.shape[0] - 1

    for band_index in rgb_bands:
        if not 0 <= band_index <= maximum_band:
            raise IndexError(
                f"RGB band index {band_index} is invalid "
                f"for a {cube.shape[0]}-band cube."
            )

    rgb = np.stack(
        [
            cube[red_index],
            cube[green_index],
            cube[blue_index],
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

    rgb = np.clip(
        rgb,
        0.0,
        1.0,
    )

    return rgb, low_values, high_values


def plot_reconstruction(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    file_name: str,
    output_dir: Path,
) -> None:
    """
    target and reconstruction: [C,H,W]
    """

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

    # Clipping is only used for display.
    target_display = np.clip(
        target_np,
        0.0,
        1.0,
    )

    reconstruction_display = np.clip(
        reconstruction_np,
        0.0,
        1.0,
    )

    target_rgb, low_values, high_values = (
        create_pseudo_rgb(
            target_display,
            RGB_BANDS,
        )
    )

    reconstruction_rgb, _, _ = create_pseudo_rgb(
        reconstruction_display,
        RGB_BANDS,
        low_values=low_values,
        high_values=high_values,
    )

    error_map = np.mean(
        np.abs(
            reconstruction_np - target_np
        ),
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

    # --------------------------------------------------------
    # Original pseudo-RGB
    # --------------------------------------------------------

    axes[0, 0].imshow(target_rgb)

    axes[0, 0].set_title(
        "Original HSI pseudo-RGB"
    )

    axes[0, 0].axis("off")

    # --------------------------------------------------------
    # Reconstructed pseudo-RGB
    # --------------------------------------------------------

    axes[0, 1].imshow(
        reconstruction_rgb
    )

    axes[0, 1].set_title(
        "VAE reconstruction pseudo-RGB"
    )

    axes[0, 1].axis("off")

    # --------------------------------------------------------
    # Mean spectral reconstruction error
    # --------------------------------------------------------

    error_image = axes[1, 0].imshow(
        error_map,
        cmap="magma",
    )

    axes[1, 0].set_title(
        "Mean absolute error over all bands"
    )

    axes[1, 0].axis("off")

    figure.colorbar(
        error_image,
        ax=axes[1, 0],
        fraction=0.046,
        pad=0.04,
    )

    # --------------------------------------------------------
    # Spectral curves
    # --------------------------------------------------------

    height = target_np.shape[1]
    width = target_np.shape[2]

    band_axis = np.arange(
        target_np.shape[0]
    )

    for location_index, (
        relative_y,
        relative_x,
    ) in enumerate(
        SPECTRAL_LOCATIONS,
        start=1,
    ):
        y = int(
            relative_y
            * (height - 1)
        )

        x = int(
            relative_x
            * (width - 1)
        )

        target_spectrum = target_np[
            :,
            y,
            x,
        ]

        reconstruction_spectrum = reconstruction_np[
            :,
            y,
            x,
        ]

        line = axes[1, 1].plot(
            band_axis,
            target_spectrum,
            linewidth=2,
            label=(
                f"Original point {location_index} "
                f"({y},{x})"
            ),
        )[0]

        axes[1, 1].plot(
            band_axis,
            reconstruction_spectrum,
            linestyle="--",
            linewidth=2,
            color=line.get_color(),
            label=(
                f"Reconstructed point "
                f"{location_index}"
            ),
        )

    axes[1, 1].set_title(
        "Spectral signatures"
    )

    axes[1, 1].set_xlabel(
        "Spectral band index"
    )

    axes[1, 1].set_ylabel(
        "Normalized intensity"
    )

    axes[1, 1].grid(
        alpha=0.3
    )

    axes[1, 1].legend(
        fontsize=8,
        ncol=2,
    )

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

        print(
            f"Saved visualization: {output_path}"
        )

    if SHOW_FIGURES:
        plt.show()

    plt.close(figure)

    print(
        f"Metrics for {file_name}: "
        f"MRAE={metrics['mrae']:.6f}, "
        f"RMSE={metrics['rmse']:.6f}, "
        f"PSNR={metrics['psnr']:.3f} dB, "
        f"SAM={metrics['sam']:.3f} degrees"
    )


# ============================================================
# Main
# ============================================================

@torch.inference_mode()
def main() -> None:
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = load_vae(
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    all_files = find_hsi_files(
        DATA_DIR
    )

    random_generator = random.Random(
        SEED
    )

    random_generator.shuffle(
        all_files
    )

    number_to_process = min(
        NUM_IMAGES,
        len(all_files),
    )

    output_dir = Path(
        OUTPUT_DIR
    )

    processed = 0

    for file_path in all_files:
        if processed >= number_to_process:
            break

        try:
            cube = load_hsi_file(
                file_path
            )

            cube = convert_to_chw(
                cube=cube,
                hsi_channels=HSI_CHANNELS,
                file_path=file_path,
            )

            if not np.isfinite(cube).all():
                raise ValueError(
                    "Cube contains NaN or infinite values."
                )

            cube = normalize_cube(
                cube,
                NORMALIZATION,
            )

            hsi = torch.from_numpy(
                np.ascontiguousarray(cube)
            ).float()

            hsi = center_crop(
                hsi,
                PATCH_SIZE,
            )

            # Add the batch dimension:
            # [C,H,W] -> [1,C,H,W]
            hsi_batch = hsi.unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            # sample=False means that the posterior mean is
            # used instead of randomly sampling from the VAE.
            reconstruction, mu, logvar, latent = model(
                hsi_batch,
                sample=False,
            )

            # Remove batch dimension:
            # [1,C,H,W] -> [C,H,W]
            reconstruction = reconstruction[0]
            target = hsi_batch[0]

            print(
                f"\nFile: {file_path.name}"
            )

            print(
                f"Input shape: "
                f"{tuple(target.shape)}"
            )

            print(
                f"Latent mean shape: "
                f"{tuple(mu.shape)}"
            )

            print(
                f"Latent log-variance shape: "
                f"{tuple(logvar.shape)}"
            )

            print(
                f"Reconstruction shape: "
                f"{tuple(reconstruction.shape)}"
            )

            plot_reconstruction(
                target=target,
                reconstruction=reconstruction,
                file_name=file_path.stem,
                output_dir=output_dir,
            )

            processed += 1

        except Exception as error:
            print(
                f"\nSkipping file: {file_path}\n"
                f"Reason: {type(error).__name__}: "
                f"{error}"
            )

    if processed == 0:
        raise RuntimeError(
            "No HSI files were successfully visualized."
        )

    print(
        f"\nCompleted visualizations: {processed}"
    )

    print(
        f"Output directory: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
