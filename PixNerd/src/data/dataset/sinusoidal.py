"""
Sinusoidal Function Dataset for Neural Field Interpolation Testing

This dataset generates 2D sinusoidal patterns for testing PixNerd's
neural field interpolation capabilities.

Key concept:
- Training: Model sees regularly sampled pixels (simulating low-resolution input)
- Testing: Model must predict all pixels (super-resolution)
- This directly tests the NerfEmbedder's interpolation quality

Super-resolution simulation:
- For 5x super-resolution: see every 5th pixel (20% visible)
- For 4x super-resolution: see every 4th pixel (25% visible)
- Regular grid sampling simulates downsampled low-res input
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
import math


def generate_sinusoidal_image(
    resolution: int,
    num_components: int = 5,
    freq_range: Tuple[float, float] = (1.0, 8.0),
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a 2D sinusoidal pattern.

    f(x, y) = sum_i A_i * sin(2π(fx_i * x + fy_i * y) + φ_i)

    Args:
        resolution: Image size (resolution x resolution)
        num_components: Number of sinusoidal components to sum
        freq_range: Range of frequencies (cycles per image)
        seed: Random seed for reproducibility

    Returns:
        Image array [H, W] with values in [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)

    # Create coordinate grid [0, 1]
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)

    # Generate random components
    image = np.zeros((resolution, resolution), dtype=np.float32)

    for _ in range(num_components):
        # Random amplitude (normalized later)
        amp = np.random.uniform(0.5, 1.5)

        # Random frequencies
        fx = np.random.uniform(freq_range[0], freq_range[1])
        fy = np.random.uniform(freq_range[0], freq_range[1])

        # Random phase
        phase = np.random.uniform(0, 2 * np.pi)

        # Random direction (makes patterns more varied)
        angle = np.random.uniform(0, 2 * np.pi)
        fx_rot = fx * np.cos(angle) - fy * np.sin(angle)
        fy_rot = fx * np.sin(angle) + fy * np.cos(angle)

        # Add component
        image += amp * np.sin(2 * np.pi * (fx_rot * X + fy_rot * Y) + phase)

    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    return image


def create_visibility_mask(
    resolution: int,
    downsample_factor: int = 5,
    mode: str = "grid",
) -> np.ndarray:
    """
    Create visibility mask for super-resolution simulation.

    Uses regular grid sampling to simulate a downsampled low-resolution input.
    This is the standard approach for super-resolution benchmarks.

    Args:
        resolution: Full image size (high-res)
        downsample_factor: Sample every Nth pixel (e.g., 5 = 5x super-res = 20% visible)
        mode: "grid" (both x and y), "columns" (x only), "rows" (y only)

    Returns:
        Boolean mask [H, W] where True = visible (low-res sample positions)

    Examples:
        downsample_factor=5: Every 5th pixel visible → 20% visible (5x super-res)
        downsample_factor=4: Every 4th pixel visible → 25% visible (4x super-res)
        downsample_factor=2: Every 2nd pixel visible → 50% visible (2x super-res)
    """
    mask = np.zeros((resolution, resolution), dtype=bool)

    if mode == "grid":
        # Regular 2D grid sampling (standard for super-resolution)
        # Visible positions: (0, 0), (0, N), (0, 2N), ..., (N, 0), (N, N), ...
        mask[::downsample_factor, ::downsample_factor] = True
    elif mode == "columns":
        # Only sample along x-axis (vertical stripes)
        mask[:, ::downsample_factor] = True
    elif mode == "rows":
        # Only sample along y-axis (horizontal stripes)
        mask[::downsample_factor, :] = True
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'grid', 'columns', or 'rows'")

    return mask


def create_visibility_mask_from_intervals(
    resolution: int,
    visible_intervals: List[Tuple[float, float]],
    mode: str = "columns",
) -> np.ndarray:
    """
    Create visibility mask from explicit intervals (legacy support).

    Args:
        resolution: Image size
        visible_intervals: List of (start, end) intervals in [0, 1]
        mode: "columns", "rows", or "grid" (both)

    Returns:
        Boolean mask [H, W] where True = visible (training region)
    """
    mask = np.zeros((resolution, resolution), dtype=bool)

    for start, end in visible_intervals:
        start_idx = int(start * resolution)
        end_idx = int(end * resolution)

        if mode == "columns" or mode == "grid":
            mask[:, start_idx:end_idx] = True
        if mode == "rows" or mode == "grid":
            mask[start_idx:end_idx, :] = True

    return mask


class SinusoidalDataset(Dataset):
    """
    Dataset of 2D sinusoidal patterns for neural field testing.

    Each sample contains:
    - image: Full sinusoidal pattern [C, H, W]
    - mask: Visibility mask (True = visible during training)
    - metadata: Dict with save function and parameters

    Super-resolution simulation:
    - Training sees regularly sampled pixels (like low-res input)
    - Model must interpolate to full resolution
    """

    def __init__(
        self,
        num_samples: int = 1000,
        resolution: int = 64,
        num_components: int = 5,
        freq_range: Tuple[float, float] = (1.0, 8.0),
        channels: int = 1,  # 1 for grayscale, 3 for RGB (same pattern per channel)
        downsample_factor: int = 5,  # 5 = 5x super-res = every 5th pixel visible
        mask_mode: str = "grid",
        seed: int = 42,
    ):
        """
        Args:
            num_samples: Number of unique sinusoidal patterns
            resolution: Image resolution (square) - the HIGH-RES target
            num_components: Number of sinusoidal components per image
            freq_range: Frequency range (cycles per image)
            channels: Number of channels (1 or 3)
            downsample_factor: Sample every Nth pixel (e.g., 5 = 20% visible, 4 = 25%)
            mask_mode: "grid" (both x,y), "columns" (x only), "rows" (y only)
            seed: Base random seed
        """
        self.num_samples = num_samples
        self.resolution = resolution
        self.num_components = num_components
        self.freq_range = freq_range
        self.channels = channels
        self.seed = seed
        self.downsample_factor = downsample_factor

        # Create visibility mask (same for all samples)
        # Regular grid sampling simulates low-resolution input
        self.visibility_mask = create_visibility_mask(
            resolution, downsample_factor, mask_mode
        )
        self.visible_ratio = self.visibility_mask.sum() / self.visibility_mask.size

        # Effective low-res size
        low_res = resolution // downsample_factor

        print(f"SinusoidalDataset: {num_samples} samples")
        print(f"  High-res: {resolution}x{resolution}")
        print(f"  Low-res equivalent: {low_res}x{low_res} (downsample_factor={downsample_factor})")
        print(f"  Visible ratio: {self.visible_ratio:.1%}")
        print(f"  Super-resolution task: {downsample_factor}x upscaling")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # Generate deterministic pattern based on index
        image = generate_sinusoidal_image(
            self.resolution,
            self.num_components,
            self.freq_range,
            seed=self.seed + idx,
        )

        # Expand to desired channels
        if self.channels == 3:
            image = np.stack([image, image, image], axis=0)
        else:
            image = image[np.newaxis, ...]  # [1, H, W]

        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(self.visibility_mask).bool()

        # Metadata for saving/visualization
        metadata = {
            "idx": idx,
            "save_fn": self._save_fn,
        }

        return image_tensor, mask_tensor, metadata

    @staticmethod
    def _save_fn(sample: np.ndarray, metadata: dict, target_dir: str):
        """Save function for visualization callback."""
        import os
        from PIL import Image

        idx = metadata.get("idx", 0)

        # Handle grayscale or RGB
        if sample.shape[-1] == 1:
            sample = sample.squeeze(-1)
            mode = "L"
        else:
            mode = "RGB"

        # Scale to 0-255
        sample = (sample * 255).clip(0, 255).astype(np.uint8)

        img = Image.fromarray(sample, mode=mode)
        img.save(os.path.join(target_dir, f"sample_{idx:04d}.png"))


class SinusoidalRandomNDataset(Dataset):
    """
    Random noise dataset for evaluation/generation (like CIFAR10RandomNDataset).

    Provides random noise tensors for the diffusion sampler.
    """

    def __init__(
        self,
        latent_shape: Tuple[int, ...] = (1, 64, 64),
        max_num_instances: int = 100,
        downsample_factor: int = 5,
        mask_mode: str = "grid",
    ):
        self.latent_shape = latent_shape
        self.max_num_instances = max_num_instances

        # Create visibility mask (regular grid sampling)
        resolution = latent_shape[-1]
        self.visibility_mask = create_visibility_mask(
            resolution, downsample_factor, mask_mode
        )

    def __len__(self) -> int:
        return self.max_num_instances

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # Random noise
        noise = torch.randn(self.latent_shape)

        # Visibility mask
        mask = torch.from_numpy(self.visibility_mask).bool()

        metadata = {
            "idx": idx,
            "save_fn": SinusoidalDataset._save_fn,
        }

        return noise, mask, metadata


# Utility functions for evaluation

def compute_interpolation_metrics(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """
    Compute metrics on visible (training) and invisible (test) regions.

    Args:
        predicted: Predicted image [B, C, H, W]
        ground_truth: Ground truth image [B, C, H, W]
        mask: Visibility mask [H, W] (True = visible/training region)

    Returns:
        Dict with MSE on visible, invisible, and full image
    """
    # Expand mask to match image shape
    mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand_as(predicted)

    # MSE on visible regions (what model trained on)
    visible_mse = ((predicted - ground_truth) ** 2)[mask_expanded].mean().item()

    # MSE on invisible regions (interpolation quality)
    invisible_mse = ((predicted - ground_truth) ** 2)[~mask_expanded].mean().item()

    # Full image MSE
    full_mse = ((predicted - ground_truth) ** 2).mean().item()

    return {
        "visible_mse": visible_mse,
        "invisible_mse": invisible_mse,
        "full_mse": full_mse,
        "interpolation_ratio": invisible_mse / (visible_mse + 1e-8),
    }
