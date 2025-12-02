"""
Extended Neural Field Embedder with Overlap and Jittering

Key innovations:
1. Extended patch boundaries: Predict beyond [0,1] to [-margin, 1+margin]
   - Allows overlapping patches for seamless blending
   - Reduces boundary artifacts in super-resolution

2. Position jittering during training:
   - Adds small Gaussian noise to coordinates
   - Forces model to learn continuous representations
   - Improves interpolation quality

3. Overlap blending during inference:
   - Adjacent patches overlap by 2*margin
   - Smooth blending in overlap regions
"""
import torch
import torch.nn as nn
from functools import lru_cache
from typing import Optional, Tuple

from src.models.layers.rope import precompute_freqs_cis_ex2d as precompute_freqs_cis_2d


class ExtendedNerfEmbedder(nn.Module):
    """
    Neural Field Embedder with extended boundaries and position jittering.

    Args:
        in_channels: Input channels (e.g., 3 for RGB)
        hidden_size: Output hidden dimension
        max_freqs: Maximum frequency for Fourier features
        margin: How far beyond [0,1] to extend (e.g., 0.25 means [-0.25, 1.25])
        jitter_std: Standard deviation for position jittering during training
        blend_mode: How to blend overlapping regions ("linear", "cosine", "none")
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        max_freqs: int = 8,
        margin: float = 0.25,
        jitter_std: float = 0.01,
        blend_mode: str = "linear",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.max_freqs = max_freqs
        self.margin = margin
        self.jitter_std = jitter_std
        self.blend_mode = blend_mode

        # Position encoding dimension
        pos_dim = max_freqs ** 2

        self.embedder = nn.Sequential(
            nn.Linear(in_channels + pos_dim, hidden_size, bias=True),
        )

    @lru_cache(maxsize=64)
    def _compute_base_positions(
        self,
        height: int,
        width: int,
        device: str,
        include_margin: bool = True,
    ) -> torch.Tensor:
        """
        Compute position grid with optional extended margins.

        Args:
            height, width: Grid dimensions
            device: Device string
            include_margin: If True, extend to [-margin, 1+margin]

        Returns:
            Position encoding tensor
        """
        if include_margin:
            # Extended range: [-margin, 1+margin]
            # This creates positions that go slightly outside the patch
            start = -self.margin
            end = 1.0 + self.margin
            # Scale factor to match the 16-based coordinate system
            scale_factor = 16.0 * (1.0 + 2 * self.margin)
        else:
            start = 0.0
            end = 1.0
            scale_factor = 16.0

        # Create position grid
        y_pos = torch.linspace(start, end, height, device=device)
        x_pos = torch.linspace(start, end, width, device=device)

        # Scale to match the standard [0, 16] range used in precompute_freqs_cis_2d
        # but now spanning [-margin*16, (1+margin)*16]
        y_pos_scaled = y_pos * 16.0
        x_pos_scaled = x_pos * 16.0

        return y_pos_scaled, x_pos_scaled

    def _apply_jitter(
        self,
        pos_y: torch.Tensor,
        pos_x: torch.Tensor,
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply position jittering during training.

        Args:
            pos_y, pos_x: Position tensors
            training: Whether in training mode

        Returns:
            Jittered position tensors
        """
        if training and self.jitter_std > 0:
            # Add Gaussian noise scaled to coordinate system
            noise_scale = self.jitter_std * 16.0  # Scale to [0,16] coordinate system
            pos_y = pos_y + torch.randn_like(pos_y) * noise_scale
            pos_x = pos_x + torch.randn_like(pos_x) * noise_scale
        return pos_y, pos_x

    def fetch_pos(
        self,
        patch_size_h: int,
        patch_size_w: int,
        device: torch.device,
        dtype: torch.dtype,
        training: bool = False,
        include_margin: bool = True,
    ) -> torch.Tensor:
        """
        Fetch position encodings with extended boundaries and optional jittering.

        Args:
            patch_size_h, patch_size_w: Patch dimensions
            device: Target device
            dtype: Target dtype
            training: Whether in training mode (enables jittering)
            include_margin: Whether to include extended margins

        Returns:
            Fourier position encoding tensor [N, pos_dim]
        """
        # Get base positions
        y_pos_scaled, x_pos_scaled = self._compute_base_positions(
            patch_size_h, patch_size_w,
            str(device), include_margin
        )

        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_pos_scaled, x_pos_scaled, indexing="ij")
        y_pos = y_grid.reshape(-1).to(device=device, dtype=dtype)
        x_pos = x_grid.reshape(-1).to(device=device, dtype=dtype)

        # Apply jittering during training
        if training and self.jitter_std > 0:
            noise_scale = self.jitter_std * 16.0
            y_pos = y_pos + torch.randn_like(y_pos) * noise_scale
            x_pos = x_pos + torch.randn_like(x_pos) * noise_scale

        # Compute Fourier features using sin/cos directly (avoids complex dtype issues with bfloat16)
        dim = self.max_freqs ** 2 * 2
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 4, device=device, dtype=dtype)[: (dim // 4)] / dim))

        x_freqs = torch.outer(x_pos, freqs)  # [N, dim/4]
        y_freqs = torch.outer(y_pos, freqs)  # [N, dim/4]

        # Use sin/cos directly instead of torch.polar (which requires complex dtype)
        x_cos = torch.cos(x_freqs)
        x_sin = torch.sin(x_freqs)
        y_cos = torch.cos(y_freqs)
        y_sin = torch.sin(y_freqs)

        # Interleave x and y components: [x_cos, y_cos, x_sin, y_sin]
        # This maintains similar structure to the complex representation
        pos_encoding = torch.cat([x_cos, y_cos, x_sin, y_sin], dim=-1)

        # Truncate to expected size (max_freqs ** 2)
        pos_encoding = pos_encoding[:, :self.max_freqs ** 2]

        return pos_encoding

    def forward(
        self,
        inputs: torch.Tensor,
        patch_size_h: int,
        patch_size_w: int,
        include_margin: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with extended boundaries.

        Args:
            inputs: [B, N, C] pixel values
            patch_size_h, patch_size_w: Patch dimensions
            include_margin: Whether to use extended boundaries

        Returns:
            [B, N, hidden_size] encoded features
        """
        B, N, C = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        # Get position encoding (with jittering during training)
        pos = self.fetch_pos(
            patch_size_h, patch_size_w,
            device, dtype,
            training=self.training,
            include_margin=include_margin,
        )

        # Broadcast to batch
        pos = pos.unsqueeze(0).repeat(B, 1, 1)

        # Concatenate inputs with position encoding
        combined = torch.cat([inputs, pos], dim=-1)

        return self.embedder(combined)


def compute_blend_weights(
    height: int,
    width: int,
    margin: float,
    blend_mode: str = "linear",
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute blending weights for overlapping regions.

    Args:
        height, width: Output dimensions
        margin: Overlap margin (fraction of patch size)
        blend_mode: "linear", "cosine", or "none"
        device: Target device

    Returns:
        Weight tensor [height, width] with values in [0, 1]
    """
    if blend_mode == "none":
        return torch.ones(height, width, device=device)

    # Compute margin in pixels
    margin_h = int(height * margin / (1 + 2 * margin))
    margin_w = int(width * margin / (1 + 2 * margin))

    weights = torch.ones(height, width, device=device)

    if blend_mode == "linear":
        # Linear ramp in margin regions
        for i in range(margin_h):
            alpha = (i + 1) / (margin_h + 1)
            weights[i, :] *= alpha
            weights[-(i + 1), :] *= alpha

        for j in range(margin_w):
            alpha = (j + 1) / (margin_w + 1)
            weights[:, j] *= alpha
            weights[:, -(j + 1)] *= alpha

    elif blend_mode == "cosine":
        # Cosine ramp for smoother blending
        import math
        for i in range(margin_h):
            alpha = 0.5 * (1 - math.cos(math.pi * (i + 1) / (margin_h + 1)))
            weights[i, :] *= alpha
            weights[-(i + 1), :] *= alpha

        for j in range(margin_w):
            alpha = 0.5 * (1 - math.cos(math.pi * (j + 1) / (margin_w + 1)))
            weights[:, j] *= alpha
            weights[:, -(j + 1)] *= alpha

    return weights


class OverlapPatchProcessor:
    """
    Utility class for processing patches with overlap.

    Handles:
    - Extracting overlapping patches from images
    - Blending overlapping regions when reconstructing
    """

    def __init__(
        self,
        patch_size: int,
        margin: float = 0.25,
        blend_mode: str = "linear",
    ):
        self.patch_size = patch_size
        self.margin = margin
        self.blend_mode = blend_mode

        # Extended patch size (with margins)
        self.margin_pixels = int(patch_size * margin)
        self.extended_size = patch_size + 2 * self.margin_pixels

    def extract_overlapping_patches(
        self,
        x: torch.Tensor,
        stride: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract overlapping patches from image.

        Args:
            x: [B, C, H, W] input image
            stride: Patch stride (default: patch_size for non-overlapping centers)

        Returns:
            patches: [B, num_patches, C, extended_size, extended_size]
            grid_shape: (num_patches_h, num_patches_w)
        """
        B, C, H, W = x.shape
        if stride is None:
            stride = self.patch_size

        # Pad image to handle boundary patches
        pad = self.margin_pixels
        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='reflect')

        # Extract patches with extended size
        patches = x_padded.unfold(2, self.extended_size, stride)
        patches = patches.unfold(3, self.extended_size, stride)

        # patches: [B, C, num_h, num_w, ext_h, ext_w]
        num_h, num_w = patches.shape[2], patches.shape[3]
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B, num_h, num_w, C, ext_h, ext_w]
        patches = patches.reshape(B, num_h * num_w, C, self.extended_size, self.extended_size)

        return patches, (num_h, num_w)

    def blend_patches(
        self,
        patches: torch.Tensor,
        grid_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Blend overlapping patches back into full image.

        Args:
            patches: [B, num_patches, C, extended_size, extended_size]
            grid_shape: (num_patches_h, num_patches_w)
            output_shape: (H, W) target output size
            device: Target device

        Returns:
            Blended image [B, C, H, W]
        """
        B, num_patches, C, ext_h, ext_w = patches.shape
        num_h, num_w = grid_shape
        H, W = output_shape

        if device is None:
            device = patches.device

        # Compute blend weights
        weights = compute_blend_weights(
            ext_h, ext_w, self.margin, self.blend_mode, device
        )

        # Initialize output and weight accumulator
        output = torch.zeros(B, C, H + 2 * self.margin_pixels, W + 2 * self.margin_pixels, device=device)
        weight_sum = torch.zeros(1, 1, H + 2 * self.margin_pixels, W + 2 * self.margin_pixels, device=device)

        # Place each patch with blending
        patches = patches.reshape(B, num_h, num_w, C, ext_h, ext_w)

        for i in range(num_h):
            for j in range(num_w):
                y_start = i * self.patch_size
                x_start = j * self.patch_size

                patch = patches[:, i, j]  # [B, C, ext_h, ext_w]
                output[:, :, y_start:y_start + ext_h, x_start:x_start + ext_w] += patch * weights
                weight_sum[:, :, y_start:y_start + ext_h, x_start:x_start + ext_w] += weights

        # Normalize by weight sum
        output = output / (weight_sum + 1e-8)

        # Crop to original size (remove padding)
        output = output[:, :, self.margin_pixels:self.margin_pixels + H,
                       self.margin_pixels:self.margin_pixels + W]

        return output
