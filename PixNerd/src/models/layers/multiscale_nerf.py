"""
Multi-Scale Neural Field Embedder

Decouples position encoding quality from encoder patch size by:
1. Using ABSOLUTE normalized coordinates [0,1] for the full image
2. Encoding at multiple frequency scales (octaves)
3. Dense position sampling regardless of patch size
4. Optional cross-scale attention for scale communication

This allows small patch sizes (good for encoder) while maintaining
dense position coverage (good for super-resolution).
"""
import torch
import torch.nn as nn
from functools import lru_cache
from typing import Optional, Tuple


class MultiScalePositionEncoder(nn.Module):
    """
    Fourier feature encoding at multiple frequency octaves.

    Unlike standard NeRF encoding tied to patch size, this uses:
    - Fixed dense sampling (e.g., 16x16 = 256 positions per patch)
    - Multiple frequency octaves for multi-scale representation
    - Absolute normalized coordinates
    """
    def __init__(
        self,
        num_octaves: int = 4,
        freqs_per_octave: int = 8,
        base_freq: float = 1.0,
    ):
        super().__init__()
        self.num_octaves = num_octaves
        self.freqs_per_octave = freqs_per_octave
        self.base_freq = base_freq

        # Output dimension: 2 (x,y) × 2 (sin,cos) × freqs × octaves
        self.out_dim = 2 * 2 * freqs_per_octave * num_octaves

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode 2D coordinates with multi-scale Fourier features.

        Args:
            coords: [N, 2] normalized coordinates in [0, 1]

        Returns:
            [N, out_dim] multi-scale Fourier features
        """
        device = coords.device
        dtype = coords.dtype

        all_features = []
        for octave in range(self.num_octaves):
            # Each octave doubles the base frequency
            octave_scale = 2 ** octave
            freqs = self.base_freq * octave_scale * (2 ** torch.linspace(
                0, self.freqs_per_octave - 1,
                self.freqs_per_octave,
                device=device, dtype=dtype
            ))

            # [N, 2, 1] * [F] -> [N, 2, F]
            scaled_coords = coords.unsqueeze(-1) * freqs * torch.pi

            # Sin and cos features
            sin_feat = torch.sin(scaled_coords)  # [N, 2, F]
            cos_feat = torch.cos(scaled_coords)  # [N, 2, F]

            # Concatenate and flatten
            octave_feat = torch.cat([sin_feat, cos_feat], dim=-1)  # [N, 2, 2F]
            all_features.append(octave_feat.reshape(coords.shape[0], -1))  # [N, 4F]

        return torch.cat(all_features, dim=-1)  # [N, 4F * num_octaves]


class CrossScaleAttention(nn.Module):
    """
    Attention mechanism for cross-scale feature communication.

    Allows global, region, and local scale features to interact
    and share information.
    """
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Project each scale to Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        global_feat: torch.Tensor,
        region_feat: torch.Tensor,
        local_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-scale attention fusion.

        Args:
            global_feat: [B, N, H] global scale features
            region_feat: [B, N, H] region scale features
            local_feat: [B, N, H] local scale features

        Returns:
            [B, N, H] fused features
        """
        B, N, H = local_feat.shape

        # Stack scales: [B, N, 3, H]
        stacked = torch.stack([global_feat, region_feat, local_feat], dim=2)

        # Self-attention across scales for each position
        # Reshape for attention: [B*N, 3, H]
        stacked = stacked.reshape(B * N, 3, H)

        q = self.q_proj(stacked).reshape(B * N, 3, self.num_heads, self.head_dim)
        k = self.k_proj(stacked).reshape(B * N, 3, self.num_heads, self.head_dim)
        v = self.v_proj(stacked).reshape(B * N, 3, self.num_heads, self.head_dim)

        # [B*N, num_heads, 3, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [B*N, num_heads, 3, head_dim]

        # Take the local scale output (or mean across scales)
        out = out.permute(0, 2, 1, 3).reshape(B * N, 3, H)
        out = self.out_proj(out)

        # Return weighted combination (emphasize local for detail)
        weights = torch.tensor([0.2, 0.3, 0.5], device=out.device, dtype=out.dtype)
        out = (out * weights.view(1, 3, 1)).sum(dim=1)  # [B*N, H]

        return out.reshape(B, N, H)


class MultiScaleNerfEmbedder(nn.Module):
    """
    Multi-Scale Neural Field Embedder.

    Key innovation: Position sampling is INDEPENDENT of patch size.

    Args:
        in_channels: Input channels (e.g., 3 for RGB)
        hidden_size: Output hidden dimension
        dense_samples: Fixed number of position samples per axis (default 16)
                      This is independent of encoder patch size!
        num_octaves: Number of frequency octaves for multi-scale encoding
        freqs_per_octave: Frequencies per octave
        fusion_type: How to combine scales ("concat", "add", "attention")
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        dense_samples: int = 16,  # Always sample 16x16=256 positions!
        num_octaves: int = 4,
        freqs_per_octave: int = 8,
        fusion_type: str = "concat",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.dense_samples = dense_samples
        self.fusion_type = fusion_type

        # Position encoder
        self.pos_encoder = MultiScalePositionEncoder(
            num_octaves=num_octaves,
            freqs_per_octave=freqs_per_octave,
        )
        pos_dim = self.pos_encoder.out_dim

        # Scale-specific embedders (different frequency ranges emphasized)
        # Global: low frequencies (coarse structure)
        self.global_embedder = nn.Sequential(
            nn.Linear(pos_dim // 4, hidden_size),  # First octave only
            nn.SiLU(),
        )

        # Region: mid frequencies
        self.region_embedder = nn.Sequential(
            nn.Linear(pos_dim // 2, hidden_size),  # First two octaves
            nn.SiLU(),
        )

        # Local: all frequencies (fine detail)
        self.local_embedder = nn.Sequential(
            nn.Linear(pos_dim, hidden_size),  # All octaves
            nn.SiLU(),
        )

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_size)

        # Scale fusion
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size * 2),
                nn.SiLU(),
                nn.Linear(hidden_size * 2, hidden_size),
            )
        elif fusion_type == "attention":
            self.cross_scale_attn = CrossScaleAttention(hidden_size, num_heads=4)
            self.fusion = nn.Linear(hidden_size * 2, hidden_size)  # input + attended
        else:  # add
            self.fusion = nn.Identity()

        # Cache for position encodings
        self._pos_cache = {}

    @lru_cache(maxsize=32)
    def _compute_dense_positions(
        self,
        height: int,
        width: int,
        device: str,
    ) -> torch.Tensor:
        """
        Compute dense normalized positions.

        Always uses self.dense_samples regardless of actual patch size!
        """
        # Dense grid in [0, 1] normalized coordinates
        y = torch.linspace(0, 1, self.dense_samples, device=device)
        x = torch.linspace(0, 1, self.dense_samples, device=device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

        # Flatten to [N, 2] where N = dense_samples^2
        coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)
        return coords

    def _interpolate_to_patch_size(
        self,
        features: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """
        Interpolate dense features to actual patch size if needed.

        Args:
            features: [B, dense_samples^2, H]
            target_h, target_w: Actual patch dimensions
        """
        if target_h == self.dense_samples and target_w == self.dense_samples:
            return features

        B, N, H = features.shape
        # Reshape to spatial: [B, H, dense, dense]
        features = features.transpose(1, 2).reshape(B, H, self.dense_samples, self.dense_samples)

        # Interpolate to target size
        features = torch.nn.functional.interpolate(
            features,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=True,
        )

        # Reshape back: [B, target_h*target_w, H]
        return features.reshape(B, H, -1).transpose(1, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        patch_size_h: int,
        patch_size_w: int,
        absolute_offset: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        """
        Multi-scale position encoding.

        Args:
            inputs: [B, patch_h * patch_w, C] pixel values
            patch_size_h, patch_size_w: Actual patch dimensions
            absolute_offset: Optional (y, x) offset for absolute positioning

        Returns:
            [B, patch_h * patch_w, hidden_size] encoded features
        """
        B, N, C = inputs.shape
        device = inputs.device

        # Get dense position grid (always dense_samples x dense_samples)
        coords = self._compute_dense_positions(
            self.dense_samples, self.dense_samples,
            str(device)
        ).to(inputs.dtype)

        # Apply absolute offset if provided (for global positioning)
        if absolute_offset is not None:
            offset = torch.tensor(absolute_offset, device=device, dtype=inputs.dtype)
            coords = coords + offset

        # Multi-scale Fourier encoding
        pos_features = self.pos_encoder(coords)  # [dense^2, pos_dim]

        # Split by octave for different scales
        pos_dim = pos_features.shape[-1]
        octave_dim = pos_dim // self.pos_encoder.num_octaves

        # Global: first octave (low freq)
        global_pos = pos_features[:, :octave_dim]
        global_feat = self.global_embedder(global_pos)  # [dense^2, H]

        # Region: first two octaves (low + mid freq)
        region_pos = pos_features[:, :octave_dim * 2]
        region_feat = self.region_embedder(region_pos)  # [dense^2, H]

        # Local: all octaves (full spectrum)
        local_feat = self.local_embedder(pos_features)  # [dense^2, H]

        # Broadcast to batch
        global_feat = global_feat.unsqueeze(0).expand(B, -1, -1)
        region_feat = region_feat.unsqueeze(0).expand(B, -1, -1)
        local_feat = local_feat.unsqueeze(0).expand(B, -1, -1)

        # Interpolate features to actual patch size
        global_feat = self._interpolate_to_patch_size(global_feat, patch_size_h, patch_size_w)
        region_feat = self._interpolate_to_patch_size(region_feat, patch_size_h, patch_size_w)
        local_feat = self._interpolate_to_patch_size(local_feat, patch_size_h, patch_size_w)

        # Project input pixels
        input_feat = self.input_proj(inputs)  # [B, N, H]

        # Fuse scales
        if self.fusion_type == "concat":
            combined = torch.cat([input_feat, global_feat, region_feat, local_feat], dim=-1)
            output = self.fusion(combined)

        elif self.fusion_type == "attention":
            # Cross-scale attention
            scale_feat = self.cross_scale_attn(global_feat, region_feat, local_feat)
            combined = torch.cat([input_feat, scale_feat], dim=-1)
            output = self.fusion(combined)

        else:  # add
            output = input_feat + global_feat + region_feat + local_feat

        return output


class HierarchicalNerfEmbedder(nn.Module):
    """
    Hierarchical Neural Field Embedder with explicit scale hierarchy.

    Uses three separate NerfBlocks at different scales with
    cross-scale communication via attention.

    This is a more structured approach than MultiScaleNerfEmbedder,
    with explicit hierarchical processing.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        dense_samples: int = 16,
        max_freqs: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.dense_samples = dense_samples
        self.max_freqs = max_freqs

        # Per-scale position encoders with different frequency ranges
        self.global_nerf = SingleScaleNerf(in_channels, hidden_size, max_freqs, scale_range=(0, 2))
        self.region_nerf = SingleScaleNerf(in_channels, hidden_size, max_freqs, scale_range=(2, 5))
        self.local_nerf = SingleScaleNerf(in_channels, hidden_size, max_freqs, scale_range=(4, 8))

        # Cross-scale communication
        self.global_to_region = nn.Linear(hidden_size, hidden_size)
        self.region_to_local = nn.Linear(hidden_size, hidden_size)

        # Final combination
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        patch_size_h: int,
        patch_size_w: int,
    ) -> torch.Tensor:
        """
        Hierarchical processing with cross-scale communication.
        """
        # Process at each scale
        global_feat = self.global_nerf(inputs, patch_size_h, patch_size_w)

        # Region gets global context
        region_input = inputs  # Could add global_feat here
        region_feat = self.region_nerf(region_input, patch_size_h, patch_size_w)
        region_feat = region_feat + self.global_to_region(global_feat)

        # Local gets region context
        local_input = inputs
        local_feat = self.local_nerf(local_input, patch_size_h, patch_size_w)
        local_feat = local_feat + self.region_to_local(region_feat)

        # Combine all scales
        combined = torch.cat([global_feat, region_feat, local_feat], dim=-1)
        return self.output_proj(combined)


class SingleScaleNerf(nn.Module):
    """Single-scale NeRF encoder with configurable frequency range."""
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        max_freqs: int = 8,
        scale_range: Tuple[int, int] = (0, 8),
    ):
        super().__init__()
        self.scale_range = scale_range
        num_freqs = scale_range[1] - scale_range[0]
        pos_dim = 4 * num_freqs  # x,y × sin,cos × freqs

        self.embedder = nn.Sequential(
            nn.Linear(in_channels + pos_dim, hidden_size),
            nn.SiLU(),
        )
        self.scale_range = scale_range

    def forward(self, inputs, patch_size_h, patch_size_w):
        B, N, C = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        # Create position grid
        y = torch.linspace(0, 16, patch_size_h, device=device, dtype=dtype)
        x = torch.linspace(0, 16, patch_size_w, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)

        # Fourier encoding in scale range
        freqs = 2 ** torch.linspace(
            self.scale_range[0],
            self.scale_range[1] - 1,
            self.scale_range[1] - self.scale_range[0],
            device=device, dtype=dtype
        )

        scaled = coords.unsqueeze(-1) * freqs * torch.pi / 16
        pos_feat = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        pos_feat = pos_feat.reshape(N, -1)

        # Broadcast and combine
        pos_feat = pos_feat.unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([inputs, pos_feat], dim=-1)

        return self.embedder(combined)
