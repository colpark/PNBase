"""
PixNerDiT with Extended Boundaries - Proper Overlap Supervision

This variant improves super-resolution quality through:
1. Extended patch boundaries: Predict actual neighbor pixels, not just rescaled coordinates
2. Overlapping region supervision: Adjacent patches predict same pixels → consistency
3. Ground truth supervision: Extended predictions compared against real pixel values

Key insight: Each patch predicts beyond its boundary into neighbor territory.
Overlapping regions are supervised multiple times, enforcing consistency.

During inference, overlapping predictions are blended for smooth output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import lru_cache
from src.models.layers.attention_op import attention
from src.models.layers.rope import apply_rotary_emb, precompute_freqs_cis_ex2d as precompute_freqs_cis_2d
from src.models.layers.time_embed import TimestepEmbedder
from src.models.layers.patch_embed import Embed
from src.models.layers.swiglu import SwiGLU as FeedForward
from src.models.layers.rmsnorm import RMSNorm as Norm


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.contiguous())
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)

        q = q.view(B, self.num_heads, -1, C // self.num_heads)
        k = k.view(B, self.num_heads, -1, C // self.num_heads).contiguous()
        v = v.view(B, self.num_heads, -1, C // self.num_heads).contiguous()

        x = attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlattenDiTBlock(nn.Module):
    """Encoder block with self-attention and adaptive layer norm."""
    def __init__(self, hidden_size, groups, mlp_ratio=4):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, pos):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ExtendedNerfEmbedder(nn.Module):
    """
    NerfEmbedder with extended boundaries for proper overlap supervision.

    Position encoding spans [-margin, 1+margin] to match extended patch pixels.
    No jittering by default - overlapping regions must be consistent.

    Args:
        in_channels: Input channels
        hidden_size: Output dimension
        max_freqs: Maximum frequency for position encoding
        margin: Extension beyond [0,1] (e.g., 0.25 means [-0.25, 1.25])
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        max_freqs: int = 8,
        margin: float = 0.25,
    ):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size = hidden_size
        self.margin = margin

        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs ** 2, hidden_size, bias=True),
        )

    @lru_cache(maxsize=64)
    def _get_pos_encoding(self, height, width, device_str, dtype_str):
        """
        Compute position encoding for extended grid (cached).

        Positions span [-margin, 1+margin] in normalized coordinates,
        mapped to [-margin*16, (1+margin)*16] in the position encoding space.
        """
        device = torch.device(device_str)
        dtype = getattr(torch, dtype_str)

        # Extended range in position encoding space
        start = -self.margin * 16
        end = (1 + self.margin) * 16

        y_pos = torch.linspace(start, end, height, device=device, dtype=dtype)
        x_pos = torch.linspace(start, end, width, device=device, dtype=dtype)

        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing="ij")
        y_flat = y_grid.reshape(-1)
        x_flat = x_grid.reshape(-1)

        # Compute Fourier features using sin/cos (bfloat16 compatible)
        dim = self.max_freqs ** 2 * 2
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 4, device=device, dtype=dtype)[: (dim // 4)] / dim))

        x_freqs = torch.outer(x_flat, freqs)
        y_freqs = torch.outer(y_flat, freqs)

        x_cos = torch.cos(x_freqs)
        x_sin = torch.sin(x_freqs)
        y_cos = torch.cos(y_freqs)
        y_sin = torch.sin(y_freqs)

        pos_encoding = torch.cat([x_cos, y_cos, x_sin, y_sin], dim=-1)
        pos_encoding = pos_encoding[:, :self.max_freqs ** 2]

        return pos_encoding

    def forward(self, inputs, patch_size_h, patch_size_w):
        """
        Forward pass with extended position encoding.

        Args:
            inputs: [B, N, C] pixel values from extended patch
            patch_size_h, patch_size_w: Extended patch dimensions

        Returns:
            [B, N, hidden_size] encoded features
        """
        B, N, C = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        # Get cached position encoding
        dtype_str = str(dtype).split('.')[-1]  # e.g., 'bfloat16'
        pos = self._get_pos_encoding(patch_size_h, patch_size_w, str(device), dtype_str)
        pos = pos.unsqueeze(0).expand(B, -1, -1)

        inputs = torch.cat([inputs, pos], dim=-1)
        return self.embedder(inputs)


class NerfBlock(nn.Module):
    """Decoder block with hypernetwork-generated MLP weights."""
    def __init__(self, hidden_size_s, hidden_size_x, mlp_ratio=4):
        super().__init__()
        self.param_generator1 = nn.Sequential(
            nn.Linear(hidden_size_s, 2 * hidden_size_x ** 2 * mlp_ratio, bias=True),
        )
        self.norm = Norm(hidden_size_x, eps=1e-6)
        self.mlp_ratio = mlp_ratio

    def forward(self, x, s):
        batch_size, num_x, hidden_size_x = x.shape
        mlp_params1 = self.param_generator1(s)
        fc1_param1, fc2_param1 = mlp_params1.chunk(2, dim=-1)
        fc1_param1 = fc1_param1.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc2_param1 = fc2_param1.view(batch_size, hidden_size_x * self.mlp_ratio, hidden_size_x)

        normalized_fc1_param1 = torch.nn.functional.normalize(fc1_param1, dim=-2)
        res_x = x
        x = self.norm(x)
        x = torch.bmm(x, normalized_fc1_param1)
        x = torch.nn.functional.silu(x)
        x = torch.bmm(x, fc2_param1)
        x = x + res_x
        return x


class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return x


class PixNerDiTExtended(nn.Module):
    """
    Class-Conditional PixNerDiT with Extended Boundaries and Overlap Supervision.

    Key features:
    1. Extended patches: Each patch includes margin pixels from neighbors
    2. Overlap supervision: Overlapping regions supervised against ground truth
    3. Consistent predictions: Adjacent patches predict same values in overlap

    Training flow:
    1. Pad image with reflection
    2. Extract extended patches (core + margin on all sides)
    3. Encoder: processes core patches for global context
    4. Decoder: predicts extended patches with NerfEmbedder
    5. Loss: compare extended predictions vs ground truth extended patches

    Inference flow:
    1. Predict extended patches
    2. Blend overlapping regions for smooth output

    Args:
        in_channels: Number of input channels (3 for RGB)
        num_groups: Number of attention heads
        hidden_size: Encoder hidden dimension
        decoder_hidden_size: Decoder hidden dimension
        num_encoder_blocks: Number of encoder transformer blocks
        num_decoder_blocks: Number of decoder NerfBlocks
        patch_size: Base patch size for encoder
        num_classes: Number of classes
        margin: Extended boundary margin (0.25 = 25% extra on each side)
    """
    def __init__(
            self,
            in_channels=3,
            num_groups=4,
            hidden_size=256,
            decoder_hidden_size=32,
            num_encoder_blocks=6,
            num_decoder_blocks=2,
            patch_size=2,
            num_classes=10,
            margin=0.25,
            weight_path=None,
            load_ema=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.decoder_hidden_size = decoder_hidden_size
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.num_blocks = self.num_encoder_blocks + self.num_decoder_blocks
        self.num_classes = num_classes
        self.margin = margin
        self.patch_size = patch_size

        # Compute extended patch size
        # For patch_size=2, margin=0.25: margin_pixels=0 (too small)
        # For patch_size=2, margin=0.5: margin_pixels=1
        # We need at least 1 pixel margin, so we use ceil
        self.margin_pixels = max(1, int(round(patch_size * margin)))
        self.extended_size = patch_size + 2 * self.margin_pixels

        # Actual margin ratio based on integer pixel count
        self.effective_margin = self.margin_pixels / patch_size

        # Decoder patch scaling for super-resolution
        self.decoder_patch_scaling_h = 1.0
        self.decoder_patch_scaling_w = 1.0

        # Embedders
        self.s_embedder = Embed(in_channels * patch_size ** 2, hidden_size, bias=True)

        # Extended NerfEmbedder (no jittering for consistency)
        self.x_embedder = ExtendedNerfEmbedder(
            in_channels=in_channels,
            hidden_size=decoder_hidden_size,
            max_freqs=8,
            margin=self.effective_margin,
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        self.final_layer = NerfFinalLayer(decoder_hidden_size, in_channels)

        # Encoder blocks
        encoder_blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_encoder_blocks)
        ])
        # Decoder blocks
        decoder_blocks = nn.ModuleList([
            NerfBlock(self.hidden_size, self.decoder_hidden_size, mlp_ratio=2) for _ in range(self.num_decoder_blocks)
        ])
        self.blocks = nn.ModuleList(list(encoder_blocks) + list(decoder_blocks))

        self.initialize_weights()
        self.precompute_pos = dict()
        self.weight_path = weight_path
        self.load_ema = load_ema

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def extract_extended_patches(self, x):
        """
        Extract extended patches from image.

        Each patch includes margin_pixels on all sides from neighbors.
        Uses reflection padding at image boundaries.

        Args:
            x: [B, C, H, W] input image

        Returns:
            extended_patches: [B, num_patches, extended_size^2 * C]
            core_patches: [B, num_patches, patch_size^2 * C]
        """
        B, C, H, W = x.shape

        # Pad image for extended patches
        x_padded = F.pad(x, (self.margin_pixels,) * 4, mode='reflect')

        # Extract extended patches using unfold
        # unfold extracts patches of size extended_size with stride patch_size
        # This gives overlapping patches where each sees its neighbors
        patches = x_padded.unfold(2, self.extended_size, self.patch_size)
        patches = patches.unfold(3, self.extended_size, self.patch_size)
        # patches: [B, C, num_h, num_w, extended_size, extended_size]

        num_h, num_w = patches.shape[2], patches.shape[3]

        # Reshape to [B, num_patches, C * extended_size * extended_size]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        extended_patches = patches.view(B, num_h * num_w, C * self.extended_size * self.extended_size)

        # Also extract core patches for encoder (non-overlapping)
        core_patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        core_patches = core_patches.transpose(1, 2)  # [B, num_patches, C * patch_size^2]

        return extended_patches, core_patches, (num_h, num_w)

    def fold_extended_patches(self, patches, output_size, grid_shape):
        """
        Fold extended patches back into image with blending.

        Overlapping regions are averaged for smooth transitions.

        Args:
            patches: [B, num_patches, C * extended_size * extended_size]
            output_size: (H, W) target output size
            grid_shape: (num_h, num_w) patch grid dimensions

        Returns:
            image: [B, C, H, W]
        """
        B = patches.shape[0]
        H, W = output_size
        num_h, num_w = grid_shape
        C = self.in_channels

        # Reshape patches
        patches = patches.view(B, num_h, num_w, C, self.extended_size, self.extended_size)

        # Create output and weight accumulator (padded)
        H_pad = H + 2 * self.margin_pixels
        W_pad = W + 2 * self.margin_pixels
        output = torch.zeros(B, C, H_pad, W_pad, device=patches.device, dtype=patches.dtype)
        weights = torch.zeros(1, 1, H_pad, W_pad, device=patches.device, dtype=patches.dtype)

        # Place each patch with accumulation
        for i in range(num_h):
            for j in range(num_w):
                y_start = i * self.patch_size
                x_start = j * self.patch_size

                patch = patches[:, i, j]  # [B, C, extended_size, extended_size]
                output[:, :, y_start:y_start + self.extended_size,
                       x_start:x_start + self.extended_size] += patch
                weights[:, :, y_start:y_start + self.extended_size,
                        x_start:x_start + self.extended_size] += 1.0

        # Average overlapping regions
        output = output / weights.clamp(min=1.0)

        # Crop to original size (remove padding)
        output = output[:, :, self.margin_pixels:self.margin_pixels + H,
                        self.margin_pixels:self.margin_pixels + W]

        return output

    def forward(self, x, t, y):
        """
        Forward pass with extended boundary supervision.

        Training: Predicts extended patches, loss computed against GT extended patches
        Inference: Predicts extended patches, blends overlapping regions

        Args:
            x: Input tensor [B, C, H, W]
            t: Timestep [B]
            y: Class labels [B]

        Returns:
            Predicted velocity field [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Compute encoder resolution
        encoder_h = int(H / self.decoder_patch_scaling_h)
        encoder_w = int(W / self.decoder_patch_scaling_w)
        decoder_patch_size_h = int(self.patch_size * self.decoder_patch_scaling_h)
        decoder_patch_size_w = int(self.patch_size * self.decoder_patch_scaling_w)

        # Scale margin pixels for decoder
        margin_h = int(self.margin_pixels * self.decoder_patch_scaling_h)
        margin_w = int(self.margin_pixels * self.decoder_patch_scaling_w)
        extended_h = decoder_patch_size_h + 2 * margin_h
        extended_w = decoder_patch_size_w + 2 * margin_w

        # Downsample for encoder path
        x_for_encoder = F.interpolate(x, (encoder_h, encoder_w)) if (encoder_h != H or encoder_w != W) else x

        # Extract patches for encoder (core only, non-overlapping)
        x_for_encoder = F.unfold(x_for_encoder, kernel_size=self.patch_size, stride=self.patch_size)
        x_for_encoder = x_for_encoder.transpose(1, 2)  # [B, num_patches, C * patch_size^2]

        # Extract extended patches for decoder (includes neighbor pixels)
        x_padded = F.pad(x, (margin_w, margin_w, margin_h, margin_h), mode='reflect')

        # Use unfold to extract extended patches with core stride
        extended_patches = x_padded.unfold(2, extended_h, decoder_patch_size_h)
        extended_patches = extended_patches.unfold(3, extended_w, decoder_patch_size_w)
        # extended_patches: [B, C, num_h, num_w, extended_h, extended_w]

        num_h, num_w = extended_patches.shape[2], extended_patches.shape[3]
        num_patches = num_h * num_w

        # Reshape for decoder: [B * num_patches, extended_h * extended_w, C]
        extended_patches = extended_patches.permute(0, 2, 3, 4, 5, 1).contiguous()
        extended_patches = extended_patches.view(B, num_patches, extended_h * extended_w, C)
        x_for_decoder = extended_patches.view(B * num_patches, extended_h * extended_w, C)

        # Position embeddings for encoder
        xpos = self.fetch_pos(encoder_h // self.patch_size, encoder_w // self.patch_size, x.device)

        # Time and class embeddings
        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        y_emb = self.y_embedder(y).view(B, 1, self.hidden_size)
        condition = F.silu(t_emb + y_emb)

        # Encoder path
        s = self.s_embedder(x_for_encoder)
        for i in range(self.num_encoder_blocks):
            s = self.blocks[i](s, condition, xpos)

        # Prepare for decoder
        s = F.silu(t_emb + s)
        s = s.view(B * num_patches, self.hidden_size)

        # Extended NerfEmbedder - predicts for positions in [-margin, 1+margin]
        x_decoded = self.x_embedder(x_for_decoder, extended_h, extended_w)

        # Decoder path
        for i in range(self.num_decoder_blocks):
            x_decoded = self.blocks[i + self.num_encoder_blocks](x_decoded, s)

        # Final layer: [B * num_patches, extended_h * extended_w, C]
        x_decoded = self.final_layer(x_decoded)

        # Reshape back to patches: [B, num_patches, C, extended_h, extended_w]
        x_decoded = x_decoded.view(B, num_patches, extended_h, extended_w, C)
        x_decoded = x_decoded.permute(0, 1, 4, 2, 3).contiguous()
        x_decoded = x_decoded.view(B, num_h, num_w, C, extended_h, extended_w)

        # Fold with blending
        H_pad = H + 2 * margin_h
        W_pad = W + 2 * margin_w
        output = torch.zeros(B, C, H_pad, W_pad, device=x.device, dtype=x.dtype)
        weights = torch.zeros(1, 1, H_pad, W_pad, device=x.device, dtype=x.dtype)

        for i in range(num_h):
            for j in range(num_w):
                y_start = i * decoder_patch_size_h
                x_start = j * decoder_patch_size_w

                patch = x_decoded[:, i, j]
                output[:, :, y_start:y_start + extended_h, x_start:x_start + extended_w] += patch
                weights[:, :, y_start:y_start + extended_h, x_start:x_start + extended_w] += 1.0

        # Average overlapping regions
        output = output / weights.clamp(min=1.0)

        # Crop to original size
        output = output[:, :, margin_h:margin_h + H, margin_w:margin_w + W]

        return output


# Alias
PixNerDiT = PixNerDiTExtended
