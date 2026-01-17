"""
PixNerDiT for EEG Signals with Non-Square Patch Support

This model supports 1D EEG signals arranged as 2D arrays:
- Input shape: (C, H, W) = (1, num_channels, num_samples)
- Example: (1, 2, 600) for 2-channel EEG with 600 time samples
- Patch size: (patch_h, patch_w) = (2, 10) for 60 patches total

Key differences from standard PixNerDiT:
- Supports non-square patches (patch_size_h, patch_size_w)
- Designed for 1-channel signal data instead of 3-channel RGB
- RoPE adjusted for non-square patch grids
"""
import torch
import torch.nn as nn

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


class NerfEmbedder(nn.Module):
    """Embeds pixel positions using continuous coordinates (NeRF-style)."""
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs ** 2, hidden_size_input, bias=True),
        )

    @lru_cache
    def fetch_pos(self, patch_size_h, patch_size_w, device, dtype):
        pos = precompute_freqs_cis_2d(
            self.max_freqs ** 2 * 2,
            patch_size_h,
            patch_size_w,
            scale=(16/patch_size_h, 16/patch_size_w)
        )
        pos = pos[None, :, :].to(device=device, dtype=dtype)
        return pos

    def forward(self, inputs, patch_size_h, patch_size_w):
        B, _, C = inputs.shape
        device = inputs.device
        dtype = inputs.dtype
        dct = self.fetch_pos(patch_size_h, patch_size_w, device, dtype)
        dct = dct.repeat(B, 1, 1)
        inputs = torch.cat([inputs, dct], dim=-1)
        inputs = self.embedder(inputs)
        return inputs


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


class PixNerDiT_EEG(nn.Module):
    """
    PixNerDiT for EEG Signals with Non-Square Patch Support.

    Designed for 1D signals arranged as 2D arrays:
    - Input: (batch, 1, num_channels, num_samples) e.g., (B, 1, 2, 600)
    - Patch: (patch_h, patch_w) e.g., (2, 10) for 60 patches

    Args:
        in_channels: Number of input channels (1 for signal amplitude)
        num_groups: Number of attention heads
        hidden_size: Encoder hidden dimension
        decoder_hidden_size: Decoder hidden dimension
        num_encoder_blocks: Number of encoder transformer blocks
        num_decoder_blocks: Number of decoder NerfBlocks
        patch_size_h: Patch height (e.g., 2 for 2 EEG channels)
        patch_size_w: Patch width (e.g., 10 for 10 time samples per patch)
        num_classes: Number of classes for conditioning
    """
    def __init__(
            self,
            in_channels=1,
            num_groups=8,
            hidden_size=256,
            decoder_hidden_size=32,
            num_encoder_blocks=6,
            num_decoder_blocks=2,
            patch_size_h=2,
            patch_size_w=10,
            num_classes=1,
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

        # Non-square patch support
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w

        # Decoder patch scaling for super-resolution
        self.decoder_patch_scaling_h = 1.0
        self.decoder_patch_scaling_w = 1.0

        # Embedders
        patch_dim = in_channels * patch_size_h * patch_size_w
        self.s_embedder = Embed(patch_dim, hidden_size, bias=True)
        self.x_embedder = NerfEmbedder(in_channels, decoder_hidden_size, max_freqs=8)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        self.final_layer = NerfFinalLayer(decoder_hidden_size, in_channels)

        # Encoder blocks (transformer with self-attention)
        encoder_blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_encoder_blocks)
        ])
        # Decoder blocks (hypernetwork NerfBlocks)
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
        # Initialize patch_embed like nn.Linear
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize label embedding table
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y):
        """
        Forward pass with non-square patch support.

        Args:
            x: Input tensor [B, C, H, W] e.g., [B, 1, 2, 600]
            t: Timestep [B]
            y: Class labels [B]

        Returns:
            Predicted velocity field [B, C, H, W]
        """
        B, _, H, W = x.shape

        # Compute encoder resolution (with potential super-resolution scaling)
        encoder_h = int(H / self.decoder_patch_scaling_h)
        encoder_w = int(W / self.decoder_patch_scaling_w)
        decoder_patch_size_h = int(self.patch_size_h * self.decoder_patch_scaling_h)
        decoder_patch_size_w = int(self.patch_size_w * self.decoder_patch_scaling_w)

        # Downsample for encoder path if doing super-resolution
        if self.decoder_patch_scaling_h != 1.0 or self.decoder_patch_scaling_w != 1.0:
            x_for_encoder = torch.nn.functional.interpolate(x, (encoder_h, encoder_w))
        else:
            x_for_encoder = x

        # Patchify for encoder (using base patch sizes)
        x_for_encoder = torch.nn.functional.unfold(
            x_for_encoder,
            kernel_size=(self.patch_size_h, self.patch_size_w),
            stride=(self.patch_size_h, self.patch_size_w)
        ).transpose(1, 2)

        # Patchify for decoder (using scaled patch sizes)
        x_for_decoder = torch.nn.functional.unfold(
            x,
            kernel_size=(decoder_patch_size_h, decoder_patch_size_w),
            stride=(decoder_patch_size_h, decoder_patch_size_w)
        ).transpose(1, 2)

        # Position embeddings for encoder
        num_patches_h = encoder_h // self.patch_size_h
        num_patches_w = encoder_w // self.patch_size_w
        xpos = self.fetch_pos(num_patches_h, num_patches_w, x.device)

        # Time and class embeddings
        t = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        y = self.y_embedder(y).view(B, 1, self.hidden_size)

        # Condition = time + class label
        condition = nn.functional.silu(t + y)

        # Encoder path
        s = self.s_embedder(x_for_encoder)
        for i in range(self.num_encoder_blocks):
            s = self.blocks[i](s, condition, xpos)

        # Prepare for decoder
        s = torch.nn.functional.silu(t + s)
        batch_size, length, _ = s.shape
        x = x_for_decoder.reshape(batch_size * length, self.in_channels, decoder_patch_size_h * decoder_patch_size_w)
        x = x.transpose(1, 2)
        s = s.view(batch_size * length, self.hidden_size)
        x = self.x_embedder(x, decoder_patch_size_h, decoder_patch_size_w)

        # Decoder path
        for i in range(self.num_decoder_blocks):
            x = self.blocks[i + self.num_encoder_blocks](x, s)

        # Final layer and unpatchify
        x = self.final_layer(x)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, length, -1)
        x = torch.nn.functional.fold(
            x.transpose(1, 2).contiguous(),
            (H, W),
            kernel_size=(decoder_patch_size_h, decoder_patch_size_w),
            stride=(decoder_patch_size_h, decoder_patch_size_w)
        )
        return x
