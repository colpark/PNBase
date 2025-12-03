#!/usr/bin/env python3
"""
Sinusoidal Function Neural Field Interpolation Benchmark

This benchmark tests PixNerd's neural field interpolation capabilities using
regular grid sampling to simulate super-resolution.

Key concept:
- Training: Model sees regularly sampled pixels (like low-res input)
- Testing: Model must predict ALL pixels (super-resolution)
- This directly tests NerfEmbedder's interpolation quality

Super-resolution simulation:
- downsample_factor=5: Every 5th pixel visible = 5x super-res (4% visible in 2D grid)
- downsample_factor=4: Every 4th pixel visible = 4x super-res (6.25% visible)
- downsample_factor=2: Every 2nd pixel visible = 2x super-res (25% visible)

Example: resolution=64, downsample_factor=4
- Low-res equivalent: 16x16 (256 pixels visible)
- High-res target: 64x64 (4096 pixels total)
- Model learns to upscale 4x via neural field interpolation

Usage:
    python train_sinusoidal.py

    # 4x super-resolution (every 4th pixel)
    python train_sinusoidal.py --downsample_factor 4

    # Higher resolution with 2x super-res
    python train_sinusoidal.py --resolution 128 --downsample_factor 2

After training, evaluate interpolation quality on unseen pixels.
"""
import os
import sys
import argparse
from pathlib import Path
from functools import partial

# Add PixNerd to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PIXNERD_DIR = SCRIPT_DIR / "PixNerd"
if PIXNERD_DIR.exists():
    os.chdir(PIXNERD_DIR)
    sys.path.insert(0, str(PIXNERD_DIR))

import torch
import torch.nn as nn
torch._dynamo.config.disable = True

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger

try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
    from lightning.pytorch.loggers import TensorBoardLogger
except (ImportError, ModuleNotFoundError):
    try:
        import tensorboardX
        TENSORBOARD_AVAILABLE = True
        from lightning.pytorch.loggers import TensorBoardLogger
    except (ImportError, ModuleNotFoundError):
        TENSORBOARD_AVAILABLE = False

from src.models.autoencoder.pixel import PixelAE
from src.models.transformer.pixnerd_c2i_heavydecoder import PixNerDiT
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.flow_matching.sampling import EulerSampler, ode_step_fn
from src.diffusion.base.guidance import simple_guidance_fn
from src.lightning_data import DataModule
from src.data.dataset.sinusoidal import SinusoidalDataset, SinusoidalRandomNDataset


class UnconditionalConditioner(nn.Module):
    """Dummy conditioner for unconditional generation."""

    def __init__(self):
        super().__init__()
        self.null_condition = 0

    def forward(self, y, metadata=None):
        # Return dummy condition and uncondition (both zeros)
        batch_size = len(y) if hasattr(y, '__len__') else 1
        device = y.device if isinstance(y, torch.Tensor) else 'cuda'
        condition = torch.zeros(batch_size, dtype=torch.long, device=device)
        uncondition = torch.zeros(batch_size, dtype=torch.long, device=device)
        return condition, uncondition

    def __call__(self, y, metadata=None):
        return self.forward(y, metadata)


class MaskedFlowMatchingTrainer(nn.Module):
    """
    Flow matching trainer with masked loss computation.

    Only computes loss on visible regions (defined by mask).
    """

    def __init__(self, scheduler, lognorm_t=True, timeshift=1.0):
        super().__init__()
        self.scheduler = scheduler
        self.lognorm_t = lognorm_t
        self.timeshift = timeshift

    def forward(self, net, x0, condition, mask=None):
        """
        Compute flow matching loss with optional masking.

        Args:
            net: Denoiser network
            x0: Clean images [B, C, H, W]
            condition: Class condition (unused for unconditional)
            mask: Visibility mask [H, W] or [B, H, W], True = compute loss

        Returns:
            loss: Scalar loss value
        """
        batch_size = x0.shape[0]
        device = x0.device
        dtype = x0.dtype

        # Sample timesteps
        if self.lognorm_t:
            # Log-normal sampling for better coverage
            t = torch.sigmoid(torch.randn(batch_size, device=device) * 1.2)
        else:
            t = torch.rand(batch_size, device=device)

        # Time shift
        t = t / (t + (1 - t) * self.timeshift)
        t = t.to(dtype)

        # Sample noise
        noise = torch.randn_like(x0)

        # Interpolate: x_t = (1 - t) * x0 + t * noise
        t_expanded = t.view(batch_size, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * noise

        # Target velocity: noise - x0
        target = noise - x0

        # Predict velocity
        pred = net(x_t, t, condition)

        # Compute loss
        loss_per_pixel = (pred - target) ** 2

        if mask is not None:
            # Expand mask to match loss shape [B, C, H, W]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B, 1, H, W]
            mask = mask.expand_as(loss_per_pixel).to(device)

            # Only compute loss on visible regions
            loss = loss_per_pixel[mask].mean()
        else:
            loss = loss_per_pixel.mean()

        return loss


class SinusoidalLightningModel(nn.Module):
    """
    Lightning model for sinusoidal benchmark with masked training.
    """

    def __init__(
        self,
        vae,
        conditioner,
        denoiser,
        diffusion_trainer,
        diffusion_sampler,
        ema_tracker,
        optimizer,
    ):
        super().__init__()
        self.vae = vae
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.diffusion_trainer = diffusion_trainer
        self.diffusion_sampler = diffusion_sampler
        self.ema_tracker = ema_tracker
        self.optimizer_fn = optimizer

        # EMA model
        self.ema_denoiser = None

    def setup_ema(self):
        """Setup EMA model after moving to device."""
        import copy
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        for param in self.ema_denoiser.parameters():
            param.requires_grad = False

    def update_ema(self):
        """Update EMA weights."""
        if self.ema_denoiser is None:
            return
        decay = self.ema_tracker.decay
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_denoiser.parameters(), self.denoiser.parameters()
            ):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


# Lightning Module wrapper
import lightning.pytorch as pl


class SinusoidalPLModule(pl.LightningModule):
    """PyTorch Lightning module for sinusoidal training."""

    def __init__(self, model, optimizer_fn):
        super().__init__()
        self.model = model
        self.optimizer_fn = optimizer_fn
        self.automatic_optimization = True

    def setup(self, stage=None):
        self.model.setup_ema()

    def forward(self, x, t, condition):
        return self.model.denoiser(x, t, condition)

    def training_step(self, batch, batch_idx):
        images, masks, metadata = batch

        # Get condition (dummy for unconditional)
        condition, _ = self.model.conditioner(images)

        # Encode (identity for pixel space)
        latents = self.model.vae.encode(images)

        # Compute masked flow matching loss
        # Handle masks - could be tensor [B, H, W] or stacked tensor
        if isinstance(masks, torch.Tensor):
            mask = masks[0] if masks.dim() == 3 else masks
        else:
            # masks is a tuple/list from DataLoader, get first element
            mask = masks[0] if isinstance(masks[0], torch.Tensor) else torch.stack(masks)[0]

        loss = self.model.diffusion_trainer(
            self.model.denoiser, latents, condition, mask=mask
        )

        # Update EMA
        self.model.update_ema()

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noise, masks, metadata = batch

        # Get condition
        condition, uncondition = self.model.conditioner(noise)

        # Sample using EMA model
        if self.model.ema_denoiser is not None:
            samples = self.model.diffusion_sampler(
                self.model.ema_denoiser,
                noise,
                condition,
                uncondition,
            )
        else:
            samples = self.model.diffusion_sampler(
                self.model.denoiser,
                noise,
                condition,
                uncondition,
            )

        # Decode
        images = self.model.vae.decode(samples)

        return images

    def configure_optimizers(self):
        return self.optimizer_fn(self.model.denoiser.parameters())


class SimpleEMA:
    """Simple EMA tracker."""
    def __init__(self, decay=0.9999):
        self.decay = decay


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sinusoidal Neural Field Interpolation Benchmark"
    )

    # Dataset config
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of unique sinusoidal patterns")
    parser.add_argument("--resolution", type=int, default=64,
                       help="Image resolution (high-res target)")
    parser.add_argument("--num_components", type=int, default=5,
                       help="Number of sinusoidal components per image")
    parser.add_argument("--channels", type=int, default=1,
                       help="Number of channels (1=grayscale, 3=RGB)")
    parser.add_argument("--downsample_factor", type=int, default=4,
                       help="Super-res factor: sample every Nth pixel (4=4x, 5=5x)")
    parser.add_argument("--mask_mode", type=str, default="grid",
                       choices=["columns", "rows", "grid"],
                       help="Sampling pattern: grid (2D), columns (1D-x), rows (1D-y)")

    # Training config
    parser.add_argument("--max_steps", type=int, default=50000,
                       help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader workers")

    # Model config
    parser.add_argument("--patch_size", type=int, default=4,
                       help="Encoder patch size")
    parser.add_argument("--hidden_size", type=int, default=256,
                       help="Encoder hidden dimension")
    parser.add_argument("--decoder_hidden_size", type=int, default=64,
                       help="Decoder hidden dimension")
    parser.add_argument("--num_encoder_blocks", type=int, default=6,
                       help="Number of encoder blocks")
    parser.add_argument("--num_decoder_blocks", type=int, default=2,
                       help="Number of decoder blocks")
    parser.add_argument("--num_groups", type=int, default=4,
                       help="Number of attention heads")

    # Sampler config
    parser.add_argument("--num_sample_steps", type=int, default=50,
                       help="Sampling steps")

    # Logging
    parser.add_argument("--exp_name", type=str, default="sinusoidal_nf_test",
                       help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="./workdirs",
                       help="Output directory")

    # Checkpointing
    parser.add_argument("--save_every_n_steps", type=int, default=5000,
                       help="Checkpoint frequency")
    parser.add_argument("--val_every_n_epochs", type=int, default=20,
                       help="Validation every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")

    # Hardware
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                       help="Training precision")
    parser.add_argument("--devices", type=int, default=1,
                       help="Number of GPUs")

    return parser.parse_args()


def build_model(args):
    """Build the model with all components."""

    scheduler = LinearScheduler()

    # VAE (identity for pixel space)
    vae = PixelAE(scale=1.0)

    # Unconditional conditioner
    conditioner = UnconditionalConditioner()

    # Denoiser
    denoiser = PixNerDiT(
        in_channels=args.channels,
        patch_size=args.patch_size,
        num_groups=args.num_groups,
        hidden_size=args.hidden_size,
        decoder_hidden_size=args.decoder_hidden_size,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        num_classes=1,  # Unconditional
    )

    # Masked flow matching trainer
    trainer = MaskedFlowMatchingTrainer(
        scheduler=scheduler,
        lognorm_t=True,
        timeshift=1.0,
    )

    # Sampler (guidance=1.0 for unconditional)
    sampler = EulerSampler(
        num_steps=args.num_sample_steps,
        guidance=1.0,  # No guidance for unconditional
        guidance_interval_min=0.0,
        guidance_interval_max=1.0,
        scheduler=scheduler,
        w_scheduler=LinearScheduler(),
        guidance_fn=simple_guidance_fn,
        step_fn=ode_step_fn,
    )

    # EMA
    ema_tracker = SimpleEMA(decay=0.9999)

    # Build model
    model = SinusoidalLightningModel(
        vae=vae,
        conditioner=conditioner,
        denoiser=denoiser,
        diffusion_trainer=trainer,
        diffusion_sampler=sampler,
        ema_tracker=ema_tracker,
        optimizer=None,  # Set later
    )

    return model


def build_datamodule(args):
    """Build the data module for sinusoidal dataset."""

    train_dataset = SinusoidalDataset(
        num_samples=args.num_samples,
        resolution=args.resolution,
        num_components=args.num_components,
        channels=args.channels,
        downsample_factor=args.downsample_factor,
        mask_mode=args.mask_mode,
        seed=42,
    )

    eval_dataset = SinusoidalRandomNDataset(
        latent_shape=(args.channels, args.resolution, args.resolution),
        max_num_instances=100,
        downsample_factor=args.downsample_factor,
        mask_mode=args.mask_mode,
    )

    pred_dataset = SinusoidalRandomNDataset(
        latent_shape=(args.channels, args.resolution, args.resolution),
        max_num_instances=100,
        downsample_factor=args.downsample_factor,
        mask_mode=args.mask_mode,
    )

    datamodule = DataModule(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        pred_dataset=pred_dataset,
        train_batch_size=args.batch_size,
        train_num_workers=args.num_workers,
        pred_batch_size=32,
        pred_num_workers=2,
    )

    return datamodule


def main():
    args = parse_args()

    # Compute effective resolution and visible ratio
    low_res = args.resolution // args.downsample_factor
    if args.mask_mode == "grid":
        visible_ratio = 1.0 / (args.downsample_factor ** 2)
    else:
        visible_ratio = 1.0 / args.downsample_factor

    print("=" * 70)
    print("Sinusoidal Neural Field Interpolation Benchmark")
    print("=" * 70)
    print()
    print("PURPOSE: Test PixNerd's super-resolution via neural field interpolation")
    print()
    print(f"  High-res target: {args.resolution}x{args.resolution}")
    print(f"  Low-res input: {low_res}x{low_res} (every {args.downsample_factor}th pixel)")
    print(f"  Super-resolution factor: {args.downsample_factor}x")
    print(f"  Visible pixels: {visible_ratio:.1%}")
    print(f"  Channels: {args.channels}")
    print(f"  Sinusoidal components: {args.num_components}")
    print(f"  Sampling mode: {args.mask_mode}")
    print()
    print("Training setup:")
    print(f"  • Regular grid sampling (like low-res input)")
    print(f"  • Loss computed ONLY on sampled {visible_ratio:.1%} pixels")
    print(f"  • NerfEmbedder must interpolate to remaining {1-visible_ratio:.1%}")
    print()
    print(f"Config: {vars(args)}")
    print()

    # Output directory
    output_dir = Path(args.output_dir) / f"exp_{args.exp_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Build model
    print("\nBuilding model...")
    model = build_model(args)

    # Optimizer
    optimizer_fn = partial(torch.optim.AdamW, lr=args.lr, weight_decay=0.0)

    # Lightning module
    pl_module = SinusoidalPLModule(model, optimizer_fn)
    print(f"Total parameters: {sum(p.numel() for p in model.denoiser.parameters()):,}")

    # Build datamodule
    print("\nBuilding datamodule...")
    datamodule = build_datamodule(args)

    # Logger
    if TENSORBOARD_AVAILABLE:
        logger = TensorBoardLogger(
            save_dir=str(output_dir),
            name="logs",
        )
    else:
        logger = CSVLogger(
            save_dir=str(output_dir),
            name="logs",
        )
        print("Using CSVLogger (tensorboard not available)")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            every_n_train_steps=args.save_every_n_steps,
            save_top_k=-1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
    trainer = Trainer(
        default_root_dir=str(output_dir),
        accelerator="auto",
        devices=args.devices,
        precision=args.precision,
        max_steps=args.max_steps,
        check_val_every_n_epoch=args.val_every_n_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        logger=logger,
        callbacks=callbacks,
    )

    # Train
    print("\nStarting training...")
    print(f"  Training on {visible_ratio:.1%} of pixels (low-res samples)")
    print(f"  Testing interpolation on {1-visible_ratio:.1%} (super-res)")
    trainer.fit(
        pl_module,
        datamodule=datamodule,
        ckpt_path=args.resume,
    )

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    print()
    print(f"Super-resolution task: {args.downsample_factor}x upscaling")
    print(f"Evaluate interpolation quality on unseen {1-visible_ratio:.1%} pixels")
    print("=" * 70)


if __name__ == "__main__":
    main()
