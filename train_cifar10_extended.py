#!/usr/bin/env python3
"""
CIFAR-10 Training with Extended Boundaries and Overlap Supervision

This training script uses PixNerDiTExtended which improves NF quality through
proper overlap supervision - each patch predicts extended regions that include
actual neighbor pixels, and overlapping predictions are supervised against
ground truth for consistency.

Key innovations:
1. Extended patches: Each patch predicts beyond its boundary into neighbor territory
2. Overlap supervision: Overlapping regions supervised against actual pixel values
3. Consistency: Adjacent patches must predict same values in shared regions
4. Blending: During inference, overlapping predictions are averaged for smooth output

How it works:
- With patch_size=2 and margin=0.5: margin_pixels=1, extended_size=4
- Each 2x2 patch predicts a 4x4 region (includes 1 pixel from each neighbor)
- The 4x4 prediction is compared against actual 4x4 pixels from the padded image
- Overlapping regions get supervised multiple times → forces consistency

No jittering: Since overlapping regions must match exactly, position jittering
is disabled. The consistency constraint provides regularization instead.

Usage:
    python train_cifar10_extended.py

    # Adjust margin (0.5 = 1 pixel overlap for patch_size=2)
    python train_cifar10_extended.py --margin 0.5

After training, use cifar10_extended_superres_inference.ipynb for generation.
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
torch._dynamo.config.disable = True

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger

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
from src.models.conditioner.class_label import LabelConditioner
from src.models.transformer.pixnerd_c2i_extended import PixNerDiTExtended
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.flow_matching.sampling import EulerSampler, ode_step_fn
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.flow_matching.training import FlowMatchingTrainer
from src.callbacks.simple_ema import SimpleEMA
from src.callbacks.save_images import SaveImagesHook
from src.lightning_model import LightningModel
from src.lightning_data import DataModule
from src.data.dataset.cifar10 import PixCIFAR10, CIFAR10RandomNDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 with Extended NerfEmbedder")

    # Training config
    parser.add_argument("--max_steps", type=int, default=100000, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Model config (~10M parameters)
    parser.add_argument("--patch_size", type=int, default=2,
                       help="Encoder patch size (32/2=16x16=256 encoder tokens)")
    parser.add_argument("--hidden_size", type=int, default=256, help="Encoder hidden dimension")
    parser.add_argument("--decoder_hidden_size", type=int, default=32, help="Decoder hidden dimension")
    parser.add_argument("--num_encoder_blocks", type=int, default=6, help="Number of encoder blocks")
    parser.add_argument("--num_decoder_blocks", type=int, default=2, help="Number of decoder blocks")
    parser.add_argument("--num_groups", type=int, default=4, help="Number of attention heads")

    # Extended NerfEmbedder config - KEY INNOVATION!
    # margin controls how much each patch extends into neighbor territory
    # With margin=0.5 and patch_size=2: margin_pixels=1, extended_size=4
    # Overlapping regions are supervised against ground truth for consistency
    parser.add_argument("--margin", type=float, default=0.5,
                       help="Extended boundary margin (0.5 = 1 pixel margin for patch_size=2)")

    # Sampler config
    parser.add_argument("--guidance", type=float, default=2.0, help="CFG guidance scale")
    parser.add_argument("--num_sample_steps", type=int, default=50, help="Sampling steps")

    # Logging
    parser.add_argument("--exp_name", type=str, default="cifar10_extended_nerf", help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="./workdirs", help="Output directory")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="pixnerd_cifar10", help="W&B project name")

    # Checkpointing
    parser.add_argument("--save_every_n_steps", type=int, default=5000, help="Checkpoint frequency")
    parser.add_argument("--val_every_n_epochs", type=int, default=10, help="Validation every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Hardware
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")

    return parser.parse_args()


def build_model(args):
    """Build the LightningModel with extended NerfEmbedder."""

    main_scheduler = LinearScheduler()

    # VAE (identity for pixel space)
    vae = PixelAE(scale=1.0)

    # Class label conditioner
    conditioner = LabelConditioner(num_classes=10)

    # Denoiser with EXTENDED NerfEmbedder
    # Each patch predicts extended region including neighbor pixels
    # Overlapping predictions are supervised against ground truth
    denoiser = PixNerDiTExtended(
        in_channels=3,
        patch_size=args.patch_size,
        num_groups=args.num_groups,
        hidden_size=args.hidden_size,
        decoder_hidden_size=args.decoder_hidden_size,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        num_classes=10,
        margin=args.margin,  # Extended boundaries with overlap supervision
    )

    # Sampler
    sampler = EulerSampler(
        num_steps=args.num_sample_steps,
        guidance=args.guidance,
        guidance_interval_min=0.0,
        guidance_interval_max=1.0,
        scheduler=main_scheduler,
        w_scheduler=LinearScheduler(),
        guidance_fn=simple_guidance_fn,
        step_fn=ode_step_fn,
    )

    # Trainer (flow matching loss)
    trainer = FlowMatchingTrainer(
        scheduler=main_scheduler,
        lognorm_t=True,
        timeshift=1.0,
    )

    # EMA
    ema_tracker = SimpleEMA(decay=0.9999)

    # Optimizer
    optimizer = partial(torch.optim.AdamW, lr=args.lr, weight_decay=0.0)

    # Build LightningModel
    model = LightningModel(
        vae=vae,
        conditioner=conditioner,
        denoiser=denoiser,
        diffusion_trainer=trainer,
        diffusion_sampler=sampler,
        ema_tracker=ema_tracker,
        optimizer=optimizer,
        lr_scheduler=None,
        eval_original_model=False,
    )

    return model


def build_datamodule(args):
    """Build the data module for CIFAR-10."""

    train_dataset = PixCIFAR10(
        root="./data",
        train=True,
        random_flip=True,
        download=True,
    )

    eval_dataset = CIFAR10RandomNDataset(
        num_classes=10,
        latent_shape=(3, 32, 32),
        max_num_instances=1000,
    )

    pred_dataset = CIFAR10RandomNDataset(
        num_classes=10,
        latent_shape=(3, 32, 32),
        max_num_instances=1000,
    )

    datamodule = DataModule(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        pred_dataset=pred_dataset,
        train_batch_size=args.batch_size,
        train_num_workers=args.num_workers,
        pred_batch_size=64,
        pred_num_workers=2,
    )

    return datamodule


def main():
    args = parse_args()

    # Compute actual pixel margins
    margin_pixels = max(1, int(round(args.patch_size * args.margin)))
    extended_size = args.patch_size + 2 * margin_pixels
    effective_margin = margin_pixels / args.patch_size

    print("=" * 70)
    print("CIFAR-10 Training with Extended Boundaries and Overlap Supervision")
    print("=" * 70)
    print()
    print("KEY INNOVATION: Proper overlap supervision for consistent predictions")
    print()
    print(f"  Patch configuration:")
    print(f"    • patch_size: {args.patch_size}")
    print(f"    • margin: {args.margin} → margin_pixels: {margin_pixels}")
    print(f"    • extended_size: {extended_size} (core {args.patch_size} + {margin_pixels} on each side)")
    print(f"    • effective_margin: {effective_margin:.2f}")
    print()
    print(f"  How it works:")
    print(f"    • Each patch predicts {extended_size}x{extended_size} pixels (not just {args.patch_size}x{args.patch_size})")
    print(f"    • Extended region includes actual neighbor pixels from the image")
    print(f"    • Overlapping regions are supervised against ground truth")
    print(f"    • Adjacent patches must predict same values → consistency!")
    print(f"    • During inference, overlapping predictions are blended")
    print()
    print(f"Full config: {vars(args)}")
    print()

    # Output directory
    output_dir = Path(args.output_dir) / f"exp_{args.exp_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Build model and data
    print("\nBuilding model...")
    model = build_model(args)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nBuilding datamodule...")
    datamodule = build_datamodule(args)

    # Logger
    if args.use_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.exp_name,
            save_dir=str(output_dir),
        )
    elif TENSORBOARD_AVAILABLE:
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
        SaveImagesHook(
            save_dir="val",
            save_compressed=True,
        ),
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
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume,
    )

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    print("\nTo generate images, use cifar10_extended_superres_inference.ipynb")
    print("=" * 70)


if __name__ == "__main__":
    main()
