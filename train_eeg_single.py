#!/usr/bin/env python3
"""
EEG Single Instance Training Script

Train a PixNerd model on a SINGLE EEG segment to verify overfitting capability.
EEG data shape: (1, 2, 600) - 1 channel, 2 electrodes, 600 time samples
Patch size: (2, 10) for 60 total patches

Usage:
    # Basic training (overfits a single EEG segment)
    python train_eeg_single.py

    # With custom config
    python train_eeg_single.py --max_steps 5000 --data_path path/to/eeg.npy

After training, use eeg_single_inference.ipynb to verify overfitting.
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

import numpy as np
import torch
torch._dynamo.config.disable = True

from torch.utils.data import Dataset, DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
    from lightning.pytorch.loggers import TensorBoardLogger
except (ImportError, ModuleNotFoundError):
    TENSORBOARD_AVAILABLE = False

from src.models.autoencoder.pixel import PixelAE
from src.models.conditioner.class_label import LabelConditioner
from src.models.transformer.pixnerd_eeg_heavydecoder import PixNerDiT_EEG
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.flow_matching.sampling import EulerSampler, ode_step_fn
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.flow_matching.training import FlowMatchingTrainer
from src.callbacks.simple_ema import SimpleEMA
from src.callbacks.save_images import SaveImagesHook
from src.lightning_model import LightningModel
from src.lightning_data import DataModule


def generate_simulated_eeg(
    num_channels: int = 2,
    num_samples: int = 600,
    fs: float = 200.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate simulated EEG data with realistic characteristics.

    Creates a signal with:
    - Alpha rhythm (8-12 Hz)
    - Beta rhythm (12-30 Hz)
    - Some noise

    Args:
        num_channels: Number of EEG channels
        num_samples: Number of time samples
        fs: Sampling frequency in Hz
        seed: Random seed

    Returns:
        EEG data array of shape (1, num_channels, num_samples)
    """
    np.random.seed(seed)
    t = np.arange(num_samples) / fs

    eeg = np.zeros((num_channels, num_samples))

    for ch in range(num_channels):
        # Alpha rhythm (8-12 Hz)
        alpha_freq = 10 + np.random.randn() * 1
        alpha = 0.5 * np.sin(2 * np.pi * alpha_freq * t + np.random.rand() * 2 * np.pi)

        # Beta rhythm (12-30 Hz)
        beta_freq = 20 + np.random.randn() * 3
        beta = 0.3 * np.sin(2 * np.pi * beta_freq * t + np.random.rand() * 2 * np.pi)

        # Some slower activity
        delta_freq = 2 + np.random.randn() * 0.5
        delta = 0.2 * np.sin(2 * np.pi * delta_freq * t + np.random.rand() * 2 * np.pi)

        # Pink noise
        noise = np.cumsum(np.random.randn(num_samples)) * 0.02

        # Combine
        eeg[ch] = alpha + beta + delta + noise

    # Normalize to [-1, 1]
    eeg = eeg / (np.abs(eeg).max() + 1e-8)

    # Add channel dimension: (1, num_channels, num_samples)
    return eeg[np.newaxis, :, :].astype(np.float32)


class SingleInstanceEEGDataset(Dataset):
    """
    Single instance EEG dataset for overfitting tests.

    Repeats a single EEG segment for training.
    """
    def __init__(
        self,
        data_path: str = None,
        num_channels: int = 2,
        num_samples: int = 600,
        num_repeats: int = 1000,
        class_label: int = 0,
    ):
        """
        Args:
            data_path: Path to .npy file with EEG data, or None to generate simulated data
            num_channels: Number of EEG channels (used for simulation)
            num_samples: Number of time samples (used for simulation)
            num_repeats: How many times to repeat (virtual dataset size)
            class_label: Class label for conditioning
        """
        if data_path is not None and Path(data_path).exists():
            # Load from file
            self.eeg_data = np.load(data_path).astype(np.float32)
            print(f"Loaded EEG data from: {data_path}")
            print(f"Shape: {self.eeg_data.shape}")

            # Ensure correct shape (1, num_channels, num_samples)
            if self.eeg_data.ndim == 2:
                self.eeg_data = self.eeg_data[np.newaxis, :, :]
            elif self.eeg_data.ndim == 3 and self.eeg_data.shape[2] == 1:
                # Shape is (2, 600, 1), reshape to (1, 2, 600)
                self.eeg_data = self.eeg_data.squeeze(-1)[np.newaxis, :, :]

            # Normalize to [-1, 1] if not already
            data_min, data_max = self.eeg_data.min(), self.eeg_data.max()
            if data_max > 1.0 or data_min < -1.0:
                self.eeg_data = 2.0 * (self.eeg_data - data_min) / (data_max - data_min + 1e-8) - 1.0
                print(f"Normalized data to [-1, 1] range")
        else:
            # Generate simulated data
            self.eeg_data = generate_simulated_eeg(
                num_channels=num_channels,
                num_samples=num_samples,
            )
            print(f"Generated simulated EEG data: {self.eeg_data.shape}")

        self.signal = torch.from_numpy(self.eeg_data)
        self.label = class_label
        self.num_repeats = num_repeats

        print(f"EEG signal shape: {self.signal.shape}")
        print(f"EEG signal range: [{self.signal.min():.3f}, {self.signal.max():.3f}]")

    def __len__(self):
        return self.num_repeats

    def __getitem__(self, idx):
        metadata = {
            "class": self.label,
            "class_name": f"eeg_class_{self.label}",
        }
        return self.signal.clone(), self.label, metadata


class SingleInstanceEEGRandomNDataset(Dataset):
    """
    Random noise dataset for single instance EEG evaluation.
    """
    def __init__(
        self,
        class_label: int,
        signal_shape: tuple = (1, 2, 600),
        max_num_instances: int = 10,
    ):
        self.class_label = class_label
        self.signal_shape = signal_shape
        self.max_num_instances = max_num_instances

    def __len__(self):
        return self.max_num_instances

    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(idx)
        noise = torch.randn(self.signal_shape, generator=generator, dtype=torch.float32)

        metadata = {
            "seed": idx,
            "class": self.class_label,
            "class_name": f"eeg_class_{self.class_label}",
        }

        return noise, self.class_label, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEG single instance (overfitting test)")

    # Training config
    parser.add_argument("--max_steps", type=int, default=5000, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")

    # Data config
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to .npy file with EEG data (or None for simulated)")
    parser.add_argument("--num_channels", type=int, default=2, help="Number of EEG channels")
    parser.add_argument("--num_samples", type=int, default=600, help="Number of time samples")
    parser.add_argument("--num_repeats", type=int, default=10000, help="Virtual dataset size")

    # Model config for EEG
    # Input: (1, 2, 600), Patch: (2, 10) -> 60 patches
    parser.add_argument("--hidden_size", type=int, default=256, help="Encoder hidden dimension")
    parser.add_argument("--decoder_hidden_size", type=int, default=32, help="Decoder hidden dimension")
    parser.add_argument("--num_encoder_blocks", type=int, default=6, help="Number of encoder blocks")
    parser.add_argument("--num_decoder_blocks", type=int, default=2, help="Number of decoder blocks")
    parser.add_argument("--patch_size_h", type=int, default=2, help="Patch height (EEG channels)")
    parser.add_argument("--patch_size_w", type=int, default=10, help="Patch width (time samples)")
    parser.add_argument("--num_groups", type=int, default=8, help="Number of attention heads")

    # Sampler config
    parser.add_argument("--guidance", type=float, default=1.0, help="CFG guidance (1.0 for overfit)")
    parser.add_argument("--num_sample_steps", type=int, default=50, help="Sampling steps")

    # Logging
    parser.add_argument("--exp_name", type=str, default="eeg_single_overfit", help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="./workdirs", help="Output directory")

    # Checkpointing
    parser.add_argument("--save_every_n_steps", type=int, default=1000, help="Checkpoint frequency")
    parser.add_argument("--val_every_n_epochs", type=int, default=50, help="Validation frequency")

    # Hardware
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")

    return parser.parse_args()


def build_model(args):
    """Build the LightningModel with EEG-specific components."""

    main_scheduler = LinearScheduler()

    # VAE (identity for signal space)
    vae = PixelAE(scale=1.0)

    # Single class conditioner (unconditional-like)
    conditioner = LabelConditioner(num_classes=1)

    # Denoiser with EEG-specific architecture
    denoiser = PixNerDiT_EEG(
        in_channels=1,  # Single channel (signal amplitude)
        patch_size_h=args.patch_size_h,  # 2 (EEG channels)
        patch_size_w=args.patch_size_w,  # 10 (time samples)
        num_groups=args.num_groups,
        hidden_size=args.hidden_size,
        decoder_hidden_size=args.decoder_hidden_size,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        num_classes=1,
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
    """Build the data module for single instance EEG training."""

    # Create single instance training dataset
    train_dataset = SingleInstanceEEGDataset(
        data_path=args.data_path,
        num_channels=args.num_channels,
        num_samples=args.num_samples,
        num_repeats=args.num_repeats,
        class_label=0,
    )

    signal_shape = train_dataset.signal.shape

    # Evaluation dataset
    eval_dataset = SingleInstanceEEGRandomNDataset(
        class_label=0,
        signal_shape=signal_shape,
        max_num_instances=10,
    )

    pred_dataset = SingleInstanceEEGRandomNDataset(
        class_label=0,
        signal_shape=signal_shape,
        max_num_instances=10,
    )

    datamodule = DataModule(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        pred_dataset=pred_dataset,
        train_batch_size=args.batch_size,
        train_num_workers=args.num_workers,
        pred_batch_size=10,
        pred_num_workers=0,
    )

    return datamodule, train_dataset


def main():
    args = parse_args()

    print("=" * 60)
    print("EEG Single Instance Training (Overfitting Test)")
    print("=" * 60)
    print(f"Config: {vars(args)}")
    print()

    # Verify patch configuration
    num_patches_h = args.num_channels // args.patch_size_h
    num_patches_w = args.num_samples // args.patch_size_w
    total_patches = num_patches_h * num_patches_w
    print(f"Patch configuration:")
    print(f"  Input shape: (1, {args.num_channels}, {args.num_samples})")
    print(f"  Patch size: ({args.patch_size_h}, {args.patch_size_w})")
    print(f"  Patches: {num_patches_h} x {num_patches_w} = {total_patches}")
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
    datamodule, train_dataset = build_datamodule(args)

    # Save the target signal for reference
    target_signal = train_dataset.eeg_data
    np.save(output_dir / "target_signal.npy", target_signal)
    print(f"Saved target signal to: {output_dir / 'target_signal.npy'}")
    print(f"Target signal shape: {target_signal.shape}")

    # Save config
    import json
    config = {
        "num_channels": args.num_channels,
        "num_samples": args.num_samples,
        "patch_size_h": args.patch_size_h,
        "patch_size_w": args.patch_size_w,
        "hidden_size": args.hidden_size,
        "decoder_hidden_size": args.decoder_hidden_size,
        "num_encoder_blocks": args.num_encoder_blocks,
        "num_decoder_blocks": args.num_decoder_blocks,
        "num_groups": args.num_groups,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

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

    # Callbacks (no SaveImagesHook for EEG)
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
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
    )

    # Train
    print("\nStarting EEG single instance training...")
    print("The model should perfectly memorize the single EEG segment.")
    trainer.fit(model, datamodule=datamodule)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    print(f"Target signal saved to: {output_dir / 'target_signal.npy'}")
    print("\nTo verify overfitting, use:")
    print("  jupyter notebook eeg_single_inference.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
