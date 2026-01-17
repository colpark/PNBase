#!/usr/bin/env python3
"""
CIFAR-10 Single Instance Training Script

Train a PixNerd model on a SINGLE CIFAR-10 image to verify overfitting capability.
This is useful for debugging and validating the model architecture.

Usage:
    # Basic training (overfits a single image)
    python train_cifar10_single.py

    # With custom config
    python train_cifar10_single.py --max_steps 5000 --image_index 0

After training, use cifar10_single_inference.ipynb to verify overfitting.
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
# Completely disable torch.compile/dynamo to avoid inductor errors
torch._dynamo.config.disable = True

from torch.utils.data import Dataset, DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

# Check if tensorboard is available
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
    from lightning.pytorch.loggers import TensorBoardLogger
except (ImportError, ModuleNotFoundError):
    TENSORBOARD_AVAILABLE = False

from src.models.autoencoder.pixel import PixelAE
from src.models.conditioner.class_label import LabelConditioner
from src.models.transformer.pixnerd_c2i_heavydecoder import PixNerDiT
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.flow_matching.sampling import EulerSampler, ode_step_fn
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.flow_matching.training import FlowMatchingTrainer
from src.callbacks.simple_ema import SimpleEMA
from src.callbacks.save_images import SaveImagesHook
from src.lightning_model import LightningModel
from src.lightning_data import DataModule
from src.data.dataset.cifar10 import PixCIFAR10, CIFAR10_CLASSES


class SingleInstanceCIFAR10Dataset(Dataset):
    """
    Single instance CIFAR-10 dataset for overfitting tests.

    Repeats a single image for training to verify the model can memorize it.
    """
    def __init__(
        self,
        root: str = './data',
        image_index: int = 0,
        num_repeats: int = 1000,
        download: bool = True,
    ):
        """
        Args:
            root: Root directory for CIFAR-10 data
            image_index: Index of the single image to use
            num_repeats: How many times to repeat the image (virtual dataset size)
            download: Whether to download CIFAR-10 if not found
        """
        # Load full CIFAR-10 to get single image
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import Compose, ToTensor, Normalize

        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        cifar = CIFAR10(root=root, train=True, download=download, transform=None)

        # Get single image and label
        pil_image, label = cifar[image_index]
        self.image = transform(pil_image)
        self.label = label
        self.class_name = CIFAR10_CLASSES[label]
        self.num_repeats = num_repeats
        self.image_index = image_index

        print(f"Single instance dataset: image_index={image_index}, "
              f"class={self.class_name} ({label}), repeats={num_repeats}")

    def __len__(self):
        return self.num_repeats

    def __getitem__(self, idx):
        metadata = {
            "class": self.label,
            "class_name": self.class_name,
            "original_index": self.image_index,
        }
        return self.image.clone(), self.label, metadata


class SingleInstanceRandomNDataset(Dataset):
    """
    Random noise dataset for single instance evaluation.

    Generates noise with the same class label as the training image.
    """
    def __init__(
        self,
        class_label: int,
        latent_shape: tuple = (3, 32, 32),
        max_num_instances: int = 10,
    ):
        self.class_label = class_label
        self.latent_shape = latent_shape
        self.max_num_instances = max_num_instances

    def __len__(self):
        return self.max_num_instances

    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(idx)
        noise = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)

        metadata = {
            "seed": idx,
            "class": self.class_label,
            "class_name": CIFAR10_CLASSES[self.class_label],
        }

        return noise, self.class_label, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 single instance (overfitting test)")

    # Training config
    parser.add_argument("--max_steps", type=int, default=5000, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size (all same image)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")

    # Single instance config
    parser.add_argument("--image_index", type=int, default=0, help="CIFAR-10 image index to overfit")
    parser.add_argument("--num_repeats", type=int, default=10000, help="Virtual dataset size")

    # Model config (same as train_cifar10.py)
    parser.add_argument("--hidden_size", type=int, default=512, help="Encoder hidden dimension")
    parser.add_argument("--decoder_hidden_size", type=int, default=64, help="Decoder hidden dimension")
    parser.add_argument("--num_encoder_blocks", type=int, default=8, help="Number of encoder blocks")
    parser.add_argument("--num_decoder_blocks", type=int, default=2, help="Number of decoder blocks")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size")
    parser.add_argument("--num_groups", type=int, default=8, help="Number of attention heads")

    # Sampler config
    parser.add_argument("--guidance", type=float, default=1.0, help="CFG guidance (1.0 for overfit)")
    parser.add_argument("--num_sample_steps", type=int, default=50, help="Sampling steps")

    # Logging
    parser.add_argument("--exp_name", type=str, default="cifar10_single_overfit", help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="./workdirs", help="Output directory")

    # Checkpointing
    parser.add_argument("--save_every_n_steps", type=int, default=1000, help="Checkpoint frequency")
    parser.add_argument("--val_every_n_epochs", type=int, default=50, help="Validation frequency")

    # Hardware
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")

    return parser.parse_args()


def build_model(args):
    """Build the LightningModel with all components."""

    main_scheduler = LinearScheduler()

    # VAE (identity for pixel space)
    vae = PixelAE(scale=1.0)

    # Class label conditioner
    conditioner = LabelConditioner(num_classes=10)

    # Denoiser with heavy decoder
    denoiser = PixNerDiT(
        in_channels=3,
        patch_size=args.patch_size,
        num_groups=args.num_groups,
        hidden_size=args.hidden_size,
        decoder_hidden_size=args.decoder_hidden_size,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        num_classes=10,
    )

    # Sampler (low guidance for overfitting)
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
    """Build the data module for single instance training."""

    # Create single instance training dataset
    train_dataset = SingleInstanceCIFAR10Dataset(
        root="./data",
        image_index=args.image_index,
        num_repeats=args.num_repeats,
        download=True,
    )

    # Get the class label from training dataset
    class_label = train_dataset.label

    # Evaluation dataset with same class
    eval_dataset = SingleInstanceRandomNDataset(
        class_label=class_label,
        latent_shape=(3, 32, 32),
        max_num_instances=10,
    )

    pred_dataset = SingleInstanceRandomNDataset(
        class_label=class_label,
        latent_shape=(3, 32, 32),
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
    print("CIFAR-10 Single Instance Training (Overfitting Test)")
    print("=" * 60)
    print(f"Config: {vars(args)}")
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

    # Save the target image for reference
    import numpy as np
    from PIL import Image
    target_img = train_dataset.image.numpy()
    target_img = ((target_img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    target_img = np.transpose(target_img, (1, 2, 0))
    Image.fromarray(target_img).save(output_dir / "target_image.png")
    print(f"Saved target image to: {output_dir / 'target_image.png'}")
    print(f"Target class: {train_dataset.class_name} ({train_dataset.label})")

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
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
    )

    # Train
    print("\nStarting single instance training...")
    print("The model should perfectly memorize the single image.")
    trainer.fit(model, datamodule=datamodule)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    print(f"Target image saved to: {output_dir / 'target_image.png'}")
    print("\nTo verify overfitting, use:")
    print("  jupyter notebook cifar10_single_inference.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
