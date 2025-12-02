"""
CIFAR-10 Dataset for PixNerd Training

CIFAR-10 consists of 60,000 32x32 color images in 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images, 10,000 test images
"""
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from PIL import Image


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class PixCIFAR10(Dataset):
    """
    CIFAR-10 dataset wrapper for PixNerd training.

    Returns normalized images in [-1, 1] range with class labels.

    Args:
        root: Root directory for CIFAR-10 data (will be downloaded if not present)
        train: If True, use training set; else test set
        random_flip: If True, apply random horizontal flips (training augmentation)
        download: If True, download the dataset if not found
    """
    def __init__(
        self,
        root: str = './data',
        train: bool = True,
        random_flip: bool = True,
        download: bool = True,
    ):
        self.train = train

        # Build transforms
        transform_list = []
        if random_flip and train:
            transform_list.append(RandomHorizontalFlip())
        transform_list.append(ToTensor())  # [0, 1]
        transform_list.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))  # [-1, 1]

        self.transform = Compose(transform_list)

        self.dataset = CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None  # We apply transforms manually
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        metadata = {
            "class": label,
            "class_name": CIFAR10_CLASSES[label],
        }

        return image, label, metadata


class CIFAR10RandomNDataset(Dataset):
    """
    Random noise dataset for CIFAR-10 class-conditional sampling/evaluation.

    Generates random noise tensors paired with class labels for inference.

    Args:
        num_classes: Number of classes (10 for CIFAR-10)
        latent_shape: Shape of noise tensor (C, H, W)
        max_num_instances: Maximum number of samples to generate
        num_samples_per_class: If specified, generate this many samples per class
        seeds: Optional list of fixed seeds for reproducibility
    """
    def __init__(
        self,
        num_classes: int = 10,
        latent_shape: tuple = (3, 32, 32),
        max_num_instances: int = 10000,
        num_samples_per_class: int = -1,
        seeds: list = None,
    ):
        self.num_classes = num_classes
        self.latent_shape = latent_shape
        self.seeds = seeds

        if num_samples_per_class > 0:
            self.max_num_instances = num_samples_per_class * num_classes
        else:
            self.max_num_instances = max_num_instances

        if seeds is not None:
            self.num_seeds = len(seeds)
            self.max_num_instances = self.num_seeds * num_classes
        else:
            self.num_seeds = (self.max_num_instances + num_classes - 1) // num_classes

    def __len__(self):
        return self.max_num_instances

    def __getitem__(self, idx):
        # Cycle through classes
        class_label = idx % self.num_classes
        seed_idx = idx // self.num_classes

        # Get seed
        if self.seeds is not None:
            seed = self.seeds[seed_idx % len(self.seeds)]
        else:
            seed = idx  # Use index as seed for reproducibility

        # Generate random noise
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)

        filename = f"{CIFAR10_CLASSES[class_label]}_{seed}"

        metadata = {
            "filename": filename,
            "seed": seed,
            "class": class_label,
            "class_name": CIFAR10_CLASSES[class_label],
        }

        return noise, class_label, metadata


class CIFAR10SuperResRandomNDataset(Dataset):
    """
    Random noise dataset for CIFAR-10 super-resolution inference.

    Generates noise at higher resolution for super-resolution sampling.

    Args:
        num_classes: Number of classes (10 for CIFAR-10)
        base_resolution: Base training resolution (32 for CIFAR-10)
        super_res_scale: Super-resolution scale factor (e.g., 4 for 4x)
        max_num_instances: Maximum number of samples
        num_samples_per_class: Samples per class
    """
    def __init__(
        self,
        num_classes: int = 10,
        base_resolution: int = 32,
        super_res_scale: int = 4,
        max_num_instances: int = 10000,
        num_samples_per_class: int = -1,
        seeds: list = None,
    ):
        self.num_classes = num_classes
        self.base_resolution = base_resolution
        self.super_res_scale = super_res_scale
        self.target_resolution = base_resolution * super_res_scale
        self.latent_shape = (3, self.target_resolution, self.target_resolution)
        self.seeds = seeds

        if num_samples_per_class > 0:
            self.max_num_instances = num_samples_per_class * num_classes
        else:
            self.max_num_instances = max_num_instances

        if seeds is not None:
            self.num_seeds = len(seeds)
            self.max_num_instances = self.num_seeds * num_classes
        else:
            self.num_seeds = (self.max_num_instances + num_classes - 1) // num_classes

    def __len__(self):
        return self.max_num_instances

    def __getitem__(self, idx):
        class_label = idx % self.num_classes
        seed_idx = idx // self.num_classes

        if self.seeds is not None:
            seed = self.seeds[seed_idx % len(self.seeds)]
        else:
            seed = idx

        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)

        filename = f"{CIFAR10_CLASSES[class_label]}_{self.super_res_scale}x_{seed}"

        metadata = {
            "filename": filename,
            "seed": seed,
            "class": class_label,
            "class_name": CIFAR10_CLASSES[class_label],
            "resolution": self.target_resolution,
            "scale": self.super_res_scale,
        }

        return noise, class_label, metadata
