import os
import logging
import time
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import glob

logger = logging.getLogger(__name__)


class BiometricDataset(Dataset):
    def __init__(
        self, base_path, num_people, transform_fp=None,
        transform_iris=None, augment=False
    ):
        if not os.path.exists(base_path):
            raise ValueError(
                f"Dataset path does not exist: {base_path}"
            )
        self.samples = []
        self.transform_fp = transform_fp
        self.transform_iris = transform_iris
        self.augment = augment
        logger.info(
            f"Loading dataset from {base_path} for " 
            f"{num_people} people..."
        )
        start_time = time.time()
        skipped = 0
        for person_id in range(1, num_people + 1):
            person_path = os.path.join(base_path, str(person_id))
            if not os.path.exists(person_path):
                skipped += 1
                continue
            fp_path = os.path.join(person_path, "Fingerprint")
            left_path = os.path.join(person_path, "left")
            right_path = os.path.join(person_path, "right")
            fp_imgs = (
                glob.glob(os.path.join(fp_path, "*.bmp")) +
                glob.glob(os.path.join(fp_path, "*.BMP"))
            )
            left_imgs = (
                glob.glob(os.path.join(left_path, "*.bmp")) +
                glob.glob(os.path.join(left_path, "*.BMP"))
            )
            right_imgs = (
                glob.glob(os.path.join(right_path, "*.bmp")) +
                glob.glob(os.path.join(right_path, "*.BMP"))
            )
            if fp_imgs and left_imgs and right_imgs:
                self.samples.append(
                    (fp_imgs[0], left_imgs[0], right_imgs[0],
                     person_id - 1)
                )
            else:
                skipped += 1
                logger.debug(f"Skipping person {person_id}: missing images")
        load_time = time.time() - start_time
        logger.info(
            f"Loaded {len(self.samples)} samples in"
            f"{load_time:.2f}s (skipped {skipped})"
        )
        if len(self.samples) == 0:
            raise ValueError(
                "No valid samples found. Check dataset structure."
            )

    
    def __len__(self):
        return len(self.samples)

    
    def __getitem__(self, idx):
        fp_path, left_path, right_path, label = self.samples[idx]
        try:
            fp = Image.open(fp_path).convert("RGB")
            left = Image.open(left_path).convert("L")
            right = Image.open(right_path).convert("L")
        except Exception as e:
            logger.error(
                f"Error loading images for sample {idx}: {e}"
            )
            raise
        if self.transform_fp:
            fp = self.transform_fp(fp)
        if self.transform_iris:
            left = self.transform_iris(left)
            right = self.transform_iris(right)
        return fp, left, right, label


def get_transforms(augment=False):
    if augment:
        transform_fp = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        transform_iris = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5], std=[0.5]
            ),
        ])
    else:
        transform_fp = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        transform_iris = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5], std=[0.5]
            ),
        ])
    return transform_fp, transform_iris


def get_dataloaders(
    base_path, num_people, batch_size, num_workers=2,
    train_split=0.7, val_split=0.15, augment_train=True
):
    logger.info(
        f"Creating dataloaders with splits: "
        f"train={train_split}, val={val_split}"
    )
    transform_fp, transform_iris = get_transforms(augment=False)
    full_dataset = BiometricDataset(
        base_path, num_people, transform_fp, transform_iris
    )
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    logger.info(
        f"Split sizes - Train: {train_size},"
        f"Val: {val_size}, Test: {test_size}"
    )
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    if augment_train:
        transform_fp_aug, transform_iris_aug = get_transforms(augment=True)
        train_dataset.dataset.transform_fp = transform_fp_aug
        train_dataset.dataset.transform_iris = transform_iris_aug
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    return train_loader, val_loader, test_loader
