import sys
import os
import tempfile
import pytest
import torch
from PIL import Image
import numpy as np
from src.data import BiometricDataset, get_transforms, get_dataloaders

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
)


def create_dummy_dataset(base_path: str, num_people: int = 3) -> None:
    for person_id in range(1, num_people + 1):
        person_path = os.path.join(base_path, str(person_id))
        os.makedirs(person_path, exist_ok=True)

        fp_path = os.path.join(person_path, 'Fingerprint')
        left_path = os.path.join(person_path, 'left')
        right_path = os.path.join(person_path, 'right')

        os.makedirs(fp_path, exist_ok=True)
        os.makedirs(left_path, exist_ok=True)
        os.makedirs(right_path, exist_ok=True)

        fp_img = Image.fromarray(
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        )
        left_img = Image.fromarray(
            np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        )
        right_img = Image.fromarray(
            np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        )

        fp_img.save(os.path.join(fp_path, 'fp.bmp'))
        left_img.save(os.path.join(left_path, 'left.bmp'))
        right_img.save(os.path.join(right_path, 'right.bmp'))


def test_dataset_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir, num_people=5)

        transform_fp, transform_iris = get_transforms(augment=False)
        dataset = BiometricDataset(
            tmpdir, 5, transform_fp, transform_iris
        )

        assert len(dataset) == 5
        assert isinstance(dataset, BiometricDataset)


def test_dataset_getitem():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir, num_people=3)

        transform_fp, transform_iris = get_transforms(augment=False)
        dataset = BiometricDataset(
            tmpdir, 3, transform_fp, transform_iris
        )

        fp, left, right, label = dataset[0]

        assert isinstance(fp, torch.Tensor)
        assert isinstance(left, torch.Tensor)
        assert isinstance(right, torch.Tensor)
        assert isinstance(label, int)

        assert fp.shape == (3, 128, 128)
        assert left.shape == (1, 64, 64)
        assert right.shape == (1, 64, 64)
        assert 0 <= label < 3


def test_get_dataloaders():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir, num_people=10)

        train_loader, val_loader, test_loader = get_dataloaders(
            tmpdir, 10, batch_size=2, num_workers=0,
            train_split=0.7, val_split=0.2
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        for batch in train_loader:
            fp, left, right, labels = batch
            assert fp.shape[0] <= 2
            break


def test_invalid_dataset_path():
    with pytest.raises(
        ValueError, match="Dataset path does not exist"
    ):
        BiometricDataset("invalid_path", 5)
