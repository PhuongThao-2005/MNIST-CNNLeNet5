from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch

from config import BATCH_SIZE, DATA_ROOT, MEDICAL_ROOT, DATASETS, SEED


def _base_transform(img_size: int = 32) -> transforms.Compose:
    """
    Resize về img_size x img_size (LeNet-5 cần 32x32),
    chuyển thành tensor, chuẩn hoá về [-1, 1].
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def _medical_transform(img_size: int = 32) -> transforms.Compose:
    """
    Medical MNIST là ảnh màu (RGB) từ ImageFolder -> cần Grayscale trước.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


# Dataset loaders

def get_handwritten_mnist(batch_size: int = BATCH_SIZE):
    """Trả về (train_loader, test_loader, class_names)."""
    cfg = DATASETS["handwritten"]
    tf  = _base_transform(cfg["img_size"])

    train_ds = datasets.MNIST(DATA_ROOT, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, cfg["class_names"]


def get_fashion_mnist(batch_size: int = BATCH_SIZE):
    """Trả về (train_loader, test_loader, class_names)."""
    cfg = DATASETS["fashion"]
    tf  = _base_transform(cfg["img_size"])

    train_ds = datasets.FashionMNIST(DATA_ROOT, train=True,  download=True, transform=tf)
    test_ds  = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, cfg["class_names"]


def get_medical_mnist(batch_size: int = BATCH_SIZE, val_ratio: float = 0.2):
    """
    Medical MNIST không có train/test split sẵn → dùng random_split 80/20.
    Trả về (train_loader, test_loader, class_names).
    """
    cfg = DATASETS["medical"]
    tf  = _medical_transform(cfg["img_size"])

    full_ds = datasets.ImageFolder(root=MEDICAL_ROOT, transform=tf)

    # Cập nhật class_names từ dataset thực tế
    class_names = full_ds.classes

    # Random split reproducible
    generator  = torch.Generator().manual_seed(SEED)
    test_size  = int(val_ratio * len(full_ds))
    train_size = len(full_ds) - test_size
    train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, class_names


def get_all_datasets(batch_size: int = BATCH_SIZE):
    """
    Trả về dict:
    {
      "handwritten":   (train_loader, test_loader, class_names, num_classes),
      "fashion": (...),
      "medical": (...),
    }
    """
    handwritten_train, handwritten_test, handwritten_classes   = get_handwritten_mnist(batch_size)
    fashion_train, fashion_test, fashion_classes = get_fashion_mnist(batch_size)
    medical_train, medical_test, medical_classes = get_medical_mnist(batch_size)

    return {
        "handwritten": {
            "train": handwritten_train, "test": handwritten_test,
            "classes": handwritten_classes, "num_classes": len(handwritten_classes),
        },
        "fashion": {
            "train": fashion_train, "test": fashion_test,
            "classes": fashion_classes, "num_classes": len(fashion_classes),
        },
        "medical": {
            "train": medical_train, "test": medical_test,
            "classes": medical_classes, "num_classes": len(medical_classes),
        },
    }
