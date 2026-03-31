from torch.utils.data import DataLoader, random_split  # DataLoader: tạo batch, random_split: chia dataset
from torchvision import datasets, transforms          # datasets: MNIST, FashionMNIST; transforms: tiền xử lý ảnh
import torch

from config import BATCH_SIZE, DATA_ROOT, MEDICAL_ROOT, DATASETS, SEED

def _base_transform(img_size: int = 32) -> transforms.Compose:
    """
    Transform dùng cho MNIST và FashionMNIST
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),   # resize ảnh về 32x32 (LeNet-5 chuẩn)
        transforms.ToTensor(),                     # chuyển ảnh PIL → Tensor (0 → 1)
        transforms.Normalize((0.5,), (0.5,)),      # chuẩn hoá về [-1, 1]
    ])


def _medical_transform(img_size: int = 32) -> transforms.Compose:
    """
    Transform cho medical dataset (ảnh RGB → grayscale)
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # chuyển ảnh màu → ảnh xám (1 channel)
        transforms.Resize((img_size, img_size)),      # resize về 32x32
        transforms.ToTensor(),                        # chuyển sang tensor
        transforms.Normalize((0.5,), (0.5,)),         # chuẩn hoá [-1, 1]
    ])


# ======================
# Dataset loaders
# ======================

def get_handwritten_mnist(batch_size: int = BATCH_SIZE):
    """Load Handwritten MNIST"""
    cfg = DATASETS["handwritten"]           # lấy config riêng của dataset
    tf  = _base_transform(cfg["img_size"])  # tạo transform

    # load train/test dataset
    train_ds = datasets.MNIST(DATA_ROOT, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)

    # tạo DataLoader để chia batch
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, cfg["class_names"]  # trả về thêm tên class


def get_fashion_mnist(batch_size: int = BATCH_SIZE):
    """Load Fashion-MNIST"""
    cfg = DATASETS["fashion"]
    tf  = _base_transform(cfg["img_size"])

    train_ds = datasets.FashionMNIST(DATA_ROOT, train=True,  download=True, transform=tf)
    test_ds  = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, cfg["class_names"]


def get_medical_mnist(batch_size: int = BATCH_SIZE, val_ratio: float = 0.2):
    """
    Medical dataset: Không có train/test sẵn, tự chia bằng random_split
    """
    cfg = DATASETS["medical"]
    tf  = _medical_transform(cfg["img_size"])  # dùng transform riêng (có grayscale)

    full_ds = datasets.ImageFolder(root=MEDICAL_ROOT, transform=tf)
    # ImageFolder: đọc dữ liệu theo folder structure:
    # class1/, class2/, ...

    class_names = full_ds.classes  # lấy tên class từ folder

    # chia train/test reproducible bằng random_split với seed cố định
    generator  = torch.Generator().manual_seed(SEED)
    test_size  = int(val_ratio * len(full_ds))     # 20% làm test
    train_size = len(full_ds) - test_size

    train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=generator)

    # tạo DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, class_names