import torch
import torch.nn as nn


#  1. LeNet-5 Baseline  (LeCun 1998)
class LeNet5(nn.Module):
    """LeNet-5 gốc: Tanh + AvgPool. Dùng làm baseline so sánh."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                          # → (6,14,14)
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                          # → (16,5,5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
            nn.Linear(120, 84),         nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

#  2. LeNet5_Handwritten
class LeNet5_Handwritten(nn.Module):
    """
    Thiết kế cho Handwritten MNIST :
        - Filter 6→12 (tăng nhẹ so với gốc 6→16 để nhanh converge)
        - Thêm Dropout(0.3) trước FC cuối để kiểm soát overfit
        - ReLU + BN + MaxPool từ Improved
        - FC nhỏ: 120→84 giữ nguyên, không cần phình to
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.BatchNorm2d(6), nn.ReLU(), nn.MaxPool2d(2, 2),  # → (6,14,14)
            nn.Conv2d(6, 12, kernel_size=5),
            nn.BatchNorm2d(12), nn.ReLU(), nn.MaxPool2d(2, 2), # → (12,5,5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 5 * 5, 84), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


#  3. LeNet5_Fashion 
class LeNet5_Fashion(nn.Module):
    """
    Thiết kế cho Fashion MNIST:
        - 3 conv layer
        - Filter tăng mạnh: 32 → 64 → 128
        - kernel 3x3 cho C2, C3 để học local texture chi tiết
        - FC lớn: 512 → 256
        - Dropout(0.5) sau mỗi FC lớn để chống overfit khi tăng capacity
        - padding=2 ở C1 giữ spatial info đầu vào
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # C1 – bắt edge tổng quát, padding giữ 32→32
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # → (32,16,16)

            # C2 – học texture chi tiết
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # → (64,8,8)

            # C3 – học cấu trúc cấp cao (dáng áo, đế giày, v.v.)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # → (128,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),         nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


#  4. LeNet5_Medical
class LeNet5_Medical(nn.Module):
    """
    Thiết kế cho Medical MNIST:
      Vấn đề: 6 lớp ảnh y tế rất khác nhau về hình dạng
              (AbdomenCT vs BreastMRI vs CXR vs ChestCT vs Hand vs HeadCT)
              -> Baseline đã quá dư thừa capacity
        - 2 conv với chỉ 4 → 8 filter
        - FC 32 node
        - Không Dropout (không cần do dataset quá dễ phân biệt)
        - BN + ReLU để training ổn định
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=5),
            nn.BatchNorm2d(4), nn.ReLU(), nn.MaxPool2d(2, 2),  # → (4,14,14)
            nn.Conv2d(4, 8, kernel_size=5),
            nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2, 2),  # → (8,5,5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 5 * 5, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


#  5. LeNet-5 Improved (ReLU + BatchNorm + MaxPool + Dropout)
class LeNet5Improved(nn.Module):
    """
    Improved chung: ReLU + BatchNorm + MaxPool + Dropout 0.4.
    Filter 6->16 giữ nguyên.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.BatchNorm2d(6), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84),         nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
    
#  Registry + factory function
MODEL_REGISTRY = {
    ("handwritten",   "baseline") : LeNet5,
    ("handwritten",   "custom")   : LeNet5_Handwritten,
    ("handwritten",   "improved") : LeNet5Improved,
    ("fashion", "baseline") : LeNet5,
    ("fashion", "custom")   : LeNet5_Fashion,
    ("fashion", "improved") : LeNet5Improved,
    ("medical", "baseline") : LeNet5,
    ("medical", "custom")   : LeNet5_Medical,
    ("medical", "improved") : LeNet5Improved,
}


def get_model(dataset: str, variant: str, num_classes: int) -> nn.Module:
    """
    Factory - trả về model đúng với dataset và variant.

    Args:
        dataset    : 'handwritten' | 'fashion' | 'medical'
        variant    : 'baseline' | 'custom' | 'improved' 
        num_classes: số lớp đầu ra
    """
    key = (dataset.lower(), variant.lower())
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Không tìm thấy model cho key={key}.\n"
            f"Các key hợp lệ: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key](num_classes=num_classes)

if __name__ == "__main__":
    dummy = torch.zeros(2, 1, 32, 32)
    configs = [
        (LeNet5,          10, "LeNet5 Baseline"),
        (LeNet5_Handwritten,  10, "LeNet5_Handwritten"),
        (LeNet5_Fashion,  10, "LeNet5_Fashion"),
        (LeNet5_Medical,   6, "LeNet5_Medical"),
        (LeNet5Improved,  10, "LeNet5 Improved (chung)"),
    ]
    print(f"{'Model':<42} {'Output':>14} {'Params':>12}")
    print("-" * 70)
    for cls, nc, name in configs:
        m      = cls(num_classes=nc)
        out    = m(dummy)
        params = sum(p.numel() for p in m.parameters())
        print(f"{name:<42} {str(out.shape):>14} {params:>12,}")