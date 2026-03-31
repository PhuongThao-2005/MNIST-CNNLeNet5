

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    #  Kiến trúc gốc LeNet-5 (LeCun 1998) cho ảnh 32x32x1:
    #  Input(1,32,32)
    #    → C1: Conv(1→6, k=5)   → Tanh → AvgPool(2)   → (6,14,14)
    #    → C3: Conv(6→16, k=5)  → Tanh → AvgPool(2)   → (16,5,5)
    #    → Flatten → 400
    #    → F5: Linear(400→120)  → Tanh
    #    → F6: Linear(120→84)   → Tanh
    #    → Output: Linear(84→num_classes)
    LeNet-5 chuẩn, hỗ trợ tuỳ chỉnh:
        in_channels  - số kênh đầu vào (1 cho grayscale)
        num_classes  - số lớp đầu ra
        use_relu     - True  → dùng ReLU (thực tế hơn)
                       False → dùng Tanh (chuẩn gốc LeCun)
    """

    def __init__(
        self,
        in_channels : int = 1,
        num_classes : int = 10,
        use_relu    : bool = False,
    ):
        super().__init__()
        act = nn.ReLU if use_relu else nn.Tanh

        self.features = nn.Sequential(
            # C1
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=0),
            act(),
            nn.AvgPool2d(kernel_size=2, stride=2),      # → (6,14,14)

            # C3
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            act(),
            nn.AvgPool2d(kernel_size=2, stride=2),      # → (16,5,5)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),    # F5
            act(),
            nn.Linear(120, 84),            # F6
            act(),
            nn.Linear(84, num_classes),    # Output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class LeNet5Improved(nn.Module):
    """
    Phiên bản cải tiến so với LeNet-5 gốc:
      • Activation: ReLU thay Tanh  → giảm vanishing gradient
      • BatchNorm sau mỗi Conv      → ổn định training nhanh hơn
      • Dropout(0.5) trước FC cuối → giảm overfitting
      • MaxPool thay AvgPool        → giữ đặc trưng nổi bật hơn
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


if __name__ == "__main__":
    dummy = torch.zeros(2, 1, 32, 32)
    for cls, name in [(LeNet5, "LeNet5"), (LeNet5Improved, "LeNet5Improved")]:
        m = cls(num_classes=10)
        out = m(dummy)
        print(f"{name}: input {dummy.shape} → output {out.shape}")
        total_params = sum(p.numel() for p in m.parameters())
        print(f"  Total parameters: {total_params:,}")
