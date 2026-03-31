import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE, EPOCHS, LEARNING_RATE
from model  import LeNet5, LeNet5Improved

def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader):
    """Tính accuracy trên loader (không tính gradient)."""
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y    = X.to(DEVICE), y.to(DEVICE)
            preds   = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

    return correct / total

def train_model(
    train_loader : torch.utils.data.DataLoader,
    test_loader  : torch.utils.data.DataLoader,
    num_classes  : int,
    name         : str,
    epochs       : int   = EPOCHS,
    lr           : float = LEARNING_RATE,
    improved     : bool  = False,
):
    """
    Train LeNet-5 (hoặc LeNet5Improved) và trả về model đã train.

    Args:
        improved: True → dùng LeNet5Improved (BatchNorm + Dropout + ReLU)
                  False → dùng LeNet5 gốc (Tanh)
    """
    variant = "Improved" if improved else "Baseline"
    print(f"\n{'='*60}")
    print(f"  Dataset: {name}  |  Variant: {variant}  |  Classes: {num_classes}")
    print(f"{'='*60}")

    # Khởi tạo model
    ModelClass = LeNet5Improved if improved else LeNet5
    model      = ModelClass(in_channels=1, num_classes=num_classes).to(DEVICE)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler – giảm lr × 0.5 mỗi 7 epoch
    scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    history = {"train_acc": [], "test_acc": [], "loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X, y    = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            loss    = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        train_acc = evaluate(model, train_loader)
        test_acc  = evaluate(model, test_loader)

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["loss"].append(total_loss)

        print(
            f"Epoch {epoch:02d}/{epochs}"
            f"  Loss: {total_loss:8.2f}"
            f"  Train: {train_acc:.4f}"
            f"  Test:  {test_acc:.4f}"
        )

    print(f"\n→ Best Test Acc: {max(history['test_acc']):.4f}")
    return model, history

def train_all(datasets: dict, improved: bool = False, epochs: int = EPOCHS):
    """
    Nhận dict từ dataset.get_all_datasets() và train lần lượt 3 dataset.
    Trả về dict {name: (model, history)}.
    """
    results = {}
    for ds_name, ds_info in datasets.items():
        model, history = train_model(
            train_loader = ds_info["train"],
            test_loader  = ds_info["test"],
            num_classes  = ds_info["num_classes"],
            name         = ds_name.upper(),
            epochs       = epochs,
            improved     = improved,
        )
        results[ds_name] = {"model": model, "history": history}
    return results

def print_summary(results: dict):
    """In bảng tổng hợp kết quả cuối cùng."""
    print("\n" + "="*50)
    print(f"{'Dataset':<12} {'Best Test Acc':>15}")
    print("-"*50)
    for ds_name, r in results.items():
        best = max(r["history"]["test_acc"])
        print(f"{ds_name:<12} {best:>14.4f}")
    print("="*50)
