import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE, EPOCHS, LEARNING_RATE, DATASETS

def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader):
    """Tính accuracy trên tập dữ liệu"""
    
    model.eval()  # chuyển model sang chế độ evaluation (tắt dropout, batchnorm ổn định)
    correct = total = 0  # biến đếm số dự đoán đúng và tổng số mẫu

    with torch.no_grad():  # tắt tính gradient
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)  # đưa dữ liệu lên GPU/CPU
            preds = model(X).argmax(dim=1)     # lấy class có xác suất cao nhất

            correct += (preds == y).sum().item()  # đếm số dự đoán đúng
            total   += y.size(0)                  # tổng số mẫu

    return correct / total  # trả về accuracy


def train_model(
    train_loader,
    test_loader,
    dataset     : str,
    variant     : str,
    num_classes : int,
    epochs      : int   = EPOCHS,
    lr          : float = LEARNING_RATE,
):
    """
    Train một model (dataset x variant) và trả về (model, history).
    epochs lấy từ DATASETS config per-dataset khi gọi từ train_all().
    """
    print(f"\n{'='*62}")
    print(f"  Dataset : {dataset.upper():<10}  Variant : {variant.upper()}")
    print(f"{'='*62}")

    model     = get_model(dataset, variant, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # StepLR: giảm lr × 0.5 mỗi 10 epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model   : {model.__class__.__name__}  ({total_params:,} params)")
    print(f"  Epochs  : {epochs}   LR: {lr}")
    print("-"*62)

    history = {"train_acc": [], "test_acc": [], "loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X, y   = X.to(DEVICE), y.to(DEVICE)
            loss   = criterion(model(X), y)
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
            f"  Test : {test_acc:.4f}"
        )

    best = max(history["test_acc"])
    print(f"\n Best Test Acc [{variant}] : {best:.4f}")
    return model, history