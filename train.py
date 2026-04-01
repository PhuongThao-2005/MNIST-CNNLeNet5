import copy

import torch
import torch.nn as nn
import torch.optim as optim
import os

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
    model,
    train_loader,
    test_loader,
    dataset     : str,
    variant     : str,
    epochs      : int   = EPOCHS,
    lr          : float = LEARNING_RATE,
    save_dir    : str   = "models",
):
    """
    Train model và lưu trọng số tốt nhất (best test acc) vào save_dir.
    Trả về (model đã load best weights, history).
    """
    print(f"\n{'='*62}")
    print(f"  Dataset : {dataset.upper():<10}  Variant : {variant.upper()}")
    print(f"{'='*62}")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # StepLR: giảm lr × 0.5 mỗi 10 epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model   : {model.__class__.__name__}  ({total_params:,} params)")
    print(f"  Epochs  : {epochs}   LR: {lr}")
    print("-"*62)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset}_{variant}.pth")

    history = {"train_acc": [], "test_acc": [], "loss": []}
    best_acc        = 0.0
    best_state_dict = None

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

        # lưu best weights
        if test_acc > best_acc:
            best_acc        = test_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, save_path)

        print(
            f"Epoch {epoch:02d}/{epochs}"
            f"  Loss: {total_loss:8.2f}"
            f"  Train: {train_acc:.4f}"
            f"  Test : {test_acc:.4f}"
            + (" <-  best" if test_acc == best_acc else "")
        )

    # load lại best weights trước khi trả về
    model.load_state_dict(best_state_dict)
    print(f"\n  Best Test Acc [{variant}] : {best_acc:.4f}  ->  saved to {save_path}")
    return model, history