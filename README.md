# MNIST–CNN LeNet-5

Dự án so sánh các biến thể **LeNet-5** (và các kiến trúc CNN cùng họ) trên ba bộ dữ liệu: **MNIST viết tay**, **Fashion-MNIST** và **Medical MNIST**. Code được viết bằng **PyTorch**; luồng huấn luyện đầy đủ nằm trong notebook Kaggle `kaggle_train.ipynb`.

## Cấu trúc mã nguồn

| Tệp | Mô tả |
|-----|--------|
| `config.py` | Thiết bị (CPU/GPU), batch size, epochs, learning rate, đường dẫn dữ liệu, metadata từng dataset. |
| `dataset.py` | Hàm tải dữ liệu và `DataLoader`: `get_handwritten_mnist`, `get_fashion_mnist`, `get_medical_mnist`. |
| `model.py` | Định nghĩa `LeNet5`, `LeNet5_Handwritten`, `LeNet5_Fashion`, `LeNet5_Medical`, `LeNet5Improved`. |
| `train.py` | `evaluate()` (tính accuracy), `train_model()` (vòng train + lưu checkpoint tốt nhất theo test accuracy). |
| `kaggle_train.ipynb` | Notebook clone repo, huấn luyện và so sánh các variant (Baseline / Custom / Improved). |

## Yêu cầu môi trường

- Python 3.10+ (khuyến nghị)
- [PyTorch](https://pytorch.org/) và **torchvision** (để dùng `datasets`, `transforms`)

Cài đặt ví dụ:

```bash
pip install torch torchvision
```

## Cấu hình đường dẫn

Trong `config.py`, mặc định đường dẫn phù hợp **Kaggle**:

- `DATA_ROOT`: nơi lưu MNIST / FashionMNIST sau khi tải.
- `MEDICAL_ROOT`: thư mục Medical MNIST định dạng `ImageFolder` (mỗi lớp một folder).

Khi chạy **trên máy local**, sửa hai biến này trỏ tới thư mục thật trên ổ đĩa của bạn.

## Các biến thể mô hình (ý tưởng)

- **Baseline (`LeNet5`)**: LeNet-5 cổ điển — Tanh + Average Pooling.
- **Custom theo dataset**: `LeNet5_Handwritten`, `LeNet5_Fashion`, `LeNet5_Medical` — độ sâu, dropout và kích thước lớp khác nhau tùy độ khó và số lớp.
- **Improved (`LeNet5Improved`)**: ReLU, BatchNorm, MaxPool, Dropout — một phiên bản “cải tiến” dùng chung để so sánh.

## Huấn luyện

- **Kaggle / Jupyter**: mở `kaggle_train.ipynb`, chạy tuần tự các ô (clone repo nếu cần, rồi import và gọi `train_model`).
- **Kiểm tra shape & số tham số mô hình** (không cần dữ liệu):

```bash
python model.py
```

Checkpoints mặc định được lưu trong thư mục `models/` với tên dạng `{dataset}_{variant}.pth` (xem tham số `save_dir` trong `train_model`).

## Ghi chú

- Trong `train.py`, **test accuracy cao nhất** trong suốt quá trình được lưu; sau khi train xong, mô hình trả về đã **load lại** trọng số tốt nhất đó.
- `dataset.py` dùng `num_workers=2` và `pin_memory=True`; nếu gặp lỗi trên Windows, có thể giảm `num_workers` hoặc tắt `pin_memory`.
