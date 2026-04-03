# Thông tin chung

Họ và tên: Nguyễn Phương Thảo - MSSV: 23127306

Môn học: Thị Giác Máy Tính - 23TGMT

# MNIST–CNN LeNet-5
Bài tập cài đặt các biến thể **CNN LeNet-5** trên ba bộ dữ liệu: **Handwritten MNIST**, **Fashion MNIST** và **Medical MNIST**. Code được viết bằng **PyTorch**; luồng huấn luyện đầy đủ nằm trong notebook Kaggle `kaggle_train.ipynb`.

## Cấu trúc mã nguồn

| Tệp | Mô tả |
|-----|--------|
| `config.py` | Thiết bị (CPU/GPU), batch size, epochs, learning rate, đường dẫn dữ liệu, metadata từng dataset. |
| `dataset.py` | Hàm tải dữ liệu và `DataLoader`: `get_handwritten_mnist`, `get_fashion_mnist`, `get_medical_mnist`. |
| `model.py` | Định nghĩa `LeNet5`, `LeNet5_Handwritten`, `LeNet5_Fashion`, `LeNet5_Medical`, `LeNet5Improved`. |
| `train.py` | `evaluate()` (tính accuracy), `train_model()` (vòng train + lưu checkpoint tốt nhất theo test accuracy). |
| `kaggle_train.ipynb` | Notebook clone repo, huấn luyện và so sánh các variant (Baseline / Custom / Improved). |

## Yêu cầu môi trường

- Python 3.10+
- [PyTorch](https://pytorch.org/) và **torchvision** (để dùng `datasets`, `transforms`)

## Cấu hình đường dẫn

Trong `config.py`, mặc định đường dẫn phù hợp **Kaggle**:

- `DATA_ROOT`: nơi lưu MNIST / FashionMNIST sau khi tải.
- `MEDICAL_ROOT`: thư mục Medical MNIST định dạng `ImageFolder` (mỗi lớp một folder).

## Huấn luyện

- **Kaggle / Jupyter**: mở `kaggle_train.ipynb`, chạy tuần tự các ô (clone repo nếu cần, rồi import và gọi `train_model`).
- **Kiểm tra shape & số tham số mô hình**:

```bash
python model.py
```

Checkpoints mặc định được lưu trong thư mục `models/` với tên dạng `{dataset}_{variant}.pth`.

## Ghi chú

- Trong `train.py`, **test accuracy cao nhất** trong suốt quá trình được lưu; sau khi train xong, mô hình trả về đã **load lại** trọng số tốt nhất đó.