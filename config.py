import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Siêu tham số huấn luyện mặc định
BATCH_SIZE   = 64      # số mẫu mỗi batch khi lấy gradient
EPOCHS       = 20      # số vòng lặp qua toàn bộ tập train
LEARNING_RATE = 0.001  # learning rate khởi tạo cho Adam

DATA_ROOT    = "/kaggle/working"
MEDICAL_ROOT = "/kaggle/input/datasets/andrewmvd/medical-mnist"

# Metadata theo từng dataset: số lớp, kích thước ảnh, kênh đầu vào, epochs/lr riêng, tên lớp (hiển thị)
DATASETS = {
    "handwritten": {
        "num_classes" : 10,
        "img_size"    : 32,
        "in_channels" : 1,
        "epochs"      : 20,
        "lr"          : 0.001,
        "class_names" : [str(i) for i in range(10)],
    },
    "fashion": {
        "num_classes" : 10,
        "img_size"    : 32,
        "in_channels" : 1,
        "epochs"      : 20,
        "lr"          : 0.001,
        "class_names" : [
            "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
        ],
    },
    "medical": {
        "num_classes" : 6,
        "img_size"    : 32,
        "in_channels" : 1,
        "epochs"      : 20,
        "lr"          : 0.001,
        "class_names" : None,     # lấy từ ImageFolder.classes lúc load
    },
}

SEED = 42
