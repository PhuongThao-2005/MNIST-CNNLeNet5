import torch

#  Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Training
BATCH_SIZE   = 64
EPOCHS       = 20
LEARNING_RATE = 0.001

#  Paths
DATA_ROOT    = "/kaggle/working"          # Handwritten MNIST / FashionMNIST download cache
MEDICAL_ROOT = "/kaggle/input/datasets/andrewmvd/medical-mnist"

#  Dataset meta
DATASETS = {
    "handwritten": {
        "num_classes": 10,
        "img_size"   : 32,          # LeNet-5 input: 32x32
        "in_channels": 1,
        "class_names": [str(i) for i in range(10)],
    },
    "fashion": {
        "num_classes": 10,
        "img_size"   : 32,
        "in_channels": 1,
        "class_names": [
            "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
        ],
    },
    "medical": {
        "num_classes": 6,           # AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT
        "img_size"   : 32,
        "in_channels": 1,
        "class_names": None,        # sẽ lấy từ ImageFolder.classes lúc load
    },
}

SEED = 42
