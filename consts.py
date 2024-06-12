PROJ_DIR = '/home/ubuntu/mask-ssl'
WEIGHT_DIR = f"{PROJ_DIR}/_weights"
OUTPUT_DIR = f"{PROJ_DIR}/outputs"
CONFIG_DIR = f"{PROJ_DIR}/configs"
LOG_DIR = f"{PROJ_DIR}/exp_logs"

SEED = 3
RESNET_INP_DIM = 224
## class subset taken from https://www.kaggle.com/datasets/ambityga/imagenet100
CLASS_SUBSET_100 = [
    117, 70, 88, 133, 5, 97, 42, 60, 14, 3,
    130, 57, 26, 0, 89, 127, 36, 67, 110, 65,
    123, 55, 22, 21, 1, 71, 99, 16, 19, 108,
    18, 35, 124, 90, 74, 129, 125, 2, 64, 92, 
    138, 48, 54, 39, 56, 96, 84, 73, 77, 52,
    20, 118, 111, 59, 106, 75, 143, 80, 140, 11,
    113, 4, 28, 50, 38, 104, 24, 107, 100, 81,
    94, 41, 68, 8, 66, 146, 29, 32, 137, 33, 
    141, 134, 78, 150, 76, 61, 112, 83, 144, 91,
    135, 116, 72, 34, 6, 119, 46, 115, 93, 7
]