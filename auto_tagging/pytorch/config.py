import torch

# devices
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
DATA_DIR = "../data/"
ANNOTATIONS_FILE = "annotations_final.csv"
TAGS_FILE = "tags.txt"

# mel spectroghram
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
POWER = 2.0
NORM = 'slaney'
MEL_SCALE = 'htk'
PAD_MODE = 'reflect'
CENTER = True

# data processing
SAMPLE_RATE = 16000
NUM_SAMPLES = 464000
IN_CHANNELS = 1
IMAGE_SIZE = (128, 256)

# hyper parameters
EPOCHS_NUM = 15
LEARNING_RATE = 0.005
BATCH_SIZE = 32
TRAIN_SIZE_PROPORTION = 0.8
TEST_SIZE_PROPORTION = 0.1
NUM_WORKERS = 0
DROPOUT_P = 0.2
CLASSES_NUM = 50
PULL_STRIDE = 2
PULL_KERNEL_SIZE = 2
KERNEL_SIZE = 3
STRIDE = 1
THRESHOLD = 0.18