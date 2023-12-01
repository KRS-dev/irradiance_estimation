# Training Hyperparameters
INPUT_SIZE = (32, 32)
INPUT_CHANNELS = 11
OUTPUT_CHANNELS = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 3
MIN_EPOCHS = 1
MAX_EPOCHS = 10


# Dataset
# DATA_DIR
NUM_WORKERS = 1

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 32
