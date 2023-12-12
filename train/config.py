# Training Hyperparameters
INPUT_SIZE = (64, 64)
INPUT_CHANNELS = 11
OUTPUT_CHANNELS = 1
LEARNING_RATE = 0.01
BATCH_SIZE = 508
NUM_EPOCHS = 2
MIN_EPOCHS = 1
MAX_EPOCHS = 30


# Dataset
# DATA_DIR
NUM_WORKERS = 12

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 32
