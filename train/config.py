# Training Hyperparameters
INPUT_SIZE = (128, 128)
INPUT_CHANNELS = 11
OUTPUT_CHANNELS = 1
LEARNING_RATE = 0.01
BATCH_SIZE = 4
NUM_EPOCHS = 100
MIN_EPOCHS = 1
MAX_EPOCHS = 100


# Dataset
# DATA_DIR
NUM_WORKERS = 12

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 32
