import os
import torch

# ============================================
# MODEL SELECTION
# ============================================
# True = Use EfficientNet (transfer learning, more accurate but slower)
# False = Use custom CNN (faster training, less accurate)
USE_TRANSFER_LEARNING = True

# ============================================
# TRAINING HYPERPARAMETERS
# ============================================
# Number of times to go through the entire dataset
EPOCHS = 20 if USE_TRANSFER_LEARNING else 10

# Number of images to process at once (reduce if running out of memory)
BATCH_SIZE = 16

# How fast the model learns (lower = more careful, higher = faster but risky)
LEARNING_RATE = 0.0003 if USE_TRANSFER_LEARNING else 0.001

# Prevents overfitting by penalizing large weights
WEIGHT_DECAY = 1e-4

# Smooths target labels slightly to prevent overconfidence
LABEL_SMOOTHING = 0.05

# Randomly drops neurons during training to prevent overfitting
DROPOUT_RATE = 0.5

# ============================================
# EARLY STOPPING SETTINGS
# ============================================
# Stop training if no improvement after this many epochs
MAX_PATIENCE = 8

# Reduce learning rate by this factor when stuck
LR_REDUCE_FACTOR = 0.5

# How many epochs to wait before reducing learning rate
LR_REDUCE_PATIENCE = 3

# ============================================
# FILE PATHS
# ============================================
# Where to save trained models
MODEL_DIR = "saved_models"
MODEL_PATH = f"{MODEL_DIR}/deepfake_cnn.pth"
OPTIMIZER_PATH = f"{MODEL_DIR}/deepfake_optimizer.pth"
FILE_COUNT_PATH = f"{MODEL_DIR}/file_count.txt"

# Where to find training/testing images
TRAIN_PATH = "dataset/train"  # Should have train/real/ and train/fake/
TEST_PATH = "dataset/test"    # Should have test/real/ and test/fake/

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================
# DEVICE CONFIGURATION
# ============================================
# Automatically use GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Print if GPU is being used
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Optimize GPU performance
    torch.backends.cudnn.benchmark = True
else:
    print("Using CPU (training will be slower)")

# Print the model architecture
model_type = "Transfer Learning (EfficientNet)" if USE_TRANSFER_LEARNING else "Custom CNN"
print(f"Model Type: {model_type}\n")