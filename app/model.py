import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
import os
from .utils import clean_dataset


def analyze_dataset():
    """Check dataset quality and distribution"""
    print("\n" + "="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    
    train_path = "dataset/train"
    test_path = "dataset/test"
    
    total_train = 0
    total_test = 0
    
    for split, split_path in [("TRAIN", train_path), ("TEST", test_path)]:
        print(f"\n{split} SET:")
        for class_name in ['real', 'fake']:
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {class_name.capitalize()}: {count} images")
                if split == "TRAIN":
                    total_train += count
                else:
                    total_test += count
            else:
                print(f"  {class_name.capitalize()}: 0 images (folder missing)")
    
    print(f"\nTOTAL: {total_train + total_test} images")
    print(f"  Train: {total_train} ({total_train/(total_train+total_test)*100:.1f}%)")
    print(f"  Test: {total_test} ({total_test/(total_train+total_test)*100:.1f}%)")
    print("="*50 + "\n")
    
    # Check for issues
    if total_train < 500:
        print("WARNING: Less than 500 training images - model may not learn well")
    if total_test < 100:
        print("WARNING: Less than 100 test images - validation may be unreliable")

# =======================================================
#                   Configuration
# =======================================================

# CHOOSE THE MODEL HERE
USE_TRANSFER_LEARNING = True  # Set to False to use the custom CNN from scratch


# Training parameters
EPOCHS = 20 if USE_TRANSFER_LEARNING else 10
BATCH_SIZE = 16  # Changed
LEARNING_RATE = 0.0003 if USE_TRANSFER_LEARNING else 0.001  # Changed

# Paths
MODEL_DIR = "saved_models"
MODEL_PATH = f"{MODEL_DIR}/deepfake_cnn.pth"
OPTIMIZER_PATH = f"{MODEL_DIR}/deepfake_optimizer.pth"
FILE_COUNT_PATH = f"{MODEL_DIR}/file_count.txt"

os.makedirs(MODEL_DIR, exist_ok=True)

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Print GPU information for debugging
# For NVIDIA GPU's use CUDA PyTorch
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Unfortunately ROCM (for AMD cards) does not work with Windows

if use_cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    torch.backends.cudnn.benchmark = True
else:
    print("No GPU detected - using CPU (slower)")

print(f"\n{'='*50}")
print(f"MODEL TYPE: {'TRANSFER LEARNING (EfficientNet)' if USE_TRANSFER_LEARNING else 'CUSTOM CNN'}")
print(f"{'='*50}\n")

# =======================================================
#                   Data Transforms & Loaders
# =======================================================

# train_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(20),  # Changed to 20
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # Increased
#     transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),  # More aggressive
#     transforms.RandomGrayscale(p=0.1),  # NEW LINE ADDED
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
# test_dataset = datasets.ImageFolder("dataset/test", transform=transform)

# train_loader = torch.utils.data.DataLoader(
#     train_dataset, 
#     batch_size=BATCH_SIZE, 
#     shuffle=True
# )
# test_loader = torch.utils.data.DataLoader(
#     test_dataset, 
#     batch_size=BATCH_SIZE
# )
def get_data_loaders():
    """Create and return data loaders for training and testing"""
    # Check if dataset directories exist and have data
    train_path = "dataset/train"
    test_path = "dataset/test"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Dataset directories not found. Please create:\n"
            f"  - {train_path}/fake/\n"
            f"  - {train_path}/real/\n"
            f"  - {test_path}/fake/\n"
            f"  - {test_path}/real/"
        )
    
    # Clean datasets before loading
    clean_dataset(train_path)
    clean_dataset(test_path)
    
    # Create datasets
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
    
    # Check if datasets have samples
    if len(train_dataset) == 0:
        raise ValueError(f"No images found in {train_path}. Please add images to train/fake/ and train/real/")
    if len(test_dataset) == 0:
        raise ValueError(f"No images found in {test_path}. Please add images to test/fake/ and test/real/")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE
    )
    
    return train_loader, test_loader

# =======================================================
#                   Model Architecture
# =======================================================

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        
        # Deeper architecture with residual connections
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        # Global Average Pooling instead of flatten
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        
        self.dropout = nn.Dropout(0.5)  # Increased dropout

    def forward(self, x):
        # Block 1: 256 -> 128
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 2: 128 -> 64
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 3: 64 -> 32
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2)
        
        # Global pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    


class TransferLearningCNN(nn.Module):
    """Transfer learning using pre-trained EfficientNet-B0"""
    def __init__(self, freeze_backbone=True):
        super(TransferLearningCNN, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        
        # Optionally freeze the backbone layers
        if freeze_backbone:
            # Freeze only early layers, unfreeze more for adaptation
            for param in list(self.backbone.parameters())[:-40]:  # More layers trainable
                param.requires_grad = False
            
            print("Backbone frozen (last 40 layers trainable)")
        else:
            print("Backbone fully trainable")
        
        # Replace the classifier head with custom layers
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.6),              # Increased
            nn.Linear(num_features, 256),  # Reduced from 512
            nn.ReLU(),
            nn.BatchNorm1d(256),          # NEW LINE
            nn.Dropout(0.5),              # Increased
            nn.Linear(256, 64),           # Reduced from 128
            nn.ReLU(),
            nn.BatchNorm1d(64),           # NEW LINE
            nn.Dropout(0.4),              # Increased
            nn.Linear(64, 2)
    )
        
        print(f"Custom classifier added ({num_features} -> 512 -> 128 -> 2)")
    
    def forward(self, x):
        return self.backbone(x)

# class DeepfakeCNN(nn.Module):
#     # Input: 256x256 images
#     # Output: Binary classification (real/fake)
#     def __init__(self):
#         super(DeepfakeCNN, self).__init__()

#         # Convolutional layers with batch normalization
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
        
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
        
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         # Fully connected layers
#         self.fc1 = nn.Linear(128 * 32 * 32, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, 2)
        
#         self.dropout = nn.Dropout(DROPOUT_RATE)

#     def forward(self, x):
#         # Conv block 1: 256x256 -> 128x128
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.max_pool2d(x, 2, 2)

#         # Conv block 2: 128x128 -> 64x64
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.max_pool2d(x, 2, 2)

#         # Conv block 3: 64x64 -> 32x32
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.max_pool2d(x, 2, 2)

#         # Flatten and fully connected layers
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
        
#         return x
# Original model (commented out for reference)
# class DeepfakeCNN(nn.Module):
#     def __init__(self):
#         super(DeepfakeCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
        
#     def forward(self, discrim_in):
#         discrim_in = self.pool(F.relu(self.conv1(discrim_in)))
#         discrim_in = self.pool(F.relu(self.conv2(discrim_in)))
#         discrim_in = discrim_in.view(-1, 16 * 5 * 5)
#         discrim_in = F.relu(self.fc1(discrim_in))
#         discrim_in = F.relu(self.fc2(discrim_in))
#         discrim_in = self.fc3(discrim_in)
#         return discrim_in
    
#     def loss(self, real, fake):
#         real_labels = torch.ones_like(real)
#         real_loss = self.loss_fn(real, real_labels)
#         fake_labels = torch.zeros_like(fake)
#         fake_loss = self.loss_fn(fake, fake_labels)
#         total_loss = real_loss + fake_loss
#         return total_loss

# Version 1 (commented out for reference)
# class DeepfakeCNN(nn.Module):
#     def __init__(self):
#         super(DeepfakeCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 61 * 61, 120)  # adjusted for 256x256 input
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 2)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Version 2 (Current Active Model)
# class DeepfakeCNN(nn.Module):
#     def __init__(self):
#         super(DeepfakeCNN, self).__init__()

#         # 1st Convolution Block
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)

#         # 2nd Convolution Block
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)

#         # 3rd Convolution Block
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         # Fully connected layers
#         self.fc1 = nn.Linear(128 * 32 * 32, 512)
#         self.dropout = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, 2)

#     def forward(self, x):
#         # Block 1
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.max_pool2d(x, 2, 2)

#         # Block 2
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.max_pool2d(x, 2, 2)

#         # Block 3
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.max_pool2d(x, 2, 2)

#         # Flatten
#         x = x.view(x.size(0), -1)

#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# =======================================================
#                   Model Setup
# =======================================================

# Create the model based on configuration

if USE_TRANSFER_LEARNING:
    model = TransferLearningCNN(freeze_backbone=False).to(device)
else:
    model = DeepfakeCNN().to(device)


# Use multiple GPUs if available
if device.type == "cuda" and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added parameter
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # Added parameter


# Learning rate scheduler
scheduler = None


# =======================================================
#                   Helper Functions
# =======================================================

def count_dataset_images():
    #Count total images in train and test datasets
    count = 0
    for folder in ["dataset/train", "dataset/test"]:
        if not os.path.exists(folder):
            continue
        for root, _, files in os.walk(folder):
            count += sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    return count

def dataset_changed():
    #Check if dataset has changed since last training
    #Returns True if images added/removed
    #Added 10/29/2025 - Walker Hall

    if not os.path.exists(FILE_COUNT_PATH):
        return True
    
    current_count = count_dataset_images()
    
    with open(FILE_COUNT_PATH, "r") as f:
        old_count = int(f.read().strip())
    
    return current_count != old_count

def save_model():
    #Save model, optimizer state, and dataset count
    torch.save(model.state_dict(), MODEL_PATH)
    torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
    
    current_count = count_dataset_images()
    with open(FILE_COUNT_PATH, "w") as f:
        f.write(str(current_count))
    
    print(f"Model saved ({current_count} images in dataset)")

def load_model():
    if not os.path.exists(MODEL_PATH):
        return False
    
    try:
        # Check if model architecture matches saved model
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        
        if os.path.exists(OPTIMIZER_PATH):
            optimizer.load_state_dict(torch.load(OPTIMIZER_PATH, map_location=device, weights_only=True))
            print("Loaded model and optimizer")
        else:
            print("Loaded model (no optimizer state)")
        
        model.eval()
        return True
    except (RuntimeError, KeyError) as e:
        # Model architecture mismatch - this happens when switching model types
        print(f"Cannot load existing model (architecture mismatch)")
        print(f"This is normal when switching between custom CNN and transfer learning")
        print(f"Will train new model from scratch\n")
        
        # Delete incompatible saved model
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(OPTIMIZER_PATH):
            os.remove(OPTIMIZER_PATH)
        if os.path.exists(FILE_COUNT_PATH):
            os.remove(FILE_COUNT_PATH)
        
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


# =======================================================
#                   Training Function
# =======================================================

def train_model():
    global model, optimizer
    
    model_exists = load_model()
    
    if model_exists and not dataset_changed():
        print("Using existing model (no dataset changes)")
        return
    
    try:
        train_loader, test_loader = get_data_loaders()
        analyze_dataset()
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: {e}")
        if model_exists:
            print("Will use existing model for inference")
        return
    
    # Create scheduler here
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}\n")
    
    model.train()
    best_accuracy = 0.0
    patience_counter = 0
    max_patience = 8  # Early stopping
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 9:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/10:.4f}')
                running_loss = 0.0
        
        train_accuracy = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(test_loader)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        print(f'\nEpoch [{epoch+1}/{EPOCHS}] Summary:')
        print(f'  Train Accuracy: {train_accuracy:.2f}%')
        print(f'  Val Accuracy: {val_accuracy:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Learning Rate: {new_lr:.6f}', end='')
        if new_lr < old_lr:
            print(f' (reduced from {old_lr:.6f})')
        else:
            print()
        print()
        
        # Save best model with early stopping
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            save_model()
            print(f"New best model saved! Accuracy: {best_accuracy:.2f}%\n")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{max_patience})\n")
            
            if patience_counter >= max_patience:
                print(f"⚠️  Early stopping triggered - no improvement for {max_patience} epochs")
                break
        
        model.train()
    
    print(f"\n{'='*50}")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%")
    print(f"{'='*50}\n")

# =======================================================
#                   Inference Function
# =======================================================

def run_inference(image):
    #Actual model confidence from neural net output layer
    # Prepare image
    image = image.resize((256, 256))
    image_tensor = test_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

        # Fix: class 0 = fake, class 1 = real
        fake_prob = probs[0][0].item() * 100
        real_prob = probs[0][1].item() * 100

    print(f"Fake: {fake_prob:.2f}% | Real: {real_prob:.2f}%")
    return fake_prob
# =======================================================
#                   Standalone Testing
# =======================================================

if __name__ == '__main__':
    train_model()