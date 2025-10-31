import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import os

# =======================================================
#                   Configuration
# =======================================================

# Training parameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3

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

# =======================================================
#                   Data Transforms & Loaders
# =======================================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
test_dataset = datasets.ImageFolder("dataset/test", transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE
)

# =======================================================
#                   Model Architecture
# =======================================================

class DeepfakeCNN(nn.Module):
    # Input: 256x256 images
    # Output: Binary classification (real/fake)
    def __init__(self):
        super(DeepfakeCNN, self).__init__()

        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        # Conv block 1: 256x256 -> 128x128
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        # Conv block 2: 128x128 -> 64x64
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        # Conv block 3: 64x64 -> 32x32
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
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

model = DeepfakeCNN().to(device)

# Use multiple GPUs if available
if device.type == "cuda" and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
    #Load model and optimizer from disk if available
    if not os.path.exists(MODEL_PATH):
        return False
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    if os.path.exists(OPTIMIZER_PATH):
        optimizer.load_state_dict(torch.load(OPTIMIZER_PATH, map_location=device))
        print("Loaded model and optimizer")
    else:
        print("Loaded model (no optimizer state)")
    
    model.eval()
    return True

# =======================================================
#                   Training Function
# =======================================================

def train_model():
    #Train the deepfake detection model
    #Skips training if model exists and dataset unchanged

    global model, optimizer
    
    # Check if training is needed
    model_exists = load_model()
    
    if model_exists and not dataset_changed():
        print("Using existing model (no dataset changes)")
        return
    
    if model_exists:
        print("Dataset changed - retraining model...")
    else:
        print("No saved model found - training new model...")
    
    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\nTraining complete!")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Test Loss: {avg_test_loss:.4f}\n")
    
    save_model()

# =======================================================
#                   Inference Function
# =======================================================

def run_inference(image):
    #Actual model confidence from neural net output layer
    # Prepare image
    image = image.resize((256, 256))
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        fake_prob = probs[0][1].item() * 100  # Class 1 = fake, convert to percentage
    
    print(f"Model confidence (fake): {fake_prob:.2f}%")
    return fake_prob

# =======================================================
#                   Standalone Testing
# =======================================================

if __name__ == '__main__':
    train_model()