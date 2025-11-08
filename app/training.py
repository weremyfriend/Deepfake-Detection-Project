import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import os

from .config import *
from .model import DeepfakeCNN, TransferLearningCNN
from .utils import clean_dataset


# ============================================
# DATA PREPROCESSING
# ============================================
# Training transforms (with augmentation to prevent overfitting)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),      # Resize all images to same size
    transforms.RandomHorizontalFlip(),  # Randomly flip images (data augmentation)
    transforms.ToTensor(),              # Convert to PyTorch tensor
    transforms.Normalize(               # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],    # Standard mean for RGB channels
        std=[0.229, 0.224, 0.225]       # Standard deviation
    )
])

# Test transforms (no augmentation, just normalize)
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================
# MODEL INITIALIZATION
# ============================================
# Create the model based on config setting
if USE_TRANSFER_LEARNING:
    model = TransferLearningCNN(freeze_backbone=False).to(device)
else:
    model = DeepfakeCNN().to(device)

# Use multiple GPUs if available
if device.type == "cuda" and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

# Loss function (measures how wrong predictions are)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# Optimizer (updates weights to minimize loss)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# ============================================
# DATASET ANALYSIS
# ============================================
def analyze_dataset():
    """Print info on training and test datasets"""
    print("\n" + "="*50)
    print("Dataset Info")
    print("="*50)
    
    total_train = 0
    total_test = 0
    
    # Count images
    for split, path in [("Train", TRAIN_PATH), ("Test", TEST_PATH)]:
        print(f"\n{split}:")
        for label in ['real', 'fake']:
            class_path = os.path.join(path, label)
            if os.path.exists(class_path):
                # Count image files
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {label}: {count} images")
                if split == "Train":
                    total_train += count
                else:
                    total_test += count
    
    # Print totals and percentages
    total = total_train + total_test
    print(f"\nTotal: {total} images")
    print(f"  Train: {total_train} ({total_train/total*100:.1f}%)")
    print(f"  Test: {total_test} ({total_test/total*100:.1f}%)")
    print("="*50 + "\n")
    
    # Warn if dataset is too small
    if total_train < 500:
        print("Warning: Less than 500 training images")
    if total_test < 100:
        print("Warning: Less than 100 test images")


# ============================================
# DATA LOADING
# ============================================
def get_data_loaders():
    """
    Load images from disk
    Returns: (train_loader, test_loader)
    """
    # Check if dataset folders exist
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Dataset directories not found. Need:\n"
            f"  - {TRAIN_PATH}/fake/ and {TRAIN_PATH}/real/\n"
            f"  - {TEST_PATH}/fake/ and {TEST_PATH}/real/"
        )
    
    # Remove corrupted or non-images before loading
    clean_dataset(TRAIN_PATH)
    clean_dataset(TEST_PATH)
    
    # Load datasets (automatically labels based on folder structure)
    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_PATH, transform=test_transform)
    
    # Make sure there are images
    if len(train_dataset) == 0:
        raise ValueError(f"No images in {TRAIN_PATH}")
    if len(test_dataset) == 0:
        raise ValueError(f"No images in {TEST_PATH}")
    
    # Create data loaders (batch images for training)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True  # Randomize order each epoch
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE
    )
    
    return train_loader, test_loader


# ============================================
# MODEL PERSISTENCE
# ============================================
def count_dataset_images():
    # Count total images in train and test folders
    count = 0
    for folder in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(folder):
            continue
        for root, _, files in os.walk(folder):
            count += sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    return count


def dataset_changed():
    """
    Check if dataset has changed since last training
    Returns True if images were added or removed
    """
    # No previous count file means first time
    if not os.path.exists(FILE_COUNT_PATH):
        return True
    
    # Compare current count to saved count
    current_count = count_dataset_images()
    with open(FILE_COUNT_PATH, "r") as f:
        old_count = int(f.read().strip())
    
    return current_count != old_count


def save_model():
    """Save model weights, optimizer state, and dataset count"""
    # Save model weights
    torch.save(model.state_dict(), MODEL_PATH)
    
    # Save optimizer state (for resuming training)
    torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
    
    # Save dataset size (to detect changes)
    current_count = count_dataset_images()
    with open(FILE_COUNT_PATH, "w") as f:
        f.write(str(current_count))
    
    print(f"Model saved ({current_count} images in dataset)")


def load_model():
    """
    Load saved model from disk
    Returns True if successful, False otherwise
    """
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        return False
    
    try:
        # Load model weights
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        
        # Load optimizer state if available
        if os.path.exists(OPTIMIZER_PATH):
            optimizer.load_state_dict(torch.load(OPTIMIZER_PATH, map_location=device, weights_only=True))
            print("Loaded model and optimizer")
        else:
            print("Loaded model only")
        
        model.eval()  # Set to evaluation mode
        return True
        
    except (RuntimeError, KeyError):
        # This happens when switching between model architectures
        print("Can't load model - architecture mismatch")
        print("Training new model from scratch\n")
        
        # Delete incompatible files
        for path in [MODEL_PATH, OPTIMIZER_PATH, FILE_COUNT_PATH]:
            if os.path.exists(path):
                os.remove(path)
        
        return False
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


# ============================================
# TRAINING
# ============================================
def train_model():
    """
    Main training loop
    Loads existing model if available and dataset unchanged
    Otherwise trains from scratch
    Saves best model based on validation accuracy
    """
    global model, optimizer
    
    # Try to load existing model
    model_exists = load_model()
    
    # Skip training if model exists and dataset unchanged
    if model_exists and not dataset_changed():
        print("Using existing model (no dataset changes)")
        return
    
    # Load data
    try:
        train_loader, test_loader = get_data_loaders()
        analyze_dataset()
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: {e}")
        if model_exists:
            print("Using existing model")
        return
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',  # Minimize validation loss
        factor=LR_REDUCE_FACTOR,  # Multiply LR by this when stuck
        patience=LR_REDUCE_PATIENCE  # Wait this many epochs before reducing
    )
    
    print(f"Training for {EPOCHS} epochs")
    print(f"LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}\n")
    
    model.train()  # Set to training mode
    best_accuracy = 0.0
    patience_counter = 0  # For early stopping
    
    # Train for multiple epochs
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase: iterate through all batches
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(inputs)  # Get predictions
            loss = criterion(outputs, labels)  # Calculate loss
            
            # Backward pass
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 10 batches
            if i % 10 == 9:
                print(f'Epoch {epoch+1}/{EPOCHS}, Step {i+1}/{len(train_loader)}, Loss: {running_loss/10:.4f}')
                running_loss = 0.0
        
        train_accuracy = 100 * correct / total
        
        # Validation phase: test on unseen data
        model.eval()  # Set to evaluation mode (disables dropout)
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():  # Don't track gradients (faster)
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
        
        # Update learning rate if stuck
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{EPOCHS}:')
        print(f'  Train: {train_accuracy:.2f}% | Val: {val_accuracy:.2f}%')
        print(f'  Loss: {avg_val_loss:.4f} | LR: {new_lr:.6f}')
        
        # Save if this is the best model so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            save_model()
            print(f"  New best: {best_accuracy:.2f}%\n")
        else:
            # No improvement
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{MAX_PATIENCE})\n")
            
            # Stop early if no improvement for too long
            if patience_counter >= MAX_PATIENCE:
                print("Early stopping - no improvement")
                break
        
        model.train()  # Back to training mode
    
    print(f"\nDone! Best accuracy: {best_accuracy:.2f}%\n")


# ============================================
# INFERENCE
# ============================================
def run_inference(image):
    """
    Run inference on a single image
    Takes a PIL image object and returns fake probability
    """
    # Preprocess image
    image = image.resize((256, 256))
    image_tensor = test_transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Run through model
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)  # Convert to probabilities
        
        # PREVENT BUGS
        # Get probabilities for each class
        # Note: class 0 = fake, class 1 = real
        fake_prob = probs[0][0].item() * 100
        real_prob = probs[0][1].item() * 100

    print(f"Fake: {fake_prob:.2f}% | Real: {real_prob:.2f}%")
    return fake_prob


# Run training if this file is executed directly
if __name__ == '__main__':
    train_model()