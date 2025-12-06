import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .config import DROPOUT_RATE


class DeepfakeCNN(nn.Module):
    """
    Custom CNN built from scratch for deepfake detection
    Architecture: 6 convolutional layers + 3 fully connected layers
    Input: 256x256 RGB images
    Output: 2 classes (real or fake)
    """
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        
        # First conv block (3 to 64 channels)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Extract basic features
        self.bn1 = nn.BatchNorm2d(64)  # Normalize for stable training
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # Deepen features
        self.bn2 = nn.BatchNorm2d(64)
        
        # Second conv block (64 to 128 channels)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # More complex patterns
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Third conv block (128 to 256 channels)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # High-level features
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        # Global average pooling (reduces spatial dimensions to 1x1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(256, 512)  # First dense layer
        self.fc2 = nn.Linear(512, 128)  # Second dense layer
        self.fc3 = nn.Linear(128, 2)    # Output layer (real vs fake)
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        """
        Forward pass through the network
        x: input image tensor (batch_size, 3, 256, 256)
        returns: raw scores for 2 classes (batch_size, 2)
        """
        # Block 1: Apply convolutions, normalize, activate, then pool
        # Size: 256x256 to 128x128
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # Reduce spatial size by half
        
        # Block 2: Size 128x128 to 64x64
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 3: Size 64x64 -> 32x32
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2)
        
        # Global pooling: 32x32 to 1x1
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to 1D vector
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Randomly zero out some neurons
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Final output (no activation, done in loss function)
        
        return x


class TransferLearningCNN(nn.Module):
    """
    Transfer learning model using pretrained EfficientNet-B0
    Only trains the final classification layers (faster + more accurate)
    """
    def __init__(self, freeze_backbone=True):
        super(TransferLearningCNN, self).__init__()
        
        # Load EfficientNet pretrained on ImageNet
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        
        if freeze_backbone:
            # Freeze early layers
            # Only train the last 40 layers to adapt to deepfake detection
            for param in list(self.backbone.parameters())[:-40]:
                param.requires_grad = False
            print("Backbone frozen (last 40 layers trainable)")
        else:
            # Train the entire network (slower but can be more accurate)
            print("Full backbone training")
        
        # Replace the classifier head with our custom layers
        num_features = self.backbone.classifier[1].in_features  # Get input size
        
        # Custom classification head for deepfake detection
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.6),              # Aggressive dropout
            nn.Linear(num_features, 256), # Reduce dimensions
            nn.ReLU(),                    # Activation
            nn.BatchNorm1d(256),          # Normalize
            nn.Dropout(0.5),              # More dropout
            nn.Linear(256, 64),           # Further reduction
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 2)              # Final output: real vs fake
        )
    
    def forward(self, x):
        """
        Forward pass using EfficientNet backbone
        x: input image tensor
        returns: raw scores for 2 classes
        """
        return self.backbone(x)