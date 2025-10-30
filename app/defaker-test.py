from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import os
import sys
import numpy as np
import time


# =======================================================
#                   Initialization
# =======================================================
MY_UTILS_PATH = './app'
if MY_UTILS_PATH not in sys.path:
    sys.path.append(MY_UTILS_PATH)

app = FastAPI()

# Serve the UI folder
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# =======================================================
#                   Initial Parameters
# =======================================================
d_noise = 100
tot_epochs = 10
max_batch_size = 512
l_rate = 0.001
use_cuda = torch.cuda.is_available()
num_gpu = torch.cuda.device_count()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.use_deterministic_algorithms(False)
num_examples = 32

# =======================================================
#                   Helper Function
# =======================================================
def model_probability_opinion(opinions: list):
    """Calculate % of times model classified image as fake"""
    return (opinions.count(False) / len(opinions)) * 100

# =======================================================
#                   Data Loading
# =======================================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
test_dataset = datasets.ImageFolder("dataset/test", transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

classes = ('real', 'fake')


# =======================================================
#                   Model Definition
# =======================================================
# #Original model
# def __init__(self):
#         super(Defaker_CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
        
# def forward(self, discrim_in):
#         discrim_in = self.pool(F.relu(self.conv1(discrim_in)))
#         discrim_in = self.pool(F.relu(self.conv2(discrim_in)))
#         discrim_in = discrim_in.view(-1, 16 * 5 * 5)
#         discrim_in = F.relu(self.fc1(discrim_in))
#         discrim_in = F.relu(self.fc2(discrim_in))
#         discrim_in = self.fc3(discrim_in)
#         return discrim_in
    
# def loss(self, real, fake):

#         real_labels = torch.ones_like(real)
#         real_loss = self.loss_fn(real, real_labels)

#         fake_labels = torch.zeros_like(fake)
#         fake_loss = self.loss_fn(fake, fake_labels)

#         total_loss = real_loss + fake_loss
#         return total_loss

#version 1
# class Defaker_CNN(nn.Module):
#     def __init__(self):
#         super(Defaker_CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 61 * 61, 120)  # adjusted for 256x256 input
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 2)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

#version 2
class Defaker_CNN(nn.Module):
    def __init__(self):
        super(Defaker_CNN, self).__init__()

        # 1st Convolution Block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 2nd Convolution Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 3rd Convolution Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# =======================================================
#                   Model Setup
# =======================================================

#Path to trained models  - 10/29/2025 Walker Hall
model_path = "saved_models/defaker_cnn.pth"
#Path to optimizer - 10/29/2025 Walker Hall
optimizer_path = "saved_models/defaker_optimizer.pth"
os.makedirs("saved_models", exist_ok=True)

defaker = Defaker_CNN().to(device)

if device.type == "cuda" and num_gpu > 1:
    defaker = nn.DataParallel(defaker)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(defaker.parameters(), lr=l_rate)



# Helper function to check if dataset changed; ensures model is not retrained on the same data -- 10/29/2025 - Walker Hall
def dataset_changed(count_file="saved_models/file_count.txt"):
    """Check if number of images has changed"""
    current_count = 0
    
    for folder in ["dataset/train", "dataset/test"]:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    current_count += 1
    
    if not os.path.exists(count_file):
        return True
    
    with open(count_file, "r") as f:
        old_count = int(f.read().strip())
    
    return current_count != old_count


# # Checks to see if there is a saved model
# if os.path.exists(model_path):
#     # Load existing model (no training needed)
#     defaker.load_state_dict(torch.load(model_path, map_location=device))
#     # Tries to load optimizer state
#     if os.path.exists(optimizer_path):
#         optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
#     print("Loaded optimizer state from disk (resuming where  it was left off).")
#     defaker.eval()
#     print("Loaded saved model from disk. No training needed.")
# else:
#     # No saved model, training new model
#     print("No saved model found, training new model...")



need_retrain = False

if os.path.exists(model_path):
    defaker.load_state_dict(torch.load(model_path, map_location=device))
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        print("Loaded optimizer state from disk (resuming where you left off).")
    defaker.eval()
    print("Loaded saved model from disk.")
    # Checks for dataset updates
    if dataset_changed():
        print("New or modified images detected — retraining...")
        need_retrain = True
    else:
        print("No new images found — skipping retraining.")
else:
    print("No saved model found training new model...")
    need_retrain = True

# =======================================================
#                   Training Loop
# =======================================================
# print("Starting training...")
# for epoch in range(tot_epochs):
#     defaker.train()
#     total_loss = 0.0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = defaker(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch {epoch+1}/{tot_epochs}, Loss: {avg_loss:.4f}")

# print("Finished Training")

if need_retrain:  #only train if needed
    print("Starting training...")
    for epoch in range(tot_epochs):
        defaker.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = defaker(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{tot_epochs}, Loss: {avg_loss:.4f}")

    print("Finished Training")

    # =======================================================
    #                   Model Evaluation (Testing)
    # =======================================================
    defaker.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = defaker(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Get predicted class (0 = real, 1 = fake)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}% | Avg Test Loss: {avg_test_loss:.4f}\n")

    # Save model + optimizer + file count
    torch.save(defaker.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # Count and save current file count
    current_count = 0
    for folder in ["dataset/train", "dataset/test"]:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    current_count += 1
    
    with open("saved_models/file_count.txt", "w") as f:
        f.write(str(current_count))
    
    print(f"Model, optimizer, and file count ({current_count} images) saved successfully.")
else:
    print("Using previously trained model (no retraining performed).")

# =======================================================
#                   Model Inference
# =======================================================
# def run_model(image): 
#     image = image.resize((256, 256))
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     defaker.eval()
#     opinions = []

#     with torch.no_grad():
#         # for _ in range(50):
#             outputs = defaker(image_tensor)
#             probs = F.softmax(outputs, dim=1)
#             #pred = torch.argmax(probs, dim=1).item()
#             # opinions.append(pred == 1)
#             fake_prob = probs[0][1].item() * 100  # Convert to percentage 10/30/2025
#             #Actual model confidence from the neural net output layer

#     # fake_prob = model_probability_opinion(opinions)
#     print(f"Model Certainty that image is fake: {fake_prob:.2f}%")
#     return fake_prob

def run_model(image): 
    # Resize and transform the input image
    image = image.resize((256, 256))
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Set model to evaluation mode
    defaker.eval()

    with torch.no_grad():
        # Run the model once on the image
        outputs = defaker(image_tensor)
        probs = F.softmax(outputs, dim=1)

        # Get the probability that the image is fake (class index 1)
        fake_prob = probs[0][1].item() * 100  # Convert to percentage

    print(f"Model Certainty that image is fake: {fake_prob:.2f}%")
    return fake_prob
# =======================================================
#                   FastAPI Setup
# =======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
        <html>
            <body>
                <h2>Use /model endpoint to upload and test an image</h2>
            </body>
        </html>
    """

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

@app.post("/model")
async def run_model_with_image(file: UploadFile = File(...)):
    file_location = UPLOAD_FOLDER / file.filename
    try:
        with open(file_location, "wb") as f:
            f.write(await file.read())
        test_img = Image.open(file_location).convert("RGB")
        fake_score = run_model(test_img)
        return {"fake_probability_percent": fake_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

# Uncomment if running directly
# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)