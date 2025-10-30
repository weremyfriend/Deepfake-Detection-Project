from torch import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
from torchvision import *
import numpy as np
import random
from PIL import Image
import os

import sys

MY_UTILS_PATH = './app'
if not MY_UTILS_PATH in sys.path:
    sys.path.append(MY_UTILS_PATH)


#import torch.utils.data.dataloader
from torch.utils.data import DataLoader
#import torch.utils.data.dataset
from torch.utils.data import Dataset

# =======================================================
#                 Initial Parameters
# =======================================================

# Initial value for Noise Dimension
d_noise = 100

# Number of epochs used for training
tot_epochs = 50

# Max Batch
max_batch_size = 512

# Rate of Learning
l_rate = 0.001

# Check if NVIDIA GPU is available for use
use_cuda = torch.cuda.is_available()

# GPU's in machine
num_gpu = 1

# Decide what the device will run
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

# Prevent deterministic outputs and algorithms
torch.use_deterministic_algorithms(False)

# Number of examples
num_examples = 32

# Random seed for use
r_seed = torch.normal(mean=0, std=d_noise, size = (num_examples,))

# Load training images
images_from_kaggle = []

for foldername, subfolders, filenames in os.walk('./van-gogh-dataset'):
    for filename in filenames:
        if filename.endswith('.jpg'):
            file_path = os.path.join(foldername, filename)
            
            # Load the image
            image = Image.open(file_path)

            # Convert to NumPy array and add to image_arrays
            np_image = np.array(image)
            np_image.resize(65536)
            np_image.reshape(256, 256)
            images_from_kaggle.append(np_image)

# Convert list of images to a NumPy array (ensure all images have the same size)
image_numpy_arrays = np.stack(images_from_kaggle)

image_arrays = torch.from_numpy(image_numpy_arrays)

# =======================================================
#                  Helper Functions
# =======================================================

def model_probability_opinion(opinions: list): # Used to calculate how many times out of length(opinions) the discriminator detected an image as fake
    dfd_average = 0 # Initial value of zero
    cycle_count = 0
    while cycle_count < len(opinions):
        if opinions[cycle_count] == False:
            dfd_average += 1
        cycle_count += 1
    return (dfd_average/len(opinions)) * 100

def img_to_tensor(image):
    t_form = np.array(image)
    t_form.resize(65536)
    t_form.reshape(256, 256)
    t_img = torch.from_numpy(t_form)
    return t_img

def tensor_array(images: list):
    tens_arr = []
    for each in images:
        t_add = img_to_tensor(each)
        tens_arr.append(t_add)
    return tens_arr

class images_data(Dataset):
    def __init__(self, images, transform=None, device=None):
        self.images = images
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        image = self.images[idx]

        i_to_t = img_to_tensor(image)
        if self.device:
            i_to_t=i_to_t.to(self.device)
        return i_to_t

#Uncomment if needed for testing
#test_img_dir = './background.jpg'
#if not os.path.exists(test_img_dir):
    #raise FileNotFoundError(f"Directory '{test_img_dir}' does not exist")
#for image_name in os.listdir(test_img_dir):
    #image_path = os.path.join(test_img_dir, image_name)   
    #if os.path.isfile(image_path):
        #image = Image.open(image_path).convert('RGB')
        #image_arrays.append(image)   



dataset = images_data(image_arrays, device=device)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


'''
def weights_init(weights_inst):
    c_name = weights_inst.__class__.__name__
    if c_name.find('Conv') != -1:
        nn.init.normal_(weights_inst.weight.data, 0.0, 0.02)
    elif c_name.find('BatchNorm') != -1:
        nn.init.normal_(weights_inst.weight.data, 1.0, 0.02)
        nn.init.constant_(weights_inst.bias.data, 0)
'''
# =======================================================
#                      Model
# =======================================================

class Defaker_CNN(nn.Module):
    def __init__(self):
        super(Defaker_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, discrim_in):
        discrim_in = self.pool(F.relu(self.conv1(discrim_in)))
        discrim_in = self.pool(F.relu(self.conv2(discrim_in)))
        discrim_in = discrim_in.view(-1, 16 * 5 * 5)
        discrim_in = F.relu(self.fc1(discrim_in))
        discrim_in = F.relu(self.fc2(discrim_in))
        discrim_in = self.fc3(discrim_in)
        return discrim_in
    
    def loss(self, real, fake):

        real_labels = torch.ones_like(real)
        real_loss = self.loss_fn(real, real_labels)

        fake_labels = torch.zeros_like(fake)
        fake_loss = self.loss_fn(fake, fake_labels)

        total_loss = real_loss + fake_loss
        return total_loss
        
defaker = Defaker_CNN()

# Setup use of multiple GPU's if present and capable
if (device.type == "cuda") and (num_examples >1):
        defaker = torch.nn.parallel.DistributedDataParallel(defaker)

x_entropy = nn.CrossEntropyLoss()
optimizer = optim.SGD(defaker.parameters(), lr=l_rate, momentum=0.9)  
    

for epoch in torch.arange(0, tot_epochs):
    running_loss = 0
    for i, data in enumerate(dataloader, 0):
        print(data)
        inputs, labels = data
        optimizer.zero_grad()
        output = defaker(inputs)
        loss = x_entropy(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0

print("Finished Training")

# =======================================================

def run_model(image): 
    # Initialize Weights

    #Display Architecture in terminal for testing purposes, not shown to user
    print("Defaker_CNN Architecture:")
    print(defaker)

    # Convert image to tensor
    tensor_img = img_to_tensor(image)
    
    #noise = torch.randn(max_batch_size, d_noise, 1, 1, device=device)
    #real_images = torch.randn(max_batch_size,3, 256, 256, device=device)
    dfd_opinions = []
    with torch.no_grad():
        for _ in range(0, 50):
            opinion_value = defaker(tensor_img)
            _, test_val = torch.max(opinion_value.data, 1)
            if test_val.item() > 0.5:
                dfd_opinions.append(True)
            else:
                dfd_opinions.append(False)
    return model_probability_opinion(dfd_opinions)
    #return "model ran without errors"

if __name__ == '__main__':
    test_img_dir = './ui/background.jpg'
    test_img = Image.open(test_img_dir)
    test_num = run_model(test_img)
    print(test_num)