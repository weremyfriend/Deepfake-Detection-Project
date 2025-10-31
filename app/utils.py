
# test and utility functions in case we upload from an image folder (could be used for batch images)

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

# =======================================================
#                  Helper Functions
# =======================================================

def img_to_tensor(image):
    # Convert PIL Image to PyTorch tensor
    t_form = np.array(image)
    t_form.resize(65536)
    t_form.reshape(256, 256)
    t_img = torch.from_numpy(t_form)
    return t_img

def tensor_array(images: list):
    # Convert list of PIL Images to list of tensors
    tens_arr = []
    for each in images:
        t_add = img_to_tensor(each)
        tens_arr.append(t_add)
    return tens_arr

# =======================================================
#                  Custom Dataset Class
# =======================================================

class ImagesDataset(Dataset):
    # Custom PyTorch Dataset for loading images
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
            i_to_t = i_to_t.to(self.device)
        return i_to_t