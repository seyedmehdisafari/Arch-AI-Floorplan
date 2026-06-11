import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class RPlanDataset(Dataset):
    def __init__(self, data_dir, img_size=(256, 256)):
        """
        Args:
            data_dir (str): Path to the folder containing RPlan images (.png)
            img_size (tuple): Target size to resize the images for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        
        # Get all .png image filenames from the directory
        self.image_filenames = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        
        # Base transforms to convert PIL Images to PyTorch Tensors
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # 1. Load image
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 2. Extract Walls (Input) and Rooms (Target)
        img_np = np.array(image)
        
        # In RPlan, walls are typically very dark pixels (close to black)
        # Thresholding pixels where R, G, B are all less than 50
        walls_mask = np.all(img_np < 50, axis=-1).astype(np.float32)
        walls_img = Image.fromarray((walls_mask * 255).astype(np.uint8))
        
        # 3. Apply transformations
        x_input = self.transform(walls_img)  # Shape: [1, H, W] - Grayscale wall mask
        y_target = self.transform(image)     # Shape: [3, H, W] - Full colored floorplan

        return x_input, y_target