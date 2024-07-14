import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import numpy as np
class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        # Convert image to tensor and normalize if needed
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)  # Assuming grayscale images
        return image_tensor

def create_dataloader(images, batch_size=32, shuffle=True):
    dataset = ImageDataset(images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def save_dataloader(dataloader, filepath):
    torch.save(dataloader.dataset, filepath)

def load_dataloader(filepath, batch_size=32, shuffle=True):
    dataset = torch.load(filepath)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def split_train_test(images, train_split=0.8):
    # Split images into train and test sets
    num_images = len(images)
    indices = list(range(num_images))
    split = int(np.floor(train_split * num_images))
    np.random.shuffle(indices)
    
    train_indices, test_indices = indices[:split], indices[split:]
    train_images = [images[i] for i in train_indices]
    test_images = [images[i] for i in test_indices]
    
    return train_images, test_images

def process_images(images, save_path, batch_size=32,split_ratio=0.8):
    # Split images into train and test sets
    train_images, test_images = split_train_test(images,split_ratio)
    
    # Create and save train DataLoader
    train_dataloader = create_dataloader(train_images, batch_size, shuffle=True)
    train_save_path = save_path + '_train.pth'
    save_dataloader(train_dataloader, train_save_path)
    print(f"Train DataLoader saved to {train_save_path}")
    
    # Create and save test DataLoader
    test_dataloader = create_dataloader(test_images, batch_size, shuffle=False)
    test_save_path = save_path + '_test.pth'
    save_dataloader(test_dataloader, test_save_path)
    print(f"Test DataLoader saved to {test_save_path}")
    
    #return train_dataloader, test_dataloader

