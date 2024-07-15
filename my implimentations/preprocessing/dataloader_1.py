from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from PIL import Image, ImageFilter
import os
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, lr_input, hr_input, transform=None, normalize=False):
        if isinstance(lr_input, str) and isinstance(hr_input, str):
            # Input is directories
            self.lr_files = [os.path.join(lr_input, f) for f in os.listdir(lr_input)]
            self.hr_files = [os.path.join(hr_input, f) for f in os.listdir(hr_input)]
        elif isinstance(lr_input, list) and isinstance(hr_input, list):
            # Input is lists of images
            self.lr_files = lr_input
            self.hr_files = hr_input
        else:
            raise ValueError("Inputs should either be directories or lists of images.")

        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return min(len(self.lr_files), len(self.hr_files))

    def __getitem__(self, idx):
        if isinstance(self.lr_files[idx], str) and isinstance(self.hr_files[idx], str):
            lr_image = Image.open(self.lr_files[idx]).convert('RGB')
            hr_image = Image.open(self.hr_files[idx]).convert('RGB')
        else:
            lr_image = self.lr_files[idx].convert('RGB')
            hr_image = self.hr_files[idx].convert('RGB')

        lr_image = lr_image.filter(ImageFilter.GaussianBlur(radius=1))

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        if self.normalize:
            lr_image = self._normalize(lr_image)
            hr_image = self._normalize(hr_image)

        return lr_image, hr_image

    def _normalize(self, image):
        # Assuming image is a tensor
        return (image - 0.5) / 0.5  # Normalize to [-1, 1]

    def split_train_test_indices(self, split_ratio=0.7, shuffle=True, random_seed=42):
        dataset_size = len(self)
        indices = list(range(dataset_size))
        split = int(np.floor(split_ratio * dataset_size))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices, test_indices = indices[:split], indices[split:]

        return train_indices, test_indices

    def get_train_loader(self, batch_size=64, num_workers=4, split_ratio=0.7, shuffle=True, random_seed=42):
        train_indices, _ = self.split_train_test_indices(split_ratio, shuffle, random_seed)
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        return train_loader

    def get_test_loader(self, batch_size=64, num_workers=4, split_ratio=0.7, shuffle=True, random_seed=42):
        _, test_indices = self.split_train_test_indices(split_ratio, shuffle, random_seed)
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoader(self, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
        return test_loader

    def save_dataloader(self, dataloader, filepath):
        torch.save(dataloader.dataset, filepath)

    def load_dataloader(filepath, batch_size=32, shuffle=True):
        dataset = torch.load(filepath)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    

class ImageDataset(Dataset):
    def __init__(self, images, transform=None, normalize=False):
        self.images = images
        self.transform = transform
        self.normalize = normalize
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image_np = np.array(image)
        
        if self.transform:
            image = self.transform(image)
            
        if self.normalize:
            image = (image - 0.5) / 0.5  # Assuming images are in range [0, 1], normalize to [-1, 1]
        
        return image

def create_dataloader(images, batch_size=32, shuffle=True):
    dataset = ImageDataset(images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def save_dataloader(dataloader, filepath):
    torch.save(dataloader.dataset, filepath)

def load_dataloader(filepath, batch_size=32, shuffle=True):
    dataset = torch.load(filepath)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def split_train_test(images, train_split=0.8):
    num_images = len(images)
    indices = list(range(num_images))
    split = int(np.floor(train_split * num_images))
    np.random.shuffle(indices)
    
    train_indices, test_indices = indices[:split], indices[split:]
    train_images = [images[i] for i in train_indices]
    test_images = [images[i] for i in test_indices]
    
    return train_images, test_images

def flatten_nested_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def process_images(lr_input, hr_input, save_path, batch_size=64, tts_ratio=0.9, normalize=True):
    # Create CustomDataset with normalization provision
    dataset = CustomDataset(lr_input, hr_input, normalize=normalize)
    
    # Split dataset into train and test sets
    train_indices, test_indices = dataset.split_train_test_indices(split_ratio=tts_ratio)
    
    # Create train DataLoader
    train_sampler = SubsetRandomSampler(train_indices)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    
    # Create test DataLoader
    test_sampler = SubsetRandomSampler(test_indices)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    # Save train DataLoader
    train_save_path = save_path + '_train.pth'
    torch.save(train_dataloader.dataset, train_save_path)
    print(f"Train DataLoader saved to {train_save_path}")
    
    # Save test DataLoader
    test_save_path = save_path + '_test.pth'
    torch.save(test_dataloader.dataset, test_save_path)
    print(f"Test DataLoader saved to {test_save_path}")
    
    return train_dataloader, test_dataloader