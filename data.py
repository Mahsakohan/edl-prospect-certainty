import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Random crop for data augmentation
    transforms.RandomHorizontalFlip(),  # Random horizontal flip for data augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color jitter
    transforms.RandomRotation(15),  # Random rotation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
data_train = CIFAR10("./data/cifar10", download=False, train=True, transform=transform_train)
data_val = CIFAR10("./data/cifar10", download=False, train=False, transform=transform_test)

# Create DataLoader for training and validation
dataloader_train = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=8)
dataloader_val = DataLoader(data_val, batch_size=128, num_workers=8)

dataloaders = {
    "train": dataloader_train,
    "val": dataloader_val,
}
