import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchvision.datasets import CelebA
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import LSUN
def get_loader(batch_s,res, data="FashionMNIST"):

    # Define transformations
    if data == "FashionMNIST":
        transform = transforms.Compose([
            transforms.Resize(res), 
            transforms.ToTensor(),  
            transforms.Normalize([0.5], [0.5])  # Normalize between -1 and 1
        ])

        # Download and load the dataset
        dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)

        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_s, drop_last=True, shuffle=True, num_workers=1)
        return dataloader
    elif data == "CelebA":
        transform = transforms.Compose([
        transforms.CenterCrop(178),  # make it square
        transforms.Resize(res),
        transforms.ToTensor(), 
        transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        # Download and load the dataset
        dataset = datasets.ImageFolder(root="./data/celeba", transform=transform)
        #dataset = CelebA(root="./data", split="train", download=True, transform=transform)
        #dataset = load_dataset("celeba", split="train[:1000]")  # first 1000 images
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_s, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)
        return dataloader
    elif data == "STL10":
        transform = transforms.Compose([
        transforms.Resize(res),  # Resize to 128x128 (common for GANs)
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.5]*3, [0.5]*3)
      
        ])

        # Download and load the dataset
        dataset = STL10(root="./data", split="unlabeled", download=True, transform=transform)
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_s, drop_last=True, shuffle=True, num_workers=0, pin_memory=True)
        return dataloader
    elif data == "Church":
        transform = transforms.Compose([
        transforms.Resize(res),  # Resize to 128x128 (common for GANs)
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.5]*3, [0.5]*3)
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize RGB channels
        ])

        # Download and load the dataset
        dataset = LSUN(root="./data",classes=["church_outdoor_train"], transform=transform)
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_s, drop_last=True, shuffle=True, num_workers=1, pin_memory=True)
        return dataloader
    else:
        return "error"