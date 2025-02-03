import torch
from torch.utils.data import Subset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, indices, transform=None):
        self.original_dataset = Subset(original_dataset, indices)
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label