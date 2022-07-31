import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WineDataset(Dataset):
    def __init__(self,transform=None ):
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=',',dtype=np.float32,skiprows = 1)
        self.x = xy[:,1:] # training_data don't need the first column #
        self.y = xy[:,[0]] # label or target_data n_feature = 1
        self.n_samples = xy.shape[0]
        # self.n_features = xy.shape[1]
        self.transform = transform

    def __getitem__(self, index):
        # dataset[]
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # len()
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
feature, label = first_data
# print(type(feature))
# print(type(label))
print(feature)

composed = torchvision.transforms.Compose([ToTensor(),MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
feature, label = first_data
# print(type(feature))
# print(type(label))
print(feature)