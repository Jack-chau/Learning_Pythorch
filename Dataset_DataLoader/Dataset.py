'''
epoch = 1 forward and backward pass of all training samples

batch_size = number of training samples in one forward & backward pass

number of iterations = number of passes, each pass using [batch_size] number of sample

e.g. 100 samples, batch_size =20 --> 100/20 = 5 interations for 1 eport

'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pandas as pd
# create own dataset

# url ='https://raw.githubusercontent.com/python-engineer/pytorchTutorial/master/data/wine/wine.csv'
# wine = pd.read_csv(url)
# # print(wine.head())
# wine_df = pd.DataFrame(wine)
# print(wine_df.shape)

class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=',',dtype=np.float32,skiprows = 1)
        self.x = torch.from_numpy(xy[:,1:]) # training_data don't need the first column #
        self.y = torch.from_numpy(xy[:,[0]]) # label or target_data n_feature = 1
        self.n_samples = xy.shape[0]
        # self.n_features = xy.shape[1]

    def __getitem__(self, index):
        # dataset[]
        return self.x[index], self.y[index]

    def __len__(self):
        # len()
        return self.n_samples

dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# training loop
num_eport = 2
total_sample = len(dataset)
n_interations = math.ceil(total_sample/4) #n_iterations = total_sample/batch_size
# math.ceil is round up
# print(total_sample, n_interations)

for eport in range(num_eport):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if (i+1) %5 ==0:
            print(f'eport {eport+1}/{num_eport}, step {i+1}/{n_interations} , inputs{inputs.shape}')

# Famous dataset
# torchvision.datasets.MNIST()
#fashion_nmist, cifar, coco