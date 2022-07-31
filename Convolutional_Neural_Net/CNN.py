import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# config device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_eport = 4
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # z-score mean and sd are 0.5 (x-u)/sd

train_dataset = torchvision.datasets.CIFAR10(root='data',train=True, download=True, transform = transform)
test_dataset = torchvision.datasets.CIFAR10(root='data',train=False, download=True, transform = transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

def imshow(img):
    img = img/2+0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
# implement conv net

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2) #2 kenal size, shift 2 pixel
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120) #output shape ((Width  of imput - Filter_size +2Padding_size)/S)+1
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward (self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

loss_fun = nn.CrossEntropyLoss() #for multimple classifications

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for eport in range(num_eport):
    for i, (images, labels) in enumerate(tqdm(train_loader, leave=True)):
        # original shape: [4,3,32,32] = 4,3,1024
        # input_layer: 3 input channel (RGB), 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # forward (feed data)
        output = model(images)
        loss = loss_fun(output, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Eport   [{eport}/{num_eport}], step: {i+1}/{n_total_steps}, Loss: {loss.item():.4f}')

print('Finished Training')

with torch.no_grad():
    n_correct = 0
    n_sample = 0
    n_class_correct = [0 for i in range(10)]
    n_class_sample = [0 for i in range(10)]

    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)

        # forward pass (feed data in CNN)
        _, predicted = torch.max(output,1)
        n_sample += labels.size(0)
        n_correct = (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] +=1
            n_class_sample[label] +=1

    acc = 100 * (n_correct/n_sample)
    print(f' accuracy of the model: {acc}%')

        # for i in range(10):
        #     acc = (n_class_correct[i]/ n_class_sample[i]) * 100.0
        #     print(f' Accuracy of {classes[i]:{acc} %}')
#https://github.com/python-engineer/pytorchTutorial/blob/master/14_cnn.py
