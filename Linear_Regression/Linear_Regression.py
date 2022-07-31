import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

#   0) prepare data
from torch.nn import Linear

x_numpy, y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)
X = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
# print(y.shape)
y = y.view(y.shape[0],1) #keep row shape, column turn chape to 1
# print(y.shape)
n_sample, n_feature = X.shape

#   1)  model
input_size = n_feature
output_size = 1
model: Linear = nn.Linear(input_size, output_size)


#   2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#   3) Training loop
num_eport = 100
for i in range(num_eport):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted,y)
    loss.backward()
    # backward pass
    optimizer.step()
    optimizer.zero_grad()
    # update
    if (i+1) %10 ==0:
        [slope,intercept] = model.parameters()
        print(f'eprot: {i+1}    Loss: {loss.item():.4f}     slope: {slope[0][0].item():.4f}    intercept: {intercept[0].item():.4f} ')

# plot
predicted = model(X).detach() # return a new tensor and without gradient
plt.plot(x_numpy,y_numpy,'ro')
plt.plot(x_numpy,predicted,'b')
plt.show()
















