import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)
x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('outputs numpy: ', outputs)

x = torch.tensor([2.0,1.0,0.1])
outputs = torch.softmax(x,dim=0)
print(outputs)

def cross_entropy(actual, predicted):
    loss = -np.sum(actual*np.log(predicted))
    return loss #/float(predicted.shape[0]) #for normalization

# create y, must be onehot encoded
Y = np.array([1,0,0])

Y_pred_good = np.array([0.7,0.2,0.1])
Y_pred_bad = np.array([0.1,0.3,0.6])
l1 = cross_entropy(Y,Y_pred_good)
l2 = cross_entropy(Y,Y_pred_bad)

print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

loss = nn.CrossEntropyLoss()
# 3 sample
Y = torch.tensor([2,0,1]) #index
# nsamples x nclasses = 3x3

Y_pred_good = torch.tensor([[0.1,1.0,2.1],[2.0,1.0,0.1],[0.1,3.0,0.1]]) # no softmax (propability)
Y_pred_bad = torch.tensor([[2.0,1.0,0.1],[0.1,1.0,2.1],[0.1,3.0,0.1]])

l1 = loss(Y_pred_good,Y)
l2 = loss(Y_pred_bad,Y)

print(l1.item())
print(l2.item())

# actual prediction we take (the highest probability)
val_1, predictions_1 = torch.max(Y_pred_good, 1)
val_2, predictions_2 = torch.max(Y_pred_bad, 1)
print(f'highest value {val_1}, tensor{predictions_1}')
print(f'highest value {val_2}, tensor{predictions_2}')




















