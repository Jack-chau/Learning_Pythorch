'''
1) Design model (input, output sizze, forward pass)
2) Construct loss and optimizer
3) Training loop
    - forward pass: compute prediction and loss
    - backward pass: gradients
    - update weight
'''
# Logistic Regression

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
bc = datasets.load_breast_cancer()
print(bc.keys())
# print(bc.DESCR)
X , y = bc.data, bc.target
print(bc.data.shape)
n_sample, n_features = X.shape

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=1234)
# scalar
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)
# 1) model
# f =wx + b, sigmoid at the end
class LogisticRegrssion (nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegrssion, self).__init__()
        self.Linear = nn.Linear(n_input_features,1)
    def forward(self,x):
        y_predicted = torch.sigmoid(self.Linear(x))
        return y_predicted

model = LogisticRegrssion(n_features)
# 2) loss and optimizer

criterion = nn.BCELoss() # Binary Cross Entropy
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_eports = 5000

for eport in range(num_eports):
    # forward pass and loss calculation
    y_predicted = model(X_train)
    loss = criterion(y_predicted,y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (eport+1) %100 == 0:
        print(f'eport {eport +1 }, loss={loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')