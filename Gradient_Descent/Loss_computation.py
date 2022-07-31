# 1) Design model (imput, output size, forward pass)
# 2) Construct loass and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

X_test = torch.tensor([5],dtype=torch.float32)
n_sample, n_feature = X.shape
print(n_sample, n_feature)
input_size = n_feature
output_size = n_feature
# model = nn.Linear(input_size,output_size)

#https://iter01.com/537202.html
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression,self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self, x):
        return self.lin(x)

# model prediction
model = LinearRegression(input_size, output_size)
# loss = MSE mean square error  R^2


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 200
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) #stochastic gradient decent

for eport in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)
    # loss
    l = loss(y_pred,Y) #input, target
    # gradient = backward pass
    l.backward() # dloss/dw (Automaticly calculated)
    # update weight
    optimizer.step()
    # zero the gradients again
    optimizer.zero_grad()
    if eport % 1 == 0:
        [w,b] = model.parameters() #weight and bias
        print(f'eport {eport+1}:w = {w[0][0].item():.3f}, loss = {l:.8f} ')
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
