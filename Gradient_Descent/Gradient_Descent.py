'''
Automatic Gradient Computation #Find the lowest slope to optimize the parameter
    - Prediction: Manually -> 4. PyTorch Model
    - Gradients Computation: Manually -> 2. Autograd
    - Loss Computation: Manually -> 3. PyTorch Loss
    - Parameter updates: Manually -> 3. Pytorch Optimization
'''

import numpy as np
# f = w * x
# f = 2 * x
X = np.array([1,2,3,4], dtype=np.float64)
Y = np.array([2,4,6,8],dtype=np.float64)
w = 0.0

# model prediction

def forward(x):
    return w*x

# loss = MSE mean square error  R^2
def loss_fun(y,y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2 # Loss Function
# dMSE/dw = 1/N * 2x *(w*x -y)^2-1 # dLoss Function/dw
'''
    y_predicted = w*x
    loss_Fun = (y_predicted - y)**2
        dLoss/dw = dLoss_Fun/dy_predicted * dy_predicted/dw
             = 2(y_predicted - y) * x
             = 2x * (y_predicted -y)
'''
def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 20

for eport in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    # loss
    loss = loss_fun(Y, y_pred)
    # gradient
    dw = gradient(X,Y,y_pred)
    # update weight
    w -= learning_rate* dw

    if eport % 1 == 0:
        print(f'eport {eport+1}:w = {w:.3f}, loss = {loss:.8f}  dw:{dw:.6f}')
print(f'Prediction after training: f(5) = {forward(5):.3f}')

