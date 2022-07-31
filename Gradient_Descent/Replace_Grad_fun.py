import torch
# f = w * x
# f = 2 * x
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8],dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction

def forward(x):
    return w*x

# loss = MSE mean square error  R^2
def loss_fun(y,y_predicted):
    return ((y_predicted-y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 3000

for eport in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    # loss
    loss = loss_fun(Y, y_pred)
    # gradient = backward pass
    loss.backward() # dloss/dw (Automaticly calculated)
    # update weight
    with torch.no_grad():
        w -=learning_rate* w.grad
    # zero the gradients again
        w.grad.zero_()
    if eport % 1 == 0:
        print(f'eport {eport+1}:w = {w:.3f}, loss = {loss:.8f}  dw:{w.grad:.6f}')
print(f'Prediction after training: f(5) = {forward(5):.3f}')