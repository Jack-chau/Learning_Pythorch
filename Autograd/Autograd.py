'''  Autograd -package
        The autograd package in pytorch is to calculate the gradients with it.
        Gradient are essential for out model optimization, it is vary import important
'''
import torch

'''Calculate gradient'''
# x = torch.randn(3,requires_grad=True) #randn follow normal distribution
# requires_grad True --> for calculate gradiant (self create computational graph)
# print(x)
# y = x+2 #fn=<AddBackward> because it is add dunction
# print(y) # grad_fn attribute is at the end of the forward path, it will point to backward (backpropagation)
# z = y*y*2 #fn=<MulBackward> #Multipication
# print(z)
# since the x is randomly generated, there are no relationship between eachother
# therefore, we need a scaler outputs
# z = z.mean() #fn=<MeanBackward> #Meantipication
# print(z)
# z.backward() # dz/dx #call backpopuatio n (find x gradient)
# print(x.grad)
# result = 4/3*y
# print(result)
# if we don't calculate the mean we will have a error, grad can be implicity created onlt for scalar outputs
# v = torch.tensor([0.1,1.0,0.001], dtype=torch.float32) # follow normal distribution
# z.backward(v)
# print(x.grad)

''' Prevent tracking the gradient (since training will keep update the gradient)'''
# method 1
# x.requires_grad_(False)
# method 2 # it will create a new tensor without requires_grad
# x.detach()
# method3
# with torch.no_grad():

# x = torch.randn(3,requires_grad=True)
# print(x)
# x.requires_grad_(False) #_ will modify the variable inplace
# # print(x)
# y = x.detach() # create an new tensor, requires_grad = False
# # print(y)
#
# with torch.no_grad():
#     y = x+2
#     print(y) # no gradient function attribute

'''
Whenever we call the backward function, 
then the gradient for this tensor will be accumulated into the dot grad attribute
'''

# weight = torch.ones(4, requires_grad=True)
#
# for epoch in range(3):
#     model_output = (weight*3).sum()
#     model_output.backward()
#     print(weight.grad)
### we have to empty the gradient everytime in training,
#otherwise it will accumulate the gradient and we will never found the minimum gradient
    # weight.grad.zero_()









