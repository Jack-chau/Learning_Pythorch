''' in pytorch, all are build-in tensor operation
        - Tensor can also be 1d,2d, 3d or nd
'''
import torch
# x = torch.empty(1) #empty tensor with 1 size
# x = torch.empty(3) #one row, three column, 1d, 3 element
# x = torch.empty (2,3) #two row, three column, 2d, 3 element
# x = torch.rand(2,2) #random gen 2x2 tensor
# x = torch.ones(2,2,dtype=torch.double)
# x = torch.ones(2,2,dtype=torch.float16)
# print(x.dtype) #check datatype
# print(x.size()) #check datatype

# x = torch.tensor([2.5,0.1])
# print(x)

'''Addition'''
# x = torch.rand(2,2)
# y = torch.rand(2,2)
# print(x)
# print(y)
# z= x+y #element rise addition
# z = torch.add(x,y)
# print(y)
# y.add_(x) #inplace addition (everything have _ is inplace addition)
# which means it will change the y tenser
# print(y)
'''Subtraction'''
# x = torch.rand(2,2)
# y = torch.rand(2,2)
# z = x-y
# z = torch.subtract(x,y)
# print(z)
'''Multipication'''
# x = torch.rand(2,2)
# y = torch.rand(2,2)
# z = x*y
# z = torch.mul(x,y)
# y.mul_(x) # inplace
'''Divide'''
# x = torch.rand(2,2)
# y = torch.rand(2,2)
# z = torch.div(x,y)
'''Sclicing'''
# x = torch.rand(5,3)
# print(x)
#print(x[:,:,step])
# print(x[:,0]) #first column, all row
# print(x[0,:]) first row
# print(x[1,:]) #second row, all columns
# print(x[1,1])
# print(x[1,1].item()) #get the actual value
'''Reshape tensor'''
# x = torch.rand(4,4)
# print(x.size())
# # print(x)
# y = x.view(16) #4x4 tensor become 1d tensor
# # print(y)
# y = x.view(-1,8) #-1 says I don't want to define the whole size of the tensor, 8 says one size I want it to be 8
# print(y.size())
'''Convert torch to numpy array'''
import numpy as np
# if you are not using cuda enable gpu, if we change one, both will change because torch and array are both saved in gpu
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)
# print(type(a))
# print(type(b))
''' Convert numby arrat to torch'''
# a = np.ones(5)
# print(type(a))
# b = torch.from_numpy(a)
# print(type(b))
# print(b) # by default, it will be float64 # we can specific the data type
# a +=1
# # be careful, both will change!!!
# print(a)
# print(b)

'''requires_grad'''
# x = torch.ones(5,requires_grad=True) #need to calculate the gradients for this tensor
# # whenever you have a variable in your model that you want to optimize, then you need the gradients, so you need to specify it, turn requires_rad tobe true
# print(x)






