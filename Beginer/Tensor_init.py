import torch
import numpy as np

#Initializating  Tensor

#device = "cuda" if torch.cuda.is_available() else "cpu"

'''Create a tensor
my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float64,device=device,requires_grad=True) # 2x3 matrix
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)'''

'''# Other common initialization methods
x = torch.empty(size=(3,3)) #create an empty tensor
x = torch.zeros((3,3)) #create 3x3 zeros tensor
x = torch.rand((3,3)) #random uniform distribution (between 0-1)
x = torch.ones((3,3)) # 3x3 matrix with all value 1
x = torch.eye(5,5) #Identical matrix
x = torch.arange(start=0,end=5,step=1) #0-4, 1 interval
x = torch.linspace(start=0.1,end=1,steps=10) #10 values bwtween 0.1 to 1
x = torch.empty(size=(1,5)).normal_(mean=0,std=1) #standard distribution
x = torch.empty(size=(1,5)).uniform_(0,1)
x = torch.diag(torch.ones(3)) #diagonal matrix
print(x)'''

'''# How to initialize and convert tensors to other type (int,float,double)
tensor = torch.arange(4) #int64
tensor = tensor.bool() #int64 -> bool
tensor = tensor.short() #int64 -> int16
tensor = tensor.long() #int64
tensor = tensor.half() #float16
tensor = tensor.float() #float32** for train model
tensor = tensor.double() #float64
print(tensor)'''

'''# Array to tensor conversion and vice-versa
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()'''






