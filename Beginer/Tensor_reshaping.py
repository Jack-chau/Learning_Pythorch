# Tensor reshaping
import torch

x = torch.arange(9) #0-8, 9 numbers
x_3x3 = x.view(3,3) # make it to a 3x3 matrix (view is a pointer, will change storage inside the memory)
x_3x3 = x.reshape(3,3) #reshape to 3x3 matrix (not a pointer)

# In pytorch, .view is pointer so you can'y alter it inplace
y = x_3x3.t()
#print(y.view(9)) #not allow
print(y.contiguous().view(9)) #allow
print(y.reshape(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2),dim=0).shape) #concatenate the first dimensions
print(torch.cat((x1,x2),dim=1).shape)

z = x1.view(-1) #2 row 5 columns -> 1 row 10 columns
print(z.shape)

batch = 64
x = torch.rand((batch,2,5))
z = x.view((batch,-1)) #concatenate the last two matrix
print(z.shape)

#switch batch,2,5 -> batch,5,2 # transpose matrix
z = x.permute(0,2,1)
print(z.shape)

x = torch.arange(10)
print(x)
print(x.unsqueeze(0).shape) #add a dimension in front of x
print(x.unsqueeze(1).shape) #add a dimension behind x

x = torch.arange(10).unsqueeze(0).unsqueeze(1)
print(x.shape)
z = x.squeeze(1) #remove 2nd tensor in x
print(z.shape)