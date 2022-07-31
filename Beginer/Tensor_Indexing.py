#   Tensor Indexing
import torch

batch_size = 10
features = 25
x = torch.rand((batch_size,features))
#print(x[0]) #first row (batch_size), 25 column (features) x[0,:]
#print(x[0].shape)
#print(x[:,0]) #all rows in first column
# print(x[2,0:10]) # the third row, and get 0 - 9 column

# Fancy indexing
x = torch.arange(10)
indices = [2,5,8]
#print(x[indices]) #the 3rd,5th,8th values
x = torch.rand((3,5))
# print(x)
rows = torch.tensor([1,0])
column = torch.tensor([4,0])
# print(x[rows,column]) #pick out two values, 2rd row and 5th column. 1st row and 1 column

# More advanced indexing
x = torch.arange(10)
# print(x[(x<2) | (x>8)] )
# print(x[(x>2) & (x>8)] )
# print(x[x.remainder(2)==0]) # grep all elements in x/2 are equal to 0

#Useful operations
# print(torch.where(x > 5, x, x*2)) #if x is >5, return x else x*2
# print(torch.tensor([0,0,1,2,2,3,4]).unique())
# print(x.ndimension())
#print(x.numel()) #count elements in x. like x.count() in numpy































