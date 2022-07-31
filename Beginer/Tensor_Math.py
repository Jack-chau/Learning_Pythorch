import torch

# Tensor Math & Comparison OPerations

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

#Addition (element wise)
z1 = torch.empty(3)
torch.add(x,y,out=z1)
z2 = torch.add(x,y)
z = x+y


#Subtraction (element wise)
z = x-y

#Division (element wise)
z = torch.true_divide(x,y)

#implace opeartion
t = torch.zeros(3)
t.add_(x) # _ is non inplace
t += x #inplace opearation

# Exponextiation (element wise)
z = x.pow(2)
z = x**2

#Simple comparison
z = x>0
z = x<0

# Matrix Multiplication (element wise)
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

# matrix exponentiation (element wise)
matrix_exp = torch.rand(5,5)
matrix_exp = matrix_exp.matrix_power(3) # a x e^3

#element wise multiplication
z = x * y

# dot product
z = torch.dot(x,y) #  sum (element wise mult.ed)

# Batch Matrix Multiplication (not elemenet wise)
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand(batch,m,p)
out_bmm = torch.bmm(tensor1,tensor2) #batch,n,p  (n x m matrix x m x p matrix = n x p matrix, since b is same size )
# print(tensor1.shape)
# print(tensor2.shape)
# print(out_bmm.shape)
# http://christopher5106.github.io/deep/learning/2018/10/28/understand-batch-matrix-multiplication.html

# Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))
z = x1 - x2 # like numpy, x2 row will 1->5
z = x1 ** x2

# Other useful tensor operation
sum_x = torch.sum(x,dim=0)
values, indices = torch.max(x,dim=0) #mix will return 2 values
values, indices = torch.min(x,dim=0) #mix will return 2 values
abs_x = torch.abs(x)
z = torch.argmax(x,dim=0) #seem as torch.max, return value only
z = torch.argmin(x,dim=0) #seem as torch.min, return value only
mean_z = torch.mean(x.float(),dim=0)
z = torch.eq(x,y) #equal? (element wise)
sorted_y, indeices = torch.sort(y,dim=0,descending=False) # sort by ascending order
z = torch.clamp(x, min=0, max=10) #all value less than 0, set to 0. max then 10, set to 10
x = torch.tensor([1,0,1,1,1],dtype=torch.bool)
z = torch.any(x) # any value in x is Ture
z = torch.all(x) # all value in x are True?
