import torch
import numpy as np

# Create torch tensor from data
data = [[1,2],[3,4]]
a = torch.Tensor(data)
print(a.shape)

# Create torch tensor from numpy
b = np.array(data)
c = torch.Tensor(b)
print(c.shape)

# Create directly
d = torch.Tensor([[1,2],[3,4]])
print(d.shape)

# Inherent properties from other tensor but different in data
e = torch.ones_like(d) # one_like = torch of 1 but with the property (size, shape, data_type as input)
print(e)

f = torch.rand_like(e) # rand_like = torch of random number
print(f)

# Create tensor of primitive constant
shape = (2,3,)
g = torch.rand(shape)
h = torch.ones(shape)
i = torch.zeros(shape)
print(g,'\n',h,'\n',i)

# Property of tensor
print(g.shape)
print(g.device)
print(g.dtype)
print(g.to("cuda").device) # push tensor to cuda



# Standard Indexing and Slicing
print("Indexing ----------")
tensor = torch.Tensor([[6],[2],[5]])
print(tensor[0]) # give -> tensor([6.])
print(tensor[1]) # give -> tensor([2.])
print(tensor[2]) # give -> tensor([5.]) 

tensor = torch.Tensor([[6,4],[2,7],[5,9]])
print(tensor[0]) # give -> tensor([6., 4.])
print(tensor[1]) # give -> tensor([2., 7.])
print(tensor[2]) # give -> tensor([5., 9.])
 
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])

tensor[:,1] = 0
print(tensor)


# Operation
tensor = torch.Tensor([[6],[2],[5]])
print(tensor)
print(tensor.shape) # look like the shape that it report is column first and row second

tensor2 = torch.ones_like(tensor)
tensor2 = torch.transpose(tensor2,0,1)

result = tensor @ tensor2
print(result)

# Following https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html