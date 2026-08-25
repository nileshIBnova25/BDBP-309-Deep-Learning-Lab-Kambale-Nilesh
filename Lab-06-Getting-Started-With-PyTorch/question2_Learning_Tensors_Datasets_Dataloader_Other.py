#--------------------------Imports--------------------------------------#
import torch
import numpy as np
#------------------------Initializing-a-Tensor--------------------------#
'''Tensors are a specialized data structure that are very similar to 
arrays and matrices. In PyTorch, we use tensors to encode the inputs 
and outputs of a model, as well as the model’s parameters.'''

data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

# From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor:  \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float32) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# With random or constant values
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n  {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")


#-----------------------------------------------------------------------#


