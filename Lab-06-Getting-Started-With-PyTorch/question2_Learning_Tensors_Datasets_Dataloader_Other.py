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

#--------------------------From a NumPy array.---------------------------#
np_array = np.array(data)
x_np = torch.from_numpy(np_array)



#--------------------------From another tensor---------------------------#
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor:  \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float32) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")



#-----------------------With random or constant values-------------------#
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n  {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")



#---------------------------Attributes of a Tensor------------------------#
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape} \n")
print(f"Datatype of tensor: {tensor.dtype} \n")
print(f"Device tensor in stored on device: {tensor.device} \n")

# Operation on Tensors
'''Each of these operations can be run on the CPU and Accelerator 
such as CUDA, MPS, MTIA, or XPU. If you’re using Colab, allocate 
an accelerator by going to Runtime > Change runtime type > GPU.'''

if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

tensor = torch.ones(4,4)
print(f"First row : {tensor[0]} \n")
print(f"First column: {tensor[1]} \n")
print(f"Last column : {tensor[...,-1]}")
tensor[:,-1] = 0 # making last column as 0
print(tensor)



#-------------------------------Joining tensors--------------------------#
t1 = torch.cat([tensor,tensor,tensor],dim =1)
print(t1)

#-----------------------------Arithmetic operations----------------------#
'''# This computes the matrix multiplication between two tensors. 
y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor'''

#           matmul rows and columns are multiplied and summed

y1 = tensor @  tensor.T                 #Multiplication of tensor by its transpose
y2 = tensor.matmul(tensor.T)            #Multiply tensor by its transpose using matmul
y3 = torch.rand_like(y1)                #Create random tensor with same shape
torch.matmul(tensor, tensor.T , out=y3) #Multiply tensor by its transpose & save as y3
print(f"printing y3 : \n { y3} \n")
#             ----------------------------------------------------------               #
'''# This computes the element-wise product. 
z1, z2, z3 will have the same value'''
# mul Element wise multiplication 
# mul → multiply each position
z1 = tensor @ tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f" printing z3: \n {y3} \n")




#---------------------------Single-element tensors------------------------#
agg = tensor.sum()
agg_item = agg.item()
print(f" Aggregate : {agg_item}, datatype : {type(agg_item)} \n")



#-----------------------------In Place operations--------------------------#
print(f" Before addition \n {tensor} \n")
tensor.add_(5)
print(f" After addition \n {tensor} \n")




#-----------------------------Bridge with Numpy----------------------------#
t = torch.ones(5)
print(f"t: {t} \n")
n = t.numpy()
print(f"n: {n} \n")
# A change in the tensor reflects in the NumPy array

t.add_(1)
print(f"t : {t} \n")
print(f"n: {n} \n")


#------------------Numpy array to Tensor -----------------------------------#
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1 ,out=n)
print(f"n: {n} \n")
print(f"t: {t} \n")

#-----------------------------------------------------------------------#


