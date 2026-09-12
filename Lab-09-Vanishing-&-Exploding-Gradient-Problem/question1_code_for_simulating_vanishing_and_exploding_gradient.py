#-------------------------Import----------------------#
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
#-----------------------------------------------------#

#----setting-random-seed
torch.manual_seed(42)
np.random.seed(42)


#--------Defining-Neural-Network-----------------------#

class NeuralNetwork(nn.Module):
    def __init__(self,activation="sigmoid",intializer_std=1.0 ):
        super(NeuralNetwork, self).__init__()