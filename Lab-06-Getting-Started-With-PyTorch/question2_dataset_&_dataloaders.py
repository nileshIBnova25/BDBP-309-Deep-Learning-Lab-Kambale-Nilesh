#-------------Imports-------------------#
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
#----------------------------------------#

#--------------------Loading-Data---------------------------_#

# training data
training_data = datasets.FashionMNIST(
    root= "data",
    train = True,
    download = True,
    transform = v2.Compose([v2.ToImage()])

)
