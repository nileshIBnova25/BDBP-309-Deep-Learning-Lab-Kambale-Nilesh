#---------------------------------------------------------#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import math
from PIL import Image
from tempfile import TemporaryDirectory
import sys

cudnn.benchmark = True       # find fastest cuda convolution algorithm
plt.ioff()  # interactive mode
#---------------------------------------------------------_#
#---------------------1)LOAD-DATA--------------------------------------#
"""
1) RandomizeResizedCrop:-Random Crop and resize and produce 224 * 224 image because resNet 
expect this size of input 
2) RandomHorizontalFlip:Creating additional random flipping variations / data augmentation. 
3) Tensor: convert image to tensor jpeg -> plt -> tensor. 
4)Normalized: Because reNetis trained using this statistics"""

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #
    ]),
    'val': transforms.Compose([
        transforms.Resize(256), # standardize image before taking center crop
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


'''1) loading data 2) datasets.ImageFolder can create classes automatically by the number of distinct folders 
3) DataLoader : batchsize 4  -> four images are processed together as batch instead of single image
4 image -> model -> loss -> update, more efficient on the gpu also give stable gradient 4) shuffling: prevent model 
to learn undesirable patterns due to ordering of the data set 5) num_workers are the processors '''


data_dir = '/home/ibab/DL-Lab/hymenoptera_data'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} # to know size of the train and validation set
class_names = image_datasets['train'].classes


# We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#-------------------Visualize a few images  -------------------------------------------#
""" create a helper function to display tensors as images , images are  normalized tensors 
during the data loading itself, Tensor -> Numpy -> CHW -> Undo normalization -> Display images
transpose due to different lib uses different dimensions conventions, the reason of this function
to use is before training to check the images are loaded correctly labels are correct crops look 
reasonable normalization is not causing any problems augmentation works """

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    #plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
plt.figure(figsize=(10,6))
imshow(out, title=[class_names[x] for x in classes])
plt.show()
print("IMAGE DISPLAYED")
#-------------------------------------------------------------#

#--------------------Training the model--------------------------------------------------#
'''Training operation: epoch -> train -> forward -> loss -> backward -> update -> save model 
'''

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0         # for loss of current epoch so it will not count for the previous epoch
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    """Because PyTorch gradient accumulate by default batch 1 -> 0.5 , batch 2 -> 0.3
                    without clearing gradient = 0.8 but normally we want gradient for batch so it 
                    clears previous gradients"""
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)             # main model training part
                        _, preds = torch.max(outputs, 1)    # which class has the highest output score
                        loss = criterion(outputs, labels)   # loss gives how wrong was prediction

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()                 # Calculate the gradient
                            optimizer.step()                # Change the weights

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                """At the beginning, larger steps can help the model learn quickly
                    Later, smaller steps help it make finer adjustment 
                    larger step -> fast learning , small steps -> fine tuning """

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights / because after 25 epoch, the model may no longer be the best one
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model
#-------------------------------Visualizing the model predictions-----------------------------------------------#
def visualize_model(model, num_images=20):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    columns = 4
    # Calculate numbers of rows automatically
    rows = math.ceil(num_images / columns)
    # Create one figure
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12,rows *3),
    )
    axes = np.array(axes).reshape(-1)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if images_so_far >= num_images:
                    break
                if images_so_far >= num_images:  # <-- add this
                    break
                ax = axes[images_so_far]

                # get image
                image = inputs.cpu().data[j].numpy().transpose((1,2,0))

                # Undo normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = std * image + mean
                image = np.clip(image, 0, 1)

                # Show image
                ax.imshow(image)

                # Show Prediction
                ax.set_title(
                    f"Prediction: {class_names[preds[j]]}",
                    fontsize=10
                )
                ax.axis('off')
                images_so_far += 1

                if images_so_far >= num_images:
                    break


        for i in range(images_so_far,len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
        model.train(mode=was_training)

#----------------------Visualizing the model predictions-------------------------------------------------_____#
"""This part is the main transfer learning is happening 
 the resnet18 already learned general visual feature such as 
 edges -> textures -> shapes -> patterns instead training from
 scratch we will start from already trained weight of resNet"""
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.

""" Imagenet has 1000 classes but the current classification problem has only 2 classes"""
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
"""lr -> learning rate, momentum -> helps SGD maintain some direction from previous updates"""
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#------------------------------------------------------------------------------------#

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
visualize_model(model_ft)
