#---------------------Imports--------------#
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import accuracy_score
#----------------------------------------------------------#
#----------------------------------------------------------#

# Define neural network
class NeuralNetwork(nn.Module):
    def __init__(self, activation_function="sigmoid", weight_std=0.01):
        super(NeuralNetwork, self).__init__()
        self.activation_function = activation_function

        # weight initialization
        self.weight_std = weight_std

        # Hidden layers
        self.linear1 = nn.Linear(
            in_features=2,
            out_features=5
        )

        self.linear2 = nn.Linear(
            in_features=5,
            out_features=5
        )
        
        self.linear3 = nn.Linear(
            in_features=5,
            out_features=5
        )
        
        self.linear4 = nn.Linear(
            in_features=5,
            out_features=5
        )
        
        self.linear5 = nn.Linear(
            in_features=5,
            out_features=1
        )
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        # weights initialize
        self.initialize_weights()
    #------------------------------------------------------------------#

    def initialize_weights(self):
        # initialize all weights from normal distribution
        for layer in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]:
            nn.init.normal_(
                layer.weight,
                mean=0.0,
                std=self.weight_std
            )
            nn.init.zeros_(layer.bias)
    #----------------------------------------------------------------------#
    def activation(self, x):
        if self.activation_function == "sigmoid":
            return self.sigmoid(x)
        elif self.activation_function == "tanh":
            return self.tanh(x)
        elif self.activation_function == "relu":
            return self.relu(x)
        else:
            raise ValueError("Unsupported activation function")
    #------------------------------------------------------------------------#

    def forward(self, x):
        # input -> hidden layer
        x = self.linear1(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.activation(x)

        x = self.linear3(x)
        x = self.activation(x)

        x = self.linear4(x)
        x = self.activation(x)

        x = self.linear5(x)
        # Output sigmoid
        out = self.sigmoid(x)

        return out
#-------------------------------------------------------------------------#
# Load_dataset
def load_data():
    # generate binary classification dataset
    X, y = make_circles(
        n_samples=1000,
        factor=0.5,
        noise=0.1,
        random_state=42
    )
    # conver numpy array to pytorch tensors
    X = torch.tensor(
        X,
        dtype=torch.float32
    )
    y = torch.tensor(
        y,
        dtype=torch.float32
    ).view(-1,1)

    return X, y
 #----------------------------------------------------------------------------#
    # Capture weights
def capture_weights(model):
    weights = {}
    for name, parameter in model.named_parameters():
        if "weight" in name:
            weights[name] = (
                parameter.detach().cpu().numpy().copy()
        )
    return weights

 #-----------------------------------------------------------------------------#
# Capture Gradient
def capture_gradients(model):
    gradients = {}
    for name, parameter in model.named_parameters():
        if "weight" in name:
            if parameter.grad is not None:
                gradients[name] = (
                parameter.grad.detach().cpu().numpy().copy()
                )
    return gradients

#------------------------------------------------------------------------------#

# Train model
def train( train_dataloader,model,loss_fn, optimizer, device ,epochs ):
    weight_history = []
    gradient_history = []
    loss_history = []

    weight_history.append(capture_weights(model))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch, (X, y ) in enumerate(train_dataloader):
            X,y = X.to(device), y.to(device)
            pred = model(X) # forward pass
            loss = loss_fn(pred, y) # calculate loss
            optimizer.zero_grad()
            loss.backward()

            if batch == 0:
                gradient_history.append(capture_gradients(model))

            # Update weights
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)
        loss_history.append(epoch_loss)

        # capture weights after epoch
        weight_history.append(capture_weights(model))
        if epoch % 10 == 0:
             print(
                 f"Epoch [{epoch+1}/{epochs}]"
                 f"Loss: {epoch_loss:.6f}"
             )

    return (
        weight_history,
        gradient_history,
        loss_history
    )

#--------------------------------------------------------------------------#
# Test model
def test(X,y,model,loss_fn,device):
    model.eval()
    X = X.to(device)
    y = y.to(device)

    with torch.no_grad():
        pred = model(X)
        loss = loss_fn(pred,y)
        predicted_labels = (
            pred >= 0.5
        ).float()
        accuracy = accuracy_score(y.cpu().numpy().flatten(), predicted_labels.cpu().numpy().flatten())

    print("\nTest Results")
    print("---------------------------")
    print(f"Loss: {loss.item():.6f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return loss.item(), accuracy

#-------------------------------------------------------------------------------------------------__#

def plot_weights(weight_history, activation_name):
    layer_names = list(weight_history[0].keys())
    fig, ax = plt.subplots(
        2,
        1,
        sharex= True,
        constrained_layout=True,
        figsize =(10,10)
        )

    # mean weight
    ax[0].set_title(
        f"Mean Weight - {activation_name}"
    )
    for layer_name in layer_names:
        mean_values = [weights[layer_name].mean() for weights in weight_history]
        ax[0].plot(
            mean_values,
            label=layer_name
        )
    ax[0].set_ylabel("Mean weight")
    ax[0].legend()
    ax[0].grid(True)

    # Standard devation
    ax[1].set_title(
        f"Weight Standard Deviation - {activation_name}"
    )
    for layer_name in layer_names:
        std_values= [weights[layer_name].std() for weights in weight_history]
        ax[1].plot(std_values,label=layer_name)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Standard deviation")
    ax[1].legend()
    ax[1].grid(True)

    plt.show()
#---------------------------------------------------------------------------------#
def plot_gradients(gradient_history,loss_history,activation_name):
    layer_names = list(gradient_history[0].keys())
    fig, ax = plt.subplots(
        3,
        1,
        sharex = True,
        constrained_layout=True,
        figsize =(10,12)
    )
    # mean gradent
    ax[0].set_title(f"Mean Gradient - {activation_name}")
    for layer_name in layer_names:
        mean_values = [gradients[layer_name].mean() for gradients in gradient_history]

        ax[0].plot(
            mean_values,
            label=layer_name
        )
    ax[0].set_ylabel("Mean gradient")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].set_title(
        f"Gradient standard deviation - {activation_name}"
    )

    for layer_name in layer_names:
        std_values = [ gradients[layer_name].std() for gradients in gradient_history]
        std_values = np.maximum(std_values,1e-12)

        ax[1].semilogy(std_values,label=layer_name)
    ax[1].set_ylabel("Gradient std (log scale)")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    ax[1].grid(True)

    ax[2].set_title(f"Training loss - {activation_name}")
    ax[2].plot(loss_history,label="Loss")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Loss")
    ax[2].legend()
    ax[2].grid(True)
    plt.show()

def run_experiment(activation_name,X,y,device,epochs=100,batch_size=32):
    print("\n========================================================")
    print(f"Activation function: {activation_name}")
    print(f"=========================================================")

    # Create dataloader
    dataset = TensorDataset(X,y)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Initialize model
    model = NeuralNetwork(
        activation_function=activation_name,
        weight_std = 1.0
    ).to(device)
    print(model)

    # binary cross entropy
    loss_fn = nn.BCELoss()

    # optimizer
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=0.001
    )

    # Accuracy before training
    print("\nBefore training")

    test(X,y,model,loss_fn,device)
    # train model
    (weight_history,gradient_history,loss_history)= train(train_dataloader,model,loss_fn,optimizer,device,epochs)

    # accuracy after training
    print("\nAfter training")

    test(X,y,model,loss_fn,device)
    plot_weights(weight_history,activation_name)
    plot_gradients(gradient_history,loss_history,activation_name,)

#-------------------------------------------------------------------------------------------------_____#

def main():
    # load dataset
    X,y = load_data()

    print("Dataset Information")
    print("-------------------------")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using {device} device")

    # Numbers of epochs
    epochs = 100
    batch_size = 32
    run_experiment(
        activation_name= "sigmoid",
        X=X,
        y=y,
        device=device,
        epochs=epochs,
        batch_size=batch_size
    )
    run_experiment(
        activation_name= "tanh",
        X=X,
        y=y,
        device=device,
        epochs=epochs,
        batch_size=batch_size
    )

    run_experiment(
        activation_name= "relu",
        X=X,
        y=y,
        device=device,
        epochs=epochs,
        batch_size=batch_size
    )
    print("\nEnd")
#------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
