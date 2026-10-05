'''Implement a 0-layer (input - output layer) neural network from scratch 
for the following dataset. This includes implementing forward and backward 
passes from scratch. Print the training loss and plot it over 999 iterations.
'''
#-----------------Imports-----------------#
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------#

class NeuralNetwork:
    def __init__(self,X,y,epochs,lr=0.01):
        self.X = X                  # Input X
        self.y = y                  # Labels
        self.lr = lr                # Learning Rate
        self.epochs = epochs        # epoches

    # activation function sigmoid used
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    # Training function
    def train(self):
        m,n = self.X.shape
        W = np.zeros((n,1)) # thus single layer
        b = 0.0
        losses = []

        for i in range(self.epochs):

            # Forward propagation
            y_hat = self.sigmoid(self.X @ W + b)

            eps = 1e-8  # prevent 0 in log eqn

            # loss function binary-cross Entropy loss
            loss = -np.mean(
                self.y * np.log(y_hat + eps) + (1 - self.y) * np.log(1 - y_hat + eps)
            )

            losses.append(loss)

            # backward pass gradient by chain rule
            dz = y_hat - self.y
            dW = (self.X.T @ dz) / m
            db = np.mean(dz)

            # updating the weights
            W -= self.lr * dW
            b -= self.lr * db

        return W, b , losses



def main():
    X = np.array([
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,1]
    ],dtype=float)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ],dtype=float)

    nn = NeuralNetwork(X,y,1000,0.1)

    W, b, losses = nn.train()

    # Prediction
    prob = nn.sigmoid(X @ W + b)
    pred = (prob >= 0.5).astype(int)

    print("Weights: \n", W)
    print("Bias: ", b)
    print("Prediction: \n", pred)
    print("Accuracy:", np.mean(pred == y) * 100, "%")
    print(losses)
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("training loss")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()








