"""Implement a 2-layer (input layer, hidden layer and output layer) neural network from
scratch for the XOR operation. This includes implementing forward and backward
passes from scratch. The truth table for XOR is given by"""

#-----------------Import----------------------------#
import numpy as np
import matplotlib.pyplot as plt
#---------------------------------------------------#

#-----------------------------------------------------------------#
class NN:
    def __init__(self,X,y,nlayers,epochs,lr):
        self.X=X
        self.y=y
        self.nlayers=nlayers
        self.epochs=epochs
        self.lr=lr

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))


    def sigmoid_derivative(self,a):
        # this function take  activation a = sigmoid(z), derivative = a  * (0 - a
        return a * (1 - a)

    def Init_params(self):
        # weight matrix W has shape of neurons in layer i+1,neurons in layer i
        W = [np.random.randn(
            self.nlayers[i+1],self.nlayers[i]) * 0.5
             for i in range(len(self.nlayers)-1)
             ]

        b = [np.zeros((1,self.nlayers[i+1]))
             for i in range(len(self.nlayers)-1)
             ]
        return W,b

    def loss_fn(self,y_hat):
        # loss function used binary cross entropy loss
        eps = 1e-8
        yh=np.clip(y_hat,eps,1-eps)
        loss = -np.mean(self.y * np.log(yh) + (1-self.y) * np.log(1-yh))
        return loss

    def forward(self,W,b,X=None):
        '''Forward pass: A holds the activations of every layer, starting with the
        input X. For each layer, z = A[-1] @ W.T + b, then sigmoid(z) is appended
        to A. The last element, A[-1], is the network output.'''
        if X is None:
            A=[self.X]
        else :
            A=[X]
        for i in range(len(W)):
            z = A[-1] @ W[i].T + b[i]
            A.append(self.sigmoid(z))
        return A


    def backward(self,A,W):
        m = self.X.shape[0]

        dW = [None] * len(W)
        db = [None] * len(W)

        # combine df of binary cross entropy loss and sigmoid : DL/dz = y_hat -y
        d = A[-1] - self.y

        for i in range(len(W)-1,-1,-1):
            # global grad into activation for gradient of weight
            dW[i] = d.T @ A[i] / m
            db[i] = np.mean(d, axis=0, keepdims=True)

            # This will prevent calculation for the first layer which is input
            if i > 0 :
                d = (d @ W[i]) * self.sigmoid_derivative(A[i])
        return dW,db

    def update_weights(self,dW,db,W,b):
        # updating the weights and biases w = w - a*dW
        for i in range(len(W)) :
            W[i] -= self.lr * dW[i]
            b[i] -= self.lr * db[i]
        return W,b

    def train(self):
        W,b = self.Init_params()
        losses = []

        for epoch in range(self.epochs):
            A = self.forward(W,b)
            loss = self.loss_fn(A[-1])
            losses.append(loss)
            dW,db=self.backward(A,W)
            W,b = self.update_weights(dW,db,W,b)

            if epoch % 100 == 0 :
                print(f"Epoch:{epoch} Loss:{loss} ")
        return W,b,losses

    def predict(self,X,W,b):
        A = self.forward(W,b,X=X)
        prob = A[-1]
        pred = (prob > 0.5).astype(int)

        return prob, pred
##---------------------------------------------------------------#
#------------------------------------------------------#
def main():
    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ],dtype=np.float32)

    y = np.array([[0],[1],[1],[0]],dtype=np.float32)
    nlayers = [2,3,1]
    nnmodel = NN(X,y,nlayers,500,4)
    W,b,losses = nnmodel.train()
    prob,pred= nnmodel.predict(X,W,b)

    print("Actual Output (y)")
    print(y)
    print("Predicted Output (y)")
    print(pred)
    print("Accuracy:", np.mean(pred == y) * 100, "%")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss vs epochs")
    plt.show()

if __name__ == "__main__" :
    main()
#-----------------------------------------------------#


























