""" Lab Goal
1. Learn the notion of computational graph and how to compute gradients for different parts
of the computational graph.
2. Understand how the gradient flows backwards in a neural network across multiple layers
3. Familiarize yourself with the stochastic gradient descent algorithm for parameter updates
in a neural network."""

#-----------------Import---------------------------#
import numpy as np
#--------------------------------------------------#

#--------------------------------------------------#
"""1. Implement backward pass for the above two networks. Print the gradient values for each
neuron in each layer."""

class BackwardPass:
    def __init__(self, hidden_layer_size, X ,y ):
        self.X = X
        self.y = y
        self.h = hidden_layer_size

    def sigmoid(self,z):
        return 1 / ( 1 + np.exp(-z) )

    def sigmoid_derive(self,a):
        return a * (1 - a)

    def forward(self,weights,biases):
         activations = [self.X]
         for w, b in zip(weights,biases):
             a = self.sigmoid(w@activations[-1] + b)
             activations.append(a)
         return activations

    def backward(self,activations,weights,y):
        grad = [None] * len(weights)

        # output error " dL/dZ
        delta = activations[-1] - self.y  # global gradient variable

        for i in range(len(weights)-1,-1,-1):
            grad[i] = delta @ activations[i].T

            if i > 0 :
                delta = (
                    weights[i].T @ delta * self.sigmoid_derive(activations[i])
                )
        return grad

    def run_network(self):
        weights = [np.random.randn(self.h[i + 1],self.h[i]) for i in range(len(self.h) -1 )]
        biases = [np.zeros((self.h[i+1],1)) for i in range(len(self.h) -1)]
        activations = self.forward(weights,biases)
        print(f"\nActivation:")
        for i, a in enumerate(activations):
            print(f"layer {i} : {a}")

        grads = self.backward(activations,weights,self.y)

        print(f"\n Weights Gradients:")
        for i, grad in enumerate(grads):
            print(f"layer {i + 1} : {grad.sum(axis=1,keepdims=True)}")
#-----------------------------------------------------------------------#

#-----------------------------------------------------------------------#
def main():
    np.random.seed(0)

    x = np.array([[1.0],[2.0],[3.0],[4.0]])
    y = np.array([[1.0]])
    hidden_layer_size = [4,3,2,1]

    # running given example network

    backprop1 = BackwardPass(hidden_layer_size, x,y)
    backprop1.run_network()

    backprop2=BackwardPass([4,1],x,y)
    backprop2.run_network()

if __name__ == "__main__":
    main()
#-----------------------------------------------------------------------#











