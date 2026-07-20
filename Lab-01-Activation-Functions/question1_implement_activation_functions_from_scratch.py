#--------------------------------------------------------------------------------------#
''' Task :- Implement the following functions in Python from scratch.
Do not use any library functions. You are allowed to use numpy
and matplotlib. Generate 100 equally spaced values between -10 and 10.
Call this list as  z. Implement the following functions and its derivative.
Use class notes to find the expression for these functions. Use z as
input and plot both the function outputs and its derivative outputs.'''
''' a) Sigmoid
    b) Tanh
    c) ReLU
    d) LeakyReLU
    e) Softmax '''
#-------------------------------------------------------------------------------------#

#-----------------------Imports------------------------------------#
import matplotlib.pyplot as plt
import numpy as np
#------------------------------------------------------------------#



#------------------------------------------------------------------#

class Lab1ActivationFunction:
    def __init__(self):
        self.z = np.linspace(-10, 10, 100)
        self.a = 0.2

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def tanh(self, z):
        return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

    def tanh_derivative(self, z):
        return 1- (self.tanh(z))**2

    def relu(self, z):
        return np.maximum(0, z)
    def relu_derivative(self, z):
        return np.where(z>0,1.0,00)
    def leaky_relu(self,z):
        return np.maximum(self.a *z, z)

    def leaky_relu_derivative(self, z):
        return np.maximum(self.a, z)

    def softmax(self,z):
        return np.exp(z)/np.sum(np.exp(z))

    def softmax_derivative1(self,z):
        return self.softmax(z) * (1 - self.softmax(z))

    def softmax_derivative2(self,z):
        return - self.softmax(z) * self.softmax(z)

    def plot_functions(self,func,dev_func,lab1,lab2,title):
        y = func(self.z)
        y_dev = dev_func(self.z)

        plt.plot(self.z,y,label=lab1,color='blue',linewidth=2)
        plt.plot(self.z,y_dev,label=lab2,color='red',linestyle='--',linewidth=2)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_flow(self):
        self.plot_functions(self.sigmoid,
                            self.sigmoid_derivative,
                            "Sigmoid Function",
                            "Derivative of Sigmoid ",
                            "Sigmoid Function & Its Derivative")

        self.plot_functions(self.tanh,
                            self.tanh_derivative,
                            "Tanh Function",
                            "Derivative of Tanh ",
                            "Tanh Function & Its Derivative")

        self.plot_functions(self.relu,
                            self.relu_derivative,
                            "ReLU Function",
                            "Derivative of ReLU ",
                            "ReLU Function & Its Derivative")

        self.plot_functions(self.leaky_relu,
                            self.leaky_relu_derivative,
                            "Leaky ReLU Function",
                            "Derivative of Sigmoid ",
                            "Sigmoid Function & Its Derivative")

        self.plot_functions(self.softmax,
                            self.softmax_derivative1,
                            "Softmax Function",
                            "Derivative of Softmax ",
                            "Softmax function and its derivative")

#-----------------------------------------------------------------------------#

def main():
    cls = Lab1ActivationFunction()
    cls.plot_flow()

if __name__ == "__main__":
    main()















