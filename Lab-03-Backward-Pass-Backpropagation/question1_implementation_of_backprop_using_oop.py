#------------Import--------------------------#
import cmath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#--------------------------------------------#

class Neuron:
    def __init__(self,activation_in,bias,g_gradient,weights):
        self.g_grad = g_gradient
        self.l_grad = None
        self.z = None
        self.t_grad = None
        self.w = weights
        self.b = bias
        self.s = None
        self.a_in = activation_in


    def sigmoid(self,derivative=False):
        z = np.clip(self.z,-500,500)
        self.s = 1/(1+ np.exp(-z))

        if derivative:
            return self.s * (1-self.s)
        return self.s


    def forward(self):
        if len(self.w) == len(self.a_in)+1:
            w_inp = self.w[:-1]
            w_bias = self.w[-1]
            self.z = w_inp @ self.a_in + w_bias + self.b
        else :
            self.z = self.w @ self.a_in + self.b
        return self.sigmoid()


    def backward(self):
        if self.g_grad is None:
            self.g_grad = 1
        self.g_grad = self.sigmoid(derivative=True) * self.g_grad

        wei_grad = self.a_in * self.g_grad
        act_grad = self.w[:len(self.a_in)] * self.g_grad
        print(self.g_grad)

        if len(self.w) == len(self.a_in) + 1:
            wei_grad=np.append(wei_grad,self.g_grad)
        return wei_grad, act_grad




a = np.array([-1,-2])
w = np.array([2,-3,-3])
b = np.array(0)
grad= np.array([1])
nn = Neuron(a,b,grad,w)
res = nn.forward()
print(res)
res,b = nn.backward()
print(res)




