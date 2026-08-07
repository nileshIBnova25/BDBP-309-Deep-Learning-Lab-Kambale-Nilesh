#----------------------------Import----------------------------------------#
import numpy as np
import pandas as pd
import cmath as math
#--------------------------------------------------------------------------#


class BackwardPass:
    def __init__(self, X, Y, config):
        self.X=X
        self.Y=Y
        self.config=config
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))


def sigmoid_derive(val,grad=None):
    ''' Moving Along Compute'''

    if grad is None:
        grad=1
    mul1 = val * -1                 # multiply by -1 gate
    exp  = math.exp(mul1)           # exponential gate
    add1 = exp +1                   # addition gate
    #div1 = 1/add1                   # division by 1/x

    '''Moving Along Gradient'''
    grd1 = -1 / add1**2             # gradient of division * global gradient
    grd2 = 1 * grd1                 # gradient of add * global gradient
    grd3 = math.exp(mul1) * grd2    # gradient of exponent * global
    grd4 = -1 * grd3                # gradient of *-1 * global
    return grd4

result = sigmoid_derive(1)


print(result)

val1=[-2,-3,-3]
val2=[-1,-2]
print(val1[-1])

def linear_derive(val1,val2,grad=None):
    if grad is None:
        grad=1
    vec1=[]
    vec2=[]
    for i in range(len(val2)):
        vec1.append(val2[i]*grad)
        vec2.append(val1[i]*grad)

    if len(val1) == len(val2)+1:
        vec1.append(val1[-1]*grad)
    return vec1, vec2
res1,res2 = linear_derive(val1,val2,result)
print(np.array(res1))
print(np.array(res2))


'''class-room example for the computational graph'''
vec = [5,-4,-2]
def class_example(vec,grad=None):
    if grad is None:
        grad=1
    # along computation
    mul = vec[0] * vec[1]          # multiplication
    add = mul * vec[2]             # addition

    # along gradient
    grd3 = 1 * grad                # thus addition
    grd2 = vec[0] * grad
    grd1 = vec[1] * grad

    return [grd1, grd2, grd3]
res_cls = class_example(vec)
print(res_cls)

































class Sigmoid:
    def __init__(self):
        pass

    def sigmoid_add(self,val):
        dat = {}
        add = val + 1
        dat["add"] = add
        mul1= add*-1 # multiply by minus 1
        dat["mul_by_minus1"] = mul1
        exp =mul1*add
        dat["exp"] = exp








