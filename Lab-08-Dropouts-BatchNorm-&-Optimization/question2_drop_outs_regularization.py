#--------------imports---------------#
import numpy as np
#------------------------------------#

#--------------Dropouts--------------#
class Dropout:
    def __init__(self,p = 0.5):
        self.p = p
    def forward(self,x, train = True):
        if not train:
            return x
        self.mask =(np.random.rand(*x.shape) > self.p)

        return x * self.mask / (1-self.p)
    def backward(self,dy):
        return dy * self.mask / (1-self.p)
#------------------------------------#
def main():
    dp = Dropout(p=0.5)
    x = np.array([
        [15,28,21,47],
        [51,32,15,65]
    ])
    print(f"Input: \n {x}")
    print(f"Training: \n {dp.forward(x,train=True)}")
    print(f"Test: \n {dp.forward(x,train=False)}")
if __name__ == "__main__":
    main()
#-------------------------------------#


