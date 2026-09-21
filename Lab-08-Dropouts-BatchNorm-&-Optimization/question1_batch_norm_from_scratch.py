#---------Imports-------------------------#
import numpy as np
#-----------------------------------------#
#-----------------------------------------#
class BatchNorm:
    def __init__(self,n,eps=1e-5):
        self.g = np.ones(n)
        self.b = np.ones(n)
        self.eps = eps

    def forward(self,x):
        self.mu = x.mean(axis=0)
        self.var = x.var(axis=0)
        self.xn = (x - self.mu) / np.sqrt(self.var + self.eps)

        return self.g * self.xn + self.b
#-----------------------------------------#

#----Main---------------------------------#
def main():
    x = np.random.randn(4,3)*10 + 5
    bn = BatchNorm(3)
    y = bn.forward(x)
    print(f"Input: \n {x}")
    print(f"Mean: {y.mean(axis=0)}")
    print(f"Variance: {y.var(axis=0)}: ")
    print(f"Batch Normalization: \n {y}")
if __name__ == '__main__':
    main()
#-----------------------------------------#
