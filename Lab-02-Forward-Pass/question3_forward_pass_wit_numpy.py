"""
Consider the following two networks.  W is a matrix, x is a vector, z is a vector,
and a is a vector. y^ is a scalar and a final prediction. Initialize x, w randomly,
z is a dot product of x and w, a is ReLU(z).  Initialize X and W randomly. Every neuron
has a bias term.
Implement forward pass for the above two networks. Print activation values for each
neuron at each layer. Print the loss value (y^).
"""
#--------------------------------------------------------------#
import numpy as np

#--------------------------------------------------------------#

class ForwardPass:
    def __init__(self,X):
        self.X=X
        self.config= self.get_config()
        self.layers=self.config["layers"]

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))


    def get_config(self):
        layers = input("Enter the number of layers: ")
        layers = int(layers)
        config = {}
        config["layers"] = layers
        for i in range(layers):

            v = input(f"Enter the numbers of neuron in layer {i + 1}:")
            v = int(v)
            if i == 0:
                config[f"weight{i + 1}"] = np.random.random((v, len(self.X)))
                config[f"bias{i + 1}"] = np.random.random(v)
            else:
                config[f"weight{i + 1}"] = np.random.random((v, len(config[f"weight{i}"])))
                config[f"bias{i + 1}"] = np.random.random(v)
        return config

    def forward_pass(self,X=None,current_layer=1,config=None):
        if config is None:
            config = self.config
        if X is None:
            X = self.X
        w = config[f"weight{current_layer}"]
        b = config[f"bias{current_layer}"]
        a = self.sigmoid((w @ X) + b)

        print(a)

        if config['layers'] == current_layer:
            return a

        return self.forward_pass(a,current_layer+1,config=config)

    def validate_by_class_example(self):
        w1 = np.array([[0.1, -1.1], [-0.1, 0.4], [0.2, 1.1]])
        w2 = np.array([[0.2, 0.1, -0.2], [0.3, -0.1, -0.1]])
        w3 = np.array([0.3, -0.3])
        b1 = np.array([0,0,0])
        b2 = np.array([0,0])
        b3 = np.array([0])
        config = {"layers":3,
                  "weight1": w1, "weight2": w2, "weight3": w3,
                  "bias1": b1, "bias2": b2, "bias3": b3
                  }
        ans = self.forward_pass(config=config)
        return ans

def main():
    X=np.array([0.3,-1.2])
    cls = ForwardPass(X)
    #answer=cls.validate_by_class_example()
    #print(answer)
    a=cls.forward_pass()

if __name__ == "__main__":
    main()