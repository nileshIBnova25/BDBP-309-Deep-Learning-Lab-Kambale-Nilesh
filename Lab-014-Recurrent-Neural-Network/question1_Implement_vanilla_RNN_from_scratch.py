"""
Learning Goals
In this lab, you will learn the basic architecture of RNN.
By the end of this lab, you should be able to
1. Form intuitions about processing sequence or time-series data
2. Understand how RNN works by unwrapping the sequence information
Exercises
1. Implement vanilla RNN from scratch
https://medium.com/@thisislong/building-a-recurrent-neural-network-from-scratch-ba9b2
7a42856
"""
#---------------------------------Import-----------------------------#
import numpy as np
import string

from torch.nn.functional import softmax

#--------------------------------------------------------------------#

inputs = np.array([
    ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
    ["Z","Y","X","W","V","U","T","S","R","Q","P","O","N","M","L","K","J","I","H","G","F","E","D","C","B","A"],
    ["B","D","F","H","J","L","N","P","R","T","V","X","Z","A","C","E","G","I","K","M","O","Q","S","U","W","Y"],
    ["M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L"],
    ["H","G","F","E","D","C","B","A","L","K","J","I","P","O","N","M","U","T","S","R","Q","X","W","V","Z","Y"]
])

expected = np.array([
    ["B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A"],
    ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
    ["C","E","G","I","K","M","O","Q","S","U","W","Y","A","B","D","F","H","J","L","N","P","R","T","V","X","Z"],
    ["N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L","M"],
    ["I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H"]
])
#---------------------------------------------------------------------#

#------------Define-One-Hot-Encoding----------------------------------#
def string_to_one_hot(inputs):
    char_to_index = {char: i for i, char in enumerate(string.ascii_uppercase)}

    one_hot_inputs = []
    for row in inputs:
        one_hot_list = []
        for char in row:
            if char.upper() in char_to_index:
                one_hot_vector = np.zeros((len(string.ascii_uppercase), 1))
                one_hot_vector[char_to_index[char.upper()]] = 1
                one_hot_list.append(one_hot_vector)
        one_hot_inputs.append(one_hot_list)
    return np.array(one_hot_inputs)
#----------------------------------------------------------------------#

#-------------------Class-InputLayers----------------------------------#
class InputLayer:
    inputs: np.ndarray
    U: np.ndarray = None
    delta_U: np.ndarray = None

    def __init__(self, inputs,hidden_size):
        self.inputs = inputs
        self.U = np.random.uniform(low=0, high=1, size=(hidden_size, len(inputs[0])))
        self.delta_U = np.zeros_like(self.U)

    def get_input(self, time_step):
        return self.inputs[time_step]

    def weighted_sum(self, time_step):
        return self.U @ self.get_input(time_step)

    def calculate_delta_per_step(self, time_step, delta_weighted_sum):
        # (h_dimension, 1) @ (1, input_size) = (h_dimension, input_size)
        self.delta_U += delta_weighted_sum @ self.get_input(time_step).T

    def update_weights_and_bias(self, learning_rate):
        self.U -= learning_rate * self.delta_U
#------------------------------------------------------------------------#

#------------------------------------------------------------------------#
class HiddenLayer:
    def __init__(self, vocab_size, size):
        self.W = np.random.uniform(low=-0.1, high=0.1, size=(size, size))
        self.bias = np.random.uniform(low=0, high=1, size=(size, 1))
        self.states = np.zeros(shape=(vocab_size, size, 1))
        self.next_delta_activation = np.zeros(shape=(size, 1))
        self.delta_bias = np.zeros_like(self.bias)
        self.delta_W = np.zeros_like(self.W)

    def get_hidden_state(self, time_step):
        # If starting out at the beginning of the sequence , a[t-1] will return zeros
        if time_step < 0:
            return np.zeros_like(self.states[0])
        return self.states[time_step]

    def set_hidden_state(self, time_step, activation):
        self.states[time_step] = activation

    def activate(self, weighted_input, time_step):
        previous_hidden_state = self.get_hidden_state(time_step - 1)
        # W @ h_prev = (h_dimension, h_dimension) @ (h_dimension, 1) = (h_dimension,1)
        weighted_hidden_state = self.W @ previous_hidden_state
        # (h_dimension, 1) + (h_dimension, 1) + (h_dimension,1 ) = (h_dimension, 1)
        weighted_sum = weighted_input + weighted_hidden_state + self.bias
        activation = np.tanh(weighted_sum) # (h_dimension,1)
        self.set_hidden_state(time_step, activation)
        return activation

    def calculate_deltas_per_step(self, time_step, delta_output):
        # (h_dimension, 1) + (h_dimension, 1) = (h_dimension, 1)
        delta_activation = delta_output + self.next_delta_activation
        # (h_dimension, 1) * scalar = (h_dimension, 1)
        delta_weighted_sum = delta_activation * (1 - self.get_hidden_state(time_step) ** 2)

        # (h_dimension, h_dimension) @ (h_dimension, 1) = (h_dimension, 1)
        self.next_delta_activation = self.W.T @ delta_weighted_sum

        # (h_dimension , 1) @ (1, h_dimension) = (h_dimension, h_dimension)
        self.delta_W += delta_weighted_sum @ self.get_hidden_state(time_step - 1).T

        # derivative of hidden bias is the same as dL_ds
        self.delta_bias += delta_weighted_sum
        return delta_weighted_sum

    def update_weights_and_bias(self, learning_rate):
        self.W -= learning_rate * self.delta_W
        self.bias -= learning_rate * self.delta_bias
#-----------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------#
class OutputLayer:
    def __init__(self, size, hidden_size):
        self.V = np.random.uniform(low=0, high=1, size=(size, hidden_size))
        self.bias = np.random.uniform(low=0, high=1, size=(size, 1))
        self.states = np.zeros(shape=(size, size, 1))
        self.delta_bias = np.zeros_like(self.bias)
        self.delta_V = np.zeros_like(self.V)

    def predict(self, hidden_state, time_step):
        # V @ h = (input_size, h_dimension) @ (h_dimension, 1) = (input_size, 1)
        # (input_size, 1) + (input_size, 1) = (input_size, 1)
        output = self.V @ hidden_state + self.bias
        exp = np.exp(output - np.max(output))
        prediction = exp / np.sum(exp)

        self.set_state(time_step, prediction)
        return prediction

    def get_state(self, time_step):
        return self.states[time_step]

    def set_state(self, time_step, prediction):
        self.states[time_step] = prediction

    def calculate_deltas_per_step(self, expected, hidden_state, time_step):
        # dL_do = dL_dyhat * dyhat_do = derivative of loss function * derivative
        # dL_do = step.y_hat - expected[step_number]

        delta_output = self.get_state(time_step) - expected # (input_size, 1)

        # (input_size, 1 ) @ (1, hidden_size) = (input_size, hidden_size)
        self.delta_V += delta_output @ hidden_state.T

        # dL_dc += dL_do
        self.delta_bias += delta_output
        return self.V.T @ delta_output

    def update_weights_and_bias(self, learning_rate):
        self.V -= learning_rate * self.delta_V
        self.bias -= learning_rate * self.delta_bias
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
class VanillaRNN:
    def __init__(self, vocab_size, hidden_size, alpha):
        self.hidden_layer = HiddenLayer(vocab_size, hidden_size)
        self.output_layer = OutputLayer(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.input_layer = None

    def feed_forward(self, inputs):
        if self.input_layer is None:
            self.input_layer = InputLayer(inputs, self.hidden_size)
        else:
            self.input_layer.inputs = inputs

        for step in range(len(inputs)):
            weighted_input = self.input_layer.weighted_sum(step)
            activation = self.hidden_layer.activate(weighted_input,step)
            self.output_layer.predict(activation, step)
        return self.output_layer

    def backpropgation(self, expected):
        for step_number in reversed(range(len(expected))):
            delta_output = self.output_layer.calculate_deltas_per_step(
                expected[step_number],
                self.hidden_layer.get_hidden_state(step_number),
                step_number)
            delta_weighted_sum = self.hidden_layer.calculate_deltas_per_step(
                step_number, delta_output
            )

            self.input_layer.calculate_delta_per_step(
                step_number, delta_weighted_sum
            )

        self.output_layer.update_weights_and_bias(self.alpha)
        self.hidden_layer.update_weights_and_bias(self.alpha)
        self.input_layer.update_weights_and_bias(self.alpha)

        self.output_layer.delta_V.fill(0)
        self.output_layer.delta_bias.fill(0)
        self.hidden_layer.delta_W.fill(0)
        self.hidden_layer.delta_bias.fill(0)
        self.hidden_layer.next_delta_activation.fill(0)

        self.input_layer.delta_U.fill(0)

    def loss(self, y_hat,y):
        """
        Cross-entropy loss function
        """
        return sum(
            -np.sum(y[i] * np.log(y_hat[i]))
            for i in range(len(y))
        )

    def train(self, inputs, expected, epochs):
        for epoch in range(epochs):
            print(f"epoch= {epoch}")
            for idx, input in enumerate(inputs):
                y_hats = self.feed_forward(input)
                self.backpropgation(expected[idx])
                print(f"Loss round: {self.loss([y for y in y_hats.states], expected[idx])}"
                      )
#-----------------------------------------------------------------------------------#

def main():
    # Conver inputs and expected outputs into one-hot encoding
    one_hot_inputs = string_to_one_hot(inputs)
    one_hot_expected = string_to_one_hot(expected)

    # Initalize Vanilla RNN
    rnn = VanillaRNN(
        vocab_size=len(string.ascii_uppercase),
        hidden_size=128,
        alpha=0.001
    )

    # Train the RNN
    rnn.train(one_hot_inputs, one_hot_expected, epochs=100)

    # Test the trained RNN with a new sequences
    new_inputs = np.array([["B", "C", "D"],
                           ["A", "B", "C"],
                           ["X", "Y", "Z"],
                           ["M", "N", "O"]
    ])

    for input_sequence in string_to_one_hot(new_inputs):
        predictions = rnn.feed_forward(input_sequence)
        print("\nInput sequence")

        for step in range(len(input_sequence)):
            output = np.argmax(predictions.states[step])
            print(
                f"Step {step + 1} :"
                f"Predicted Character = {string.ascii_uppercase[output]}"
            )

if __name__ == "__main__":
    main()






