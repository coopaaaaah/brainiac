import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# sets default data type
# sets same random seed
nnfs.init()

# neuron
# O - w \
# O - w - b -> O
# O - w /

# 1.2 - 3.1 \
# 5.1 - 2.1 - 3 -> 35.7
# 2.1 - 8.7 /

# tensor -> object that CAN be represented as an array

# dot product - way to multiply vectors


def output_layer():
    # 4 unique inputs from the previous layer
    inputs = [1, 2, 3, 2.5]

    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]

    biases = [2, 3, 0.5]

    layer_outputs = []

    # is this not the dot product?
    # ei: ([0.2, 0.8, -0.5, 1.0], 2)
    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        # ei: ([1, 0.2], [2, 0.8], [3, -0.5], [2.5, 1.0])
        for n_input, weight in zip(inputs, neuron_weights):
            neuron_output += n_input*weight
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)

    print(layer_outputs)


def dot_product():
    inputs = [[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]

    first_weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]

    second_weights = [
        [0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33],
        [-0.44, 0.73, -0.13]
    ]

    first_biases = [2, 3, 0.5]

    second_biases = [-1, 2, -0.5]

    # move weights into inputs, we want things indexed by the weight set
    layer1_outputs = np.dot(inputs, np.array(first_weights).T) + first_biases

    layer2_outputs = np.dot(layer1_outputs, np.array(second_weights).T) + second_biases

    print(layer2_outputs)


def feature_set():
    # custom for training data set
    X = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

    # set by nnfs init
    # np.random.seed(0)
    layer1 = LayerDense(4, 5)
    layer2 = LayerDense(5, 2)
    layer1.forward(X)
    layer2.forward(layer1.output)
    print(layer2.output)


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# batches
#   calculate stuff in parallel
#   helps with generalization


'''
def create_data(points, classes):
    X = np.zeros((points * classes, 2))  # data matrix (each row = single example)
    y = np.zeros(points * classes, dtype='uint8')  # class labels
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_number
    return X, y

import matplotlib.pyplot as plt

X, y = create_data(100, 3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()
'''


def activation_functions():
    # custom for training data set
    X, y = spiral_data(100, 3)

    layer1 = LayerDense(2, 5)
    activation1 = Activation_ReLU()
    layer1.forward(X)
    activation1.forward(layer1.output)

    import matplotlib.pyplot as plt

    plt.scatter(activation1.output[:, 0], activation1.output[:, 1], c=y, cmap="brg")
    plt.show()


activation_functions()
