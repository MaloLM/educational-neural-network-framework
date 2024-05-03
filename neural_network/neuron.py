import math
import numpy as np
import neural_network.utils.activation as activations
import neural_network.utils.optimizers as optimizers


class Neuron:

    def __init__(self, id, activation: str) -> None:

        self.id = id
        self.random_generator = np.random.default_rng()

        self.set_activation(activation)

        self.bias = 0
        self.x0 = 1.0
        self.opt = None
        self.weights = None
        self.input_values = []
        self.output = None
        self.weights_gradients = None
        self.bias_gradient = None

    def set_activation(self, activation_name):
        try:
            self.activation = activations.activation_functions[activation_name]
        except KeyError:
            raise ValueError(
                f"Activation function '{activation_name}' is not defined. Please select between {list(activations.activation_functions.keys())}.")

    def set_optimizer(self, opt: str):
        try:
            self.opt = optimizers.optimizers[opt]
        except KeyError:
            raise ValueError(
                f"Activation function '{opt}' is not defined. Please select between {list(optimizers.optimizers.keys())}.")

    def initialize_weights(self, input_size, method='glorot'):
        if method == 'glorot':
            borders = self.glorot_initialization(input_size)
        elif method == 'he':
            borders = self.he_initialization(input_size)
        elif method == 'uniform':
            borders = self.uniform_initialization()
        elif method == 'normal':
            borders = self.normal_initialization()
        elif method == 'sparse':
            self.weights = self.sparse_initialization(input_size)
            return
        else:
            raise ValueError("Unknown initialization method.")

        self.weights = np.random.uniform(
            borders[0], borders[1], size=input_size)

    def glorot_initialization(self, input_size):
        limit = math.sqrt(6 / input_size)
        return -limit, limit

    def he_initialization(self, input_size):
        limit = math.sqrt(2 / input_size)
        return -limit, limit

    def uniform_initialization(self):
        return -1.0, 1.0  # Customize as needed

    def normal_initialization(self):
        mean = 0
        std_dev = 1
        self.weights = self.random_generator.normal(
            mean, std_dev, self.input_size)
        return None  # No borders to return

    def sparse_initialization(self, input_size, sparsity=0.1):
        sparse_weights = np.random.choice(
            [0, 1], size=input_size, p=[1-sparsity, sparsity])
        return np.random.randn(input_size) * sparse_weights

    def initialize_gradients(self, input_size):
        self.weights_gradients = np.zeros(input_size)
        self.bias_gradient = 0.0

    def forward(self):

        if len(self.weights) != len(self.input_values):
            raise ValueError(
                f"Neuron weights and neuron inputs are not of the same number: {len(self.weights)} vs {len(self.input_values)}")

        self.logits = self.linear_combination()

        self.output = self.activate()

    def linear_combination(self):
        if len(self.input_values) != len(self.weights) and len(self.input_values) <= 0:
            print(self.input_values)
            print(self.weights, "\n")
            raise ValueError("Inputs or weights are not initialized.")

        return np.dot(self.weights, self.input_values) + self.bias

    def activate(self):
        if isinstance(self.activation, activations.Softmax):  # DIRTY CODE TO REFACTOR with OOP
            return self.logits
        else:
            return self.activation.function(self.logits)

    def update_weights(self, layer_name):
        self.weights = self.opt.update(
            self.weights, self.weights_gradients, layer_name)

    def update_bias(self, layer_name="None"):
        self.bias = self.opt.update(
            [self.bias], [self.bias_gradient], layer_name)
        self.bias = self.bias[0]

    def zero_grads(self):
        self.bias_gradient = None
        self.weights_gradients = None
