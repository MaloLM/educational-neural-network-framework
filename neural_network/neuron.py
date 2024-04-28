
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
        self.weights = []
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

    def initialize_weights(self, input_size):

        def glorot_initialization(input_size):
            return -1/math.sqrt(input_size), 1/math.sqrt(input_size)

        def he_initialization(input_size):
            return -math.sqrt(6/input_size), math.sqrt(6/input_size)

        border = glorot_initialization(input_size)

        self.weights = self.random_generator.uniform(
            border[0], border[1], input_size)

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

        if not (self.input_values and self.weights).any():
            raise ValueError("Inputs or weights are not initialized.")

        net_input = np.dot(self.weights, self.input_values) + self.bias
        return net_input

    def activate(self):
        if isinstance(self.activation, activations.Softmax):  # DIRTY CODE TO REFACTOR with OOP
            return self.logits
        else:
            return self.activation.function(self.logits)

    def update_weights(self):
        self.weights = self.opt.update(self.weights, self.weights_gradients)

    def update_bias(self):
        self.bias = self.opt.update(self.bias, self.bias_gradient)
