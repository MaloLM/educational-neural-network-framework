import numpy as np
import neural_network.utils.activation as activation


class Neuron:
    """
    Represents a single neuron within a neural network layer, encapsulating the activation function, bias, and weights.

    Attributes:
        id (int): A unique identifier for the neuron.
        activation_func (callable): The activation function to use for this neuron.
        bias (float): The bias term for the neuron, initialized randomly.
        weights (numpy.ndarray): The weights for the inputs connected to this neuron.
        input_values (list): The inputs to this neuron from the previous layer.

    Methods:
        initialize_weights(num_inputs): Initializes weights for the neuron based on the number of inputs.
        forward_propagate(): Conducts forward propagation through this neuron by calculating the activated output.
        summing(): Computes the weighted sum of inputs plus the bias.
        activate(): Applies the activation function to the sum of inputs.
    """

    def __init__(self, id, activation_func: str) -> None:
        """
        Initializes a Neuron instance with a specified activation function.

        Args:
            id (int): A unique identifier for the neuron.
            activation_func (str): The name of the activation function to use.

        Raises:
            ValueError: If the specified activation function is not available.
        """
        self.id = id
        try:
            self.activation_func = activation.activation_functions[activation_func]
        except KeyError:
            raise ValueError(
                f"Activation function '{activation_func}' is not defined. Please select between {list(activation.activation_functions.keys())}.")

        self.bias = np.random.rand()
        self.x0 = 1.0
        self.weigths = []
        self.input_values = []

    def initialize_weights(self, num_inputs):
        """
        Initializes random weights for this neuron based on the number of input connections.

        Args:
            num_inputs (int): The number of input connections to this neuron.
        """
        self.weights = np.random.randn(num_inputs)

    def forward_propagate(self):
        """
        Processes the inputs through this neuron using its activation function.

        Raises:
            ValueError: If the number of weights does not match the number of inputs.
        """
        if len(self.weigths) != len(self.input_values):
            raise ValueError(
                "Neuron weigths and neuron inputs are not of the same number.")
        net_input = self.summing()
        self.activate(net_input)

    def summing(self):
        """
        Calculates the weighted sum of the neuron's inputs plus its bias.

        Returns:
            float: The net input to the activation function.

        Raises:
            ValueError: If inputs or weights have not been initialized.
        """
        if not self.input_values or not self.weights:
            raise ValueError("Inputs or weights are not initialized.")

        net_input = np.dot(self.weights, self.input_values) + self.bias
        return net_input

    def activate(self, net_input):
        """
        Applies the activation function to the net input to get the neuron's output.

        Returns:
            float: The output of the neuron after activation.
        """
        return self.activation_func(net_input)
