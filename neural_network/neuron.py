import math
import numpy as np
import neural_network.utils.activation as activations
import neural_network.utils.optimizers as optimizers


class Neuron:
    """
    Represents a single neuron within a neural network layer.

    A neuron is a fundamental unit of the brain and in artificial neural networks, it similarly acts as
    a basic processing unit. It receives input, processes it through a weighted sum (linear combination)
    and an activation function, and outputs the result.

    Attributes:
        id (int): An identifier for the neuron.
        activation (callable): The activation function of the neuron, determining how input is transformed.
        bias (float): The bias value of the neuron, part of its trainable parameters.
        x0 (float): A constant input (often 1.0) used for the bias weight.
        opt (callable): The optimizer used for updating weights and biases during training.
        weights (np.ndarray): The weights assigned to inputs of the neuron.
        input_values (list): The inputs received by the neuron.
        output (float): The output of the neuron after applying the activation function.
        weights_gradients (np.ndarray): Gradients of the weights used for optimization.
        bias_gradient (float): Gradient of the bias used for optimization.
    """

    def __init__(self, id, activation: str) -> None:
        """
        Initializes a Neuron with an id and activation function.

        Args:
            id (int): The identifier for the neuron.
            activation (str): The name of the activation function to be used.
        """
        self.id = id
        self.random_generator = np.random.default_rng()

        self.__set_activation(activation)

        self.bias = 0
        self.x0 = 1.0
        self.opt = None
        self.weights = None
        self.input_values = []
        self.output = None
        self.weights_gradients = None
        self.bias_gradient = None

    def __set_activation(self, activation_name):
        """
        Sets the activation function of the neuron based on the provided activation name.

        Raises:
            ValueError: If the specified activation function is not recognized.
        """
        try:
            self.activation = activations.activation_functions[activation_name]
        except KeyError:
            raise ValueError(
                f"Activation function '{activation_name}' is not defined. Please select between {list(activations.activation_functions.keys())}.")

    def set_optimizer(self, opt: str):
        """
        Sets the optimizer for the neuron based on the provided optimizer name.

        Raises:
            ValueError: If the specified optimizer is not recognized.
        """
        try:
            self.opt = optimizers.optimizers[opt]
        except KeyError:
            raise ValueError(
                f"Activation function '{opt}' is not defined. Please select between {list(optimizers.optimizers.keys())}.")

    def initialize_weights(self, input_size, method='glorot'):
        """
        Initializes weights according to the specified method and input size.

        Args:
            input_size (int): The number of inputs the neuron is expected to receive.
            method (str): The method to use for initialization (e.g., 'glorot', 'he').

        Raises:
            ValueError: If an unknown initialization method is specified.
        """
        if method == 'glorot':
            borders = self.__glorot_initialization(input_size)
        elif method == 'he':
            borders = self.__he_initialization(input_size)
        elif method == 'uniform':
            borders = self.__uniform_initialization()
        elif method == 'normal':
            borders = self.__normal_initialization()
        elif method == 'sparse':
            self.weights = self.__sparse_initialization(input_size)
            return
        else:
            raise ValueError("Unknown initialization method.")

        self.weights = np.random.uniform(
            borders[0], borders[1], size=input_size)

    def __glorot_initialization(self, input_size):
        """
        Initializes weights using the Glorot (Xavier) uniform initialization method.

        This method sets the initial weights to values drawn uniformly from a range
        that considers the number of input connections, helping to keep the variance
        of the outputs of each neuron at initialization stable.

        Args:
            input_size (int): The number of input connections to the neuron.

        Returns:
            tuple: A tuple containing the lower and upper bounds for the uniform distribution.
        """
        limit = math.sqrt(6 / input_size)
        return -limit, limit

    def __he_initialization(self, input_size):
        """
        Initializes weights using the He initialization method.

        This method is particularly suited for layers followed by a ReLU activation,
        as it sets the scale of the initial weights according to the number of inputs,
        which helps avoid the vanishing gradients problem.

        Args:
            input_size (int): The number of input connections to the neuron.

        Returns:
            tuple: A tuple containing the lower and upper bounds for the uniform distribution.
        """
        limit = math.sqrt(2 / input_size)
        return -limit, limit

    def __uniform_initialization(self):
        """
        Initializes weights using a simple uniform distribution over a fixed range.

        Returns:
            tuple: A tuple containing the lower and upper bounds for the uniform distribution.
        """
        return -1.0, 1.0  # Customize as needed

    def __normal_initialization(self):
        """
        Initializes weights using a normal distribution with a mean of 0 and standard deviation of 1.

        This method sets the initial weights to values drawn from a normal distribution,
        not typically bound to input size, and should be used cautiously due to potential
        issues with the scale of the weights.

        Returns:
            None: This method directly modifies the weights array and does not return any value.
        """
        mean = 0
        std_dev = 1
        self.weights = self.random_generator.normal(
            mean, std_dev, self.input_size)
        return None  # No borders to return

    def __sparse_initialization(self, input_size, sparsity=0.1):
        """
        Initializes weights using a sparse method where a certain percentage of the weights are set to zero.

        This method can help in promoting simpler and more interpretable models, reducing the chance
        of overfitting by reducing the number of active connections at the start.

        Args:
            input_size (int): The number of input connections to the neuron.
            sparsity (float): The fraction of weights to be set to zero.

        Returns:
            np.ndarray: An array of initialized weights, some of which are set to zero.
        """
        sparse_weights = np.random.choice(
            [0, 1], size=input_size, p=[1-sparsity, sparsity])
        return np.random.randn(input_size) * sparse_weights

    def initialize_gradients(self, input_size):
        """
        Initializes the gradients of the weights and bias to zeros.

        This method is used to reset the gradients before each new update cycle during training,
        ensuring that gradient values from previous updates do not interfere with new computations.

        Args:
            input_size (int): The number of input connections to the neuron.
        """
        self.weights_gradients = np.zeros(input_size)
        self.bias_gradient = 0.0

    def forward(self):
        """
        Processes the inputs through the neuron, applying the linear combination and then the activation function.

        Raises:
            ValueError: If the number of weights does not match the number of inputs.
        """
        if len(self.weights) != len(self.input_values):
            raise ValueError(
                f"Neuron weights and neuron inputs are not of the same number: {len(self.weights)} vs {len(self.input_values)}")

        self.logits = self.__linear_combination()

        self.output = self.__activate()

    def __linear_combination(self):
        """
        Computes the weighted sum of inputs plus bias.

        Returns:
            float: The result of the linear combination.

        Raises:
            ValueError: If inputs or weights are not properly initialized.
        """
        if len(self.input_values) != len(self.weights) and len(self.input_values) <= 0:
            print(self.input_values)
            print(self.weights, "\n")
            raise ValueError("Inputs or weights are not initialized.")

        return np.dot(self.weights, self.input_values) + self.bias

    def __activate(self):
        """
        Applies the activation function to the linear combination output.

        Returns:
            float: The activated output.
        """
        if isinstance(self.activation, activations.Softmax):  # DIRTY CODE TO REFACTOR with OOP
            return self.logits
        else:
            return self.activation.function(self.logits)

    def update_weights(self, layer_name):
        """
        Updates the weights using the assigned optimizer.

        Args:
            layer_name (str): The name of the layer to which this neuron belongs, used for logging or tracking.
        """
        self.weights = self.opt.update(
            self.weights, self.weights_gradients, layer_name)

    def update_bias(self, layer_name="None"):
        """
        Updates the bias using the assigned optimizer.

        Args:
            layer_name (str): The name of the layer to which this neuron belongs, used for logging or tracking.
        """
        self.bias = self.opt.update(
            [self.bias], [self.bias_gradient], layer_name)
        self.bias = self.bias[0]

    def zero_grads(self):
        """
        Resets the gradients of the weights and bias to prepare for the next update cycle.
        """
        self.bias_gradient = None
        self.weights_gradients = None
