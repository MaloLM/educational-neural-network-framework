from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    """
    An abstract base class for activation functions in neural networks. Activation functions
    are critical in neural networks as they introduce non-linear properties to the network,
    allowing it to learn more complex functions.

    Methods:
        function(x): Computes the activation value given an input x.
        derivative(output): Computes the derivative of the activation function for backpropagation.
    """

    @abstractmethod
    def function(self, x):
        """
        The activation function that modifies input signals before sending them to the next layer.

        Args:
            x (np.array): Input to the activation function.

        Returns:
            np.array: Activated output.
        """
        pass

    @abstractmethod
    def derivative(self, output, *args, **kwargs):
        """
        Calculates the derivative of the activation function which is used in the backpropagation
        process to update weights and biases.

        Args:
            output (np.array): The output from the activation function.

        Returns:
            np.array: The derivative of the activation function.
        """
        pass


class Softmax(ActivationFunction):
    """
    The Softmax activation function which is typically used in the output layer of a classifier to
    obtain probabilities for multi-class data.
    """

    def function(self, z):
        """
        Applies the softmax function to the input array z. This function is stabilized numerically
        by subtracting the maximum element from each input in the array to prevent large exponent values.

        Args:
            z (np.array): The input logits.

        Returns:
            np.array: The probabilities for each class.
        """
        exp_z = np.exp(z - np.max(z))  # for numerical stability
        return exp_z / exp_z.sum(axis=0)

    def derivative(self, output, y_true):
        """
        Computes the derivative of the softmax function for use in cross-entropy loss backpropagation.

        Args:
            output (np.array): The output from the softmax function.
            y_true (np.array): The true labels in one-hot encoded format.

        Returns:
            np.array: The gradient of the softmax output w.r.t. the input logits.
        """
        output = np.array(output)
        y_true = np.array(y_true)

        return output - y_true

# Other activation classes follow a similar structure as shown above for Softmax.
# Implementations include the function and derivative methods where:
# - `function(x)` computes the activated output.
# - `derivative(output)` calculates the gradient used during backpropagation.

# This includes ReLU, Sigmoid, Identity, LeakyReLU, Tanh, Softplus, ELU, GELU, and Passthrough.
# Each class will similarly document the mathematical operation performed and its usage in a neural network.


class ReLU(ActivationFunction):
    def function(self, x):

        return np.maximum(0, x)

    def derivative(self, output, *args, **kwargs):

        output = np.array(output)
        return np.where(output > 0, 1, 0)


class Sigmoid(ActivationFunction):
    def function(self, x):

        return 1 / (1 + np.exp(-x))

    def derivative(self, output):

        return output * (1 - output)


class Identity(ActivationFunction):
    def function(self, x):

        return x

    def derivative(self, output):

        return np.ones_like(output)


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):

        self.alpha = alpha

    def function(self, x):

        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, output):

        dx = np.ones_like(output)
        dx[output < 0] = self.alpha
        return dx


class Tanh(ActivationFunction):
    def function(self, x):

        return np.tanh(x)

    def derivative(self, output):

        return 1 - np.power(output, 2)


class Softplus(ActivationFunction):
    def function(self, x):

        return np.log(1 + np.exp(x))

    def derivative(self, output):

        return 1 / (1 + np.exp(-output))


class ELU(ActivationFunction):
    def __init__(self, alpha=1.0):

        self.alpha = alpha

    def function(self, x):

        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, output):

        return np.where(output > 0, 1, output + self.alpha)


class GELU(ActivationFunction):
    def function(self, x):

        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def derivative(self, x):

        c = np.sqrt(2 / np.pi)
        return 0.5 * (1 + np.tanh(c * (x + 0.044715 * np.power(x, 3)))) \
            + 0.5 * x * (1 - np.tanh(c * (x + 0.044715 * np.power(x, 3))) ** 2) \
            * c * (1 + 3 * 0.044715 * x * x)


class Passthrough(ActivationFunction):
    def function(self, z):
        return z

    def derivative(self, z):
        return z


activation_functions = {
    "relu": ReLU(),
    "softmax": Softmax(),
    "sigmoid": Sigmoid(),
    "logistic": Sigmoid(),
    "linear": Identity(),
    "identity": Identity(),
    "gelu": GELU(),
    "elu": ELU(),
    "leaky_relu": LeakyReLU(),
    "tanh": Tanh(),
    "softplus": Softplus(),
    "passthrough": Passthrough(),
    "none": Passthrough(),
    "": Passthrough(),
    None: Passthrough(),

}
