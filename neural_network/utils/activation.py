from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def function(self, x):
        pass

    @abstractmethod
    def derivative(self, output, *args, **kwargs):
        pass


class Softmax(ActivationFunction):
    def function(self, z):
        exp_z = np.exp(z - np.max(z))  # for numerical stability
        return exp_z / exp_z.sum(axis=0)

    def derivative(self, output, y_true):
        output = np.array(output)
        y_true = np.array(y_true)

        return output - y_true


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
