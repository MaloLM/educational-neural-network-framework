from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms used in training neural networks.

    Optimizers are used to update the weights of the network during training based on the gradients
    calculated by backpropagation. They control how much the weights change in response to the error
    they produce.

    Attributes:
        learning_rate (float): The step size used for each update, a hyperparameter that controls how
                               much the weights change in response to the gradient.
    """

    def __init__(self, learning_rate=0.01, *args, **kwargs):
        """
        Initializes the optimizer with a specified learning rate.

        Args:
            learning_rate (float): The learning rate.
        """
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, params, grads):
        """
        Updates parameters based on their gradients.

        Args:
            params (list or np.ndarray): The parameters to update.
            grads (list or np.ndarray): The gradients of the parameters.
        """
        pass


class SGD(Optimizer):
    """
    Implements stochastic gradient descent (SGD), a simple yet very effective approach to fitting
    neural network weights.

    Attributes:
        name (str): The name of the optimizer.
    """

    def __init__(self, learning_rate=0.0001, name="sgd"):
        """
        Initializes the SGD optimizer with a specified learning rate.

        Args:
            learning_rate (float): The learning rate.
            name (str): A name for the optimizer.
        """
        super().__init__(learning_rate)
        self.name = name

    def update(self, weights, grads, *args, **kwargs):
        """
        Updates the weights using stochastic gradient descent.

        Args:
            weights (list or np.ndarray): The weights to be updated.
            grads (list or np.ndarray): The gradients of the weights.

        Returns:
            list or np.ndarray: The updated weights.
        """
        for param, grad in zip(weights, grads):
            param -= self.learning_rate * grad

        return weights


class GradientDescent:
    """
    A basic gradient descent optimizer not derived from the abstract class, for educational purposes.

    Attributes:
        learning_rate (float): The learning rate.
        name (str): The name of the optimizer.
    """

    def __init__(self, learning_rate=0.0001, name="gradient descent"):
        """
        Initializes the gradient descent optimizer.

        Args:
            learning_rate (float): The learning rate.
            name (str): A name for the optimizer.
        """
        self.learning_rate = learning_rate
        self.name = name

    def update(self, params, gradient, *args, **kwargs):
        """
        Updates the parameters using basic gradient descent.

        Args:
            params (list or np.ndarray): The parameters to update.
            gradient (list or np.ndarray): The gradient of the parameters.

        Returns:
            list or np.ndarray: The updated parameters.
        """
        params = np.array(params)
        gradient = np.array(gradient)

        return params - self.learning_rate * gradient


class GradientDescentWithMomentum:
    """
    Implements gradient descent with momentum, which helps accelerate gradients vectors in the right
    directions, thus leading to faster converging.

    Attributes:
        learning_rate (float): The learning rate.
        momentum (float): The momentum factor.
    """

    def __init__(self, learning_rate=0.0001, momentum=0.8):
        """
        Initializes the gradient descent optimizer with momentum.

        Args:
            learning_rate (float): The learning rate.
            momentum (float): The momentum factor.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def update(self, weights, gradient, layer_name):
        """
        Updates the weights using gradient descent with momentum.

        Args:
            weights (np.ndarray): The weights to be updated.
            gradient (np.ndarray): The gradients of the weights.
            layer_name (str): The name of the layer for which weights are being updated.

        Returns:
            np.ndarray: The updated weights.
        """
        weights = np.array(weights)
        gradient = np.array(gradient)

        if layer_name not in self.velocities:
            self.velocities[layer_name] = np.zeros_like(weights)

        self.velocities[layer_name] = self.momentum * \
            self.velocities[layer_name] + self.learning_rate * gradient

        return weights - self.velocities[layer_name]


class RMSprop(Optimizer):
    """
    Implements RMSprop, an adaptive learning rate method. It is designed to work well even on non-stationary
    problems and with very noisy or sparse gradients.

    Attributes:
        rho (float): Decay rate.
        epsilon (float): Small value to prevent division by zero.
        name (str): The name of the optimizer.
    """

    def __init__(self, learning_rate=0.01, rho=0.9, epsilon=1e-8, name="RMSprop"):
        """
        Initializes RMSprop optimizer with specified parameters.

        Args:
            learning_rate (float): The learning rate.
            rho (float): The decay rate.
            epsilon (float): The epsilon value to prevent division by zero errors.
            name (str): A name for the optimizer.
        """
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.mean_square = None
        self.name = name

    def update(self, params, grads):
        """
        Updates parameters using the RMSprop optimization algorithm.

        Args:
            params (list or np.ndarray): The parameters to update.
            grads (list or np.ndarray): The gradients of the parameters.

        Returns:
            list or np.ndarray: The updated parameters.
        """
        if self.mean_square is None:
            self.mean_square = [np.zeros_like(param) for param in params]

        for param, grad, ms in zip(params, grads, self.mean_square):
            ms[:] = self.rho * ms + (1 - self.rho) * np.square(grad)
            param -= (self.learning_rate / (np.sqrt(ms) + self.epsilon)) * grad


class Adam(Optimizer):
    """
    Implements the Adam optimizer, an algorithm for first-order gradient-based optimization of stochastic
    objective functions, based on adaptive estimates of lower-order moments.

    Attributes:
        beta1 (float): The exponential decay rate for the first moment estimates.
        beta2 (float): The exponential decay rate for the second moment estimates.
        epsilon (float): Small value to prevent division by zero.
        name (str): The name of the optimizer.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="adam"):
        """
        Initializes Adam optimizer with specified parameters.

        Args:
            learning_rate (float): The learning rate.
            beta1 (float): The decay rate for the first moment.
            beta2 (float): The decay rate for the second moment.
            epsilon (float): The epsilon value to prevent division by zero errors.
            name (str): A name for the optimizer.
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.name = name

    def update(self, params, grads, *args, **kwargs):
        """
        Updates parameters using the Adam optimization algorithm.

        Args:
            params (list or np.ndarray): The parameters to update.
            grads (list or np.ndarray): The gradients of the parameters.

        Returns:
            np.array: The updated parameters.
        """
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        alpha_t = self.learning_rate * \
            (np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t))

        new_params = []

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + \
                (1 - self.beta2) * np.square(grad)

            param_update = self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)
            new_param = param - alpha_t * param_update
            new_params.append(new_param)

        return np.array(new_params)


def is_optimizer_defined(opt: str):
    """
    Checks if a given optimizer name is defined within the available optimizers.

    This function is typically used to verify if an optimizer, specified by its name,
    is included in the optimizers dictionary which should contain all supported optimizers
    by the neural network framework or application.

    Args:
        opt (str): The name of the optimizer to check.

    Returns:
        bool: True if the optimizer is defined, False otherwise.
    """
    # Assumes that `optimizers` is a dictionary with keys as optimizer names.
    if opt in list(optimizers.keys()):
        return True
    else:
        return False


optimizers = {
    "gd": GradientDescent(),
    "gradient descent": GradientDescent(),
    "sgd": SGD(),
    "stochastic gradient descent": SGD(),
    "momentum gradient descend": GradientDescentWithMomentum(),
    "momentum": GradientDescentWithMomentum(),
    "RMSprop": RMSprop(),
    "rms": RMSprop(),
    "adam": Adam(),
}
