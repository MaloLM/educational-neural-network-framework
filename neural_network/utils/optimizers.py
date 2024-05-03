from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, learning_rate=0.01, *args, **kwargs):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, params, grads):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.0001, name="sgd"):
        super().__init__(learning_rate)
        self.name = name

    def update(self, weights, grads, *args, **kwargs):
        for param, grad in zip(weights, grads):
            param -= self.learning_rate * grad

        return weights


class GradientDescent:
    def __init__(self, learning_rate=0.0001, name="gradient descent"):
        self.learning_rate = learning_rate
        self.name = name

    def update(self, params, gradient, *args, **kwargs):

        params = np.array(params)
        gradient = np.array(gradient)

        return params - self.learning_rate * gradient


class GradientDescentWithMomentum:
    def __init__(self, learning_rate=0.0001, momentum=0.8):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def update(self, weights, gradient, layer_name):
        weights = np.array(weights)
        gradient = np.array(gradient)

        if layer_name not in self.velocities:
            self.velocities[layer_name] = np.zeros_like(weights)

        self.velocities[layer_name] = self.momentum * \
            self.velocities[layer_name] + self.learning_rate * gradient

        return weights - self.velocities[layer_name]


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.01, rho=0.9, epsilon=1e-8, name="RMSprop"):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.mean_square = None
        self.name = name

    def update(self, params, grads):
        if self.mean_square is None:
            self.mean_square = [np.zeros_like(param) for param in params]

        for param, grad, ms in zip(params, grads, self.mean_square):
            ms[:] = self.rho * ms + (1 - self.rho) * np.square(grad)
            param -= (self.learning_rate / (np.sqrt(ms) + self.epsilon)) * grad


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="adam"):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.name = name

    def update(self, params, grads, *args, **kwargs):
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
