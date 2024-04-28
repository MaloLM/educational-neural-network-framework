from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    def __init__(self, learning_rate=0.01, *args, **kwargs):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, params, grads):

        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, name="sgd"):
        super().__init__(learning_rate)
        self.name = name

    def update(self, params, grads):
        # Itère sur tous les paramètres et leurs gradients correspondants
        for param, grad in zip(params, grads):
            # Mise à jour des paramètres avec SGD
            param -= self.learning_rate * grad


class GradientDescent:
    def __init__(self, learning_rate=0.01, name="gradient descent"):
        self.learning_rate = learning_rate
        self.name = name

    def update(self, weights, gradient):
        res = weights - self.learning_rate * gradient
        return res


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, name="momentum gradient descend"):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        self.name = name

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in params]

        for param, grad, vel in zip(params, grads, self.velocity):
            vel[:] = self.momentum * vel + self.learning_rate * grad
            param -= vel


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

    def update(self, params, grads):
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        alpha_t = self.learning_rate * \
            np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for param, grad, m, v in zip(params, grads, self.m, self.v):
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * np.square(grad)
            param -= alpha_t * m / (np.sqrt(v) + self.epsilon)


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
    "momentum gradient descend": Momentum(),
    "momentum": Momentum(),
    "RMSprop": RMSprop(),
    "rms": RMSprop(),
    "adam": Adam(),
}
