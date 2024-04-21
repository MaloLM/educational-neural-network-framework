import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid activation function on input z.

    Parameters:
        z (array_like): Input data (scalar, vector, or matrix).

    Returns:
        array_like: The sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))


def linear(z):
    """
    Linear activation function.

    Parameters:
        z (array_like): Input data (scalar, vector, or matrix).

    Returns:
        array_like: Same as input z.
    """
    return z


def relu(z):
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters:
        z (float): Input scalar value.

    Returns:
        float: max(0.0, z).
    """
    return max(0.0, z)


def leaky_relu(z, alpha=0.01):
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.

    Parameters:
        z (array_like): Input data (scalar, vector, or matrix).
        alpha (float, optional): Coefficient of leakage. Default is 0.01.

    Returns:
        array_like: Leaky ReLU of z.
    """
    return np.where(z >= 0, z, alpha * z)


def tanh(z):
    """
    Hyperbolic tangent activation function.

    Parameters:
        z (array_like): Input data (scalar, vector, or matrix).

    Returns:
        array_like: The hyperbolic tangent of z.
    """
    return np.tanh(z)


def softmax(z):
    """
    Softmax function for converting input vector to a probability distribution.

    Parameters:
        z (array_like): Input vector, typically raw logit values from the last neural network layer.

    Returns:
        array_like: A vector of probabilities where each element is the probability of corresponding class.
    """
    exp_z = np.exp(z - np.max(z))  # for numerical stability
    sum_exp_z = np.sum(exp_z)
    return exp_z / sum_exp_z


def softplus(z):
    """
    Softplus activation function.

    Parameters:
        z (array_like): Input data (scalar, vector, or matrix).

    Returns:
        array_like: The softplus of z.
    """
    return np.log(1 + np.exp(z))


def elu(z, alpha=1.0):
    """
    Exponential Linear Unit (ELU) activation function.

    Parameters:
        z (array_like): Input data (scalar, vector, or matrix).
        alpha (float, optional): The scaling factor for negative values. Default is 1.0.

    Returns:
        array_like: The ELU of z.
    """
    return np.where(z >= 0, z, alpha * (np.exp(z) - 1))


def gelu(z):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    Parameters:
        z (array_like): Input data (scalar, vector, or matrix).

    Returns:
        array_like: The GELU of z.
    """
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))


activation_functions = {
    "relu": relu,
    "softmax": softmax,
    "sigmoid": sigmoid,
    "linear": linear,
    "gelu": gelu,
    "elu": elu,
    "leaky_relu": leaky_relu,
    "tanh": tanh,
    "softplus": softplus,
}
