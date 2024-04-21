from abc import ABC, abstractmethod

from neural_network.neuron import Neuron


class Layer(ABC):
    """
    Abstract base class for different types of layers in a neural network. Defines common attributes and requires implementation of specific functionality in subclasses.

    Attributes:
        name (str): The name of the layer, which helps in identifying it within the network.
        neurons (list of Neuron): A list containing the neuron objects in this layer.
        activation_func (str): The activation function used by neurons in this layer.
        prev_lay_num_neurons (int): The number of neurons in the previous layer, used to link layers together.

    Methods:
        count_params(): Abstract method that should return the total number of trainable parameters in this layer.
    """

    def __init__(self, name, num_neurons, activation_func) -> None:
        """
        Initializes a Layer with a given number of neurons, each using the specified activation function.

        Args:
            name (str): The name of the layer.
            num_neurons (int): The number of neurons to initialize in the layer.
            activation_func (str): The activation function name for all neurons in the layer.

        Notes:
            All neurons in the layer are initialized with the same activation function. The `Neuron` initialization may raise
            a ValueError if the activation function name is not recognized.
        """

        self.name = name
        self.neurons = []
        self.activation_func = activation_func
        self.prev_lay_num_neurons = None

        for neuron in range(num_neurons):
            self.neurons.append(Neuron(neuron, "relu"))

    def __str__(self) -> str:
        """
        Provides a string representation of the layer, including its name, number of neurons, activation function, and total parameters.

        Returns:
            str: A formatted string that describes the layer.
        """
        return f"name: {self.name}\nneurons: {len(self.neurons)}\nactivation function: {self.activation_func} \nparams: {self.count_params()}"

    @abstractmethod
    def count_params(self):
        """
        Abstract method to be implemented by subclasses that returns the number of trainable parameters in this layer.

        Returns:
            int: The number of trainable parameters in the layer.
        """
        pass
