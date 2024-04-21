from neural_network.layers.layer import Layer


class DenseLayer(Layer):
    """
    Represents a fully connected (dense) layer in a neural network, where each neuron in this layer is connected to all neurons in the previous layer.

    Inherits from:
        Layer: The abstract base class for network layers.

    Attributes:
        name (str): The name of the layer.
        neurons (list of Neuron): A list of neurons in this layer, each using the specified activation function.
        activation_func (str): The activation function applied to the output of each neuron in this layer.
        prev_lay_num_neurons (int): The number of neurons in the previous layer.

    Methods:
        count_params(): Calculates and returns the total number of trainable parameters in this dense layer.
    """

    def __init__(self, name, num_neurons, activation_func) -> None:
        """
        Initializes a DenseLayer with a specified number of neurons and activation function.

        Args:
            name (str): The name of the layer, which can be used for identification within the network.
            num_neurons (int): The total number of neurons in this dense layer.
            activation_func (str): The activation function name for the neurons in this layer.

        Notes:
            The initialization ensures that all neurons are connected to every neuron in the previous layer, and each neuron in this layer will use the given activation function.
        """
        super().__init__(name, num_neurons, activation_func)

    def count_params(self):
        """
        Calculates the total number of trainable parameters in this dense layer, including weights and biases.

        The formula for calculating parameters is given by:
        (number of neurons in the previous layer * number of neurons in this layer) + number of biases (one per neuron).

        Returns:
            int: The total number of trainable parameters in the layer.
        """
        return self.prev_lay_num_neurons * len(self.neurons) + len(self.neurons)
