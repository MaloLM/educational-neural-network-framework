from neural_network.layers.layer import Layer


class OutputLayer(Layer):
    """
    Represents the output layer of a neural network. This layer is responsible for producing the final output of the network,
    using a specified activation function.

    Inherits from:
        Layer: The abstract base class for network layers.

    Methods:
        count_params(): Overrides the abstract method to calculate the number of trainable parameters specific to the output layer.
    """

    def __init__(self, name, num_neurons, activation_func) -> None:
        """
        Initializes an OutputLayer with a given number of neurons and a specified activation function.

        Args:
            name (str): The name of the layer.
            num_neurons (int): The number of neurons in this output layer.
            activation_func (str): The activation function name for the neurons in this layer.

        Notes:
            Inherits the initialization of Layer and passes the necessary parameters to the base class.
        """
        super().__init__(name, num_neurons, activation_func)

    def count_params(self):
        """
        Calculates the total number of trainable parameters in the output layer. It considers the number of connections
        from the previous layer's neurons to this layer's neurons plus the bias for each neuron.

        Returns:
            int: The total number of trainable parameters which includes all weights from previous layer neurons to each
                 neuron in this layer plus a bias term for each neuron in this layer.
        """
        return self.prev_lay_num_neurons * len(self.neurons) + len(self.neurons)
