from neural_network.layers.layer import Layer


class InputLayer(Layer):
    """
    Represents the input layer of a neural network, primarily handling the input data size without containing any neurons or activation functions.

    Inherits from:
        Layer: The abstract base class for network layers.

    Attributes:
        input_size (int): The size of the input data that this layer will receive.

    Methods:
        count_params(): Returns the number of trainable parameters in the input layer, which is always zero as this layer contains no neurons.
    """

    def __init__(self, name, input_size) -> None:
        """
        Initializes an InputLayer that sets up the input dimensions for a neural network.

        Args:
            name (str): The name of the layer, useful for identifying the layer within the network.
            input_size (int): The number of input features or the size of the input data vector.

        Notes:
            This layer does not have neurons or an activation function since its primary role is to receive input data.
        """
        super().__init__(name, 0, None)
        self.input_size = input_size

    def count_params(self):
        """
        Overrides the count_params method from Layer to return zero, as there are no trainable parameters in an input layer.

        Returns:
            int: Returns 0, indicating no trainable parameters in the input layer.
        """
        return 0
