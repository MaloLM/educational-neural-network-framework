from neural_network.layers.layer import Layer
from itertools import chain
import numpy as np


class InputLayer(Layer):
    """
    Represents the input layer of a neural network which is the first layer in the network architecture.

    The input layer is responsible for receiving and optionally pre-processing the raw input data before
    passing it to subsequent layers. Unlike other layers, it does not have neurons in the traditional sense
    and thus does not perform any parameterized transformations on its inputs.

    Attributes:
        input_size (int): The expected number of elements in the input data. This defines the width of the layer.

    Methods:
        count_params: Returns the number of trainable parameters, which is 0 for the input layer.
        flatten: Flattens a nested list or array-like structure into a 1D list that matches the input size.
        forward: Processes the input by flattening it and then passes it unchanged to the next layer.
        get_width: Returns the size of the input that this layer is set to handle.
    """

    def __init__(self, name, input_size) -> None:
        """
        Initializes the InputLayer with a name and expected input size.

        Args:
            name (str): The name of the layer.
            input_size (int): The number of input features expected by the layer.
        """

        super().__init__(name, 0, None)
        self.input_size = input_size

    def count_params(self):
        """Returns the number of trainable parameters in the layer, which is always 0 for an input layer."""
        return 0

    def flatten(self, input):
        """
        Flattens the input into a 1-dimensional list or array, ensuring it matches the expected input size.

        Args:
            input (list or np.ndarray): The input data to be flattened.

        Returns:
            list: A flattened version of the input data.

        Raises:
            ValueError: If the flattened input does not match the expected input size.
        """
        if isinstance(input, list):
            try:
                flat_output = list(chain.from_iterable(input))
            except TypeError:
                flat_output = [item for sublist in input for item in (
                    sublist if isinstance(sublist, list) else [sublist])]
        else:
            flat_output = np.ravel(input)

        if len(flat_output) == self.input_size:
            return flat_output
        else:
            raise ValueError(
                f"The length of the flattened input is {len(flat_output)}, which does not match the expected size of {self.input_size}. Ensure that your input data is correctly formatted and contains the right number of elements.")

    def forward(self, input):
        """
        Processes the input by flattening it and storing both the original and flattened input for potential backpropagation.

        Args:
            input: The input data to the layer.

        Returns:
            tuple: The flattened input and None (since there are no weights or transformations applied).
        """
        flattened_input = self.flatten(input)
        self.inputs.append(input)
        self.outputs.append(flattened_input)

        return flattened_input, None

    def initiate_weights_and_gradients(self, *args, **kwargs):
        """Placeholder method; no implementation needed as the input layer does not have weights or gradients."""
        pass

    def initialize_gradients(self):
        """Placeholder method; no implementation needed as the input layer does not manage gradients."""
        pass

    def get_width(self):
        """Returns the number of elements in the input data expected by the layer."""
        return self.input_size
