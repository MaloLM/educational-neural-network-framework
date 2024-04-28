from neural_network.layers.layer import Layer
import numpy as np


class InputLayer(Layer):

    def __init__(self, name, input_size) -> None:

        super().__init__(name, 0, None)
        self.input_size = input_size

    def count_params(self):

        return 0

    def flatten(self, input):

        if isinstance(input, list):
            flat_output = [item for sublist in input for item in (
                sublist if isinstance(sublist, list) else [sublist])]
        else:
            flat_output = np.ravel(input).tolist()

        if len(flat_output) == self.input_size:
            return flat_output
        else:
            raise ValueError(
                f"The length of the flattened input is {len(flat_output)}, which does not match the expected size of {self.input_size}. Ensure that your input data is correctly formatted and contains the right number of elements.")

    def forward(self, input):
        # print(f"forward to {self.name}")
        flattened_input = self.flatten(input)
        self.inputs.append(input)
        self.outputs.append(flattened_input)

        return flattened_input, None

    def initiate_weights_and_gradients(self, *args, **kwargs):
        pass

    def initialize_gradients(self):
        pass

    def get_width(self):
        return self.input_size
