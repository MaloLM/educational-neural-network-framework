from neural_network.layers.layer import Layer


class InputLayer(Layer):

    def __init__(self, name, input_size) -> None:
        super().__init__(name, 0, None)
        self.input_size = input_size

    def count_params(self):
        return 0
