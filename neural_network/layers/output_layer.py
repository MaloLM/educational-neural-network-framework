
from neural_network.layers.layer import Layer


class OutputLayer(Layer):

    def __init__(self, name, num_neurons, activation_func) -> None:
        super().__init__(name, num_neurons, activation_func)

    def count_params(self):
        return self.prev_lay_num_neurons * len(self.neurons) + len(self.neurons)
