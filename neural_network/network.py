from neural_network.layers.layer import Layer
from neural_network.layers.input_layer import InputLayer
from neural_network.layers.dense_layer import DenseLayer
from neural_network.layers.output_layer import OutputLayer


class NeuralNetwork:
    """quoicoubeh"""

    def __init__(self, layers, opt) -> None:
        self.loss_func = 0
        self.backpropagation_func = 0
        self.optimizer = opt

        self.build_network(layers)

    def build_network(self, layers):

        if not all(isinstance(layer, Layer) for layer in layers):
            raise ValueError(
                "All elements of layers must be instances of the Layer class")

        if not isinstance(layers[0], InputLayer):
            raise ValueError(
                "First layer must be instance of InputLayer class")

        if not isinstance(layers[-1], OutputLayer):
            raise ValueError(
                "Last layer must be instance of OutputLayer class")

        if not all(isinstance(layer, DenseLayer) for layer in layers[1:-1]):
            raise ValueError(
                "Hidden layers should be instances of DenseLayer class")

        self.layers = layers

        previous_num_neurons = 0
        for layer in self.layers:
            layer.prev_lay_num_neurons = previous_num_neurons
            previous_num_neurons = len(layer.neurons)

        for i in range(len(layers) - 1):
            if len(layers[i].neurons) != layers[i+1].prev_lay_num_neurons:
                raise ValueError(
                    f"input of layer {i+1} called '{layers[i+1].name}' does not match output of layer {i} called '{layers[i].name}'. {len(layers[i].neurons)} vs {layers[i+1].prev_lay_num_neurons}")

    def back_probagate(self):
        pass

    def count_total_params(self):
        tot = 0
        for layer in self.layers:
            tot += layer.count_params()
        return tot

    def feedforward(self, input):
        output = 0 + input
        return output

    def __str__(self) -> str:
        content = []

        content.append(f"Total params: {self.count_total_params()}")

        for layer in self.layers:
            content.append(str(layer))
            content.append("------------------------")

        return "\n".join(content)
