from neural_network.layers.dense_layer import DenseLayer
import neural_network.utils.activation as activations


class OutputLayer(DenseLayer):

    def __init__(self, name, num_neurons, activation) -> None:

        super().__init__(name, num_neurons, activation)
        self.weights_init_method = "glorot"

    def post_forward(self, output, logits):

        self.logits.append(logits)

        activation = activations.activation_functions[self.activation]

        layer_output = activation.function(logits)
        self.outputs.append(layer_output)
        return layer_output, logits
