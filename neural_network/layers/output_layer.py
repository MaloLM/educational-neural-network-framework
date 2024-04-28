from neural_network.layers.layer import Layer
import neural_network.utils.activation as activations


class OutputLayer(Layer):

    def __init__(self, name, num_neurons, activation) -> None:

        super().__init__(name, num_neurons, activation)

    def count_params(self):

        return len(self.input_layer.neurons) * len(self.neurons) + len(self.neurons)

    def forward(self, input):
        # DRY issue with dense layer
        layer_logits = []
        layer_output = []

        for neuron in self.neurons:
            neuron.input_values = input
            neuron.forward()
            layer_logits.append(neuron.logits)
            layer_output.append(neuron.output)

        # DIRTY CODE TO REFACTOR with OOP
        activation = activations.activation_functions[self.activation]
        # DIRTY CODE TO REFACTOR with OOP
        layer_output = activation.function(layer_logits)

        self.inputs.append(input)
        self.logits.append(layer_logits)

        self.outputs.append(layer_output)

        return layer_output, layer_logits

    def initiate_weights_and_gradients(self, init_weights=True, init_gradients=True):
        input_size = self.input_layer.get_width()
        for neuron in self.neurons:
            if init_weights:
                neuron.initialize_weights(input_size)
            if init_gradients:
                neuron.initialize_gradients(input_size)

    def get_width(self):
        # DRY issue with dense layer
        return len(self.neurons)
