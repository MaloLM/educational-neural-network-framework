from neural_network.layers.layer import Layer


class DenseLayer(Layer):

    def __init__(self, name, num_neurons, activation) -> None:

        super().__init__(name, num_neurons, activation)
        self.weights_init_method = "he"

    def count_params(self):

        num_bias = len(self.neurons)
        input_len = len(self.input_layer.neurons)
        num_neurons = len(self.neurons)

        return input_len * num_neurons + num_bias

    def forward(self, input):
        output = []
        logits = []

        for neuron in self.neurons:
            neuron.input_values = input
            neuron.forward()
            output.append(neuron.output)
            logits.append(neuron.logits)

        self.inputs.append(input)

        output, logits = self.post_forward(output, logits)

        return output, logits

    def post_forward(self, output, logits):
        self.logits.append(logits)
        self.outputs.append(output)
        return output, logits

    def initiate_weights_and_gradients(self, init_weights=True, init_gradients=True):
        input_size = self.input_layer.get_width()
        for neuron in self.neurons:
            if init_weights:
                neuron.initialize_weights(
                    input_size, method=self.weights_init_method)
            if init_gradients:
                neuron.initialize_gradients(input_size)

    def get_width(self):
        return len(self.neurons)
