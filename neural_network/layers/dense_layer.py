from neural_network.layers.layer import Layer


class DenseLayer(Layer):
    """
    Represents a dense (fully connected) layer in a neural network, which is a type of layer where each 
    neuron is connected to every neuron in the previous layer.

    A dense layer's primary function is to compute a weighted sum of the inputs it receives, add a bias, 
    and then apply an activation function to produce the output. This layer is common in many types of 
    neural networks and is fundamental in learning features and patterns from the input data.

    Attributes:
        weights_init_method (str): The method used to initialize weights. "he" initialization is common
        for layers with ReLU activation, as it considers the size of the previous layer in the network.
    """

    def __init__(self, name, num_neurons, activation) -> None:
        """
        Initializes the DenseLayer with a specified number of neurons, a name, and an activation function.

        Args:
            name (str): The name of the layer.
            num_neurons (int): The number of neurons in the dense layer.
            activation (callable): The activation function used by neurons in the dense layer.
        """
        super().__init__(name, num_neurons, activation)
        self.weights_init_method = "he"

    def count_params(self):
        """
        Counts the total number of trainable parameters in the dense layer, which includes weights and biases.

        Returns:
            int: The total number of parameters, which is the sum of all weights for each neuron and one bias 
            per neuron.
        """
        num_bias = len(self.neurons)
        input_len = len(self.input_layer.neurons)
        num_neurons = len(self.neurons)

        return input_len * num_neurons + num_bias

    def forward(self, input):
        """
        Processes the input through the dense layer by computing the output of each neuron.

        Args:
            input (list): The input data to the layer, expected to be a list of numerical values.

        Returns:
            tuple: A tuple containing the outputs of the layer and the corresponding logits.
        """
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
        """
        Stores the logits and outputs of the layer, potentially for later use in backpropagation.

        Args:
            output (list): The outputs of the layer after activation.
            logits (list): The logits (pre-activation outputs) of the layer.

        Returns:
            tuple: The same outputs and logits passed as arguments.
        """
        self.logits.append(logits)
        self.outputs.append(output)
        return output, logits

    def initiate_weights_and_gradients(self, init_weights=True, init_gradients=True):
        """
        Initializes the weights and gradients of each neuron in the layer based on the size of the input layer
        and the chosen initialization method.

        Args:
            init_weights (bool): If True, initializes the weights of each neuron.
            init_gradients (bool): If True, initializes the gradients for weight updates.
        """
        input_size = self.input_layer.get_width()
        for neuron in self.neurons:
            if init_weights:
                neuron.initialize_weights(
                    input_size, method=self.weights_init_method)
            if init_gradients:
                neuron.initialize_gradients(input_size)

    def get_width(self):
        """
        Returns the number of neurons in the layer, which is its 'width'.

        Returns:
            int: The number of neurons in the layer.
        """
        return len(self.neurons)
