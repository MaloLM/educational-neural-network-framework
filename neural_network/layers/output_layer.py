from neural_network.layers.dense_layer import DenseLayer
import neural_network.utils.activation as activations


class OutputLayer(DenseLayer):
    """
    Represents the output layer of a neural network, which is a specialized type of DenseLayer.
    The output layer is responsible for producing the final output of the network, which can be used for
    making predictions or decisions based on the input data provided to the network.

    This layer typically uses a specific activation function suitable for the type of prediction task at hand,
    such as softmax for multi-class classification or sigmoid for binary classification.

    Attributes:
        weights_init_method (str): The method used to initialize weights. The "glorot" (also known as Xavier)
        initialization is commonly used for output layers with sigmoid or tanh activation functions, as it
        considers the number of inputs and outputs to the neurons to maintain a variance that avoids vanishing
        or exploding gradients.
    """

    def __init__(self, name, num_neurons, activation) -> None:
        """
        Initializes the OutputLayer with a specified number of neurons, a name, and an activation function.

        Args:
            name (str): The name of the layer.
            num_neurons (int): The number of neurons in the output layer.
            activation (callable): The activation function used by neurons in the output layer.
        """
        super().__init__(name, num_neurons, activation)
        self.weights_init_method = "glorot"

    def post_forward(self, output, logits):
        """
        Processes the logits through the activation function specific to the output layer and appends
        the results to the outputs attribute.

        This method adjusts the raw logits computed by the neurons to produce the final outputs of the
        neural network, which are typically probabilities in classification tasks or real values in regression tasks.

        Args:
            output (list): The raw output from neurons before applying the activation function.
            logits (list): The logits (pre-activation outputs) computed by the layer.

        Returns:
            tuple: A tuple containing the final output of the layer after applying the activation function,
                   and the logits.
        """
        self.logits.append(logits)

        activation = activations.activation_functions[self.activation]

        layer_output = activation.function(logits)
        self.outputs.append(layer_output)
        return layer_output, logits
