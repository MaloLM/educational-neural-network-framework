from neural_network.layers.layer import Layer
from neural_network.layers.input_layer import InputLayer
from neural_network.layers.dense_layer import DenseLayer
from neural_network.layers.output_layer import OutputLayer


class NeuralNetwork:
    """
    Represents a simple neural network for educational purposes to demonstrate the fundamentals of deep learning architectures.

    Attributes:
        loss_func (callable): The function used to calculate the loss during training.
        backpropagation_func (callable): The function used to perform backpropagation.
        optimizer (Optimizer): An instance of an optimizer class which will be used to update the network weights.
        layers (list): A list of layers comprising the neural network.

    Methods:
        build_network(layers): Validates and constructs the neural network from a list of layers.
        back_propagate(): Placeholder method for backpropagation logic.
        count_total_params(): Returns the total number of parameters in the network.
        feedforward(input): Processes the input through the network and returns the output.
        __str__(): Provides a string representation of the neural network including the total number of parameters and layer descriptions.
    """

    def __init__(self, layers, opt) -> None:
        """
        Initializes the NeuralNetwork with the specified layers and optimizer.

        Args:
            layers (list): A list of Layer instances that will form the neural network. 
                           Must start with an InputLayer and end with an OutputLayer, 
                           with any number of DenseLayer instances in between.
            opt (Optimizer): An instance of an optimizer class for adjusting the weights during training.

        Raises:
            ValueError: If the layers do not meet the criteria (not starting with InputLayer or ending with OutputLayer).
        """
        self.loss_func = 0
        self.backpropagation_func = 0
        self.optimizer = opt

        self.build_network(layers)

    def build_network(self, layers):
        """
        Validates and constructs the neural network from a list of provided layers.

        Args:
            layers (list): A list of Layer instances to be validated and structured into a neural network.

        Raises:
            ValueError: If the first layer is not an InputLayer, if the last layer is not an OutputLayer,
                        if intermediate layers are not instances of DenseLayer, or if there are mismatches
                        in the number of neurons between consecutive layers.
        """

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
        """
        Implements the backpropagation algorithm for training the network.
        Placeholder for future implementation.
        """
        pass

    def count_total_params(self):
        """
        Calculates the total number of trainable parameters in the network.

        Returns:
            int: Total number of parameters.
        """
        tot = 0
        for layer in self.layers:
            tot += layer.count_params()
        return tot

    def feedforward(self, input):
        """
        Feeds the input through the network and returns the output.

        Args:
            input (array-like): The input data to be fed into the neural network.

        Returns:
            output (array-like): The output from the neural network.
        """
        output = 0 + input
        return output

    def __str__(self) -> str:
        """
        Provides a string representation of the neural network, detailing the total parameters and description of each layer.

        Returns:
            str: Formatted string describing the neural network structure.
        """
        content = []

        content.append(f"Total params: {self.count_total_params()}")

        for layer in self.layers:
            content.append(str(layer))
            content.append("------------------------")

        return "\n".join(content)
