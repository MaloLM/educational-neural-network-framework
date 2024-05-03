from abc import ABC, abstractmethod
from neural_network.neuron import Neuron


class Layer(ABC):
    """
    Abstract base class representing a layer in a neural network.

    A layer is a collection of neurons that processes input data and produces output, 
    which may serve as input for subsequent layers. Layers are the fundamental building 
    blocks of neural networks, defining the architecture and behavior of the network.

    Attributes:
        name (str): The name of the layer, helps in identifying the layer within the network.
        num_neurons (int): The number of neurons in this layer.
        activation (callable): The activation function used by neurons in this layer.
        neurons (list): A list of `Neuron` objects representing the neurons in this layer.
        inputs (list): A list to store inputs to the layer.
        logits (list): A list to store the linear combination of inputs and weights.
        outputs (list): A list to store the output of the layer after applying the activation function.

    Methods:
        clear_data: Resets the inputs, logits, and outputs lists, and clears gradients in each neuron.
        set_optimizer: Assigns an optimizer to each neuron for parameter updates.
        count_params: Abstract method to calculate the total number of trainable parameters.
        forward: Abstract method to compute the output of the layer given an input.
        initiate_weights_and_gradients: Abstract method to initialize weights and gradients for training.
    """

    def __init__(self, name, num_neurons, activation) -> None:
        """
        Initializes a new layer with a specified number of neurons and an activation function.

        Args:
            name (str): The name of the layer.
            num_neurons (int): The number of neurons to be included in the layer.
            activation (callable): The activation function to be used by the neurons.
        """

        self.name = name
        self.opt = None
        self.activation = activation
        self.input_layer = None
        self.neurons = [Neuron(id, activation)
                        for id in range(num_neurons)]

        self.inputs = []
        self.logits = []
        self.outputs = []

    def __str__(self) -> str:
        """Returns a string representation of the layer with its basic attributes."""
        return (f"name: {self.name}"
                f"neurons: {len(self.neurons)}"
                f"nactivation function: {self.activation}"
                f"params: {self.count_params()}")

    def clear_data(self):
        """Clears the data of the layer and resets the gradients in each neuron."""
        self.inputs = []
        self.logits = []
        self.outputs = []

        for neuron in self.neurons:
            if isinstance(neuron, Neuron):
                neuron.zero_grads()

    @abstractmethod
    def initiate_weights_and_gradients(self):
        """Abstract method to initialize weights and gradients for the neurons in the layer."""
        pass

    def set_optimizer(self, opt: str):
        """Assigns an optimizer to each neuron in the layer for parameter updates."""
        for neuron in self.neurons:
            neuron.set_optimizer(opt)

    @abstractmethod
    def count_params(self):
        """Abstract method to calculate the total number of trainable parameters in the layer."""
        pass

    @abstractmethod
    def forward(self, input):
        """Abstract method that defines how the layer processes input to produce output."""
        pass
