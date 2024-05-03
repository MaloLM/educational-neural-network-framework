from abc import ABC, abstractmethod
from neural_network.neuron import Neuron


class Layer(ABC):

    def __init__(self, name, num_neurons, activation) -> None:

        self.name = name
        self.opt = None
        self.activation = activation
        self.input_layer = None
        self.neurons = [Neuron(id, activation)
                        for id in range(num_neurons)]

        self.inputs = []
        self.logits = []
        self.outputs = []

    def clear_data(self):
        self.inputs = []
        self.logits = []
        self.outputs = []

        for neuron in self.neurons:
            if isinstance(neuron, Neuron):
                neuron.zero_grads()

    def __str__(self) -> str:

        return f"name: {self.name}\nneurons: {len(self.neurons)}\nactivation function: {self.activation} \nparams: {self.count_params()}"

    def set_optimizer(self, opt: str):
        for neuron in self.neurons:
            neuron.set_optimizer(opt)

    @abstractmethod
    def count_params(self):
        pass

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def initiate_weights_and_gradients(self):
        pass
