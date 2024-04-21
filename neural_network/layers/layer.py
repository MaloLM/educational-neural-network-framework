from abc import ABC, abstractmethod

from neural_network.neuron import Neuron


class Layer(ABC):

    def __init__(self, name, num_neurons, activation_func) -> None:
        self.name = name
        self.neurons = []
        self.activation_func = activation_func
        self.prev_lay_num_neurons = None

        for neuron in range(num_neurons):
            self.neurons.append(Neuron(neuron, "relu"))

    def __str__(self) -> str:
        return f"name: {self.name}\n num neurons: {len(self.neurons)}\n activation function: {self.activation_func} \n params: {self.count_params()}"

    @abstractmethod
    def count_params(self):
        """Return number of parameters of the layer."""
        pass
