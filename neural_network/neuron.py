class Neuron:

    def __init__(self, id, activation_func: callable) -> None:
        self.id = id
        self.activation_func = activation_func
        self.bias = []  # 1 per neuron ?
        self.x0 = 1.0
        self.weigths = []
        self.input = []

    def forward_propagate(self):
        pass

    def summing(self):
        net_input = 0
        return net_input
