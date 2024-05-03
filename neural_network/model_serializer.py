import os
import h5py
from neural_network.layers.input_layer import InputLayer
from neural_network.layers.dense_layer import DenseLayer
from neural_network.layers.output_layer import OutputLayer

input_layer_instance = InputLayer
dense_layer_instance = DenseLayer
output_layer_instance = OutputLayer


class ModelSerializer:
    def __init__(self, filename, directory='../data/models'):
        self.filename = filename + ".hdf5"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.filepath = os.path.join(directory, self.filename)

    def save_model(self, model):
        with h5py.File(self.filepath, 'w') as file:
            file.attrs['optimizer'] = model.optimizer
            file.attrs['loss_func'] = model.loss_func.__class__.__name__
            for i, layer in enumerate(model.layers):
                group = file.create_group(f'layer_{i}')
                group.attrs['name'] = layer.name
                group.attrs['type'] = layer.__class__.__name__
                group.attrs['activation'] = layer.activation.__class__.__name__
                group.attrs['nb_neurons'] = len(layer.neurons)
                group.create_dataset(
                    'weights', data=[neuron.weights for neuron in layer.neurons])
                group.create_dataset(
                    'biases', data=[neuron.bias for neuron in layer.neurons])

    def load_model(self):
        with h5py.File(self.filepath, 'r') as file:
            optimizer = file.attrs['optimizer']
            loss_func = file.attrs['loss_func']
            layers = []

            for i in range(len(file.keys())):
                group = file[f'layer_{i}']
                layer_type = group.attrs['type']

                klass = globals()[layer_type]

                nb_neurons = group.attrs['nb_neurons']
                activation = group.attrs['activation']

                name = group.attrs['name']
                layer = klass(name, nb_neurons, activation)
                # MALO: ce n'est pas la bonne approche de sauvegarder en dur et de load en dur:
                # il faut surement mieux que save model soi plus dynamique pour que tout type d'argments de classe y soient sauvés
                for neuron in layer.neurons:
                    weights = group['weights'][:]
                    biases = group['biases'][:]
                for j, neuron in enumerate(layer.neurons):
                    neuron.weights = weights[j]
                    neuron.bias = biases[j]
                layers.append(layer)

            return layers, optimizer, loss_func
