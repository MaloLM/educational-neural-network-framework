import os
import h5py
from neural_network.layers.input_layer import InputLayer
from neural_network.layers.dense_layer import DenseLayer
from neural_network.layers.output_layer import OutputLayer

input_layer_instance = InputLayer
dense_layer_instance = DenseLayer
output_layer_instance = OutputLayer


class ModelSerializer:
    """
    Provides functionality to serialize (save) and deserialize (load) neural network models.
    The models are saved into HDF5 files, which is a popular format for storing large quantities of numerical
    data. This format is particularly suited for handling data for deep learning models.

    Attributes:
        filename (str): The name of the file where the model will be saved.
        filepath (str): The full path to the file, derived from the specified directory and filename.
    """

    def __init__(self, filename, directory='../data/models'):
        """
        Initializes the serializer with a filename and directory.

        Args:
            filename (str): The base name of the file to save the model.
            directory (str): The directory path where the model files will be stored.
        """
        self.filename = filename + ".hdf5"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.filepath = os.path.join(directory, self.filename)

    def save_model(self, model):
        """
        Saves a neural network model to an HDF5 file.

        Args:
            model (Model): The neural network model to be saved. It should have attributes like
                           optimizer, loss function, and layers which should also have their own
                           attributes like weights and biases.
        """
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
        """
        Loads a neural network model from an HDF5 file.

        Returns:
            tuple: A tuple containing the list of layers, optimizer, and loss function used in the model.
        """
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
                # Assumes a dynamic load approach, currently hardcoded which should be improved
                for neuron in layer.neurons:
                    weights = group['weights'][:]
                    biases = group['biases'][:]
                for j, neuron in enumerate(layer.neurons):
                    neuron.weights = weights[j]
                    neuron.bias = biases[j]
                layers.append(layer)

            return layers, optimizer, loss_func
