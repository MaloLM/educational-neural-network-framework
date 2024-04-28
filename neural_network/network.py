from neural_network.layers.layer import Layer
from neural_network.layers.input_layer import InputLayer
from neural_network.layers.dense_layer import DenseLayer
from neural_network.layers.output_layer import OutputLayer
from neural_network.model_serializer import ModelSerializer
import neural_network.utils.optimizers as optimizers

from neural_network.utils.activation import Softmax
import neural_network.utils.loss as loss
import neural_network.utils.optimizers as opt
import numpy as np


class NeuralNetwork:

    def __init__(self, layers: list, opt: opt.Optimizer, loss: str) -> None:

        self.model_serializer = ModelSerializer("mnist_model")
        self.loss_func = loss
        self.optimizer = opt
        self.cumulative_batch_accuracy = []
        self.cumulative_batch_loss = []
        self.model_compatibility(layers)
        self.compile(layers)

    def model_compatibility(self, layers):

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

        if not self.optimizer:
            raise ValueError("Optimizer must be given.")

        if not optimizers.is_optimizer_defined(self.optimizer):
            raise ValueError(
                f"Optimiser '{self.optimizer}' is unknown. Please select between {list(opt.optimizers.keys())}.")

    def compile(self, layers, init_weights: bool = True, init_gradients: bool = True):

        self.layers = layers

        previous_layer = None
        for layer in self.layers:
            layer.set_optimizer(self.optimizer)
            layer.input_layer = previous_layer
            previous_layer = layer
            layer.initiate_weights_and_gradients(init_weights, init_gradients)

        try:
            self.loss_func = loss.loss_functions[self.loss_func]
        except KeyError:
            raise ValueError(
                f"Loss function '{self.loss_func}' is not defined. Please select between {list(loss.loss_functions.keys())}.")

        for i in range(len(layers) - 1):
            if len(layers[i].neurons) != len(layers[i+1].input_layer.neurons):
                raise ValueError(
                    f"input of layer {i+1} called '{layers[i+1].name}' does not match output of layer {i} called '{layers[i].name}'. {len(layers[i].neurons)} vs {len(layers[i+1].input_layer.neurons)}")

    def count_total_params(self):

        tot = 0
        for layer in self.layers:
            tot += layer.count_params()
        return tot

    def sample_forward(self, sample_input):
        layer_input = sample_input

        for layer in self.layers:
            layer_input, logit = layer.forward(layer_input)

        res_output = layer_input

        return res_output, logit

    def predict(self, input):
        return self.sample_forward(input)

    def batch_forward(self, batch):
        # print("batch forwarding: ", len(batch), " samples")

        if len(batch) <= 0:
            raise ValueError("Batch should not be empty")

        predictions = []
        logits = []

        for sample in batch:
            sample_prediction, logit = self.sample_forward(sample)
            predictions.append(sample_prediction)
            logits.append(logit)

        return predictions, logits

    def batching(self, data, batch_size):
        batches = []
        for i in range(0, len(data), batch_size):
            batches.append(data[i:i + batch_size])
        return batches

    def fit_training(self, samples, labels, batch_size, epochs=1):

        if len(samples) != len(labels):
            raise ValueError("Samples and labels are not of the same number.")

        if batch_size > len(samples):
            raise ValueError("Batch size can not be more than sample size.")

        self.reset_data()

        x_batches = self.batching(samples, batch_size)
        y_batches = self.batching(labels, batch_size)

        for epoch_index in range(epochs):
            for batch_idx, batch in enumerate(x_batches):
                print(f"--- Batch n°{batch_idx}, batchlen= {len(batch)}\n")
                preds, _ = self.batch_forward(batch)

                true_labels = y_batches[epoch_index]

                one_hot_encoded_labels = self.one_hot_encode(true_labels, 10)

                loss = self.loss_func.loss(preds, one_hot_encoded_labels)

                self.cumulative_batch_loss.append(loss)

                current_batch_acc = self.calculate_batch_accuracy(
                    preds, one_hot_encoded_labels)

                self.cumulative_batch_accuracy.append(current_batch_acc)

                self.backpropagate(one_hot_encoded_labels)

                self.update_params()

                self.reset_data()

        print("LOSS (batch):")
        print(self.cumulative_batch_loss, "\n")

        print("ACCURACY (batch):")
        print(self.cumulative_batch_accuracy, "\n")

    def reset_data(self):  # find a better name
        for layer in self.layers:
            layer.reset_data()

    def backpropagate(self, one_hot_encoded_labels):
        # Initialisation de l'objet Softmax pour les calculs de gradient # dirty
        softmax = Softmax()
        # Récupération de la couche de sortie du réseau
        output_layer = self.layers[-1]

        # Calcul du gradient de la fonction de perte par rapport aux logits de la couche de sortie
        loss_gradient = softmax.derivative(
            output_layer.logits, one_hot_encoded_labels)

        delta = loss_gradient  # Initialisation du delta avec le gradient de la perte

        # Itération sur les couches du réseau en ordre inverse (à partir de la couche de sortie)
        for layer_index in reversed(range(len(self.layers))):
            # Récupération de la couche courante
            layer = self.layers[layer_index]
            # Affichage des informations de la couche pour le débogage
            # print("index= ", layer_index, " name: ",
            #       layer.name, " activation= ", layer.activation)

            # Calcul des gradients pour chaque neurone de la couche
            for n, neuron in enumerate(layer.neurons):
                # Si ce n'est pas la première couche, utiliser les sorties de la couche précédente comme entrées
                if layer_index > 0:
                    inputs = np.array(self.layers[layer_index - 1].outputs)
                else:
                    # Si c'est la première couche, utiliser les entrées du réseau
                    inputs = np.array(self.input_layer.outputs)

                # Calcul du gradient des poids et du biais pour chaque neurone
                neuron.weights_gradients = np.dot(inputs.T, delta[:, n])
                neuron.bias_gradient = np.sum(delta[:, n], axis=0)

            # Propagation du gradient vers la couche précédente si nécessaire
            if not isinstance(layer.input_layer, InputLayer) and not isinstance(layer, InputLayer):
                previous_layer = self.layers[layer_index - 1]
                weighted_deltas = np.zeros(
                    (delta.shape[0], len(previous_layer.neurons)))

                for n, neuron in enumerate(layer.neurons):
                    for j in range(delta.shape[0]):
                        weighted_deltas[j, :] += neuron.weights * delta[j, n]

                new_delta = np.zeros_like(weighted_deltas)
                for i, neuron in enumerate(previous_layer.neurons):
                    neuron_outputs = [sample[i]
                                      for sample in previous_layer.outputs]

                    # Calcul des dérivées des activations pour chaque sortie de neurone
                    activation_derivatives = neuron.activation.derivative(
                        neuron_outputs)
                    new_delta[:, i] = activation_derivatives * \
                        weighted_deltas[:, i]

                delta = new_delta  # Mise à jour du delta pour la prochaine itération

    def update_params(self):
        # Mettre à jour les poids et les biais en utilisant les gradients calculés et un taux d'apprentissage η
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update_weights()
                neuron.update_bias()

    def one_hot_encode(self, labels, num_classes):
        # Create an empty list to store the one-hot encoded labels
        one_hot_encoded = []

        # Iterate through each label in the input list
        for label in labels:
            # Create a list of zeros of length num_classes
            encoding = [0] * num_classes
            # Set the appropriate index to 1
            encoding[label] = 1
            # Append the encoding to the list of one-hot encoded labels
            one_hot_encoded.append(encoding)

        return one_hot_encoded

    def calculate_batch_accuracy(self, y_preds, y_true):
        predicted_classes = np.argmax(y_preds, axis=1)
        true_classes = np.argmax(y_true, axis=1)
        num_correct = np.sum(predicted_classes == true_classes)
        num_samples = len(y_true)
        accuracy = num_correct / num_samples

        return accuracy

    def get_highest_probability_index(self, probabilities):
        highest_index = np.argmax(probabilities)
        return highest_index

    def __str__(self) -> str:

        content = []

        content.append(f"Total params: {self.count_total_params()}")

        for layer in self.layers:
            content.append(str(layer))
            content.append("------------------------")

        return "\n".join(content)

    def save(self):
        self.model_serializer.save_model(self)

    def load(self):
        layers, optimizer, loss = self.model_serializer.load_model()
        self.optimizer = optimizer
        self.loss_func = loss

        self.compile(layers, init_weights=False, init_gradients=False)
