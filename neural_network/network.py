from neural_network.layers.layer import Layer
from neural_network.layers.input_layer import InputLayer
from neural_network.layers.dense_layer import DenseLayer
from neural_network.layers.output_layer import OutputLayer
from neural_network.model_serializer import ModelSerializer
import neural_network.utils.optimizers as optimizers
from neural_network.utils.model_utils import calculate_batch_accuracy, precision_score, recall_score, f1_score, accuracy_score
from neural_network.utils.plots import plot_confusion_matrix, plot_history
import neural_network.utils.optimizers as opt
import neural_network.utils.loss as loss

import numpy as np
from IPython.display import clear_output


class NeuralNetwork:

    def __init__(self, layers: list, opt: opt.Optimizer, loss: str) -> None:

        self.loss_func = loss
        self.optimizer = opt
        self.cumulative_batch_accuracy = []
        self.cumulative_batch_loss = []
        self.model_compatibility(layers)
        self.__compile(layers)
        self.model_serializer = ModelSerializer("mnist_model")

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

    def __compile(self, layers, init_weights: bool = True, init_gradients: bool = True):

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
        return sum(layer.count_params() for layer in self.layers)

    def sample_forward(self, sample_input):
        layer_input = sample_input

        for layer in self.layers:
            layer_input, logit = layer.forward(layer_input)

        res_output = layer_input

        return res_output, logit

    def predict(self, input):
        prediction, _ = self.sample_forward(input)
        return prediction

    def validate_old(self, data_samples, data_labels, batch_size=2):

        if len(data_samples) != len(data_labels):
            raise ValueError("Samples and labels must be of the same length")

        if len(data_samples) < batch_size:
            raise ValueError(
                "Samples number should be superior or equal to provided batch size")

        predictions = []
        true_labels = []

        sample_batches = self.batching(data_samples, batch_size)
        labels_batches = self.batching(data_labels, batch_size)

        for idx, batch in enumerate(sample_batches):

            labels = labels_batches[idx]

            for j, sample in enumerate(batch):

                sample_prediction = self.predict(sample)
                sample_prediction = np.array(sample_prediction)
                class_prediction = np.argmax(sample_prediction, axis=0)

                predictions.append(class_prediction)
                true_labels.append(labels[j])

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        accuracy = accuracy_score(true_labels, predictions)

        print(f"Precision: {precision:.2f}/1.00")
        print(f"Recall: {recall:.2f}/1.00")
        print(f"F1 Score: {f1:.2f}/1.00")
        print(f"Accuracy: {accuracy:.2f}/1.00")

    def validate(self, data_samples, data_labels, batch_size=2):
        import matplotlib.pyplot as plt
        if len(data_samples) != len(data_labels):
            raise ValueError("Samples and labels must be of the same length")

        if len(data_samples) < batch_size:
            raise ValueError(
                "Samples number should be superior or equal to provided batch size")

        predictions = []
        true_labels = []

        sample_batches = self.batching(data_samples, batch_size)
        labels_batches = self.batching(data_labels, batch_size)

        num_classes = len(np.unique(data_labels))

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for idx, batch in enumerate(sample_batches):
            labels = labels_batches[idx]

            for j, sample in enumerate(batch):
                sample_prediction = self.predict(sample)
                sample_prediction = np.array(sample_prediction)
                class_prediction = np.argmax(sample_prediction, axis=0)

                predictions.append(class_prediction)
                true_labels.append(labels[j])

                # Update confusion matrix
                confusion_matrix[labels[j], class_prediction] += 1

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        accuracy = accuracy_score(true_labels, predictions)

        print(f"Precision: {precision:.2f}/1.00")
        print(f"Recall: {recall:.2f}/1.00")
        print(f"F1 Score: {f1:.2f}/1.00")
        print(f"Accuracy: {accuracy:.2f}/1.00")

        class_labels = [str(i) for i in range(num_classes)]
        plot_confusion_matrix(confusion_matrix, class_labels)

    def batch_forward(self, batch):
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
        data_array = np.array(data)
        return [data_array[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def fit(self, samples, labels, batch_size, epochs=1):

        if len(samples) != len(labels):
            raise ValueError("Samples and labels are not of the same number.")

        if batch_size > len(samples):
            raise ValueError("Batch size can not be more than sample size.")

        self.__clear_data()

        x_batches = self.batching(samples, batch_size)
        y_batches = self.batching(labels, batch_size)

        for epoch_index in range(epochs):
            for idx, batch in enumerate(x_batches):
                clear_output(wait=True)

                preds, _ = self.batch_forward(batch)

                true_labels = y_batches[idx]

                one_hot_encoded_labels = self.one_hot_encode(true_labels, 10)

                loss = self.loss_func.loss(preds, one_hot_encoded_labels)
                current_batch_acc = calculate_batch_accuracy(
                    preds, one_hot_encoded_labels)
                print(
                    f"EPOCH n°{epoch_index+1}/{epochs}\n ----- Batch n°{idx}/{len(x_batches)}, batchlen= {len(batch)}\n Batch accuracy: {current_batch_acc}/1.0")

                self.cumulative_batch_loss.append(loss)
                self.cumulative_batch_accuracy.append(current_batch_acc)

                self.__backpropagate(one_hot_encoded_labels)
                self.__update_params()
                self.__clear_data()

        plot_history(self.cumulative_batch_loss,
                     self.cumulative_batch_accuracy)

    def __clear_data(self):
        for layer in self.layers:
            layer.clear_data()

    def __backpropagate(self, one_hot_encoded_labels):

        output_layer = self.layers[-1]

        loss_gradient = self.loss_func.gradient(
            output_layer.logits, one_hot_encoded_labels)

        delta = loss_gradient

        for layer_index in reversed(range(len(self.layers))):

            layer = self.layers[layer_index]

            inputs = np.array(
                self.layers[layer_index - 1].outputs if layer_index > 1 else self.layers[0].outputs)

            self.update_gradients(layer, inputs, delta)

            if layer_index > 1:  # No need to propagate to the input layer

                delta = self.propagate_to_previous_layer(layer, delta)

    def calculate_initial_loss_gradient(self, loss, output_layer, labels):
        return loss.derivative(output_layer.logits, labels)

    def update_gradients(self, layer, inputs, delta):
        for neuron_index, neuron in enumerate(layer.neurons):
            neuron.weights_gradients = np.dot(inputs.T, delta[:, neuron_index])
            neuron.bias_gradient = np.sum(delta[:, neuron_index], axis=0)

    def propagate_to_previous_layer(self, current_layer, delta):
        previous_layer = self.layers[self.layers.index(current_layer) - 1]
        weighted_deltas = np.zeros(
            (delta.shape[0], len(previous_layer.neurons)))

        for n, neuron in enumerate(current_layer.neurons):
            for j in range(delta.shape[0]):

                weighted_deltas[j, :] += neuron.weights * delta[j, n]

        new_delta = np.zeros_like(weighted_deltas)
        for i, neuron in enumerate(previous_layer.neurons):
            neuron_outputs = [sample[i] for sample in previous_layer.outputs]
            activation_derivatives = neuron.activation.derivative(
                neuron_outputs)

            new_delta[:, i] = activation_derivatives * weighted_deltas[:, i]

        return new_delta

    def __update_params(self):
        for layer in self.layers:
            if not isinstance(layer, InputLayer):
                for neuron in layer.neurons:
                    neuron.update_weights(layer.name)
                    neuron.update_bias()

    def one_hot_encode(self, labels, num_classes):
        one_hot_encoded = np.zeros((len(labels), num_classes), dtype=int)
        one_hot_encoded[np.arange(len(labels)), labels] = 1

        return one_hot_encoded

    def __str__(self) -> str:
        total_params = self.count_total_params()
        return "\n".join([
            f"Total params: {total_params}",
            *(f"{layer}\n------------------------" for layer in self.layers)
        ])

    def save(self):
        self.model_serializer.save_model(self)

    def load(self):
        layers, optimizer, loss = self.model_serializer.load_model()
        self.optimizer = optimizer
        self.loss_func = loss

        self.__compile(layers, init_weights=False, init_gradients=False)

    # --- TESTING

    def fit_training_with_gradient_verification(self, samples, labels, batch_size, epochs=1):
        self.__clear_data()

        x_batches = self.batching(samples, batch_size)
        y_batches = self.batching(labels, batch_size)

        epsilon = 1e-5  # Small perturbation for numerical gradient calculation
        # Check up to 3 layers or fewer if the network is smaller
        num_layers_to_check = min(3, len(self.layers))

        for epoch_index in range(epochs):
            print(f"--- EPOCH n°{epoch_index+1}\n")
            for batch_idx, batch in enumerate(x_batches):
                print(
                    f"----- Batch n°{batch_idx + 1}, batchlen= {len(batch)}\n")

                self.batch_forward(batch)

                true_labels = y_batches[epoch_index]

                one_hot_encoded_labels = self.one_hot_encode(true_labels, 10)

                self.backpropagate(one_hot_encoded_labels)

                # Gradient verification for various weights and layers
                for layer_idx, layer in enumerate(self.layers[:num_layers_to_check]):
                    # Check up to 5 neurons per layer
                    num_neurons_to_check = min(5, len(layer.neurons))
                    for neuron_idx, neuron in enumerate(layer.neurons[:num_neurons_to_check]):
                        # Check up to 10 weights per neuron
                        for weight_idx in range(min(10, len(neuron.weights))):
                            original_weight = neuron.weights[weight_idx]

                            # Analytical gradient
                            grad_analytical = neuron.weights_gradients[weight_idx]

                            # Calculation of numerical gradient
                            # Positive perturbation
                            neuron.weights[weight_idx] = original_weight + epsilon
                            loss_plus = self.loss_func.loss(self.batch_forward(batch)[
                                                            0], one_hot_encoded_labels)

                            # Negative perturbation
                            neuron.weights[weight_idx] = original_weight - epsilon
                            loss_minus = self.loss_func.loss(self.batch_forward(batch)[
                                0], one_hot_encoded_labels)

                            # Restore the original weight value
                            neuron.weights[weight_idx] = original_weight

                            # Numerical gradient
                            grad_numerical = (
                                loss_plus - loss_minus) / (2 * epsilon)

                            # Comparison of gradients
                            error = abs(grad_analytical - grad_numerical) / \
                                max(abs(grad_analytical), abs(
                                    grad_numerical), 1e-8)
                            print(
                                f"Layer {layer_idx},name {layer.name}, Neuron {neuron_idx}, Weight {weight_idx}:")
                            print(
                                f" Analytical Gradient: {grad_analytical}, Numerical Gradient: {grad_numerical}")
                            print(f" Relative Error: {error}")

                self.update_params()

                self.clear_data()

    def samples_checking(self, samples, labels, batch_size, epochs=1):

        import matplotlib.pyplot as plt
        import numpy as np

        if len(samples) != len(labels):
            raise ValueError("Samples and labels are not of the same number.")

        if batch_size > len(samples):
            raise ValueError("Batch size can not be more than sample size.")

        self.__clear_data()

        x_batches = self.batching(samples, batch_size)
        y_batches = self.batching(labels, batch_size)

        for _ in range(epochs):
            for idx, batch in enumerate(x_batches):
                if idx in [1, 3, 5]:
                    preds, _ = self.batch_forward(batch)

                    true_labels = y_batches[idx]

                    one_hot_encoded_labels = self.one_hot_encode(
                        true_labels, 10)

                    for idx, _ in enumerate(batch):

                        string = str(one_hot_encoded_labels[idx])

                        image = batch[idx]
                        label = true_labels[idx]

                        pred = np.argmax(preds[idx], axis=0)
                        pred = str(pred)

                        # sample display
                        plt.imshow(image, cmap='gray')
                        plt.title(
                            f'Class: {string}, True: {label}, Pred: {pred}')
                        plt.colorbar()
                        plt.grid(False)
                        plt.show()
