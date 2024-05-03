
from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):

    @abstractmethod
    def loss(self, preds, one_hot_true_labels):
        pass

    @abstractmethod
    def gradient(self, preds, one_hot_true_labels):
        pass


class MeanSquaredError(Loss):

    def loss(self, preds, one_hot_true_labels):
        one_hot_true_labels = np.array(one_hot_true_labels)
        preds = np.array(preds)
        return np.mean((preds - one_hot_true_labels) ** 2)

    def gradient(self, preds, one_hot_true_labels):
        n_samples = len(preds)
        return (2 / n_samples) * (preds - one_hot_true_labels)

    def __str__(self) -> str:
        return "mean squared error"


class MeanAbsoluteError(Loss):

    def loss(self, preds, one_hot_true_labels):
        one_hot_true_labels = np.array(one_hot_true_labels)
        preds = np.array(preds)
        return np.mean(np.abs(preds - one_hot_true_labels))

    def gradient(self, preds, one_hot_true_labels):
        n_samples = len(preds)
        gradients = np.where(preds > one_hot_true_labels, 1, -1)
        return gradients / n_samples

    def __str__(self) -> str:
        return "mean aboslute error"


class BinaryCrossEntropy(Loss):

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, preds, one_hot_true_labels):
        one_hot_true_labels = np.array(one_hot_true_labels)
        preds = np.array(preds)
        probs = self.sigmoid(preds)

        log_probs = -one_hot_true_labels * \
            np.log(probs) - (1 - one_hot_true_labels) * np.log(1 - probs)
        return np.mean(log_probs)

    def gradient(self, preds, one_hot_true_labels):
        probs = self.sigmoid(preds)
        n_samples = len(preds)
        return (probs - one_hot_true_labels) / n_samples

    def __str__(self) -> str:
        return "categorical cross entropy"


class CategoricalCrossEntropy(Loss):

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def loss(self, preds, one_hot_true_labels):
        one_hot_true_labels = np.array(one_hot_true_labels)
        preds = np.array(preds)
        n_samples = len(preds)
        log_probs = -np.log(preds[range(n_samples),
                            one_hot_true_labels.argmax(axis=1)])
        return np.mean(log_probs)

    def gradient(self, preds, one_hot_true_labels):
        one_hot_true_labels = np.array(one_hot_true_labels)
        probs = self.softmax(preds)
        n_samples = len(preds)
        probs[range(n_samples), one_hot_true_labels.argmax(axis=1)] -= 1
        return probs / n_samples


loss_functions = {
    "mean squared error": MeanSquaredError(),
    "mse": MeanSquaredError(),
    "mean aboslute error": MeanAbsoluteError(),
    "mae": MeanAbsoluteError(),
    "binary cross entropy": BinaryCrossEntropy(),
    "bce": BinaryCrossEntropy(),
    "categorical cross entropy": CategoricalCrossEntropy(),
    "cce": CategoricalCrossEntropy(),
}
