
from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """
    Abstract base class for loss functions in a neural network.

    Loss functions measure the inconsistency between predicted outputs and the actual target values,
    providing a numeric representation of model performance. This class serves as a template for loss
    functions, requiring the implementation of `loss` and `gradient` methods.

    Methods:
        loss(preds, one_hot_true_labels): Calculates the loss value.
        gradient(preds, one_hot_true_labels): Computes the gradient of the loss function.
    """

    @abstractmethod
    def loss(self, preds, one_hot_true_labels):
        """
        Computes the loss between predictions and true labels.

        Args:
            preds (list or np.ndarray): The predictions made by the network.
            one_hot_true_labels (list or np.ndarray): The true labels in a one-hot encoded format.

        Returns:
            float: The calculated loss.
        """
        pass

    @abstractmethod
    def gradient(self, preds, one_hot_true_labels):
        """
        Computes the gradient of the loss function with respect to the predictions.

        Args:
            preds (list or np.ndarray): The predictions made by the network.
            one_hot_true_labels (list or np.ndarray): The true labels in a one-hot encoded format.

        Returns:
            np.ndarray: The gradients of the loss function.
        """
        pass


class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) loss, used typically for regression problems.

    Calculates the average of the squares of the differences between predicted values and actual values.
    """

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
    """
    Mean Absolute Error (MAE) loss, commonly used for regression.

    Calculates the average of the absolute differences between predicted values and actual values.
    """

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
    """
    Binary Cross Entropy loss, used for binary classification problems.

    Calculates the loss for binary classification tasks by penalizing the probability based on how far
    it is from the actual label.
    """

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
    """
    Categorical Cross Entropy loss, used for multi-class classification problems.

    Measures the performance of a classification model whose output is a probability value between 0 and 1.
    Categorical cross entropy loss increases as the predicted probability diverges from the actual label.
    """

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
