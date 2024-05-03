import numpy as np


def true_positive(y_true, y_pred, class_label):
    """
    Calculate the number of true positives for a specific class.

    Parameters:
    y_true (numpy.ndarray): Array of ground truth labels.
    y_pred (numpy.ndarray): Array of predicted labels.
    class_label (int or float): The label of the class to calculate true positives for.

    Returns:
    int: The number of true positives for the specified class.
    """
    return np.sum((y_true == class_label) & (y_pred == class_label))


def true_negative(y_true, y_pred, class_label):
    """
    Calculate the number of true negatives for a specific class.

    Parameters:
    y_true (numpy.ndarray): Array of ground truth labels.
    y_pred (numpy.ndarray): Array of predicted labels.
    class_label (int or float): The label of the class to calculate true negatives for.

    Returns:
    int: The number of true negatives for the specified class.
    """
    return np.sum((y_true != class_label) & (y_pred != class_label))


def false_positive(y_true, y_pred, class_label):
    """
    Calculate the number of false positives for a specific class.

    Parameters:
    y_true (numpy.ndarray): Array of ground truth labels.
    y_pred (numpy.ndarray): Array of predicted labels.
    class_label (int or float): The label of the class to calculate false positives for.

    Returns:
    int: The number of false positives for the specified class.
    """
    return np.sum((y_true != class_label) & (y_pred == class_label))


def false_negative(y_true, y_pred, class_label):
    """
    Calculate the number of false negatives for a specific class.

    Parameters:
    y_true (numpy.ndarray): Array of ground truth labels.
    y_pred (numpy.ndarray): Array of predicted labels.
    class_label (int or float): The label of the class to calculate false negatives for.

    Returns:
    int: The number of false negatives for the specified class.
    """
    return np.sum((y_true == class_label) & (y_pred != class_label))


def precision_score(y_true, y_pred, average='macro'):
    """
    Calculate the precision score for predicted labels, using macro averaging.

    Parameters:
    y_true (numpy.ndarray): Array of ground truth labels.
    y_pred (numpy.ndarray): Array of predicted labels.
    average (str): The type of averaging performed on the data ('macro' by default).

    Returns:
    float: The calculated precision score.
    """
    classes = np.unique(y_true)
    precision = 0
    for c in classes:
        tp = true_positive(y_true, y_pred, c)
        fp = false_positive(y_true, y_pred, c)
        precision += tp / (tp + fp) if (tp + fp) > 0 else 0
    return precision / len(classes) if average == 'macro' else precision


def recall_score(y_true, y_pred, average='macro'):
    """
    Calculate the recall score for predicted labels, using macro averaging.

    Parameters:
    y_true (numpy.ndarray): Array of ground truth labels.
    y_pred (numpy.ndarray): Array of predicted labels.
    average (str): The type of averaging performed on the data ('macro' by default).

    Returns:
    float: The calculated recall score.
    """
    classes = np.unique(y_true)
    recall = 0
    for c in classes:
        tp = true_positive(y_true, y_pred, c)
        fn = false_negative(y_true, y_pred, c)
        recall += tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall / len(classes) if average == 'macro' else recall


def f1_score(y_true, y_pred, average='macro'):
    """
    Calculate the F1 score, which is the harmonic mean of precision and recall.

    Parameters:
    y_true (numpy.ndarray): Array of ground truth labels.
    y_pred (numpy.ndarray): Array of predicted labels.
    average (str): The type of averaging performed on the data ('macro' by default).

    Returns:
    float: The calculated F1 score.
    """
    prec = precision_score(y_true, y_pred, average)
    rec = recall_score(y_true, y_pred, average)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0


def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy of predictions, defined as the ratio of correct predictions to the total number of predictions.

    Parameters:
    y_true (numpy.ndarray): Array of ground truth labels.
    y_pred (numpy.ndarray): Array of predicted labels.

    Returns:
    float: The accuracy of the predictions.
    """
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions


def calculate_batch_accuracy(y_preds, y_true):
    """
    Calculate batch accuracy using the accuracy_score function.

    Parameters:
    y_preds (numpy.ndarray): Array of predicted probabilities for each class.
    y_true (numpy.ndarray): One-hot encoded array of true labels.

    Returns:
    float: The accuracy of the predictions in the batch.
    """
    # Convert the probabilities to class indices
    predicted_classes = np.argmax(y_preds, axis=1)
    true_classes = np.argmax(y_true, axis=1)

    # Use the previously defined accuracy_score function
    accuracy = accuracy_score(true_classes, predicted_classes)

    return accuracy
