import numpy as np
import matplotlib.pyplot as plt


def plot_history(loss_history, accuracy_history):
    """
    Plot the training loss and accuracy history with fixed y-axis for accuracy.

    Parameters:
    loss_history (list of float): List containing the history of loss values.
    accuracy_history (list of float): List containing the history of accuracy values.
    """
    epochs = range(1, len(loss_history) + 1)

    fig, ax1 = plt.subplots()

    color = 'tab:orange'
    ax1.set_xlabel('Batches')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss_history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, accuracy_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)

    plt.title('Training Loss and Accuracy History')

    plt.show()


def plot_confusion_matrix(confusion_matrix, class_labels, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.

    Parameters:
    confusion_matrix (numpy.ndarray): The confusion matrix to display.
    class_labels (list): List of labels for the classification classes.
    title (str): Title of the plot.
    cmap (matplotlib.colors.Colormap): Color map to use for plotting.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=class_labels, yticklabels=class_labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'  # Integer formatting
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
