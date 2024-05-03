# Educational-Neural-Network

This project is a Python implementation of an artificial neural network framework aimed at deeply understanding the fundamental concepts behind neural networks and deep learning. This project is inspired by Andrej Karpathy and Russ Salakhutdinov, who strongly recommend implementing such frameworks for educational purposes.

## Results on MNIST

As an educational project (focusing on my own mastery of deep learning concepts), this framework is primarily compatible with solving multi-class problems (>2). It was designed to address the problem of digit recognition using the MNIST database.

The results highlight the learning capabilities of the model.

![](/data/images/loss_and_acc.png "Loss and accuracy during training.")
![](/data/images/testset_perf.png "Validation performances")

![](/data/images/testset_conf.png "Validation confusion matrix")

The validation test shows accuracy ranging from 92-95%. Goal of this project was not to achieve 99% accuracy but decent learning capabilities, since this implementation is from scratch.

However, a test with a second, smaller dataset of hand-written digits formatted in the same way yields poorer results (shown bellow).

![](/data/images/devset_perf.png "Devset performances")

![](/data/images/devset_conf.png "Devset confusion matrix")

I have not focused much on improving the results of this test since it is not the project's goal, but there is room for improvement, especially in data formatting more than in the neural network's capability. For more details on the data from the second validation test, see the notebook titled: `image_extraction_for_devset_validation.ipynb`.

The `mnist_with_tensorflow.ipynb` notebook compares this implementation with TensorFlow, a leading industry deep learning framework.

## Areas for Improvement

- Enhance the complexity of the framework by implementing additional types of layers such as Dropout, convolutions...
- Improve the compatibility of activation functions, optimizers, and loss functions to enrich the framework.
- Enable saving and loading a model (hyperparameters, weights, biases...).
- Optimize memory usage (numpy, Cython...).
