class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, gradient):
        return weights - self.learning_rate * gradient


optimizers = {
    "gd": GradientDescent()
}
