class NeuralNet:
    def __init__(self, loss_fn, learning_rate, layers):
        self.loss_fn = loss_fn
        self.layers = layers
        self.set_learning_rates(learning_rate)

    def forward(self, x):
        """
        x needs to be a list of 2d arrays
        """
        assert len(x.shape) == 3
        for l in self.layers:
            x = l.forward(x)
        return x

    def set_learning_rates(self, learning_rate):
        for l in self.layers:
            try:
                l.set_learning_rate(learning_rate)
            except AttributeError:
                continue

    def compute_loss(self, pred, y):
        return self.loss_fn.loss(pred, y)

    def backward(self):
        x = self.loss_fn.backward()
        for l in self.layers[::-1]:
            x = l.backward(x)
        return x
