
class NeuralNet:
    def __init__(self, loss_fn, *layers):
        self.loss_fn = loss_fn
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def compute_loss(self, pred, y):
        return self.loss_fn.loss(pred, y)

    def backward(self):
        x = self.loss_fn.backward()
        for l in self.layers[::-1]:
            x = l.backward(x)
        return x
