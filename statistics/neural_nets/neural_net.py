
class NeuralNet:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

