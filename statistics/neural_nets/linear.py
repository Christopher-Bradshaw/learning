import numpy as np

# The forward computes w*in + b
class Linear:
    def __init__(self, n_in, n_out, weights=None, bias=None):

        self.weights = weights if weights is not None else np.random.normal(size=(n_out, n_in))
        self.bias = bias if bias is not None else np.random.random(n_out)

        print(self.weights.shape)
        assert self.weights.shape == (n_out, n_in)
        assert self.bias.shape == (n_out, )

    def forward(self, x):
        return np.matmul(
                self.weights,
                x,
        ) + self.bias

