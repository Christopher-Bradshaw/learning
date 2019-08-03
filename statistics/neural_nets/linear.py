import numpy as np

# The forward computes w*in + b
class Linear:
    def __init__(self, n_in, n_out, weights=None, bias=None):

        self.weights = weights if weights is not None else np.random.normal(size=(n_out, n_in))
        self.bias = bias if bias is not None else np.random.random(n_out)

        assert self.weights.shape == (n_out, n_in)
        assert self.bias.shape == (n_out, )

        self.last_x = None
        # This is ... pretty trivial
        self.d_out_d_in = self.weights

    def forward(self, x):
        self.last_x = x
        return np.matmul(
                self.weights,
                x,
        ) + self.bias

    def backward(self, d_loss_d_out):
        """
        We want dC/dw and dC/db, we have dC/dy (from the next layer)
        dC/dx = dC/dy dy/dx
        """
        # Update the weights

        # And prop the d backwards
        return np.matmul(
                self.d_out_d_in.T,
                d_loss_d_out,
        )
