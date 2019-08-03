import numpy as np

class ReLU:
    def __init__(self):
        self.d_out_d_in = None

    def forward(self, x):
        self._compute_d_out_d_in(x)
        return np.maximum(0, x)

    def _compute_d_out_d_in(self, x):
        self.d_out_d_in = np.zeros_like(x)
        self.d_out_d_in[x > 0] = 1


    def backward(self, d_loss_d_out):
        """
        We want dC/dw_i for all the weights in our network.
        We don't have any weights in this layers.
        But, in previous layers there might be weights.
        d/dw relu(f(x, w)) = d
        In those layers we will calculate dout/dw
        we have dC/dy (from the next layer)
        dC/dx = dC/dy dy/dx
        """

        # Don't need to update weights

        # So just backprop
        return d_loss_d_out * self.d_out_d_in
