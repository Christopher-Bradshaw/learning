import numpy as np

# The forward computes w*in + b
class Linear:
    def __init__(self, n_in, n_out, weights=None, bias=None):
        self.weights = (
            weights.astype(np.float32)
            if weights is not None
            else np.random.normal(size=(n_out, n_in))
        )
        self.bias = (
            bias.astype(np.float32) if bias is not None else np.random.random(n_out)
        )

        assert self.weights.shape == (n_out, n_in)
        assert self.bias.shape == (n_out,)

        self.last_x = None

        # This is ... pretty trivial
        # Note that this does not actually copy - it is just a reference. This is good,
        # because it means that updates to the weights automatically update this. This is bad
        # because updates to the weights automatically update this.
        self.d_out_d_in = self.weights

        self.d_out_d_weights = None
        self.learning_rate = None

    def forward(self, x):
        self.last_x = x
        return np.matmul(self.weights, x) + self.bias

    def _update_d_out_d_weights(self, x):
        """
        weights1 -> top row of weights
        weights_1 -> left column of weights
        out1 = weights1 dot x
        So, dout1/dweights = x

        |y1||x1, x2, ..., x3|
        |y2|
        or the other way?
        """

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def backward(self, d_loss_d_out):
        """
        We want dC/dw and dC/db, we have dC/dy (from the next layer)
        dC/dx = dC/dy dy/dx
        """
        # Find out what to prop backwards. This needs to happen before the weights are updated
        # as that also changes self.d_out_d_in
        back = np.matmul(self.d_out_d_in.T, d_loss_d_out)

        # Update the weights
        d_out_d_weights = d_loss_d_out[:, np.newaxis] * self.last_x[np.newaxis, :]
        self.weights -= self.learning_rate * d_out_d_weights

        return back
