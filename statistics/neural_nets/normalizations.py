import numpy as np


class BatchMinMaxNorm:
    def __init__(self):
        self.mins = None
        self.maxes = None

    def forward(self, x):
        assert len(x.shape) == 3
        # Can't normalize when there is only one data point
        if x.shape[0] == 1:
            return x

        self.maxes, self.mins = np.amax(x, axis=0), np.amin(x, axis=0)
        # First part puts in [0, 1], then [0, 2], then [-1, 1]
        return ((x - self.mins) / (self.maxes - self.mins)) * 2 - 1

    def backward(self, d_loss_d_out):
        # Single item, no change
        if self.mins is None:
            return d_loss_d_out

        # Not sure if this is right...
        ret = d_loss_d_out / (self.maxes - self.mins) * 2

        # Reset before returning
        self.mins, self.maxes = None, None
        return ret
