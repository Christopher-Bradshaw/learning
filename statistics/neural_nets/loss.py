import numpy as np

class MeanSquaredError:
    def __init__(self):
        self.d_out_d_in = None

    def loss(self, pred, true):
        self._compute_d_out_d_in(pred, true)
        return np.sum(np.power(pred - true, 2))

    def _compute_d_out_d_in(self, pred, true):
        self.d_out_d_in = 2 * (pred - true)

    def backward(self):
        """
        Return d loss / d pred
        """
        if self.d_out_d_in is None:
            raise Exception("Haven't computed the loss!")
        return self.d_out_d_in
