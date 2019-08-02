import numpy as np

from linear import Linear

class TestLinear:
    def test_forward(self):

        n_in, n_out = 3, 2

        bias = np.arange(n_out)
        weights = np.arange(n_in*n_out).reshape((n_out, n_in))

        layer = Linear(n_in, n_out, weights, bias)
        x = np.arange(n_in)
        # |0 1 2| |0|
        # |3 4 5| |1| + |0, 1|= |5, 14| + |0, 1| = |5, 15|
        #         |2|

        assert np.array_equal(layer.forward(x), [5, 15])

