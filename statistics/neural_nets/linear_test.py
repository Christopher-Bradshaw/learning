import numpy as np

from linear import Linear


class TestLinear:
    def test_forward(self):

        n_in, n_out = 3, 2

        bias = np.arange(n_out)[:, np.newaxis]
        weights = np.arange(n_in * n_out).reshape((n_out, n_in))

        layer = Linear(n_in, n_out, weights, bias)
        x = np.array([[[0], [1], [2]]])
        # |0 1 2| |0|   |0|   | 5|   |0|   | 5|
        # |3 4 5| |1| + |1| = |14| + |1| = |15|
        #         |2|

        # breakpoint()
        assert np.array_equal(layer.forward(x), [[[5], [15]]])
        assert np.array_equal(layer.d_out_d_in, weights)
