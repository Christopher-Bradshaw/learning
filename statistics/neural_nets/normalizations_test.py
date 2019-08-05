import numpy as np
from normalizations import BatchMinMaxNorm


class TestBatchMinMaxNorm:
    def test_does_nothing_with_single_item_batches(self):
        n = BatchMinMaxNorm()
        x = np.array([[[1], [2]]])
        assert np.array_equal(n.forward(x), x)

    def test_correctly_min_maxes(self):
        n = BatchMinMaxNorm()
        x = np.array([[[1], [2]], [[2], [1]], [[3], [0]]])
        y = np.array([[[-1], [1]], [[0], [0]], [[1], [-1]]])
        assert np.array_equal(n.forward(x), y)

    def test_correcty_runs_backward(self):
        n = BatchMinMaxNorm()

        # As the spread of the input doesn't change, it just gets shifted,
        # nothing should change in backward
        x = np.array([[[1], [2]], [[2], [1]], [[3], [0]]])
        n.forward(x)
        inp_back = np.array([[1], [1]])
        assert np.array_equal(n.backward(inp_back), inp_back)

        # Now the spread is halved in this layer.
        # I'm still not sure I have this the right way around...
        n.forward(x * 2)
        inp_back = np.array([[1], [1]])
        assert np.array_equal(n.backward(inp_back), inp_back / 2)
