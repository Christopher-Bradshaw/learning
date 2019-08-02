import numpy as np

from activations import ReLU

class TestReLU:
    def test_ReLU(self):
        x = np.random.random(100)
        x[::2] *= -1

        y = ReLU().forward(x)
        assert np.all(y[::2] == 0) and np.all(y[1::2] == x[1::2])
