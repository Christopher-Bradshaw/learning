import numpy as np

from neural_net import NeuralNet
from activations import ReLU
from linear import Linear

class TestNeuralNet:
    def test_neural_net(self):
        n_in, n_out = 3, 2

        weights = np.array([[0, -1, 2], [-3, 4, -5]])
        bias = np.arange(n_out)

        nn = NeuralNet(
                Linear(n_in, 2, weights, bias),
                ReLU(),
        )
        x = np.arange(n_in)
        # |0 -1 2 | |0|
        # |-3 4 -5| |1| + |0, 1|= |3, -6| + |0, 1| = |3, -5| -> |3, 0|
        #           |2|

        assert np.array_equal(nn.forward(x), [3, 0])
