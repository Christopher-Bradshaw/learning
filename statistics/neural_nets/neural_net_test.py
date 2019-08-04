import numpy as np

from neural_net import NeuralNet
from activations import ReLU
from linear import Linear
from loss import MeanSquaredError


class TestNeuralNet:
    def test_neural_net(self):
        n_in, n_out = 3, 2

        weights = np.array([[0, -1, 2], [-3, 4, -5]])
        bias = np.arange(n_out)

        nn = NeuralNet(
            MeanSquaredError(), 1e-3, layers=[Linear(n_in, 2, weights, bias), ReLU()]
        )
        x = np.arange(n_in)
        y = np.array([2, 3])
        assert len(y) == n_out
        # |0 -1  2| |0|   |0|   | 3|   |0|   | 3|    |3|
        # |-3 4 -5| |1| + |1| = |-6| + |1| = |-5| -> |0|
        #           |2|

        pred = nn.forward(x)
        assert np.array_equal(pred, [3, 0])

        nn.compute_loss(pred, y)
        dL_dx = nn.backward()

        # |0 -1  2| |0 + dx1|   | 3 + 0    -  dx2 + 2dx3|   | 3 + ...|    |3 - dx2 + 2dx3|
        # |-3 4 -5| |1 + dx2| = |-6 - 3dx1 + 4dx2 - 5dx3| = |-5 + ...| -> |0|
        #           |2 + dx3| The second component is ReLU'ed away
        # MSE loss results in 2( ... ) so dL = -2dx2 + 4dx3, dL/dx = |0, -2, 4|

        assert np.array_equal(dL_dx, [0, -2, 4])

    def test_neural_net_tends_to_correct(self):
        n_in, n_out = 4, 2

        np.random.seed(12)
        weights = np.random.normal(size=(n_out, n_in))
        bias = np.zeros(n_out)

        nn = NeuralNet(
            MeanSquaredError(), 1e-2, layers=[Linear(n_in, 2, weights, bias)]
        )

        x = np.array([-1, 0.5, -0.33, 0.75])
        y = np.array([-0.5, 0.2])

        for _ in range(1000):
            pred = nn.forward(x)
            loss = nn.compute_loss(pred, y)
            nn.backward()

        assert np.isclose(loss, 0)
