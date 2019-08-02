import numpy as np

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)
