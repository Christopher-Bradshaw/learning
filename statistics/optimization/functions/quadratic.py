import numpy as np

# Given a hessian, a gradient and an X, returns the value of a quadratic
def quadratic(h, g, x):
    dims = x.shape[0]
    assert x.shape() == (dims, 1)
    assert g.shape() == (dims, 1)
    assert h.shape() == (dims, dims)
    return np.matmul(x.T, np.matmul(h, x)) + np.matmul(g.T, x)
