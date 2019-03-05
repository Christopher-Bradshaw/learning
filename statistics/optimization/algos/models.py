import numpy as np

# Return a quadrative model for f around x
def quadratic(f, grad_f, hess_f, x):
    dims = x.shape[0]
    assert x.shape == (dims,)

    return lambda p: f(x) + np.matmul(grad_f(x), p) + 0.5 * np.matmul(p, np.matmul(hess_f(x), p))
