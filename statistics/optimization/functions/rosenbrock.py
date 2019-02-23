import numpy as np

# x either comes in with shape (ndims,) and leaves as a float
# or it comes in with shape (n_dims, n_samples) and leaves as (n_samples,)
def rosenbrock(x, c1=1, c2=100):
    if len(x.shape) == 1:
        return np.sum(c2 * np.power(x[:-1]**2 - x[1:], 2) + np.power(x[:-1] - c1, 2))
    elif len(x.shape) == 2:
        return np.sum(
                c2 * np.power(x[:-1,:]**2 - x[1:,:], 2) + np.power(x[:-1,:]**2 - c1, 2),
                axis=0
        )
    else:
        raise Exception("No clue what's going on here")
