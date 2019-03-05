import autograd.numpy as np


# These aren't important enough to have their own files.
# But they are a bit of a schlep to ensure all the broadcasting works out
# x can either be passes as a 2d or 1d vector. In the 2d case the columns are
# individual xs


def f_one(Q, x):
    dims = len(Q)
    assert Q.shape == (dims, dims) # square matrix

    if len(x.shape) == 1:
        assert x.shape == (dims,)
        return np.log(1 + np.matmul(x, np.matmul(Q, x)))#.flatten()
    # This is for plotting
    elif len(x.shape) == 2:
        len_x = x.shape[1]
        assert x.shape == (dims, len_x) # but can have many columns!

        # While the input shape makes sense to me, it actually needs to be
        # a list of column vectors to get properly broadcasted.
        x = x.T[:,:,np.newaxis]
        assert x.shape == (len_x, dims, 1)

        xr = x
        xl = x.transpose((0, 2, 1))

        return np.log(1 + np.matmul(xl, np.matmul(Q, xr))).flatten()
    else:
        raise Exception("No clue what's going on here")
