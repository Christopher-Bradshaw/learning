import numpy as np

# Remember that solving Ax = b is the same as minimzing,
# Phi = 1/2 x^t A x - b^t x as del Phi is Ax - b
# I'm not really sure what to call this...
def linear_system_phi(A, b, x):
    dims = len(A)
    len_x = x.shape[1]
    assert A.shape == (dims, dims) # square matrix
    assert b.shape == (dims, 1) # column vector
    assert x.shape == (dims, len_x) # but can have many columns!

    # While the input shape makes sense to me, it actually needs to be
    # a list of column vectors to get properly broadcasted.
    x = x.T[:,:,np.newaxis]
    assert x.shape == (len_x, dims, 1)

    xr = x
    xl = x.transpose((0, 2, 1))
    return (0.5 * np.matmul(xl, np.matmul(A, xr)) - np.matmul(b.T, xr)).flatten()
