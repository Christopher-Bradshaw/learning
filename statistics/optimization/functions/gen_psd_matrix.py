import numpy as np

# Generate a PSD matrix of the given size.
def gen_psd_matrix(dims, eigenvalues=None):
    eigenvalues = eigenvalues or np.random.uniform(size=dims)
    assert np.all(eigenvalues > 0) # required for it to be PSD

    r = np.random.random((dims, dims))
    q, _ = np.linalg.qr(r)

    assert np.isclose(np.dot(q[:,0], q[:,1]), 0) # columns are orthogonal

    A = np.matmul(q.T, np.matmul(np.diag(eigenvalues), q))
    return A
