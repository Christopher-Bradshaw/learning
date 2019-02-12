import numpy as np

def conjugate_gradient(A, b, x):
    dims = A.shape[0]
    assert A.shape == (dims, dims)
    assert x.shape == (dims,)
    assert b.shape == (dims,)

    # Calculate the initial residual. This is also the gradient as grad(phi(x)) = Ax - b
    r = np.matmul(A, x) - b
    # Go in the direction of steepest descent
    p = -r
    positions = [x]

    for _ in range(dims):
        # Step length as derived in my notes
        # Different
        alpha = np.matmul(r, r) / np.matmul(p, np.matmul(A, p))

        # Update x, taking a step in direction p
        x = x + alpha * p
        positions.append(x)

        # Update the residual in this new location. Bail if solved
        r_next = r + alpha * np.matmul(A, p)
        if np.linalg.norm(r_next) == 0:
            break

        # See notes. Ensures that p_k is orthogonal to p_k-1
        beta = np.matmul(r_next, r_next) / np.matmul(r, r)
        r = r_next

        # Set the next search direction
        p = -r + beta * p

    return np.array(positions)

# Algorithm 5.1 in Nocedal/Wright
# This is less efficient than the other one
def conjugate_gradient_5_1(A, b, x):
    dims = A.shape[0]
    assert A.shape == (dims, dims)
    assert x.shape == (dims,)
    assert b.shape == (dims,)

    # Calculate the initial residual. This is also the gradient as grad(phi(x)) = Ax - b
    r = np.matmul(A, x) - b
    # Go in the direction of steepest descent
    p = -r

    positions = [x]
    for _ in range(dims):
        # Step length as derived in my notes
        alpha = - np.matmul(r, p) / np.matmul(p, np.matmul(A, p))

        # Update x, taking a step in direction p
        x = x + alpha * p
        positions.append(x)

        # Update the residual in this new location. Bail if solved
        r = np.matmul(A, x) - b
        if np.linalg.norm(r) == 0:
            break

        # See notes. Ensures that p_k is orthogonal to p_k-1
        beta = np.matmul(r, np.matmul(A, p)) / np.matmul(p, np.matmul(A, p))

        # Set the next search direction
        p = -r + beta * p

    return np.array(positions)
