import numpy as np
from .line_search import get_line_length

# FR is a non-linear CG method. See algo 5.4 in Nocedal/Wright
def fletcher_reeves(f, grad_f, x, restart=None, use_pr=False):
    assert not (restart and use_pr) # Either FR, FR w restarts, or PR
    # Come back and look at http://www.unilim.fr/laco/rapports/2003/R2003_10.pdf
    if use_pr: print("This might not work - this needs a modification to wolfe to ensure descent that I can't find...")

    a_max = 1

    # Important to start in the direction of steepest descent
    gf = grad_f(x)
    p = -gf

    # To evaluate performance
    positions = [x]
    while np.linalg.norm(gf) > 1e-7:
        # Our line length computation expects p to be of unit length
        # So scale the input, then scale the output to be correct for our p
        # N.B. **don't actually scale p here else setting the new p will be wrong**
        p_scale = np.linalg.norm(p)
        alpha = get_line_length(f, grad_f, x, p / p_scale, a_max, c1=1e-3, c2=0.4) / p_scale

        x = x + alpha * p
        positions.append(x)

        gf_next = grad_f(x)

        # FR with restarts
        if restart and np.dot(gf, gf_next) / np.linalg.norm(gf_next)**2 > restart:
            beta = 0
        # PR
        elif use_pr:
            beta = max(
                    np.dot(gf_next, gf_next - gf) / np.linalg.norm(gf)**2,
                    0
            )
        # Default FR
        else:
            beta = np.matmul(gf_next, gf_next) / np.matmul(gf, gf)

        gf = gf_next

        p = -gf + beta * p

    return np.array(positions)

# The below are both for convex functions

# Algorithm 5.2 in Nocedal/Wright
# See below for a slightly less efficient but more annotated algo
# See my (written) notes for a description of why this is the same as that one.
def conjugate_gradient(A, b, x):
    dims = A.shape[0]
    assert A.shape == (dims, dims)
    assert x.shape == (dims,)
    assert b.shape == (dims,)

    r = np.matmul(A, x) - b
    p = -r

    # Some things to evaluate performance
    positions = [x]
    residual_norm = [np.linalg.norm(r)]

    for _ in range(dims):
        alpha = np.matmul(r, r) / np.matmul(p, np.matmul(A, p))

        x = x + alpha * p
        positions.append(x)

        r_next = r + alpha * np.matmul(A, p)
        residual_norm.append(np.linalg.norm(r_next))
        if np.linalg.norm(r_next) == 0:
            break

        beta = np.matmul(r_next, r_next) / np.matmul(r, r)
        r = r_next

        p = -r + beta * p
    return np.array(positions), np.array(residual_norm)

# Algorithm 5.1 in Nocedal/Wright
# This is less efficient than the other one
# Kept here because it is a bit easier to follow
def _inefficient_but_easy_to_read_CG(A, b, x):
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
