import autograd.numpy as np

def dogleg(grad_f, hess_f, x, max_step):
    g = grad_f(x)
    B = hess_f(x)
    # This only works if B is PD
    assert np.all(np.linalg.eig(B)[0] > 0)

    # The step to the absolute minimum - use this if valid
    pB = -np.matmul(np.linalg.inv(B), g)

    if np.linalg.norm(pB) < max_step:
        return pB

    # The step along the gradient. Step length modulated by
    # the relative strength of g and B
    pU = -np.matmul(g, g) / np.matmul(g, np.matmul(B, g)) * g

    # Use the combination of the two steps to step to the boundary
    s1, s2 = pU, pB - pU

    step = lambda tau: min(tau, 1) * s1 + max(0, tau - 1) * s2
    tau = _binary_search(lambda tau: np.linalg.norm(step(tau)) - max_step)

    return step(tau)

def _binary_search(f):
    tau = 1
    delta = 0.5
    for _ in range(20):
        # Step is too long
        if f(tau) > 0:
            tau -= delta
        else:
            tau += delta
        delta /= 2
    return tau
