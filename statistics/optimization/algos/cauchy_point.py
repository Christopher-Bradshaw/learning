import autograd.numpy as np


# Find the direction with the gradient (steepest descent) and the
# step length with grad, hess and the max step to find an approximate
# minimum.
# This is a solution of the subproblem of trust_region
def cauchy_point(grad_f, hess_f, x, max_step):
    g = grad_f(x)
    B = hess_f(x)
    if np.matmul(x, np.matmul(B, x)) <= 0:
        tau = 1
    else:
        tau = min(1, np.linalg.norm(g)**3 / (max_step * np.matmul(g, np.matmul(B, g))))
    return -tau * max_step * g / np.linalg.norm(g)
