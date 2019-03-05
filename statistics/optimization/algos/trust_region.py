import autograd.numpy as np
from .models import quadratic

# Runs a trust region optimization using f_sub to choose the step
# given the gradient, hessian and region size.
def trust_region(f, grad_f, hess_f, x, f_sub, max_step=1, initial_step=None, eta=0.1):
    dims = len(x)
    step = initial_step or 1
    assert 0 < eta < 1/4

    positions = [x]
    while np.linalg.norm(grad_f(x)) > 1e-7:
        p = f_sub(grad_f, hess_f, x, step)

        m = quadratic(f, grad_f, hess_f, x)
        model_ratio = (f(x) - f(x + p)) / (m(np.zeros(dims)) - m(p))

        # Model predicted a much larger change that true, reduce step size
        if model_ratio < 1/4:
            step = 1/4 * step
        # Model was pretty good and at the max length, increase step size
        elif model_ratio > 3/4 and np.linalg.norm(p) == step:
            step = min(max_step, 2*step)

        # Negative model ratio means the step resulted in an increase
        # In general a small model ratio means that the model was no good, so don't update.
        if model_ratio > eta:
            x = x + p
            positions.append(x)

    return np.array(positions)
