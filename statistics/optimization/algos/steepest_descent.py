import numpy as np
from .line_search import get_line_length

def steepest_descent(f, grad_f, x):
    all_x = []
    while True:
        all_x.append(x)
        direction = -grad_f(x)
        rate = np.linalg.norm(direction)
        # This is roughly floating point accuracy. Below this we are ~
        # at a stationary point.
        if rate < 1e-7:
            break
        direction /= rate
        a = get_line_length(f, grad_f, x, direction, a_max=10)
        x = x + direction * a
    return np.array(all_x)
