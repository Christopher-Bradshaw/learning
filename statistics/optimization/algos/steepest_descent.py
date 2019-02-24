import autograd.numpy as np
from .line_search import get_line_length

def steepest_descent(f, grad_f, x):
    positions = [x]
    while np.linalg.norm(grad_f(x)) > 1e-7:
        direction = -grad_f(x)
        direction /= np.linalg.norm(direction)
        a = get_line_length(f, grad_f, x, direction, a_max=10)
        x = x + direction * a
        positions.append(x)
    return np.array(positions)
