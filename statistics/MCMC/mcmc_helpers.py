import numpy as np
from scipy.stats import uniform

def get_next_x(current_x, sampling_func, proposal_pdf):
    potential_next_x = current_x + proposal_pdf.rvs()
    to_move = uniform.rvs()
    if sampling_func(potential_next_x) / sampling_func(current_x) >  to_move:
        return potential_next_x
    else:
        return current_x


def hidden_func(x):
    if type(x) is np.ndarray:
        y = np.zeros(len(x))
        y[np.logical_and(0.5 < x, x < 1.5)] = x[np.logical_and(0.5 < x, x < 1.5)]
        return y

    if 0.5 < x < 1.5:
        return x
    return 0
