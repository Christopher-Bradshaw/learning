import autograd.numpy as np

def many_matmul(*args):
    res = args[-1]
    for arg in args[::-1][1:]:
        res = np.matmul(arg, res)
    return res
