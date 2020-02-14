import numpy as np
from numba import cuda, int32


@cuda.reduce
def sum_reducer(a, b):
    return a + b


def main():
    a = np.arange(100)
    d_a = cuda.to_device(a)

    print(sum_reducer(a))
    # We can't use the reducer from within a jit compiled function
    # But we can just keep the array on the device.
    # And then reduce it from outside!
    print(sum_reducer(d_a))


main()
