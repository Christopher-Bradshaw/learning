import numpy as np
from numba import cuda

threadsPerBlock = 1024
blocks = 1024


@cuda.jit
def double(inp, out):
    # By default CUDA will copy both inp and out onto the device
    # and then back off the device.
    # But, we don't need to copy inp back off - it is only read.
    # And we don't need to copy out onto the device - it is empty!

    idx = cuda.grid(1)
    while idx < len(inp):
        out[idx] = inp[idx] * 2
        idx += cuda.gridsize(1)


@cuda.reduce
def sum_reducer(a, b):
    return a + b


def main():
    inp = np.arange(10_000_000)
    d_out = cuda.device_array_like(inp)
    double[blocks, threadsPerBlock](cuda.to_device(inp), d_out)

    s = sum_reducer(d_out)
    exp = (len(inp) - 1) * len(inp)

    assert s == exp


main()
