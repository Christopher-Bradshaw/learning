import numpy as np
from numba import cuda


@cuda.jit
def double(inp, out):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x

    idx = tx + bx * bw
    while idx < len(inp):
        out[idx] = inp[idx] * 2
        idx += bw


def main():
    inp = np.arange(1000)
    out = np.zeros_like(inp)

    threadsPerBlock = 1000
    blocks = 1
    double[blocks, threadsPerBlock](inp, out)
    print(out)


if __name__ == "__main__":
    main()
