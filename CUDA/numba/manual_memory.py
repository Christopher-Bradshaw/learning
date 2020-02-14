import numpy as np
from numba import cuda

# See http://numba.pydata.org/numba-doc/latest/cuda/memory.html

threadsPerBlock = 1024
blocks = 1


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


def lots_of_copies():
    inp = np.arange(10000)
    out = np.zeros_like(inp)
    double[blocks, threadsPerBlock](inp, out)


def few_copies():
    inp = np.arange(10000)
    d_inp = cuda.to_device(inp)
    d_out = cuda.device_array_like(inp)
    double[blocks, threadsPerBlock](d_inp, d_out)

    out = d_out.copy_to_host()


def main():
    # Compile
    lots_of_copies()
    few_copies()

    # Now benchmark
    start, end = cuda.event(timing=True), cuda.event(timing=True)
    n = 200
    for f in [lots_of_copies, few_copies]:
        times = []
        for _ in range(n):
            start.record()
            f()
            end.record()
            end.synchronize()
            t = cuda.event_elapsed_time(start, end)
            times.append(t)
        print(f.__name__, np.mean(times), np.std(times) / np.sqrt(n))


if __name__ == "__main__":
    main()
