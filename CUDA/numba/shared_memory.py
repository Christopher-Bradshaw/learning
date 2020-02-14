import numpy as np
from numba import cuda, int32

threadsPerBlock = 1024
blocks = 16

n_calls = 10


@cuda.jit
def mult_by_x_shared(inp, out, x):
    shared = cuda.shared.array(1, int32)
    tix = cuda.threadIdx.x

    while tix < len(shared):
        shared[tix] = x[tix]
        tix += cuda.blockDim.x
    cuda.syncthreads()

    idx = cuda.grid(1)
    while idx < len(inp):
        out[idx] = inp[idx]
        # This only becomes valuable when this is used many times in a loop
        for _ in range(n_calls):
            out[idx] *= shared[0]
            out[idx] += shared[0]
        idx += cuda.gridsize(1)


@cuda.jit
def mult_by_x_not_shared(inp, out, x):
    idx = cuda.grid(1)
    while idx < len(inp):
        out[idx] = inp[idx]
        for _ in range(n_calls):
            out[idx] *= x[0]
            out[idx] += x[0]
        idx += cuda.gridsize(1)


def main():
    inp = np.arange(1000000, dtype=np.int32)
    factor = 4
    start, end = cuda.event(True), cuda.event(True)

    reses = []
    for (name, f) in [
        ("not shared", mult_by_x_not_shared),
        ("shared", mult_by_x_shared),
        ("not shared", mult_by_x_not_shared),
    ]:
        times = []
        for i in range(100):

            d_out = cuda.device_array_like(inp)

            start.record()
            f[blocks, threadsPerBlock](
                cuda.to_device(inp), d_out, cuda.to_device(np.array([factor]))
            )
            end.record()
            end.synchronize()

            out = d_out.copy_to_host()

            # Compilation...
            if i != 0:
                times.append(cuda.event_elapsed_time(start, end))
        print(
            f"{name}: {np.mean(times):.2f} +/- {np.std(times) / np.sqrt(len(times)):.3f} (max: {np.max(times):.2f})"
        )
        reses.append(out)
    assert np.all([reses[0] == reses_i for reses_i in reses])


main()
