# GPU programming with CUDA

Note that none of the cuda tools come with useful man pages, but they have great `--help` pages.

## Basic terminology

Computation is either done on the `host` (the CPU) or a `device` (a CUDA enabled GPU).

GPU's excel when the problem contains lots of *data parallelism*. See [the different types of parallelism](./types_of_parallelism.md).



## Compilation

Compile with `nvcc`. This appears to compile in C/C++ mode if the file extension is .c/.cpp. Make sure to have the extension of `.cu` if you want to compile the CUDA stuff! See [the docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#supported-input-file-suffixes)

If you compile a cuda object file `nvcc -c lib.cu -o lib.o` and want to link this into your main project `clang++ main.o lib.o` you need to explicitly include the cuda library `-lcudart`.

### Useful flags
* `nvcc --run file.cu`: Run it
* `-g`: Adds debug info


### Debugging

Call `cuda-gdb a.out`. Then its just like the normal gdb.


### Built in variables

Their are some vars that the compiler inserts (not sure about this). e.g., `blockIdx`, `threadIdx`. See [the docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-variables)


## Syntax

When we call a CUDA function, we do it with,

```
<<<# of blocks, # of threads>>>f( args )
```

See [the docs, section 2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model) for a good description (and images!) of this hierarchy. The basics,

There is a grid on which sits blocks. These blocks have an index in 3D space. Inside each block there are threads. These threads can share memory (per block shared memory). Each threads also has an index in 3D space.


## Hardware stuff

### Hardware execution

While this is the logical way things are organised, the hardware implementation is different. Threads in a single logical block get executed in hardware blocks of 32 called warps. I guess this means that if you have a logical block of e.g. 33 you schedule 2 warps, the second of which is almost entirely empty.

The total number of warps is `ceil(ThreadsPerBlock / WarpSize)`

### Compute Capabilities

See [the docs, appendix H](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities). To know what is going on, you also need to know your hardware's compute capability. See [these tables](https://developer.nvidia.com/cuda-gpus#compute) (you need to click into e.g. CUDA-enabled GeForce Products) to get that number.
My 1060 has compute capability of 6.1.

### Getting info

* `nvidia-smi`
