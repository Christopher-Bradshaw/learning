# GPU

![](./GPU_arch.png)


## GeForce 1060

According to the [specs](https://www.geforce.com/hardware/desktop-gpus/geforce-gtx-1060/specifications) my 1060 has 1280 CUDA cores.

### Highest level

The GPU contains an L2 cache of global memory. 

### Multiprocessors

The GPU as a whole is split into multiple *multiprocessors*. Each of these contain,

Processing
* 128 CUDA cores
* 4 warp schedulers
Memory
* 48 KB L1 cache of global memory
* 96 KB of shared memory
* A read-only constant cache, which speeds up (caches) reads from the device's constant memory.


Each multiprocessor consists of (see [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x)) 
The 

See [the docs, appendix H](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities). To know what is going on, you also need to know your hardware's compute capability. See [these tables](https://developer.nvidia.com/cuda-gpus#compute) (you need to click into e.g. CUDA-enabled GeForce Products) to get that number.
My 1060 has compute capability of 6.1.

## Multiprocessor

Per [the docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x)

## Threads

We launch a kernel with a grid of blocks, each of which contain some number of threads.

### The block

See [CUDA docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy)

There is a limit on the number of threads per block, as the entire block needs to fit on a single processor core.

Each block needs to be independent of all others. It needs to be possible to execute the blocks in any order, in parallel, or in series. This makes scheduling very easy, and makes it easier to write code that is portable to different processors.

## Memory

The GPU (device) has a separate memory space to the CPU (host). In the same way that CPU's have various levels of memory (RAM, L2, L1, registers), the GPU has different types of memory with different properties.


### Global memory

The main pool of devices memory is called global memory. This memory is the GDDR-SDRAM (pretty similar to normal CPU memory which is DDR-SDRAM, but with optimizations for graphics workloads. I don't know what these are...) whose size (e.g. my 1060 has 6GB) and bandwith (192 GB/s) are advertised.

However, global memory has relatively long access latencies (100s of clock cycles) and finite bandwidth. These can mean that some SM

### Shared memory

Each processor core
Each block has 




### Constant memory

