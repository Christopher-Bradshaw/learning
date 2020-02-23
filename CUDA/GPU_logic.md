# Logic

## Threads

We launch a kernel with a grid of blocks, each of which contain some number of threads.

### The block

See [CUDA docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy).

Each block has "shared memory" that is accessible to any of its threads. This is stored in the shared memory sector of single SM. Thus, all threads in a given block *must* be scheduled on a single SM. This places a limit on the number of threads per block (though I'm not entirely sure what governs this limit). As they share memory, threads with in a block must be able to, and can, synchronize their execution (in order to avoid race conditions) in a lightweight way.

While all threads in a given block will be assigned to the same SM, blocks may be run in any order, in series or in parallel, on any SM. Thus, each block needs to be fully independent of all others. This makes scheduling very easy, and also makes it easy to write code that is portable to different processors with more/fewer SM.
If you need blocks to share memory, the kernel needs to be split into (at least) two calls. The first that writes the data that needs to be shared and the second that reads it.


## Memory

The GPU (device) has a separate memory space to the CPU (host). In the same way that CPU's have various levels of memory (RAM, L2, L1, registers), the GPU has different types of memory with different properties.


### Global memory

The main pool of devices memory is called global memory. This memory is the GDDR-SDRAM (pretty similar to normal CPU memory which is DDR-SDRAM, but with optimizations for graphics workloads. I don't know what these are...) whose size (e.g. my 1060 has 6GB) and bandwith (192 GB/s) are advertised.

However, global memory has relatively long access latencies (100s of clock cycles) and finite bandwidth. These can mean that some SM

### Shared memory

Each processor core
Each block has 




### Constant memory


