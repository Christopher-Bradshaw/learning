# Parallelization with Fractals

![Julia image](/cython/parallelization/julia.png?raw=true, "Julia")

**I don't really understand parallelization properly. Take what follows with a grain of salt.**

## Big Picture

In the other examples we've shown how cython speeds up code running in a single thread/process. Once we have hit the limit with this, we can attempt to parallelize.

### Process level parallelization

This is a fairly heavy weight process. In python on linux this is done with a call to `os.fork` which I expect eventually runs the system call `fork`. This creates a new process with its own copy of everything. This sounds awfully slow and memory intensize (especially if you are working with large numpy arrays), but fork implements copy-on-write so it will use the parent's copy of everything until it starts to change things. If all it does it create new objects then the only memory allocated will be for those objects, which it can then pass back to the parent which has been (hopefully) waiting for it.

### Thread level parallelization

Less heavy weight. Memory is shared between threads.

OpenMP is a way to describe how to run things in parallel. This info is used by compilers to automatically multithread that section of code.


### When/How to parallelize

When it makes sense:
* you are CPU bound
* you are frequently waiting on things e.g. disk reads, http requests

When it doesn't make sense:
* If you are saturating memory/io I think there is nothing you can do.


Use processes if the work is not completely independent and each process needs to modify global memory.


## Details of what we did here

1. Wrote efficient, single threaded, cython
2. Saw that each iteration of our loop was completely independent. It just does work on one pixel. This workload is also not memory intensize. This bodes well for MP
3. Replaced range with `prange` and added the `open_mp` compiler args.
4. Realized that with the default scheduler some threads (the ones that have regions that -> infinity fast) finished early. Moved to guided scheduling to improve this.

That's it! For this tiny bit of extra work we get (on an 8 core, 16 thread machine) an ~8x speedup! I didn't compare to the pure python implementation but I suspect that overall we are comfortably 2 orders of magnitude faster.
