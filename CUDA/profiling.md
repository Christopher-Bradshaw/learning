# Profiling CUDA

See [profiling docs](https://docs.nvidia.com/cuda/profiler-users-guide/index.html). Also **N.B.** you might need to run these tools as root, see [here](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters). Getting X programs to work as root was a bit of a pain but `xhost +si:localuser:root` worked.

If you are using runtime tools, remember to compile with helpful option. `-g -lineinfo`.

## Events

The simplest thing you can do is work out how long some code took to run.

```
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);

ACTUAL CODE DOING THINGS GOES HERE

cudaEventRecord(stop);
// ^ tells us to record an event when we get here. But we can't read the time off it until we've got there
// So, we synchronize on that event.
cudaEventSynchronize(stop);
float t;
cudaEventElapsedTime(&t, start, stop);
std::cout << "Time taken: " << t << "ms" << std::endl;
```

## nvvp

If you would usually run it with `./main`, run,

```
nvvp ./main
```

This basically guides you through a lot of analysis. It's pretty sick...


## nvprof

See all the options [here](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview).
```
nvprof ./main
```

Useful args:
* `--print-gpu-trace`: Prints out the main things that the GPU does (copy memory, run kernel, etc) in order.
* `--metrics all`: Prints out a whole ton of metrics. Can also chose individual ones with `--metrics a,b,c`.
