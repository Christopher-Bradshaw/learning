# CUDA

## Compilation

Compile with `nvcc`. This appears to compile in C/C++ mode if the file extension is .c/.cpp. Make sure to have the extension of `.cu`. See [the docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#supported-input-file-suffixes)

### Useful flags
* `nvcc --run file.cu`: Run it
* `-g`: Adds debug info

### Debugging

Call `cuda-gdb a.out`. Then its just like the normal gdb.


## Syntax

When we call a CUDA function, we do it with,

```
<<<# of blocks, # of threads>>>f( args )
```



