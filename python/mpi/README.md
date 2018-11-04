# Python MPI

I will look through the specifics of the python implementation here. For general MPI stuff, see https://github.com/Christopher-Bradshaw/C_learning/tree/master/mpi.

[Full API reference](http://mpi4py.scipy.org/docs/apiref/index.html)

[Tutorials](http://mpi4py.readthedocs.io/en/stable/tutorial.html)


## Library

mpi4py is the python lib that you want.


## Hello world

See `hello_world.py`. Run with `mpirun -n 2 python3 hello_world.py`. It's pretty cool that it is that easy.


## Pi Computation

A slightly larger program. Shows the interesting behaviour with broadcast (I assumed that you would broadcast/recv). Also shows MPI working with Cython. Nothing too exciting but a potential gotcha is that `pyximport` doesn't appear to work. I guess because all n processes are trying to create it at the same time and so there are race conditions.

Run with:
```
python3 setup.py build_ext --inplace
mpirun -n 15 python3 pi_computation.py
```
