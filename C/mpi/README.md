# The basics of MPI

See [Intro to parallel Programming chapter 3](https://www.elsevier.com/books/an-introduction-to-parallel-programming/pacheco/978-0-12-374260-5), and [my notes/examples on mpi4py](https://github.com/Christopher-Bradshaw/learning/tree/master/mpi).

## What is MPI

If you want the full details, the 868 page spec is here https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf.

The overview (section 1.1) is useful though. MPI is a **message-passing library interface specification**.
* Message passing: Describes how data is moved from the memory of one machine into that of another.
* Library interface: This isn't a language, it describes the interface for a library.
* Specification: This isn't an implementation. MPI is a spec for which there are implementation in some languages.


## Hello World (locally)

http://mpitutorial.com/tutorials/mpi-hello-world/, https://pleiades.ucsc.edu/hyades/Hyades_QuickStart_Guide

Unless you have a cluster ready and waiting (and even if you do) you can get started locally. First you need an implementation of MPI. On Fedora I can get this with:

```
dnf install openmpi openmpi-devel
```

We can now write a `mpi_hello.c` that uses some features of mpi. To build and run this with gcc we can:

```
gcc mpi_hello.c -I/usr/include/openmpi-x86_64/ -L/usr/lib64/openmpi/lib/ -lmpi
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/openmpi/lib ./a.out
```

Which is a bit of a faff (though useful to know for YCM because I think that needs to use clang). Instead we should use the compilers that come with `openmpi-devel`. Add `/usr/lib64/opemmpi/bin/` to your path and run:

```
$ mpicc mpi_hello.c
$ ./a.out
Note that even before MPI_Init we have multiple processes running!
Hello world from processor cb.ucsc.edu, rank 0 out of 1 processors
```

Much easier.

While we now have something running that "uses MPI" the output of hello world shows that we are only runnning on 1 processor. We can run multiple processes in parallel with `mpirun` or `mpiexec` (these are synomyns). Man pages are packaged in a non standard place so don't work out of the box. This is so that you don't have overlapping man pages if you install multiple MPI implementations, see [here](https://fedoraproject.org/wiki/Packaging:MPI#Packaging_of_MPI_compilers). After running `module load mpi/openmpi-x86_64` we have access to them. I have never used modules before but they sound pretty cool! Anyway, back to MPI and the quickstart looks like:

```
$ mpirun -n 4 a.out
Note that even before MPI_Init we have multiple processes running!
Note that even before MPI_Init we have multiple processes running!
Note that even before MPI_Init we have multiple processes running!
Note that even before MPI_Init we have multiple processes running!
Hello world from processor cb.ucsc.edu, rank 1 out of 4 processors
Hello world from processor cb.ucsc.edu, rank 2 out of 4 processors
Hello world from processor cb.ucsc.edu, rank 0 out of 4 processors
Hello world from processor cb.ucsc.edu, rank 3 out of 4 processors
```

An important difference (I think) between MPI and other methods of parallelization (Open-MP multi-threading, fork) - we start off with multiple processes and they persist throughout. In e.g. Open-MP you just parallelize the section you want.


## Sending and Receiving

Our hello world example spins up a number of processes but they don't communicate at all. Let's look at some message passing with a pretty trivial implementation in `mpi_send_and_receive.c`.

The two MPI functions we use here are:
```
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest  , int tag, MPI_Comm comm);
int MPI_Recv(void *buf      , int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
```

For a send and recieve to match, they must use the same communicator, use the same tags and the dest (src) must be the rank of the recieving (sending) node. For this to be successful their types must also match.

These functions are potentially **blocking** meaning that while a node is listening for a result, or waiting for a receiver to pick up it is doing nothing else. I say potentially - receive is always blocking. Sends may buffer if they are small.
This is a bit of a problem if we as a master node don't know which slave will be done first. To handle this there are the `MPI_ANY_SOURCE` and `MPI_ANY_TAG` macros, when you just results as they come. This can only be used on the receiver side (you must specify to whom you are sending and what the tag is).
But this causes another problem - if you use these wildcards you could receive data but not know where it came from or how it was tagged. Even though you didn't care who sent you data first, you probably need to know this info (e.g. in a simulation you don't care which region completes first but you still need to know which region reported). This is put in the `MPI_Status` struct.

This doesn't actually send the data over. However, we now have enough info to get it with a call to:

```
int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
```

I could get away without doing this because I am running this on a single machine and so the master process can access the memory of the slaves directly. This won't be possile in a distributed memory architecture.


Sends from N1 -> N2 are also ordered. If N1 sends something first, N2 will receive it first. This is not true for messages between different nodes.

## Hello World (cluster)

See [cluster setup](cluster_setup.md).

## Detail diving

In the example I glossed over quite a bit of details and boilerplate. Code for open-mpi is available at https://github.com/open-mpi/ompi (see [here](https://github.com/open-mpi/ompi/tree/390d72addd61f08986538bc54b6d609dafde2769/ompi/mpi/c) for the C implementation) and has comments that make it pretty easy to skim.

Note that some of these details are in the interface and some will be implementation dependent.

Naming conventions:
* All MPI identifiers start with `MPI_`
* All MPI functions and types start with `MPI_[A-Z]`
* All MPI constants and macros start with `MPI_` and are fully capitalized.

`int MPI_Init(int *argc, char*** argv)`: A summary of what it does can be found [here](http://cw.squyres.com/columns/2004-02-CW-MPI-Mechanic.pdf), but "generally its job is to create, initialize, and make available all aspects of the message passing layer." This might include allocating memory it needs, network communication channels (TCP connections/ports), and/or learning abour network topology. The details will be implementation dependent (e.g. [open-mpi](https://github.com/open-mpi/ompi/blob/390d72addd61f08986538bc54b6d609dafde2769/ompi/runtime/ompi_mpi_init.c)). Note that this doesn't open connections to all nodes. That is done lazily - as required.

`int MPI_Finalize()`: Close/free the things init opened/malloced.

`MPI_COMM_WORLD`: This is a pointer to a **global** communicator object. How is a `#define`d object a global variable? Implementation is `#define MPI_COMM_WORLD ((MPI_Comm)&(ompi_mpi_comm_world))` where `MPI_Comm` is a type that we are casting the pointer to. I don't really understand how this is created... And I think I might be wrong - I think each node might need its own copy.


`MPI_Comm_[rank,size]`: These two functions get info about the communicator (the number of processes that are communicating and their rank within it). I think that this doesn't require any communication - all of that has been setup by init and this just reads from the MPI_COMM_WORLD object to find this.

Output: We've seen that processes are able to print to stdout - even processes running on remote machines log to stdout on the machine that we launched the code from. There is no guarantee of ordering - if you want that have all nodes send their output to a single node which can coordinate it. Open question - what about writing to file?

Input: Only process 0 will read from stdin

Aliasing: MPI prohibits inputs and outputs refering to the same block of memory! This is because this is not allowed infortran and they wanted to keep things simple.


## Point to point vs Collective Communication

Pretty much everything so far has been point to point communication. Node 0 sends out some work to nodes 1 though n. When they are done with it they send it back to node 0. This isn't hugely efficient if there is any work in collating the results of the previous nodes. And even if there isn't you still have this O(n) of picking up the data (or sending it out). In many cases both of these could be parallelised.

MPI has functions that do this. They are things like scatter (take an array and break it into pieces and send piece n to node n), reduce (which receives data from all the nodes and reduces it in some way (e.g. finds the sum or the max)). These are both more convenient (one line of code vs a couple doing the manual send and receive) and almost certainly (depending on the implementation) more performant. For example on a reduce sum the implementation probably trees down, resulting in O(log(n)) time compared to O(n).

Some other collective communication funcs: Bcast, which broadcasts the data to all nodes. Gather, the opposite of scatter which receives pieces (e.g. of an array) and creates a single full array. Once we have done something like gather or reduce we sometimes just need it on one node (e.g. if we just want to save the result) but sometime want it on all of them (e.g. if it is an input to the next function). These functions tend to have an AllXXX compatriot which does a similar thing but leaves the result accesible to all nodes.

**N.B. ALL the processes in a communicator need to call the collective communication func**. You can't have special case node n off doing its own thing. If you do, you probably want to create a new communicator.
