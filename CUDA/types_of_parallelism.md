# Parallelism

## Flynn's Taxonomy

* SISD: Single instruction, single data stream. This is a basic, serial computer.
* MISD: Multiple instruction, single data stream. Apply multiple operations to a single data stream. Not practical.
* SIMD: Single instruction, multiple data streams. Apply the same operation to multiple pieces of data at the same time. For example, computing a the vector sum in parallel.
* MIMD: Multiple instruction, multiple data streams. Multiple processors executing different instructions of different data. For example, compute a ...


## Types of parallelism

These two types of parallelism don't map directly onto MIMD/SIMD. Those are slightly lower level ways of looking at it.

### Data Parallelism

Data parallelism distributes the data over different nodes which each perform the same computation on their local data. If necessary, the intermediate results from each node are combined.

For example, computing the sum of an array can be parallelized by sending items [0, n) to processor 1, [n, 2n) to processor 2, ... The results of each computation can then be summed, reducing execution time by a factor of the number of processors (assuming no extra overhead, memory limitations, etc).
Mergesort is easily parallelized in a similar fashion. Split the array into n subarrays where n is the number of processors. Have each sort their sublist. Then merge the results.

A vector sum is another example of data parallelism. Using `openMP`, elements are sent do different CPU threads. Using `CUDA` they will be sent to multiple GPU threads.

### Task parallelism

Task parallelism splits the work into multiple, independent tasks that can be executed in parallel.

For example, in an N-body simulation, we want to compute the local forces by looking at all the pairwise interactions but also want to compute the large scale forces using a tree/fft method. These two tasks (local, global) can be computed independently of each other and the results summed at the end.

CUDA streams take advantage of task parallelism. You can fire off multiple, independent jobs in a stream. These jobs should all be independent of each other. You can synchronize at the end when they are all complete.

### Summary of differences

From the [wiki](https://en.wikipedia.org/wiki/Data_parallelism#Data_parallelism_vs._task_parallelism),

* In data parallelism, the same operations are performed by each node on their data. In task parallelism, different operations are performed on the same or different data.
* In data parallelism, the amount of parallelization possible is proportional to the amount of data. In task parallelism it is proportional to the number of tasks.
