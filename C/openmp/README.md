# The basics of OpenMP

See [Intro to parallel Programming chapter 5](https://www.elsevier.com/books/an-introduction-to-parallel-programming/pacheco/978-0-12-374260-5). I've also played with the cython implementation of OpenMP [here](https://github.com/Christopher-Bradshaw/python_learning/tree/master/cython/parallelization).

The [8 page spec summary](https://www.openmp.org/wp-content/uploads/OpenMP3.0-SummarySpec.pdf) is also pretty useful.


## What is OpenMP

It is a way to do shared memory multiprocessing. Contrast this with MPI which is for distributed memory multiprocessing. OpenMP is a relatively simple, and potentially less powerful that other methods, way to parallelize code using compiler directives.

Despite using compiler directives, the choice of the number of threads to use can be made at run time.

## Variable Scope

Something to be careful with is variable scope. Remember that in C variables are lexically or block scoped. Contrast this to e.g. python where variables are scoped to the function (not block) where they are defined.

When writing parallel code with openMP, we need to be careful whether our variables are shared across threads, or unique to each thread. Anything that is declared when we are in single threaded mode will be shared. See the comment about `partial_area` in `integrate.c` - originally I was defining that outside the parallel block and was getting race conditions in its assignment.

The easy way to ensure that everything is correctly scope is to have the section openMP runs on be as small as possible. For example, rather than write integrate as I did with the `{ <block> }` for openMP, just have a single `integrate_parallel` function that handles this. With hindsight I'm not sure wy I didn't do this...

Note that we have the same pattern as we saw in MPI - do some work in parallel, reduce in series. This is fine here as the reduction is trivial but it makes the code a bit more complicated and bug prone. OpenMP, being pretty cool, gives us a way to do reductions better. See the second integration in `integrate.c`.

## Ways of parallelizing:

### Pragmas

* Blocks/function: `# pragma omp parallel` followed by a block/function
* For loops: `# pragma omp for` followed by `for ... { <code> }`. Note there are some caveats of the types of for loops that can be parallelized.

### Clauses

These are optional additional info for pragamas.
* `num_threads(int)`: How many threads to use
* `reduction(operator: var`: Takes an externally defined variable and creates private versions, initialized to the identity value (e.g. 1 for *, 0 for +) of it for each thread. Once threads have completed, reduces those vars using the operator onto the shared variable.
* `private(var)`: Takes a shared variable and specifies that the compiler should create private copies of it.
* `default(none | shared)`: If not present, it is assumed that all variables that are not declared in the parallel section are shared (each thread associates that varible with the same memory location). If it is set as none, any shared variables need to be explicitly declared with a `shared(var1, var2)` clause.
* `shared(var)`: See default.
* `schedule`: In a parallel for loop, describes how work is assigned. See [the docs](https://msdn.microsoft.com/en-us/library/b5b5b6eb.aspx) for details.

## Enforcing mutual exclusion

There are 3 ways: The critical pragma, the atomic pragma, locks.

The Intro to PP book suggests we prefer atomic to critical to locks if all are possible.

There are also named critical sections - so that different threads can each be in one of the critical sections but not both in the same one.


## Cache Issues

Remembering CS323 - there are different levels of memory. E.g. on my system (with [access times](http://norvig.com/21-days.html#answers) and main memory added).
```
$ lscpu
L1d cache:           32K            0.5ns
L1i cache:           64K
L2 cache:            512K           7ns
L3 cache:            8192K
Main memory:         32G            100ns
CPU max MHz:         3400.0000      0.3ns per cycle
```

The summary of this is that many cache misses will make your code run slowly. You won't be making full use of your CPU.

This is important with serial code but there are many more new and fun ways to ruin your performance for cache reasons when you are multithreaded!
