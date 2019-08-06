# Pair counter

A common problem in astronomy - compute the probability that two points will be separated by some distance.
There are good solutions to this (e.g. [Corrfunc](https://github.com/manodeep/Corrfunc), [Halotools](https://github.com/astropy/halotools)) so this is mostly just an exercise to learn the algos + practice some rust. The ambitious goal would be to match the performance of those tools in some specific case.

Currently I have a naive algo. This is necessary to make testing more complicated algos easy!

## Todo

Decide what faster thing to implement. I'm guessing something like a [Quadtree](https://en.wikipedia.org/wiki/Quadtree) or [k-d tree](https://en.wikipedia.org/wiki/Quadtree).

Though maybe the easiest is to just divide the space up into boxes the size of the bin. You then only need to look in the same box and the 8 neighbors.

## Profiling

Where the binary is the performance test,
```
cargo profiler callgrind --bin target/debug/deps/performance-8b1fad358b4c5d86 -- --test simple_grid_perf
```

This is great.
```
flamegraph target/debug/deps/performance-8b1fad358b4c5d86 --test simple_grid_perf
```
