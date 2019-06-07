# Profiling

See [../cython/profiling/README.md](the cython section) for some more information about profiling (and profiling cython code).

## Tools
* Snakeviz: Graphical viewer of `.prof` files


## Scripts


## Jupyter


## Tests

Often, the first time you find out that your code is slow is when you unit test the function (before incorporating it into a notebook/script). It would be great if we could profile the test directly, rather than having to pull the code out into a script/notebook.

Let's say that the slow test is `test_that_is_slow`. Run `pytest -k test_that_is_slow --profile`. This will create `prof/test_that_is_slow.prof`. Snakeviz that file `snakeviz prof/test_that_is_slow.prof`.
