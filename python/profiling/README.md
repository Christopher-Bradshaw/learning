# Profiling

See [the cython section](../cython/profiling/README.md) for some more information about profiling (and profiling cython code).

## Tools
* [Snakeviz](https://jiffyclub.github.io/snakeviz/): Graphical viewer of `.prof` files. This is really cool...
* Docs for [the profilers](https://docs.python.org/3/library/profile.html)


## Output

However you are profiling, the output consists of lines that look something like this:

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.637    0.637 {built-in method builtins.exec}
        1    0.000    0.000    0.637    0.637 code_to_profile.py:1(<module>)
2692507/29    0.636    0.000    0.636    0.022 code_to_profile.py:1(fibonacci)
       29    0.000    0.000    0.000    0.000 {built-in method builtins.print}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```

These columns tell you:
* ncalls: How many times this function was called.
* tottime: The total time spent in this function (not including time spent in subfunctions)
* percall: tottime / ncalls
* cumtime: The total time spent in this function (including subfunctions)
* percall: cumtime / ncalls
* filename: The filename, line number and function name that the previous stats were referring to

I've sorted this output by cumtime, so we see that all time is spent in the `code_to_profile.py` module (that is the script I am calling). We then made *many* calls to a `fibonacci` function. Each of these was very fast (tottime-percall is 0) but because we made many calls, it dominates the tottime.

## Scripts

To generate the example output above, I ran:

```
 python3 -m cProfile -s cumtime code_to_profile.py
```

Other options to sort by include `ncalls` and `tottime`. `cumtime` I think is useful for getting a nice hierarchy of where time is spent. `tottime` probably drills closer to where exactly time is spent.

To output to a file, run

```
python3 -m cProfile -s tottime -o code_to_profile.prof code_to_profile.py
```

This can then be loaded and analysed in a script using [pstats](https://docs.python.org/3/library/profile.html#module-pstats). You can also do this directly in the pstats shell,

```
python3 -m pstats prof/test_fibonacci_works.prof
```

Once in the shell, type `help` and maybe read through [this useful blog post](https://www.stefaanlippens.net/python_profiling_with_pstats_interactive_mode/) to see what you can do.

The profile can also be visualized (though this example is pretty boring) with

```
snakeviz code_to_profile.prof
```

## Jupyter

There is a built-in profiler magic for jupyter, [prun](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-prun). Use

```
%% prun # at the start of the cell to profile the whole cell
% prun <code> # line magic
```

This will pop up the ouput table at the bottom of the browser tab.

Or, you can be fancy with,

```
%load_ext snakeviz
%%snakeviz # at the start of the cell to profile the whole cell
% snakeviz <code> # line magic
```

## Tests

Often, the first time you find out that your code is slow is when you unit test the function (you are unit testing, right? right??). It would be great if we could profile the test directly, rather than having to pull the code out into a script/notebook.

You can, with [pytest-profiling](https://pypi.org/project/pytest-profiling/). Install this, then:

```
pytest --profile
```

You can combine this with the usual pytest options, e.g.

```
pytest -k test_fibonacci_works --profile
```

This outputs the usual table, but I don't find it that useful for two reasons:
1. There is a ton of pytest gunk that gets in the way
2. I can't find a way to sort by anything other than the default, `cumtime`.

Fortunately, this also outputs a `.prof` file in `prof/<test_name>.prof`. So use the same tricks (pstats, snakeviz) that we talked about earlier.
