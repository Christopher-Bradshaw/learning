# Compiling

## High level
Compiling has 2 stages.
1. Compile cython into C or C++
2. Convert the C(++) into a shared library that can be imported to Python.

## Pyximport
At the start of the code run:
```
import pyximport
pyximport.install()
```
Then you can import cython is if it were python! In the background it will be compiled (and cached!). This is fine for local code, but as it depends on both having cython and a C compiler (and has the compilation overhead) it isn't advisable for e.g. a library you users. But for your personal science code it is probably easiest!

Note the downsides (unable to reload) when using in jupyter notebooks. So, this is not great for dev but once the code is good...

## Cythonize
Compile cython to C: `cythonize file.pyx` -> `file.c`

But that isn't actually useful.

Compile cython to shared object file: `cythonize -b file.pyx` -> `file.cpython-<version>-<arch>.so`


You can also use globs: `cythonize -b *.pyx`.

*So to get our two cython modules in this dir working just run `cythonize -b *.pyx` and then `python3 main.py` should work.


## Distutils
The simple way will break down when you have something more complicated (e.g. when you start needing to link things). Then you should use distutils. See the docs: https://docs.python.org/3/distutils/setupscript.html.

This is probably best learned by example but some useful things to know:

`ext_modules`: defines a list of extension modules and the C files that are needed to build them. Calling `build_ext` will build these C files into shared object files - these can then be imported by Python.   Ensure that you include the pacakge you want the shared object to live in: `Extension('pkg.foo', ['src/foo1.c', 'src/foo2.c'])`

The shorthand for this (if you just want to compile `file.c` to `file. ... .so`) is just the file name. By default (in the shorthand) if the cython lives in a package the shared object will stay in that package. So ensure you have `__init__.py` in the right places.

But we have cython code so we first need to `cythonize([ modules ])` and then use the output of that (the C files) as the args to `ext_modules`.

The `build_ext` command is used to build these extensions. `--inplace` does it inplace rather than (I am guessing) a build dir.

If you need header files you can include them with `include_dirs`, either just for the extensions that need them, or globally.


## Extra checks
By default the compiled C code includes some extra checks. E.g. that the object is not None, that we are within array bounds. The cython compiler can also convert negative indexes to positive ones. However if we don't want these features/safety checks we can disable them for performance. This can be done in many different ways:

* Decorator: `@cython.boundscheck(False)` (requires `cimport cython`)
* Context manager: `with cython.boundscheck(False):`
* Compiler directive: `# cython: nonecheck=False` at top of file
* Compiler arg: `cython --directive nonecheck=False`

## Looking at the C code
You probably don't want to. But, if you do, grep for the cython code you are looking for. The C code contains comments with the cython code. Realistically you should be doing this with `cythonize -a`.
