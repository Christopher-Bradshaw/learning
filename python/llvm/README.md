# LLVM from Python

This was motivated by wanting to know how [Numba](http://numba.pydata.org/) works. I closely followed this [excellent talk](http://dev.stephendiehl.com/numpile/). Though, I think that talk is a bit old and some of the libraries are deprecated so there are some differences.

The goal here is to automatically convert a Python function to an LLVM function. We can then directly call that LLVM func (with all optimizations from the LLVM compiler), dropping out of the python interpreter!

## Python AST

What the python AST looks like.

## LLVM basics

The basics of LLVM syntax and how to compile/run it/call it from C. A couple of hello world standard scripts.

## LLVM python

The python tools to build up/compile/run LLVM code.

## AST to LLVM (Poormans Numba)

See [toplevel.py](./ast_to_llvm/toplevel.py) for the `autojit` function. The outline of what this does:

* Python code
* Python AST
* A subset of the python AST that we call the *core* language.
* Works out the constraints on the types of the variables in the core language (e.g. if `c = a + b` the type of `c` must be the same as `a` and the same as `b`).
* Works out the actual types given the arguments to the function (e.g. if the function is called with floats, the local variables/return value types might be different to if it is called with ints).
* Maps this typed core language to LLVM

## Prereqs

* llvm-devel (package manager)
* llvmlite (pypi)
* Probably some other things I forgot...
