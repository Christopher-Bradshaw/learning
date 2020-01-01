# LLVM Python funcs

See http://dev.stephendiehl.com/numpile/

The goal here is to automatically convert a Python function to an LLVM function. We can then directly call that LLVM func, dropping out of the python interpreter!

The rough layers that we will go through to do this are:

* Python code
* Python ast
* An intermediate representation (IR) which we will call the *core*, that:
    * Supports a subset of python (i.e. bails if you)
    * Infers some extra things (types)
* LLVM

Our job is to go from the AST to this *core* language, and then from it to LLVM.

## Python AST

## LLVM

See [basics](./basics) for a basic intro to LLVM.

## Prereqs

* llvm-devel (package manager)
* llvmlite (pypi)
