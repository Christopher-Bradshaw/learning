# LLVM

Remember that you never want to directly write LLVM. But, here are some of the basics so that I have a rough idea of how it works.

## Syntax Basics

* `;` starts a comment
* Variables are either local or global. Local vars are prefixed with `%`, global with `@`.

## Compiling/Running

* `llc example.ll` compiles to assembly (`example.s`) for your architecture (or a specified arch). Can specify optimization level with `-O=[0-3]` (default is 0).
* `lli example.ll` runs in an interpreter.
* `opt example.ll -o example.bc` converts to bitcode. Not entirely sure what this is...
* `opt -O3 example.ll -o example.opt.ll -S` output optimized LLVM.
