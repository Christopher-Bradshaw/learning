# C

Notes to help me remember how to write C.

## C variable special cases
I don't even know what to call this...

* `extern`:
* `static`: When applied to a local (automatic) variable in a function, means that it persists through multiple function calls. When applied to a function, means that that function should not be visible outside of the compilation unit. The first (variable) usage sounds tricky and possibly to be avoided. The second (function) usage sounds good - encapsulation, minimize namespace noise.

## Summary

* `unmatched.c`: Simple program to find unmatched braces in stdin. First thing I wrote to get me back into C.
