# Absolute basics of lisp

## Resources

http://www.gigamonkeys.com/book/

## Execution

Run scripts with `sbcl --script FILENAME`

## Installing libraries

I have installed `quicklisp`. Now, to install software, run in a REPL,

```
(ql:quickload "cl-ppcre")
;(ql:quickload :cl-ppcre) Think this is the same
```

Then import with,

```
(require "cl-ppcre")
```

Access elements with,

```
(cl-ppcre:regex-replace ...)
```
