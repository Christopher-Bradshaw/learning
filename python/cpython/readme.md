# Cpython

Notes on the reference implementation of Python

## Main Resources Used
* Philip Guo lectures: http://pgbovine.net/cpython-internals.htm
* Allison Kaptur talk: https://www.youtube.com/watch?v=HVUTjQzESeo
* PythonTutor (also by Philip Guo): http://www.pythontutor.com/visualize.html#


## Where things are

* Main interpreter loop: Python/ceval.c:916. Helpfully labelled `main_loop`.
* Include/Objects: Include has many .h files. Are the include files for the .c files in Objects. Though not a 1 to 1 mapping.
* Lib: standard library (part written in python)
