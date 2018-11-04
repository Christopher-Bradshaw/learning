import time
import numpy as np
"""What we learned
A decorator is just a function that takes the function it is decorating as an argument
and returns another function. That returned function is the thing that is called.

Use cases:
    A `ensure_logged_in` decorator that ensures the user is logged in before executing whatever
    is in the function (returning the home page)
    A `timeit` decorator that prints the time a func takes while performance testing
    A `memoize` decorator (e.g. https://github.com/cython/cython/blob/084a25f55d0b4cf8b4c3cd496ec57bb3e2f57f71/runtests.py#L406-L414)
"""

# A function that takes a function, and then returns a function that uses the passed
# function by modified in some way
def manual_decorator(some_fn):

    def modified_some_fn():
        print("blah")
        some_fn()

    return modified_some_fn


def fn():
    print("hi")


decorated_fn = manual_decorator(fn)
decorated_fn()

# If we always want fn decorated we can
fn = manual_decorator(fn)
fn()


# Or we could do it with the decorator syntax
# Decorator takes the function as an argument and should return a function
@manual_decorator
def fn1():
    print("hi1")

fn1()

# A more practical use case
def timeit(some_fn):
    def modified_some_fn(*args):
        start = time.time()
        ret = some_fn(*args)
        print("Func took {}s".format(time.time() - start))
        return ret
    return modified_some_fn

@timeit
def double(x):
    return x*2
doubled = double(np.arange(10000))
print(doubled)
