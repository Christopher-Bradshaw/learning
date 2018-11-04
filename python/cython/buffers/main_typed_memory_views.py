import pyximport
pyximport.install()

import numpy as np
import time

import typed_memory_views as tmv

# Using a summation function that wants bytes
some_bytes = bytearray("abcdefg", "utf8")

s = tmv.summer(some_bytes)
# Pretty nice, this is 700. ord("d") == 100. Useful to remember!
assert s == len(some_bytes) * ord("d")
print(s)

np_rands = np.random.random(size=int(1e8)).reshape((int(1e4), int(1e4)))

try:
    tmv.summer(np_rands)
    raise Exception("We should never get here")
except ValueError as e:
    print("This ValueError'ed because: {}".format(e))
try:
    tmv.summer(np_rands.flatten())
    raise Exception("We should never get here")
except ValueError as e:
    print("This ValueError'ed because: {}".format(e))



# Using a mean finding function that wants a double, in C contig order
t_start = time.time()
mean_cython = tmv.mean(np_rands)
t_cython = time.time() - t_start

t_start = time.time()
mean_numpy = np.mean(np_rands)
t_numpy = time.time() - t_start

assert np.isclose(mean_numpy, mean_cython)
print(t_cython, t_numpy) # We are around 2x slower than numpy. Not bad


try:
    tmv.mean(np.asfortranarray(np_rands))
    raise Exception("We should never get here")
except ValueError as e:
    print("This ValueError'ed because: {}".format(e))



# What if we want a generic function that supports all numeric types? Fused types are the answer
np_double = np.random.random(size=int(1e6)).astype(np.float64)
np_int = np.arange(1e6).astype(np.int32)

print(tmv.generic_mean(np_double))
print(tmv.generic_mean(np_int))



# Memory view slicing
rands = np.random.random(size=int(1e6)).reshape((int(1e3), int(1e3)))
print(tmv.minimum_with_slicing(rands))


# Typed, structured memory views
x = np.zeros(10, dtype=[("a", np.int32), ("b", np.float32)])
x["a"] = np.arange(10)
x["b"] = np.random.random(10)

x_new = tmv.doubler_structured_memory_views(x)
# Because everything was done in place the data in x == x_new
assert np.all(x_new == x)

x2 = tmv.doubler_structured_memory_views(np.copy(x))
assert np.all(x2["b"] == x["b"]*2) and np.all(x2["b"] == x["b"]*2)
