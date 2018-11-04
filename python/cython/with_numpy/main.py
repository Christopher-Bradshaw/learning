import numpy as np
import prime_check
import time
import math
""" What we learned:
    Cython can be 30x faster than python.
    This is a bad test within cython becaues most of the time is spent in the common _is_prime
    function. However we can still see that converting to a memoryview early is good.
    Pretty crazy you can find all primes under 1M in less than a second...
"""

def naive_python(inp):
    is_prime = np.zeros(len(inp), dtype=np.bool_)
    for i, v in enumerate(inp):
        is_prime[i] = _is_prime(v)
    return is_prime

def _is_prime(v):
    max_div = int(math.ceil(math.sqrt(v+1)))
    for i in range(2, max_div):
        if v % i == 0:
            return False
    return True


def call_and_time(f, summary=None):
    inp = np.arange(1000000)
    start = time.time()
    x = f(inp)
    if summary: print(summary)
    print("seconds taken", time.time() - start)
    print(np.count_nonzero(x))

call_and_time(naive_python, "naive python")
call_and_time(prime_check.is_prime_numpy, "numpy")
call_and_time(prime_check.is_prime_half_half, "half-half")
call_and_time(prime_check.is_prime_memoryview, "memoryview")
"""
naive python
seconds taken 20.1
numpy
seconds taken 0.68
half-half
seconds taken 0.67
memoryview
seconds taken 0.62
"""
