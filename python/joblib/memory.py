import os
import time

import numpy as np
from joblib import Memory

memory = Memory((os.path.dirname(__file__) or ".") + "/joblib_cache", verbose=2)

@memory.cache()
def slow_function(arr):
    time.sleep(3)
    return np.sum(arr)

x = np.random.random(10)

# This will be slow the first time, fast the second!
print(slow_function(x))
print(slow_function(x))
