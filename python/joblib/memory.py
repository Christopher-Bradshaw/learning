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

print("Calling the function the first time is slow")
print(slow_function(x))
print("\n")
print("But running it a second time is fast because it is cached")
print(slow_function(x))

# To clear the cache run this, or else just delete the stuff in the cache dir
# memory.clear(warn=False)
