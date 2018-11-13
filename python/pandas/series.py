import pandas as pd
import numpy as np

# The basic data structure is the series
s = pd.Series([1,2,3,4,5,6])
# print(dir(s)) # Is very long - series can do many things

# Slicing works as expected
s0 = s[0]
ss = s[2:4]

print(s0, type(s0)) # type is np.int64
print(ss, type(ss)) # type is Series


# Underling the series is a np array
nps = s.values
assert type(nps) is np.ndarray
