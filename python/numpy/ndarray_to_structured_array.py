#!/usr/bin/env python3
import numpy as np

x = np.random.random((2, 3))
# Ravel is really important, but also really simple.
# It returns a contiguous flattened array
# Which we can then view as a structured array!
x_struc = x.ravel().view([("x", np.float64), ("y", np.float64), ("z", np.float64)])

# Going back is really easy
x_again = x_struc.ravel().view(np.float64).reshape((2, 3))

print(x)
print(x_struc)
print(x_again)
