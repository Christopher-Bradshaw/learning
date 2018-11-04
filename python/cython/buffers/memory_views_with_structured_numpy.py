import numpy as np

x = np.zeros(10, dtype=[("a", np.int32), ("b", np.float64)])
x["a"] = np.arange(10)
x["b"] = np.geomspace(1, 10, num=10)

mv = memoryview(x)
print([i for i in dir(mv) if not i.startswith("_")])

assert mv.format == "T{i:a:=d:b:}" # int a, double b
assert mv.shape == (10,) # just a length 10 array
assert mv.strides == (12,) # Item n+1 is 4+8 bytes away from item n
assert mv.itemsize == 12 # As above
