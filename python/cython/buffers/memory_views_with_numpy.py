import numpy as np

np_2d = np.arange(16, dtype=np.int64).reshape(4, 4)

mv = memoryview(np_2d)
print([i for i in dir(mv) if not i.startswith("_")])

assert mv.c_contiguous and not mv.f_contiguous # np is (I think) by default C contig
# To take a step in the second dimension you move 8 bytes (1 item)
# To take a step in the first dimension you move 32 bytes (4 items)
assert mv.strides == (32, 8)
assert mv.format == "l" and mv.itemsize == 8 # The items are longs
assert mv.shape == (4,4)

print(mv.tolist()) # What you would expect
# [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

assert mv.readonly == False # np arrays are not readonly

# Note that you have to access like this - mv[0][0] fails. This kinda makes sense.
# You can't access a subview - you have to get to the element in one go
# Even mv[0] fails
mv[0,0] = 16
mv[0,1] = 17
# Of course numpy can be addressed with [0,0] or [0][0].
# And modifying the memoryview has modified the numpy array.
# This makes sense - it is just a view.
assert np_2d[0,0] == 16 and np_2d[0][1] == 17


# The memory view flags are not a view on the data's flags (I can't imagine how that
# would work) so if things change they aren't propagated to memoryviews.
# New memoryviews work as expected though.
np_2d.setflags(write=False)
assert mv.readonly == False and memoryview(np_2d).readonly == True
