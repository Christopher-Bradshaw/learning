# https://docs.python.org/3/library/stdtypes.html#memoryview
some_bytes = b"Just some bytes, sitting around doing byte things..."

mv = memoryview(some_bytes)

# Memory views don't have that many methods/properties.
print([i for i in dir(mv) if not i.startswith("_")])

assert mv.readonly # Byte arrays are immutable so the view of it must also be
assert mv.shape == (len(some_bytes),)
assert mv.obj is some_bytes # This is the base obj on which the memview was taken
assert mv.itemsize == 1 # It is a byte array. Each item is 1 byte

# We can take another view of this memoryview
mv1 = mv[2:22:2]
assert mv1.obj is some_bytes # Still based off the original obj
assert mv1.itemsize == 1 # Still looking at bytes
assert mv1.strides == (2,) # We only take every second object

# Let's create a new memoryview from this one.
# We can take a slightly different view of the data.
mv2 = mv.cast("i")

assert mv2.itemsize == 4 # Now it is made of ints
assert len(mv2) == len(mv)/4 # but is much shorter (in terms of number of items)
assert mv2.nbytes == mv.nbytes # but the same length (in terms of number of bytes)
assert mv2.strides == (4,) # We don't skip anything so this is just itemsize


# What if we delete the original reference to the underlying obj?
del some_bytes

# Taking the memoryview added a reference to it so it is still here
print(mv.obj)
print(mv2.obj)
