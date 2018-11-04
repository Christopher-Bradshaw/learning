import numpy as np

x = np.arange(5)
assert x.shape == (5, )
assert np.expand_dims(x, 0).shape == (1, 5)
assert np.expand_dims(x, 1).shape == (5, 1)
assert np.expand_dims(x, -1).shape == (5, 1)

# I find vstack/hstack/dstack very hard to understand...
# On the other hand, concatenate is very easy to understand.
# So I'm going to use that.
# All you need to know - shapes must be the same except along the axis we are concatenating!

assert np.concatenate((np.ones(5), np.ones(5))).shape == (10, )
assert np.concatenate((np.ones((2, 5)), np.ones((1, 5)))).shape == (3, 5)
assert np.concatenate((np.ones((5, 2)), np.ones((5, 1))), axis=1).shape == (5, 3)
