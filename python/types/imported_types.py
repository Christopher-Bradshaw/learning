# See https://www.python.org/dev/peps/pep-0561/
import numpy as np


def np_sum(a: np.ndarray) -> complex:
    return np.sum(a)

print(np_sum(np.arange(10)))
print(np_sum([1, 2]))
try:
    print(np_sum("cat"))
except TypeError as e:
    print(e)

# mypy is fine with all of this! The reason is because it doesn't know about imported types
# see https://github.com/python/mypy/issues/5480

# Let's think about this...
# mypy is a static analysis tool - so it does nothing at runtime.
# mypy has no way of knowing what `np.ndarray` is. (does it not?)
# so all it can reasonably do is interpret it as any?
