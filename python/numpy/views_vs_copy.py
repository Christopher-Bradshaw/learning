#!/usr/bin/env python3
import numpy as np
"""
When slicing arrays in numpy sometimes you get a view (changes will be reflected in the
array you sliced from) and sometimes a copy (changes will be reflected ...).

You only get a view in 2 cases (that I know of so far:
    1) start:stop:step slice (normal list slice). E.g. x[4:15:3]
    2) Structured/record array column slice. E.g. x["col1"]
        (Note that at the moment if you pass a list of cols you get a copy but the
        next version will give you a view)
"""

def basic_slicing():
    x = np.arange(10)
    y = x[1:6:2]
    print("start:stop:step slicing results in a view? {}".format(id(y.base) == id(x)))

def true_false_slicing():
    x = np.arange(10)
    indexes = x > 4
    y = x[indexes]

    print("true/false slicing results in a view? {}".format(id(y.base) == id(x)))

def integer_slicing():
    x = np.arange(3, 15)
    indexes = [1,4,7]
    y = x[indexes]
    print("integer slicing results in a view? {}".format(id(y.base) == id(x)))

    indexes = [1,2,3]
    y = x[indexes]
    print("integer slicing (with consecutive integers) results in a view? {}".format(id(y.base) == id(x)))

    indexes = [1]
    y = x[indexes]
    print("integer slicing (with a single integer) results in a view? {}".format(id(y.base) == id(x)))

def col_slicing_structured_array():
    x = np.zeros(10, dtype=[("x", np.int), ("y", np.int)])
    # Just the fact that we can do this shows it is a view :)
    x["x"], x["y"] = np.arange(5, 15), np.arange(2, 12)

    y = x["x"]
    print("structured array col slicing (for a single col) results in a view? {}".format(id(y.base) == id(x)))
    y = x[["x", "y"]]
    print("structured array col slicing (for many cols) results in a view? {}".format(id(y.base) == id(x)))
    print("This is changing soon! And you get a FutureWarning if you write to it")


def stop_writing_to_view():
    print("Do you worry that you will accidentally mess up and edit your main data (when you just want to edit a subsample?)")
    print("Set flags to stop it")
    x = np.arange(10)
    x[4] = 1000
    x.setflags(write=False)
    try:
        x[3] = 1000
    except ValueError:
        print("After setting write=False this threw!")
    finally:
        print("Did it throw?")


if __name__ == "__main__":
    basic_slicing()
    true_false_slicing()
    integer_slicing()
    col_slicing_structured_array()

    stop_writing_to_view()
