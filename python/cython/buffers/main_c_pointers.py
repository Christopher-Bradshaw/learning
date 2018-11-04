import pyximport
pyximport.install()

import numpy as np

import c_pointers

def do_busy_work():
    pass

# We are sick of numpy's arange having 1 r so we make a new one!
assert np.all(c_pointers.arrange(3) == np.arange(3))

# And it appears to all work fine. However there is a problem - the np array thinks it has
# a view of the data and so it won't clean it up when it is deleted (as it is now because
# we didn't assign it to anything).
# We have a memory leak.

# Let us try assign on the stack. We can create an arange function for every int. Templating or something!
x = c_pointers.arrange10()
print(x) # looks all right!

# But notice that the numpy memory is not the same as the C memory. This appears to have made a copy?
print(x.__array_interface__)



# Working with C arrays is not trivial. Here is an example that works
our_arranged = c_pointers.arrange_safe(3)
print(our_arranged.__array_interface__)
assert np.all(our_arranged == np.arange(3))

print("About to free, look out for freeing")
del our_arranged
print("Did you see it?")
# We should


# What about if there are multiple views?

print("Now with multiple views")
initial = c_pointers.arrange_safe(5)
vw = initial[:2]

del initial
print("About to free the last thing, look out for freeing")
del vw
print("Did you see it?")
