# This has the effect of adding a `#include "c_funcs.h" in the generated source
cdef extern from "c_funcs.h":
    # Here we duplicate any definitions that we want to use here

    int doubler(int var_name) # We include the var_name so we can call it with kwargs but this isn't mandatory
    double speed_of_light


    ctypedef struct point:
        float x
        float y

    float distance_between(point*, point*)

# Note that the extern block **doesn't** wrap the functions for us (yet?). We need to manually do that.
# But it exists to make sure that we are using them correctly.
# Once we have defined them in the extern block, we can use them as if we cdef'ed them here.

# We might also see syntax like
# cdef extern from *:
# I'm not quite sure what this means/does


def py_doubler(int x):
    return doubler(x)


def get_speed_of_light():
    return speed_of_light

def py_get_distance_between(x1, y1, x2, y2):
    cdef point p1, p2
    p1 = point(x=x1, y=y1)
    p2 = point(x2, y2)

    return distance_between(&p1, &p2)
