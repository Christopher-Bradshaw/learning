import numpy as np

# See algo 3.5 and 3.6 in Nocedal/Wright
# Given a function f, the gradient grad_f, the current location x0, the step direction p,
# and the maximum step size a_max.
# Return a step size that fulfills the strong wolfe conditions.
def get_line_length(f, grad_f, x0, p, a_max, c1=1e-4, c2=0.9):
    # try:
    #     assert np.isclose(np.linalg.norm(np.linalg.norm(p)), 1)
    # except AssertionError:
    #     raise Exception("This expects a normalized direction!")

    phi = lambda a: f(x0 + a * p)
    gphi = lambda a: np.dot(grad_f(x0 + a * p), p)

    v0 = phi(0)
    g0 = gphi(0)
    assert g0 < 0, "p needs to be in a descent direction"


    # True if fulfills strong wolfe 1 - sufficient decrease
    sw1 = lambda a: phi(a) <= v0 + c1 * a * g0
    # True if fulfills strong wolfe 2 - flat grad
    sw2 = lambda a: np.abs(gphi(a)) <= c2 * np.abs(g0)

    # Somewhere in this range is an acceptable value. Note that we don't know
    # whether a_lo < a_hi or vice versa.
    def zoom(a_lo, a_hi):
        while True:
            if abs(a_lo - a_hi) < 1e-9:
                raise Exception(f"Couldn't find a good line length, {a_lo}, {a_hi}, {a_max}")

            # We need to choose this between a_lo and a_hi. There might be cleverer ways
            # to do it, but bisecting is easy.
            a_j = (a_lo + a_hi) / 2

            if not sw1(a_j) or phi(a_j) >= phi(a_lo):
                a_hi = a_j
                continue

            # sw1 must be fulfilled if we are here
            if sw2(a_j):
                return a_j
            elif gphi(a_j) * (a_hi - a_lo) >= 0:
                a_hi = a_lo
                continue
            else:
                a_lo = a_j
                continue

    # I want to assert that we actually fulfills sw1 and sw2 before returning
    def ret_with_assert(a):
        assert sw1(a) and sw2(a), "Failure!"
        return a

    a_i = min(1, a_max / 2)
    a_prev = 0
    is_first_iteration = True
    while True:
        # Note that a_prev doesn't fulfill sw2 - too large negative gradient,
        # and fulfills sw1 (except on the first iteration).

        # If we don't fulfill sw1 now (and we must have the last time) we must have bent
        # up (or flattened) between the previous and now. There is a solution between
        # the previous and now
        # or
        # While still fulfilling sw1 we have had a positive slope recently. Again, there
        # must be a solution between the previous and now
        if (not sw1(a_i)) or (not is_first_iteration and phi(a_i) >= phi(a_prev)):
            return ret_with_assert(zoom(a_prev, a_i))

        # sw1 must be fulfilled if we are here

        if sw2(a_i):
            return ret_with_assert(a_i)
        # If greater than 0 and not sw2 then it must be too steep upward.
        # there is a minima between a_i (large +slope) and a_prev (large -slope)
        elif gphi(a_i) >= 0:
            return ret_with_assert(zoom(a_i, a_prev))
        # We have sw1 and not sw2 because too steep negative. Look at larger a
        else:
            a_prev = a_i
            a_i = (a_i + a_max)/2

        is_first_iteration = False
