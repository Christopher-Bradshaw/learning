from mpi4py import MPI
from helpers import nprint
from math import pi

from engines.integrate import integrate

MASTER_RANK = 0

def get_partial_area(g_start, g_stop, g_num_pts):
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    delta = g_stop - g_start

    # And these are the local variables - different for each process
    l_start = g_start + rank * delta / size
    l_stop = l_start + delta / size
    l_num_pts = g_num_pts / size
    return integrate(l_start, l_stop, l_num_pts, "sin")

# g_* to indicate that these are the global start/stop/pts
# global in the sense that they are the same for each process
def main(g_start, g_stop, g_num_pts):
    partial_area = get_partial_area(g_start, g_stop, g_num_pts)

    # reduce is faster (and easier to write!) than having everything send back to
    # the master node and then have it manually add things up.
    # This isn't a huge deal here where the reduce is simple, but imagine a complex,
    # costly reduce (large matrix sum)
    area = MPI.COMM_WORLD.reduce(partial_area, op=MPI.SUM, root = MASTER_RANK)
    if MPI.COMM_WORLD.Get_rank() == MASTER_RANK:
        nprint(area)
    else:
        assert area is None


if __name__ == "__main__":
    main(0, 2*pi, 10000000)

