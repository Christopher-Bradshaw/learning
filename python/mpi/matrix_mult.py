from mpi4py import MPI
from helpers import nprint

import numpy as np

MASTER_RANK = 0

def vector_mult(matrix, vector):
    # The # of rows in the matrix is the number of output vector components
    # This slave is computing
    ret = np.empty(len(matrix), dtype=np.float64)
    for i in range(len(matrix)):
        ret[i] = np.dot(matrix[i], vector)
    return ret

def main(test = True):
    # The way that we are going to scatter things will depend on the number of processes.
    # We also need to be able to evenly divide things.
    dims = 8000
    num_procs = MPI.COMM_WORLD.Get_size()
    assert dims // num_procs == dims / num_procs

    # Memory assigned for the pieces of the matrix each node gets and the full vector
    l_matrix = np.empty((dims//num_procs, dims), dtype=np.float64)
    l_vector = np.empty(dims, dtype=np.float64)


    if MPI.COMM_WORLD.Get_rank() == MASTER_RANK:
        # The large matrix is only assigned on node 0
        matrix = np.random.random(dims**2).reshape(dims, dims)
        l_vector = np.random.random(dims)

        MPI.COMM_WORLD.Scatter(matrix, l_matrix, root = MASTER_RANK)

        # Each node needs to know about the vector, but because we assign at random
        # We can't generate it on each one!
        MPI.COMM_WORLD.Bcast(l_vector, root = MASTER_RANK)
    else:
        MPI.COMM_WORLD.Scatter(None, l_matrix, root = MASTER_RANK)
        MPI.COMM_WORLD.Bcast(l_vector, root = MASTER_RANK)

    # We've gone from a (Y, Y) x (Y, 1) to
    # (l, Y) x (Y, 1) where l < Y. Now we can mult in parallel
    MPI.COMM_WORLD.Barrier()
    start = MPI.Wtime()
    res = vector_mult(l_matrix, l_vector)
    MPI.COMM_WORLD.Barrier()
    # Barriers sync everything up so we see how long we are in the actual mult
    if MPI.COMM_WORLD.Get_rank() == MASTER_RANK:
        print("mpi time: ", MPI.Wtime() - start)

    # Now we need to gather the pieces of the final vector together
    if MPI.COMM_WORLD.Get_rank() == MASTER_RANK:
        result = np.empty_like(l_vector)
        MPI.COMM_WORLD.Gather(res, result, root = MASTER_RANK)
        # nprint(result)
        if test:
            test_start = MPI.Wtime()
            test_result = np.matmul(matrix, l_vector)
            print("test time:", MPI.Wtime() - test_start)
            # nprint(n_result)
            assert np.allclose(result, test_result)
    else:
        MPI.COMM_WORLD.Gather(res, None, root = MASTER_RANK)

if __name__ == "__main__":
    main()
