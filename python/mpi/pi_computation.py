from mpi4py import MPI
from helpers import nprint

import numpy as np

from engines.pi_computation_engine import estimate_pi

MASTER_RANK = 0

def master():
    # size = MPI.COMM_WORLD.Get_size()
    # Root is the rank of the sending process
    MPI.COMM_WORLD.bcast(1000000, root=MASTER_RANK)

    pi_estimates = []
    for i in range(MPI.COMM_WORLD.Get_size()):
        if i == MASTER_RANK:
            continue
        pi_estimates.append(MPI.COMM_WORLD.recv(source=i))

    nprint(np.mean(pi_estimates))

def slave():
    data = MPI.COMM_WORLD.bcast(None, root=MASTER_RANK)
    pi_estimate = estimate_pi(data)

    MPI.COMM_WORLD.send(pi_estimate, dest=MASTER_RANK)



def main():
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == MASTER_RANK:
        master()
    else:
        slave()

if __name__ == "__main__":
    main()
