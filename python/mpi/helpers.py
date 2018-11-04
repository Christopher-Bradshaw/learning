from mpi4py import MPI

def nprint(string):
    rank = MPI.COMM_WORLD.Get_rank()
    print("Node {}: {}".format(rank, string))
