#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int master();
int slave(int rank);
int do_work(int rank, int data);

#define JOB_SUCCESS 0
#define JOB_FAILED -1

int main() {

    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Example master/slave hierarchy. One machine (in this case the one with rank == 0) sends
    // out work
    if (world_rank == 0) {
        master();
    } else {
        slave(world_rank);
    }

    MPI_Finalize();
}


int master() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Send out jobs
    int data_to_send, res, tag = 1234;
    for (int i = 1; i < world_size; i++) {
        data_to_send = i*i;
        printf("Sending data to node %d\n", i);
        res = MPI_Ssend(&data_to_send, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
        printf("Result of sending to node %d: %d\n", i, res);
    }

    // Get results back
    printf("Send jobs out, waiting for results\n");
    int job_status;
    for (int i = 1; i < world_size; i++) {
        MPI_Recv(&job_status, 1, MPI_INT, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (job_status == JOB_SUCCESS) {
            printf("Job %d sucess!\n", i);
        } else {
            printf("Job %d failed. We should retry or something...\n", i);
        }
    }

    return 0;
}

int slave(int rank) {
    int rec_data, source_node = 0, tag = 1234, res;
    printf("Waiting on data (node %d)\n", rank);
    res = MPI_Recv(&rec_data, 1, MPI_INT, source_node, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Got data (node %d)\n", rank);

    int job_result = do_work(rank, rec_data);
    // About to block!
    if (rank == 3) {
        sleep(3);
    }
    MPI_Ssend(&job_result, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
    return 0;
}


int do_work(int rank, int data) {
    printf("Node %d, Received %d as data\n", rank, data);
    return JOB_SUCCESS;
}
