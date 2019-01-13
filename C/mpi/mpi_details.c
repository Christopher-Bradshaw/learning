#include <mpi.h>
#include <stdio.h>
#include <assert.h>

#define TAG_POINTER 11
#define TAG_PROCESSOR_NAME 12

void asset_all_comm_worlds_identical();

int main() {
    MPI_Init(NULL, NULL);

    asset_all_comm_worlds_identical();

    MPI_Finalize();
}


void asset_all_comm_worlds_identical() {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int name_len;

    if (world_rank == 0) {
        long recv_comm_world_pointer;
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        MPI_Get_processor_name(processor_name, &name_len);
        printf("rank 0 on %s has %ld\n", processor_name, (long)MPI_COMM_WORLD);
        for (int i = 1; i < world_size; i++) {
            MPI_Recv(&recv_comm_world_pointer, 1, MPI_LONG, i, TAG_POINTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, i, TAG_PROCESSOR_NAME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("rank %d on %s has %ld\n", i, processor_name, recv_comm_world_pointer);
        }
    } else {
        long comm_world_pointer = (long)MPI_COMM_WORLD;
        MPI_Send(&comm_world_pointer, 1, MPI_LONG, 0, TAG_POINTER, MPI_COMM_WORLD);

        char processor_name[MPI_MAX_PROCESSOR_NAME];
        MPI_Get_processor_name(processor_name, &name_len);
        MPI_Send(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, TAG_PROCESSOR_NAME, MPI_COMM_WORLD);
        printf("rank %d on %s has %ld\n", world_rank, processor_name, (long)MPI_COMM_WORLD);
    }
}
