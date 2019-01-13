#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void hello(void);
void hello2(void);

int main(int argc, char* argv[]) {

    int threads;
    if (argc < 2) {
        threads = 4;
    } else {
        threads = strtol(argv[1], NULL, 10);
    }

    # pragma omp parallel num_threads(threads)
    hello();

    # pragma omp parallel num_threads(threads/2)
    {
        hello2();
        hello2();
    }



    return 0;
}

void hello(void) {
    int rank = omp_get_thread_num();
    int total_threads = omp_get_num_threads();
    printf("Hello from thread %d of %d\n", rank, total_threads);
}

void hello2(void) {
    int rank = omp_get_thread_num();
    int total_threads = omp_get_num_threads();
    printf("Still here: thread %d of %d\n", rank, total_threads);
}
