#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

double integrate(double (*f)(double), double start, double stop, double steps);
double square(double x);
double cube(double x);

int main() {
    int steps = 10000, threads = 12;
    double global_area;

    // Our integration algorithm relies on steps being divisible by the number of threads
    steps -= steps % threads;

    global_area = 0;
# pragma omp parallel num_threads(threads)
    {
        // We need to declare partial_area in here else each thread will share it
        // and there will be race conditions in its assignment
        double partial_area = integrate(square, -3, 3, steps);
        // Critical section - this is effectively a mutex around the next line
# pragma omp critical
        global_area += partial_area;

    }
    printf("%lf\n", global_area);


    global_area = 1;
# pragma omp parallel num_threads(threads) reduction(+: global_area)
    global_area = integrate(sin, 0, 3*M_PI, steps);
    printf("%lf\n", global_area);


    // Imagine we want to add the area of two function, say we go from
    // an x^3 region to an x^2 region at 0.
    // We reduce down the temporary (partial_area like) variables with +
    // The starting values for these are the identity for addition (0).
    // We then reduce those into the initial value of global_area, again with +
    // And we return that value.
    // I still don't quite understand this...
    global_area = 0;
# pragma omp parallel num_threads(threads) reduction(+: global_area)
    global_area = integrate(cube, -3, 0, steps);
# pragma omp parallel num_threads(threads) reduction(+: global_area)
    global_area = integrate(square, 0, 3, steps);

    printf("%lf\n", global_area);
    return 0;
}

double integrate(double (*f)(double), double start, double stop, double steps) {
    int rank = omp_get_thread_num();
    int tot_threads = omp_get_num_threads();

    double delta = stop - start;
    double stepsize = delta/steps;

    double l_start = start + rank * delta / tot_threads;
    double l_stop = start + (rank + 1) * delta / tot_threads;

    double x = l_start + stepsize/2;

    double sum = 0;
    while (x < l_stop) {
        sum += f(x);
        x += stepsize;
    }
    return delta * sum / steps;
}


double square(double x) {
    return x*x;
}

double cube(double x) {
    return x*x*x;
}
