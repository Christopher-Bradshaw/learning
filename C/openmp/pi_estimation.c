#include <stdio.h>
#include <omp.h>
//
// See http://mathworld.wolfram.com/PiFormulas.html
int main() {

    long long num_components = 50000000, num_threads = 8;
    double pi_est = 0, numerator;


# pragma omp parallel for num_threads(num_threads) \
    private(numerator) reduction(+: pi_est)
    for (long long i = 0; i < num_components; i++) {
        if (i % 1000000 == 0) {
            printf("%lld ", i);
        }
        // rather than have a double numerator inside the loop (which will reassign memory
        // each loop), mark it private. I'm pretty sure the compiler does something a bit
        // cleverer and initilises it outside the loop.
        numerator = (i % 2 == 0) ? 1 : -1;
        // Because pi_est is part of a reduction, the pi_est here != the outer pi_est.
        // It is a temporary, local, variable. It is initialised to the identity value (0 for sum)
        // The outer pi_est is set to be the sum(all the inner pi_ests) at the end.
        // Using reduction is good else we would need to wrap this in a critical/atomic and that
        // would be slow.
        pi_est += numerator / (2*i + 1);
    }
    pi_est *= 4;
    printf("%d: %lf\n", num_threads, pi_est);


    // Keeping track of these variables - what is local to each thread, what is shared, etc etc
    // is tricky. Wouldn't it be nice if we could be 100% explicit about everything that we use?
    // Enter default (https://msdn.microsoft.com/en-us/library/242dw0dc.aspx)
    //
    // Same pi computation again, except we now need to be explicit about the shared vars

    pi_est = 0;
# pragma omp parallel for num_threads(num_threads) \
    default(none) shared(num_components) private(numerator) \
    reduction(+: pi_est)
    for (int i = 0; i < num_components; i++) {
        numerator = (i % 2 == 0) ? 1 : -1;
        pi_est += numerator / (2*i + 1);
    }
    pi_est *= 4;
    printf("%d: %lf\n", num_threads, pi_est);
}
