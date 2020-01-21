#include <stdio.h>
#include "helpers.h"

#define N 124 // The length of the vectors

int main(void) {
    int a[N], b[N];
    arange(a, N);
    arange(b, N);

    print(a, N);

    return 0;
}
