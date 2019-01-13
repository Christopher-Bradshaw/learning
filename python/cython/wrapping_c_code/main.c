// This is just to prove that the c_funcs are valid C!
#include <stdio.h>
#include "c_funcs.h"

int main() {
    int x = 2;
    int y;
    y = doubler(x);
    printf("%d\n", y);


    point p1 = {1, 2};
    point p2 = {1, 3};

    printf("%f\n", distance_between(&p1, &p2));
}
