#include "c_funcs.h"
#include <math.h>

int doubler(int x) {
    return 2*x;
}

int not_imported(int x) {
    return x;
}

float distance_between(point* p1, point* p2) {
    return sqrt(
            powf((p1->x - p2->x), 2) +
            powf((p1->y - p2->y), 2)
    );
}

