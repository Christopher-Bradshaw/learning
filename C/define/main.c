#include <stdio.h>

#include "import_gparent.h"
#include "import_parent.h"

// Defines are often called "symbolic constants"
// Useful to give names to magic numbers

#define PI 3.14159
#define AREA(R) PI*R*R


int main() {

    float r = 3;
    printf("Area of circle with radius %f is %f\n", r, AREA(r));

#undef AREA
    // If you uncomment this line, this won't compile.
    /* printf("Area of circle with radius %f is %f\n", r, AREA(r)); */

    // Uncomment these lines. It will still compile! But will segfault
    // Defines are not type checked!

    /* #define X 1 */
    /* printf("%s\n", X); */

    printf("%d\n", p());
}
