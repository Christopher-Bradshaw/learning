#include <stdio.h>
#include "simple_lib.h"

int main() {
    int x = 2;
    printf("%d\n", doubler(x));
    printf("%p\n", &doubler);
}
