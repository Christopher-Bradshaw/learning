#include <stdio.h>
#include "helpers.h"

void arange(int *res, int len_res) {
    for (int i=0; i<len_res; i++) {
        res[i] = i;
    }
}

void print(int *res, int len_res) {
    for (int i=0; i<len_res; i++) {
        printf("%d, ", res[i]);
    }
    printf("\n");
}
