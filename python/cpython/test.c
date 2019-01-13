#include <stdio.h>
#include <stdlib.h>

int main() {
    // Testing stuff with labels
    printf("Hello, ");
test_label:
    printf("World!\n");

    int *arr;
    int arr_len = 5;

    arr = malloc(sizeof(int) * arr_len);
    arr[-1] = 4;

    for (int i = -1; i < arr_len; i++) {
        printf("%d\n", arr[i]);
    }
}
