// Compile with gcc caller.c simple_func.s -o caller
// You need to have comipled the lib with llc simple_func.ll
#include <stdio.h>

// Not sure how you do headers, so will just declare it here
int mult(int, int);
int return_first(int, int);
float add(float, float);

int main(){
    printf("%d\n", mult(2, 8));
    printf("%f\n", add(2., 8.));
    printf("%d\n", return_first(4, 8));
    return 0;
}
