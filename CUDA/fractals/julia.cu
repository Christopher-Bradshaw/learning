/*
   nvcc
*/
#include <iostream>
#include "file_output.hpp"

int main(void) {
    file_output::test();
    int x = file_output::test2();
    std::cout << x;
    return 0;
}
