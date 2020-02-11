/*
 * g++ basic_io.cpp -o basic_io && ./basic_io
 */
#include <iostream>

int main(void) {
    // All names defined in the standard library are in the `std` namespace
    // The << operator takes two arguments.
    // The left needs to be an ostream (outstream) and the right the thing to print
    // It returns the left arg which is why we can chain like this.
    std::cout << "Enter two numbers: " << std::endl;

    int v1, v2;
    // Similar to above, the >> operator takes two args.
    // The left is an istream (instream) and the right the variable that the data
    // is stored in. Again, it returns the left arg which is why we can chain.
    std::cin >> v1 >> v2;

    std::cout << "The sum of " << v1 << " and " << v2 << " is " << v1 + v2 << std::endl;
    return 0;
}
