#include <iostream>

int main() {

  // Declare x - starts of as a NULL pointer
  int *x;
  std::cout << (x == nullptr) << std::endl;

  // Assign x to be the address of b
  int b = 2;
  x = &b;
  std::cout << x << ", " << *x << std::endl;

  // y is a reference to b. It is implemented as a pointer and has pointer
  // like performance, but we access it as if it were the underlying thing
  // i.e. no dereferencing, y. rather than x-> (if it were and object)
  int &y = b;
  std::cout << y << std::endl;

  // Like a pointer, if you change the underlying data it is reflected in
  // the reference
  b += 1;
  std::cout << y << std::endl;

  // We can change what we reference, but we cannot create an unassigned
  // reference. This will fail: int &unassign;
  int c = 4;
  y = c;
  std::cout << y << std::endl;
}
