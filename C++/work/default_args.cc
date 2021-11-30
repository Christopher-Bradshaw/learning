#include <iostream>

// The default can be specified in the definition ...
int add(int x, int y);
int add(int x, int y = 2) {
  return x + y;
}

// ... but it can also be in the declaration
int multiply(int, int = 2);
int multiply(int x, int y) {
  return x * y;
}

int main() {
  std::cout << add(3, 4) << std::endl;
  std::cout << add(3) << std::endl;

  std::cout << multiply(3, 4) << std::endl;
  std::cout << multiply(3) << std::endl;

}

