#include <iostream>

void increment(int &x) {
  x += 1;
}

int main() {
  int x = 2;
  std::cout << x << std::endl;
  increment(x);
  std::cout << x << std::endl;
}
