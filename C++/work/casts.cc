#include <iostream>

int main() {
  double tot = 3.3;
  double seg = 1.0;
  int x = static_cast<int>(tot / seg);

  std::cout << x << std::endl;
}
