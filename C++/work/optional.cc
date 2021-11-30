// Only in c++17 and later. Compile with:
// g++ -std=c++17 optional.cc
#include <iostream>
#include <optional>

int add(const int x, const std::optional<int> y) {
  if (y.has_value()) {
    return x + *y;
  }
  return x + 1;
}

int main() {
  // Note how you just call it with an int! You don't need to wrap it in
  // anything. That's pretty cool.
  std::cout << add(2, 3) << std::endl;

  /* std::cout << add(2) << std::endl; */
}
