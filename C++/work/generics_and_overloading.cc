#include <iostream>

template <typename T>
T add(T, T);

template <typename T, typename U>
T add(T, U);

int main() {
  int x = 1;
  int y = 2;
  std::cout << add(x, y) << std::endl;

  double a = 1.1;
  double b = 2.2;
  std::cout << add(a, b) << std::endl;

  // Return type is the same as the type of the first arg
  std::cout << add(x, a) << std::endl;
  std::cout << add(a, x) << std::endl;

}

template <typename T>
T add(T x, T y) {
  std::cout << "Same T" << std::endl;
  return x + y;
}
template <typename T, typename U>
T add(T x, U y) {
  std::cout << "Diff T" << std::endl;
  return x + y;
}
