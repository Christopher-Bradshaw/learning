#include <iostream>

int add(int x, int y) {
  return x + y;
}

float add(float x, float y) {
  return x + y;
}

int add(int x) {
  return x + 1;
}

int main() {
  std::cout << add(2, 3) << std::endl;
  std::cout << add(2.2f, 3.0f) << std::endl;
  std::cout << add(2) << std::endl;
}
