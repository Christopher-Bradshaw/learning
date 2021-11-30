#include <iostream>

enum Color {
  red,
  green,
  blue,
};

int main() {
  Color red = Color::red;
  std::cout << (red == Color::red) << std::endl;
  // These are actually just ints internally
  std::cout << Color::red << std::endl;
}
