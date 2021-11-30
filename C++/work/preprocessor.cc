#include <iostream>

// Literally just macro substituted
#define SPEED_OF_LIGHT 3e8 // m/s
// Substituted sanely... i.e. doesn't do it here!
void print_SPEED_OF_LIGHT() {
  std::cout << SPEED_OF_LIGHT << std::endl;
}

// With args. Putting things in parems is usually good to isolate them
#define min(a, b) (a < b ? a : b)
// Can also span multiple lines
#define max(a,\
            b)\
    (a > b\
     ? a : b)

int main() {
  std::cout << SPEED_OF_LIGHT << std::endl;
  print_SPEED_OF_LIGHT();

  std::cout << min(1, 2) << std::endl;
  std::cout << min(3, 2) << std::endl;

  std::cout << max(1, 2) << std::endl;
  std::cout << max(3, 2) << std::endl;
}

