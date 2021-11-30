#include <iostream>
#include <string>

struct person {
  std::string name;
  int age;

  bool operator==(const person& other) const {
        return age == other.age;
  }
  bool operator==(const int other) const {
    return age == other;
  }
};


int main() {

  // Two ways to initiate a struct, with and without designators
  person bob{"Bob", 20};
  person tom{.name="Tom", .age=17};

  std::cout << bob.age << std::endl;
  std::cout << (bob == tom) << std::endl;
  // True because of how we defined the equality operator
  std::cout << (bob == 20) << std::endl;

  // If you don't need to use a struct multiple times, you can define and assign
  // it in one go
  struct {
    int x;
    int y;
  } ru{1, 2};
  std::cout << ru.x << " " << ru.y << std::endl;

  // You can also define default values
  struct {
    int a = 1;
    bool b = false;
    double c;
  } v;
  std::cout << "a: " << v.a << std::endl;
  std::cout << "b: " << v.b << std::endl;
  std::cout << "c: " << v.c << std::endl;

  struct {
    double x1;
    double x2;
    double x3;
    double x4;
    double x5;
    double x6;
  } w;

  std::cout << w.x1 << std::endl;
  std::cout << w.x2 << std::endl;
  std::cout << w.x3 << std::endl;
  std::cout << w.x4 << std::endl;
  std::cout << w.x5 << std::endl;
  std::cout << w.x6 << std::endl;
}
