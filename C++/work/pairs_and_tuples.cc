#include <iostream>
#include <complex>
#include <tuple>
#include <utility>

int main() {

  std::pair<int, int> x (0, 2);
  std::pair<int, int> y (1, 1);

  std::cout << (x > y) << std::endl;
  std::cout << x.first << " "<< x.second << std::endl;

  // And generalised to n-tuples
  typedef std::tuple<std::complex<double>, std::string, double> my_tuple;
  my_tuple z(std::complex<double>(0, 1), "Love", 3.14);
  std::cout << std::get<1>(z) << std::endl;
  std::cout << std::tuple_size<my_tuple>::value << std::endl;

  // Most things work on pairs
  typedef std::pair<int, int> ip;
  ip xymax;
  xymax = std::max(ip(1, 1), ip(1, 0));
  std::cout << xymax.first << " " << xymax.second << std::endl;

  xymax = std::max(ip(1, 1), ip(1, 2));
  std::cout << xymax.first << " " << xymax.second << std::endl;
}
