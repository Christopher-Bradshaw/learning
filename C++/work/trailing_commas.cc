int add(int a, int b) {
  return a + b;
}
int main() {
  int my_long_variable_name = 1;
  int my_other_long_variable_name = 3;
  add(
      my_long_variable_name,
      my_other_long_variable_name //, This is not allowed!
  );
}

