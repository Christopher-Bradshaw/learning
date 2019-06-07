from code_to_profile import fibonacci

class TestFibonacci():
    def test_fibonacci_works(self):
        assert fibonacci(1) == 1
        assert fibonacci(2) == 1
        assert fibonacci(5) == 5
        assert fibonacci(10) == 55

