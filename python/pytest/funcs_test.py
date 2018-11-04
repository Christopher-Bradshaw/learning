import pytest

import funcs

def test_add_1():
    assert funcs.add_1(3) == 4
    print("simple test passed")

class TestAdd1():
    def helper_function_that_adds_one(self, x):
        return x + 1

    def test_add_1_for_negative_numbers(self):
        assert funcs.add_1(-5) == -4

    def test_add_1_for_0(self):
        assert funcs.add_1(0) == self.helper_function_that_adds_one(0)

    def test_add_1_for_positive_numbers(self):
        assert funcs.add_1(4) == 5

class TestEnsurePositive():
    def test_happy_path(self):
        assert funcs.ensure_positive(2) == 2

    def test_throws_when_not_positive(self):
        with pytest.raises(Exception) as err:
            funcs.ensure_positive(-3)

        assert type(err.value) == Exception
        assert err.value.args[0] == "Not positive"
