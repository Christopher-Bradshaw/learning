import pytest
from autodiff import autodiff
from finite_diff import finite_diff

@pytest.mark.skip()
def test_failing():
    f = lambda x1: 2
    print(autodiff(f, 2))


class TestBasic():
    def test_d_of_constant(self):
        res = autodiff(lambda x1, x2: x2, 2, 3)
        assert res.value == 3 and res.derivative == 0

    def test_d_of_itself(self):
        res = autodiff(lambda x1: x1, 2)
        assert res.value == 2 and res.derivative == 1

class TestPlusMinus():
    def test_d_of_sum_with_constant(self):
        res = autodiff(lambda x1: x1 + 99, 2)
        assert res.value == 101 and res.derivative == 1

        res = autodiff(lambda x1: 99 + x1, 2)
        assert res.value == 101 and res.derivative == 1

    def test_d_of_sum(self):
        res = autodiff(lambda x1: x1 + x1, 3)
        assert res.value == 6 and res.derivative == 2

    def test_d_of_sub(self):
        res = autodiff(lambda x1: x1 - x1, 3)
        assert res.value == 0 and res.derivative == 0

    def test_d_of_sub_with_constant(self):
        res = autodiff(lambda x1: x1 - 99, 2)
        assert res.value == -97 and res.derivative == 1

        res = autodiff(lambda x1: 99 - x1, 2)
        assert res.value == 97 and res.derivative == -1

class TestMultDivide():
    def test_d_of_product(self):
        res = autodiff(lambda x1: x1 * x1, 3)
        assert res.value == 9 and res.derivative == 6

    def test_d_of_power(self):
        res = autodiff(lambda x1: x1 ** 2, 3)
        assert res.value == 9 and res.derivative == 6

        res = autodiff(lambda x1: x1 ** 0, 3)
        assert res.value == 1 and res.derivative == 0

        res = autodiff(lambda x1, x2: x1 ** x2, 3, 2)
        assert res.value == 9 and res.derivative == 6


    def test_d_of_exp(self):
        res = autodiff(lambda x1: 2 ** x1, 3)
        assert res.value == 8 and res.derivative == pytest.approx(5.545177)

        res = autodiff(lambda x1, x2: x2 ** x1, 3, 2)
        assert res.value == 8 and res.derivative == pytest.approx(5.545177)

    def test_fx_to_gx(self):
        res = autodiff(lambda x1: x1 ** x1, 3)
        assert res.value == 27 and res.derivative == pytest.approx(56.6625)

        res = autodiff(lambda x1: (x1**1.1) ** (x1**1.1), 3)
        assert res.value == pytest.approx(57.1921) and res.derivative == pytest.approx(155.072)


class TestStress():
    @pytest.mark.parametrize("f, args", [
        (lambda x1: 2 * ((x1 + 1)**2 * 7), (5,)),
        (lambda x1, x2, x3: 5 * (2 + (x1 * x2) ** x3) ** x2, (1, 2, 3)),
        (lambda x1: 2 ** (x1 * (x1 + x1 - 3) * 2), (3,)),
        (lambda x1: (2 ** (x1 * (x1 + x1 - 3) * 2)) ** x1, (1.1,)),
    ])
    def test_chain(self, f, args):
        # import pdb; pdb.set_trace()
        res = autodiff(f, *args)
        assert res.value == f(*args)
        assert res.derivative == pytest.approx(finite_diff(f, *args), rel=1e-3)
