import math

# Given a function and the arguments, autodiff computes both the
# value and the derivative wrt the first argument.
def autodiff(f, *args):
    args = [ad_num(args[i], int(i == 0)) for i in range(len(args))]
    return(ad_num(f(*args)))

class ad_num():
    def __init__(self, val, der=0):
        if isinstance(val, ad_num):
            self.value, self.derivative = val.value, val.derivative
        else:
            self.value, self.derivative = val, der

    def __str__(self):
        return "value: {}, derivative: {}".format(self.value, self.derivative)

    def __repr__(self):
        return self.__str__()


    def __add__(self, other):
        other = self._lift(other)
        return ad_num(
            self.value + other.value,
            self.derivative + other.derivative,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        return (self * -1).__add__(other)


    def __mul__(self, other):
        other = self._lift(other)
        return ad_num(
                self.value * other.value,
                self.derivative * other.value + self.value * other.derivative,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    # self ** other
    def __pow__(self, other):
        other = self._lift(other)
        # f(x) ** g(x)
        if self.derivative != 0 and other.derivative != 0:
            return ad_num(
                    self.value ** other.value,
                    self.value ** other.value * (
                        other.derivative * math.log(self.value) +
                        other.value / self.value * self.derivative
                    )
            )
        # Power: f(x) ** a
        elif other.derivative == 0:
            return ad_num(
                    self.value ** other.value,
                    other.value * self.value ** (other.value - 1) * self.derivative
            )
        # Exponent: a ** f(x)
        elif self.derivative == 0:
            return ad_num(
                    self.value ** other.value,
                    self.value ** other.value * math.log(self.value) * other.derivative,
            )
        else:
            raise Exception("This should never happen")

    # other ** self
    def __rpow__(self, other):
        return self._lift(other) ** self

    def _lift(self, x):
        if not isinstance(x, ad_num):
            return ad_num(x, 0)
        return x
