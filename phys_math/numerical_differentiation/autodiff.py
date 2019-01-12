import math

# Given a function and the arguments, autodiff computes both the
# value and the derivative wrt the first argument.
def autodiff(f, *args):
    args = [ad_num(args[i], int(i == 0)) for i in range(len(args))]
    return(f(*args))

class ad_num():
    def __init__(self, val, der=0):
        self.value = val
        self.derivative = der

    def __str__(self):
        return "value: {}, derivative: {}".format(self.value, self.derivative)

    def __repr__(self):
        return self.__str__()


    def __add__(self, other):
        try:
            return ad_num(
                self.value + other.value,
                self.derivative + other.derivative,
            )
        except AttributeError:
            return ad_num(
                self.value + other,
                self.derivative,
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        return (self * -1).__add__(other)


    def __mul__(self, other):
        try:
            return ad_num(
                    self.value * other.value,
                    self.derivative * other.value + self.value * other.derivative,
            )
        except AttributeError:
            return ad_num(
                    self.value * other,
                    self.derivative * other,
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    # Note this is self ** other. Self is obviously an ad_num. Other might be.
    def __pow__(self, other):
        try:
            # f(x) ** g(x)
            if other.derivative != 0 and self.derivative != 0:
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
        except AttributeError:
            # Power: f(x) ** a
            return ad_num(
                    self.value ** other,
                    other * self.value ** (other - 1) * self.derivative,
            )

    # other ** self
    def __rpow__(self, other):
        # Exponent: a ** f(x)
        # The case where both are ad_nums is handled in __pow__. This is the same except
        # self and other are inverted
        return ad_num(
                other ** self.value,
                other ** self.value * math.log(other) * self.derivative,
        )
