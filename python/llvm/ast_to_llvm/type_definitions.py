# This is a type constructor
# See below for how it builds our types!
class TCon:
    def __init__(self, s):
        self.s = s

    def __eq__(self, other):
        if isinstance(other, TCon):
            return self.s == other.s
        else:
            return False

    def __hash__(self):
        return hash(self.s)

    def __str__(self):
        return self.s

    __repr__ = __str__


# int32 = TCon("Int32")
int64 = TCon("Int64")
# float32 = TCon("Float")
float64 = TCon("Float64")


# This is a variable that stands in for a type. In the type inference
# step, each variable (and constant) will be assign a type variable (Tvar). We will
# then use the constraints to work out what all the types are
class TVar:
    def __init__(self, s):
        self.s = s

    def __hash__(self):
        return hash(self.s)

    def __eq__(self, other):
        if isinstance(other, TVar):
            return self.s == other.s
        else:
            return False

    def __str__(self):
        return self.s

    __repr__ = __str__
