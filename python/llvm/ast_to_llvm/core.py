# Define the core language

# I think we are making this a subclass of ast so that we can use
# the ast parsing tools to parse this too.
# Also, it is an ast! So it makes sense.

import ast


class Var(ast.AST):
    _fields = ("ref", "type")

    def __init__(self, ref, typ=None):
        self.ref, self.type = ref, typ
        super().__init__()


class Assign(ast.AST):
    _fields = ("ref", "val", "type")

    def __init__(self, ref, val, typ=None):
        self.ref, self.val, self.type = ref, val, typ
        super().__init__()


class PrimOp(ast.AST):
    _fields = ("fn", "args")

    def __init__(self, fn, args):
        self.fn, self.args = fn, args
        super().__init__()


class Func(ast.AST):
    _fields = ("fname", "args", "body")

    def __init__(self, fname, args, body):
        self.fname, self.args, self.body = fname, args, body
        super().__init__()


class Return(ast.AST):
    _fields = ("val",)

    def __init__(self, val):
        self.val = val
        super().__init__()


class Num(ast.AST):
    _fields = ("n", "type")

    def __init__(self, n, typ=None):
        self.n, self.type = n, typ
        super().__init__()
