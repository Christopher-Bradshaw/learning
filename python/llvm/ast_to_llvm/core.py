# Define the core language

# I think we are making this a subclass of ast so that we can use
# the ast parsing tools to parse this too.
# Also, it is an ast! So it makes sense.

import ast


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
