import ast

from .ast_printers import pformat_ast
from .type_definitions import TVar, int64, float64


class TypeInfer(ast.NodeVisitor):
    def __init__(self):
        self.names = gen_names()
        self.arg_types = None
        self.ret_type = None

        # Once we've seen a variable, keep track of its type as it cannot change!
        self.seen_vars = {}

        # A list of 2 tuples whose pairs need to have the same type
        self.constraints = []

    def __call__(self, node, p=False):
        if p:
            print("CORE")
            print(pformat_ast(node))

        self.visit(node)

        if p:
            print("TYPED CORE")
            print(pformat_ast(node))

            print()
            print("Constraints")
            for c in self.constraints:
                print(c[0], c[1])

            print()
            print("Signature")
            print(self.arg_types, "->", self.ret_type)
        return node, self.constraints

    # This generates a fresh variables name.
    def fresh(self):
        return TVar("$" + next(self.names))

    def visit_Func(self, node):

        # The argument types are unknown
        self.arg_types = [self.fresh() for i in node.args]
        for i, _ in enumerate(node.args):
            node.args[i].type = self.arg_types[i]
            self.seen_vars[node.args[i].ref] = self.arg_types[i]

        # As is the return type
        self.ret_type = TVar("$ret_type")

        # Parse the body
        list(map(self.visit, node.body))
        return node

    def visit_Num(self, node):
        typ = self.fresh()
        node.type = typ
        if isinstance(node.n, int):
            self.constraints.append((typ, int64))
        elif isinstance(node.n, float):
            self.constraints.append((typ, float64))
        return typ

    def visit_Var(self, node):
        typ = self.seen_vars[node.ref]
        node.type = typ
        return typ

    def visit_Assign(self, node):
        typ = self.visit(node.val)

        # If we've seen this variable name before, we now know that its type
        # must be the same as type returned from the node.val
        if node.ref in self.seen_vars:
            self.constraints.append((typ, self.seen_vars[node.ref]))

        # Set this type to be the same as the return type
        self.seen_vars[node.ref] = typ
        node.type = typ
        # Assignment doesn't return anything, so it doesn't imply anything about types
        return None

    def visit_PrimOp(self, node):
        if node.fn == "add#":
            typ_l = self.visit(node.args[0])
            typ_r = self.visit(node.args[1])
            self.constraints.append((typ_l, typ_r))
            return typ_l
        else:
            raise NotImplementedError

    def visit_Return(self, node):
        typ = self.visit(node.val)
        self.constraints.append((typ, self.ret_type))
        return None


def gen_names():
    k = 1
    while True:
        yield str(k)
        k += 1
