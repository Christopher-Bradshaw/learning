import ast
from textwrap import dedent

from .ast_printers import pformat_ast
from . import core
from .common import primops


# Walk the python ast and return the core language
class PythonVisitor(ast.NodeVisitor):
    def __init__(self):
        self._source = None
        self._ast = None

    # Source might be given in many formats (string, func obj, etc)
    def __call__(self, source):
        if isinstance(source, str):
            source = dedent(source)
        else:
            raise NotImplementedError
        print(source)

        self._source = source
        self._ast = ast.parse(source)

        print("PYTHON")
        print(pformat_ast(self._ast))

        res = self.visit(self._ast)

        print("CORE")
        print(pformat_ast(res[0]))

        return res

    # Define all the visitors
    def visit_Module(self, node):
        # The module contains multiple objects. Visit each of them
        return list(map(self.visit, node.body))

    def visit_FunctionDef(self, node):
        statements = node.body
        statements = list(map(self.visit, statements))
        args = list(map(self.visit, node.args.args))

        return core.Func(node.name, args, statements)

    def visit_Assign(self, node):
        ref = node.targets[0].id
        val = self.visit(node.value)
        return core.Assign(ref, val)

    def visit_BinOp(self, node):
        return core.PrimOp(
            primops[node.op.__class__], [self.visit(node.left), self.visit(node.right)]
        )

    def visit_Name(self, node):
        return core.Var(node.id)

    def visit_Return(self, node):
        return core.Return(self.visit(node.value))

    def visit_arg(self, node):
        return core.Var(node.arg)

    def visit_Num(self, node):
        return core.Num(node.n)

    def visit_Name(self, node):
        return core.Var(node.id)
