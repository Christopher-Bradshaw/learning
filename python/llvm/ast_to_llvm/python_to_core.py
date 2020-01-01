import ast
from textwrap import dedent

from .ast_printers import pformat_ast
from . import core

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
        self.visit(self._ast)

    # Define all the visitors
    def visit_Module(self, node):
        print("Module visitor")
        print(pformat_ast(node))
        # The module contains multiple objects. Visit each of them
        res = list(map(self.visit, node.body))
        print(res)

    def visit_FunctionDef(self, node):
        print("Function visitor")
        statements = node.body
        print(statements)
        statements = list(map(self.visit, statements))

        return core.Func(node.name, node.args, node.body)

    def visit_Assign(self, node):
        print("Assign visitor")
        ref = node.targets[0].id
        val = self.visit(node.value)
        print(ref, val)
        # core.Assign(

    def visit_BinOp(self, node):
        print("BinOp visitor")
        primops = {ast.Add: "add#", ast.Mult: "mult#"}
        return core.PrimOp(
            primops[node.op.__class__], [self.visit(node.left), self.visit(node.right)]
        )

    def visit_Return(self, node):
        print(node)
        print("Return visitor")
