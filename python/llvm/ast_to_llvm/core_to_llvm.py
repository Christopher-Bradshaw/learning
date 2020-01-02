import ast
import llvmlite.ir as ir

from .ast_printers import pformat_ast


class LLVMEmitter(ast.NodeVisitor):
    def __init__(self):
        self.function = None
        self.function_name = "fname"
        self.module_name = "mname"
        # Our function takes 2 ints and returns 1 int
        self.func_type = ir.FunctionType(
            ir.IntType(32), (ir.IntType(32), ir.IntType(32))
        )

    def start_function(self):
        self.function = ir.Function(
            self.module_name, self.func_type, self.function_name
        )

    def visit_Func(self, node):
        print("Visit Func")
        print(pformat_ast(node))
        list(map(self.visit, node.body))

    def visit_PrimOp(self, node):
        print("Visit PrimOp")

    def visit_Assign(self, node):
        print("Visit Assign")
        val = self.visit(node.val)

    def visit_Return(self, node):
        print("Visit Return")
