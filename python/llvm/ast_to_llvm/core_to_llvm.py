import ast
import llvmlite.ir as ir
import llvmlite.binding as binding

from .ast_printers import pformat_ast
from .common import primops

int32 = ir.IntType(32)


# This gets applied to a function
class LLVMEmitter(ast.NodeVisitor):
    def __init__(self):
        # These will be set while we build the function
        self.function = None
        self.builder = None

        # These are needed so that we can find the address of local variables
        # locals[var_name] = ptr
        self.locals = {}

        self.function_name = "fname"
        self.module_name = ir.Module("mname")
        # Our function takes 2 ints and returns 1 int
        self.func_type = ir.FunctionType(int32, (int32, int32))

    def visit_Func(self, node):
        print("CORE")
        print(pformat_ast(node))

        # Create the basic structure
        self.function = ir.Function(
            self.module_name, self.func_type, self.function_name
        )
        block = self.function.append_basic_block("entry")
        self.builder = ir.IRBuilder(block)

        for (core_arg, func_arg) in zip(node.args, self.function.args):
            typ = int32
            name = core_arg.ref
            alloc = self.builder.alloca(typ, name=name)
            self.builder.store(func_arg, alloc)
            self.locals[name] = alloc

        # Now that we have the builder, we just visit each statement in
        # turn and build up the LLVM IR
        list(map(self.visit, node.body))

        # And done, print + return the result
        print("\nLLVM")
        print(self.function)
        return self.function

    def visit_PrimOp(self, node):
        if node.fn == "add#":
            l = self.visit(node.args[0])
            r = self.visit(node.args[1])
            return self.builder.add(l, r)
        else:
            raise NotImplementedError

    def visit_Assign(self, node):
        name = node.ref
        if name in self.locals:
            raise NotImplementedError
        else:
            typ = int32
            val = self.visit(node.val)
            alloc = self.builder.alloca(typ, name=name)
            self.builder.store(val, alloc)
            self.locals[name] = alloc

    def visit_Return(self, node):
        self.builder.ret(self.visit(node.val))

    def visit_Num(self, node):
        return ir.Constant(int32, node.n)

    def visit_Var(self, node):
        return self.builder.load(self.locals[node.ref])


def llvm_init():
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()


def create_llvm_engine():
    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    # And an execution engine with an empty backing module
    backing_mod = binding.parse_assembly("")
    engine = binding.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def compile_code(engine, code):
    mod = binding.parse_assembly(code)
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
