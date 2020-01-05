import ast
import llvmlite.ir as ir
import llvmlite.binding as binding

from .ast_printers import pformat_ast
from .type_definitions import TFun, core_to_llvm_types, int64, float64


# This gets applied to a function
class LLVMEmitter(ast.NodeVisitor):
    def __init__(self):
        # These will be set while we build the function
        self.function = None
        self.func_type = None
        self.builder = None

        # These are needed so that we can find the address of local variables
        # locals[var_name] = ptr
        self.locals = {}

        self.function_name = "fname"
        self.module_name = ir.Module("mname")

    def __call__(self, node, func_type: TFun, p=False):
        if p:
            print("CORE")
            print(pformat_ast(node))

        self.func_type = ir.FunctionType(
            core_to_llvm_types[func_type.ret_type],
            tuple([core_to_llvm_types[t] for t in func_type.arg_types]),
        )

        llvm_func = self.visit(node)

        if p:
            print("\nLLVM")
            print(self.function)
        return llvm_func

    def visit_Func(self, node):
        # Create the basic structure
        self.function_name = node.fname
        self.function = ir.Function(
            self.module_name, self.func_type, self.function_name
        )
        block = self.function.append_basic_block("entry")
        self.builder = ir.IRBuilder(block)

        for (core_arg, func_arg, func_arg_type) in zip(
            node.args, self.function.args, self.func_type.args
        ):
            typ = func_arg_type
            name = core_arg.ref
            alloc = self.builder.alloca(typ, name=name)
            self.builder.store(func_arg, alloc)
            self.locals[name] = alloc

        # Now that we have the builder, we just visit each statement in
        # turn and build up the LLVM IR
        list(map(self.visit, node.body))

        return self.function

    def visit_PrimOp(self, node):
        if node.fn == "add#":
            l = self.visit(node.args[0])
            r = self.visit(node.args[1])
            # if node.type is
            if l.type is core_to_llvm_types[float64]:
                return self.builder.fadd(l, r)
            elif l.type is core_to_llvm_types[int64]:
                return self.builder.add(l, r)
            else:
                return NotImplementedError

        else:
            raise NotImplementedError

    def visit_Assign(self, node):
        name = node.ref
        if name in self.locals:
            raise NotImplementedError
        else:
            typ = core_to_llvm_types[node.type]
            val = self.visit(node.val)
            alloc = self.builder.alloca(typ, name=name)
            self.builder.store(val, alloc)
            self.locals[name] = alloc

    def visit_Return(self, node):
        self.builder.ret(self.visit(node.val))

    def visit_Num(self, node):
        typ = core_to_llvm_types[node.type]
        return ir.Constant(typ, node.n)

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
