from ctypes import CFUNCTYPE, c_double, c_int

from .python_to_core import PythonVisitor
from .type_infer import TypeInfer
from .core_to_llvm import LLVMEmitter, llvm_init, create_llvm_engine, compile_code
from .ast_printers import pformat_ast


def autojit(func):
    untyped_core = PythonVisitor()(func)
    typed_core, constraints = TypeInfer()(untyped_core)
    # llvmlite.ir.values.Function
    llvm = LLVMEmitter()(typed_core)

    # Now make it callable
    llvm_init()
    engine = create_llvm_engine()
    compile_code(engine, str(llvm))

    # And now wrap this cfunction in a python func

    def _wrapper(*args):
        func_ptr = engine.get_function_address(func.__name__)
        cfunc = CFUNCTYPE(c_int, c_int, c_int)(func_ptr)
        return cfunc(*args)

    return _wrapper
