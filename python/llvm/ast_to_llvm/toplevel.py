from ctypes import CFUNCTYPE, c_double, c_int

from .python_to_core import PythonVisitor
from .type_infer import TypeInfer, TypeReify, compute_mgu, function_sig_reify
from .core_to_llvm import LLVMEmitter, llvm_init, create_llvm_engine, compile_code
from .type_definitions import arg_type

from .ast_printers import pformat_ast


def autojit(func):
    name = func.__name__

    def _wrapper(*args):
        # Work out what the mangled name is
        arg_types = tuple(map(arg_type, args))
        mangled_name = mangler(arg_types, name)

        # Compile to Core
        core = PythonVisitor()(func)
        func_type, constraints = TypeInfer()(core)

        # Work out what all the types are
        mgu = compute_mgu(constraints)
        TypeReify(mgu)(core)
        print(func_type)
        func_type = function_sig_reify(func_type, mgu)
        print(func_type)

        print(pformat_ast(core))

        # Compile to LLVM
        llvm = LLVMEmitter()(core)

        # Make that LLVM calable
        llvm_init()
        engine = create_llvm_engine()
        compile_code(engine, str(llvm))

        func_ptr = engine.get_function_address(name)
        cfunc = CFUNCTYPE(c_int, c_int, c_int)(func_ptr)
        return cfunc(*args)

    return _wrapper


def mangler(arg_types, name):
    return name + "_" + str(hash(arg_types))
