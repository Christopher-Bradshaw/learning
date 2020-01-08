from ctypes import CFUNCTYPE

from .python_to_core import PythonVisitor
from .type_infer import TypeInfer, TypeReify, compute_mgu, function_sig_reify
from .core_to_llvm import LLVMEmitter, llvm_init, create_llvm_engine, compile_code
from .type_definitions import arg_type, core_to_ctypes

from .ast_printers import pformat_ast


def autojit(func):
    name = func.__name__

    def _wrapper(*args):
        # Work out what the mangled name is. Really we should cache this
        arg_types = tuple(map(arg_type, args))
        mangled_name = mangler(arg_types, name)

        # Compile to Core
        core = PythonVisitor()(func)
        core, func_type, constraints = TypeInfer()(core, arg_types, p=False)

        # Work out what all the types are
        mgu = compute_mgu(constraints)
        if not mgu:
            raise Exception(
                f"Could not compute MGU from ast and constraints:\n{pformat_ast(core)}\n{get_constraints(constraints)}"
            )
        core = TypeReify(mgu)(core, p=False)
        func_type = function_sig_reify(func_type, mgu)

        # Compile to LLVM
        llvm = LLVMEmitter()(core, func_type, p=False)

        # Make that LLVM calable
        llvm_init()
        engine = create_llvm_engine()
        compile_code(engine, str(llvm))

        func_ptr = engine.get_function_address(name)
        cfunc = CFUNCTYPE(
            core_to_ctypes[func_type.ret_type],
            *[core_to_ctypes[t] for t in func_type.arg_types],
        )(func_ptr)
        return cfunc(*args)

    return _wrapper


def get_constraints(constraints):
    res = []
    for c in constraints:
        res.append(f"{c[0]} ~ {c[1]}")
    return "\n".join(res)


def mangler(arg_types, name):
    return name + "_" + str(hash(arg_types))
