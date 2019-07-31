import ctypes

lib = ctypes.cdll.LoadLibrary("engine/target/release/libengine.so")

# Define the types of the various funcs
lib.doubler.argtypes = [ctypes.c_double]
lib.doubler.restype = ctypes.c_double

print(lib.doubler(3))
