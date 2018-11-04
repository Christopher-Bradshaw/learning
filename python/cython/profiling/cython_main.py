from math import pi

# from cython_funcs_basic import integrate, sin_squared
# from cython_funcs_basic_with_profiling import integrate, sin_squared
from cython_funcs_fast_with_profiling import integrate, sin_squared, integrate_sin_squared

res = integrate_sin_squared(0, 2*pi)#, sin_squared)
print(res)
