import dis

code = """
x = 2
y = 3
print(x + y)
"""
x = compile(code, "the filename of the code we loaded", "exec")
print(type(x))
print(dir(x))

print(x.co_code)

# This is the bytecode
bytecode = [i for i in x.co_code]
print(bytecode)
print(len(bytecode))

# [100, 0, 90, 0, 100, 1, 90, 1, 101, 2, 101, 0, 101, 1, 23, 0, 131, 1, 1, 0, 100, 2, 83, 0]
dis.dis(code)

"""
101 - LOAD_NAME
1 - POP_TOP
100 - LOAD_CONST
83 - return value
"""

"""
  1           0 LOAD_CONST               0 (2)
              2 STORE_NAME               0 (x)

  2           4 LOAD_CONST               1 (3)
              6 STORE_NAME               1 (y)

  3           8 LOAD_NAME                2 (print)
             10 LOAD_NAME                0 (x)
             12 LOAD_NAME                1 (y)
             14 BINARY_ADD
             16 CALL_FUNCTION            1
             18 POP_TOP
             20 LOAD_CONST               2 (None)
             22 RETURN_VALUE
"""
