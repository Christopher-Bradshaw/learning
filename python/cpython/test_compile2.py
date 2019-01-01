import dis
code = """
x = 10
y = [1,2,3]

def double(x):
    x *= 2
    return x

def double_last_in_list(l):
    new = double(l[-1])
    l[-1] = new

z = double(x)
double_last_in_list(y)
print(x, y, z)
"""

dis.dis(code)

"""
  2           0 LOAD_CONST               0 (10)
              2 STORE_NAME               0 (x)

  3           4 LOAD_CONST               1 (1)
              6 LOAD_CONST               2 (2)
              8 LOAD_CONST               3 (3)
             10 BUILD_LIST               3
             12 STORE_NAME               1 (y)

  5          14 LOAD_CONST               4 (<code object double at 0x7f10aad70c90, file "<dis>", line 5>)
             16 LOAD_CONST               5 ('double')
             18 MAKE_FUNCTION            0
             20 STORE_NAME               2 (double)

  9          22 LOAD_CONST               6 (<code object double_last_in_list at 0x7f10aab0c810, file "<dis>", line 9>)
             24 LOAD_CONST               7 ('double_last_in_list')
             26 MAKE_FUNCTION            0
             28 STORE_NAME               3 (double_last_in_list)

 13          30 LOAD_NAME                2 (double)
             32 LOAD_NAME                0 (x)
             34 CALL_FUNCTION            1
             36 STORE_NAME               4 (z)

 14          38 LOAD_NAME                3 (double_last_in_list)
             40 LOAD_NAME                1 (y)
             42 CALL_FUNCTION            1
             44 POP_TOP

 15          46 LOAD_NAME                5 (print)
             48 LOAD_NAME                0 (x)
             50 LOAD_NAME                1 (y)
             52 LOAD_NAME                4 (z)
             54 CALL_FUNCTION            3
             56 POP_TOP
             58 LOAD_CONST               8 (None)
             60 RETURN_VALUE
"""
