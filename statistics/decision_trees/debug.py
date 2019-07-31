from tree import DecisionTree
import numpy as np


# import sys


# def info(typ, value, tb):
#     if hasattr(sys, "ps1") or not sys.stderr.isatty() or typ != AssertionError:
#         # we are in interactive mode or we don't have a tty-like
#         # device, so we call the default hook
#         sys.__excepthook__(typ, value, tb)
#     else:
#         import traceback, pdb

#         # we are NOT in interactive mode, print the exception...
#         traceback.print_exception(typ, value, tb)
#         # ...then start the debugger in post-mortem mode.
#         pdb.pm()


# sys.excepthook = info

# lim = 40


# def f(x1, x2):
#     val = x1 ** 2 + x2 ** 2 + 40 * np.sin(x1)
#     res = np.zeros_like(x1)
#     res[val > lim] = 2
#     res[(val > 20) & (val < lim)] = 1
#     return res


# n_obj, n_feat = 1000, 2
# np.random.seed(2)
# x = np.random.random((n_obj, n_feat)) * 10
# y = f(x[:, 0], x[:, 1])

# t = DecisionTree()
# t.fit(x, y)

s = np.random.randint(0, 5, (10, 100))
r = np.bincount(s)
print(r)
