# Fibonaci diagonalization (see Intro to Linear Algebra, Strang, Page 300 4th ed)

# Initial values + transform
v = vector([1, 0])
A = matrix([[1, 1], [1, 0]])

# The slow way
# for i in range(0, 10):
#     print(A^i*v)

# The fast way using diagonalization
ev = A.eigenvectors_right()


# Entry in ii is the ith eigenvalue
diag_matrix_lambda = diagonal_matrix([ev[0][0], ev[1][0]])
# ith columns is the corresponding eigenvector
ev_matrix_s = column_matrix([ev[0][1][0], ev[1][1][0]])
ev_matrix_s_inv = ev_matrix_s.inverse()

# This was a faff to set up, but raising the diagonal matrix to a large power is
# much faster!
# for i in range(10):
#     print(ev_matrix_s * diag_matrix_lambda^i * ev_matrix_s_inv * v)

print(diag_matrix_lambda)
print()
print(ev_matrix_s)
print()
print(ev_matrix_s_inv)
print()

# Though weirdly the slow version is fast and the fast version slow...
# I guess sage optimizes this internally. That makes sense
def slow(n):
    # m = identity_matrix(2)
    # for i in range(n):
    #     m *= A
    return (A^n * v)[0]

def fast(n):
    return (ev_matrix_s * diag_matrix_lambda^n * ev_matrix_s_inv * v)[0]


timeit("slow(100000)")
timeit("fast(100000)")
