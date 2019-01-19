# Matrix docs: http://doc.sagemath.org/html/en/reference/matrices/sage/matrix/constructor.html

A = matrix([[1, 0], [0, 1]])
print(A)

# Identity matrixes can also easiy be created
assert A == matrix.identity(2)

# Scalar ops
print(A * 2)
print(A + 2)

# Matrix-vector ops
B = matrix([[2, 3], [4, 5]])
v = vector([2, 3])
print(B*v)

# Matrix ops
print(B * B)
print(B.inverse())
print(B.det()) # or .determinant()


# Eigen
var("a b")
H = matrix([[8*a^2*b + 2, -4 * a * b], [-4 * a * b, 2 * b]])
print(H.substitute(a=1, b=100).eigenvalues())
