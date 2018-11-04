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
