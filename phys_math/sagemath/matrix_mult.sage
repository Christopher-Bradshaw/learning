# LU decomposition

A = Matrix([
    [2, 4, -2],
    [4, 9, -3],
    [-2, -3, 7],
])
b = vector([2, 8, 10])

# I calculated these by hand - see lecture notes
e1 = identity_matrix(3)
e1[1, 0] = -2

e2 = identity_matrix(3)
e2[2, 0] = 1

e3 = identity_matrix(3)
e3[2, 1] = -1

# Upper matrix
print(e3*e2*e1*A)

print((e3*e2*e1).inverse())
