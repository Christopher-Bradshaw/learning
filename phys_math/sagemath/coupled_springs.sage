# Coupled masses on springs from 11.1 in Taylor
# The equation we are trying to solve is: M a = -K x

var("k1 k2 k3 x1 x2 a1 a2 m1 m2")
M = diagonal_matrix([m1, m2])
a = vector([a1, a2])
K = matrix([
    [k1 + k2,   k2],
    [k2,        k1+k2],
])
x = vector([x1, x2])

kev = K.eigenvectors_right()

# Using the two eigenvectors we look for solutions of the form
# x = e ^ (i*w*t) * eigenvector
# a = -w^2 x * eigenvector
# I'm pretty sure w needs to be sqrt(eigenvalue)
