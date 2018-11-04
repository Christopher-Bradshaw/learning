# Simple variable
var("x")

print(10*x)

A = matrix.identity(2) * x
print(A)

# Create multiple variables separated by a space
var("y z")

# Create variables from other variables
k = x*y*z
print(k)

# Not limited to single char vars
var("long_var")
print(long_var)
