var("s1 s2 k")
cov_matrix = matrix([[s1^2, k], [k, s2^2]])

print(sqrt(cov_matrix.det()))

print(cov_matrix)
print(cov_matrix.inverse())


inv_matrix = 1/(s1^2*s2^2 - k*k) * matrix([[s2^2, -k], [-k, s1^2]])
print(inv_matrix == cov_matrix.inverse())
res = cov_matrix * inv_matrix
print(res.substitute(s1 = 10, s2 = 16, k = 100))
# print(inv_matrix * cov_matrix)

# inv_matrix = matrix([
#         [-k^2/((k^2/s1 - s2)*s1^2) + 1/s1, k/((k^2/s1 - s2)*s1)],
#         [k/((k^2/s1 - s2)*s1), -1/(k^2/s1 - s2)],
# ])
# print(inv_matrix * cov_matrix)

res = cov_matrix * cov_matrix.inverse()
print(res.substitute(s1 = 10, s2 = 16, k = 100))
