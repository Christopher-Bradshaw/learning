from math import sin, pi

def sin_squared(x):
    return sin(x)**2

def integrate(start, stop, func, n=200000):
    dx = (stop - start) / n
    total = 0
    for i in range(n):
        total += func(start + (i+0.5)*dx)
    return total * dx


res = integrate(0, 2*pi, sin_squared)
print(res)
