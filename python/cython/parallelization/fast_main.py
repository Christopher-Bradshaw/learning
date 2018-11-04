import time

from engine.mandelbrot import mandelbrot, julia
from plotter import plot

start = time.time()

# output = mandelbrot(3840, 2160, 1.3, -2.6, 1.3, 100000, 5000)
output = julia(
        3840, 2160, complex(0, -0.8),
        1.3, -1.3,
        1.3,
        1000000, 50000)

print("Fast took: {}".format(time.time() - start))
plot(output)
