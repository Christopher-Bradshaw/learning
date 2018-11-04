import time
import numpy as np

from plotter import plot


def naive_mandelbrot(height, width, img_max_x, img_max_y, max_iter, max_value):
    arr = np.zeros((height, width), np.uint8)

    def complex_from_coords(real, imag):
        return complex(
                (real - width/2)/width * img_max_x,
                (imag - height/2)/height* img_max_y,
        )


    def mandelbrot(c, max_iter, max_value):
        z = 0
        for k in range(max_iter):
            z = z*z + c
            if abs(z) > max_value:
                return k
        return max_iter

    for i in range(height):
        for j in range(width):
            arr[i, j] = mandelbrot(complex_from_coords(j, i), max_iter, max_value)

    return arr


start = time.time()
output = naive_mandelbrot(256, 256, 4, 4, 100, 10)
print("Naive took: {}".format(time.time() - start))

plot(output)
# arr = (255 * (arr.astype(np.float32) / np.max(arr))).astype(np.uint8)

# img = Image.fromarray(arr, "L")
# img.save('my.tiff')
# img.show()
