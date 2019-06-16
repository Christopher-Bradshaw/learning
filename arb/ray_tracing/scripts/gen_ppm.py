import numpy as np
import ray_tracer as rt

width, height = 200, 100
a = np.zeros((width, height, 3))

a[:,10] = 1

rt.helpers.ppm.write(a, "outfile.ppm")
