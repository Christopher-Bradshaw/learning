import numpy as np
from .ray_tracer import compute_image

class TestComputeImage():
    def test_compute_image(self):
        width, height = 200, 100
        img = np.zeros((width, height, 3))
        img[:,10] = 1

        compute_image(
                np.array([0, 0, 1]),
                30,
                15,
                200,
                100,
                img,
                np.array([0, 0, 0]),
                2,
                1,
        )

