import numpy as np

# Assuming for now that the image is 2d. And that is lives in the
# z = 0 plane
def compute_image(
        camera_location,
        camera_h_half_fov, # degrees
        camera_v_half_fov,
        camera_h_pixels,
        camera_v_pixels,
        input_image,
        input_lower_left_corner_location, # of the 2d image
        input_width, # in physical units
        input_height, # in physical units
):
    assert input_lower_left_corner_location[2] == 0 # Check image lives at z = 0

    img = np.zeros((camera_h_pixels, camera_v_pixels, 3))

    for i in range(camera_h_pixels):
        h_angle = (-1 + i/camera_h_pixels) * np.radians(camera_h_half_fov)
        for j in range(camera_v_pixels):
            v_angle = (-1 + j/camera_v_pixels) * np.radians(camera_v_half_fov)

            input_x = camera_location[0] + (np.tan(h_angle) * camera_location[2])
            input_y = camera_location[1] + (np.tan(v_angle) * camera_location[2])
