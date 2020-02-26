#include <Magick++.h>
#include <fstream>
#include <iostream>

namespace file_output {

void write_ppm(int **res, int x_pixels, int y_pixels, int max_value) {
    std::ofstream f;
    f.open("out.ppm");
    f << "P2" << std::endl;
    f << x_pixels << " " << y_pixels << std::endl;
    f << max_value << std::endl;
    for (int x = 0; x < x_pixels; x++) {
        for (int y = 0; y < y_pixels; y++) {
            f << res[x][y] << " ";
        }
        f << std::endl;
    }
    f.close();
}

void write_jpg(int **res, int x_pixels, int y_pixels, int max_value) {
    Magick::Geometry geo = Magick::Geometry(x_pixels, y_pixels);
    Magick::Image img = Magick::Image(geo, "white");

    for (int x = 0; x < x_pixels; x++) {
        for (int y = 0; y < y_pixels; y++) {
            img.pixelColor(x, y,
                           // See http://hslpicker.com/
                           Magick::ColorHSL(
                               // Color
                               0.65,
                               // Fully saturated
                               1,
                               // Lightness
                               pow((float)res[x][y] / max_value, 0.6)));
        }
    }
    img.write("out.jpg");
}

}  // namespace file_output
