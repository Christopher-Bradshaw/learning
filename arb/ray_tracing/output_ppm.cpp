#include <iostream>

#include "vec3.h"

int main() {
    int nx = 200, ny = 100;

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";

    for (int row = 0; row < ny; row++) {
        for (int col = 0; col < nx; col++) {
            vec3 pix(float(row) / ny, float(col) / nx, 0);
            std::cout << int(pix[0] * 255) << " " << int(pix[1] * 255) << " 0" << "\n";
        }
    }
}
