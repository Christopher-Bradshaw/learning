#include <emscripten/bind.h>
#include <gsl/gsl_sf_bessel.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(my_module) {
    function("bessel_J0", &gsl_sf_bessel_J0);
}
