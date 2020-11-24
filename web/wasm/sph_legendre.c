/* #include <gsl/gsl_sf_legendre.h> */

/* double legendre_Plm(const int l, const int m, const double x) { */
/*     return gsl_sf_legendre_Plm(l, m, x); */
/* } */

#include <gsl/gsl_sf_bessel.h>
double bessel(double x) {
    return gsl_sf_bessel_J0(x);
}
