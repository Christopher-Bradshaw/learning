#include "common.hpp"

namespace julia {
void julia(int, int, cfloat, float, float, float, float, int, int *);
void julia_gpu(int, int, cfloat, float, float, float, float, int, int *);

cfloat iter_julia(cfloat z_old, cfloat c);
}  // namespace julia
