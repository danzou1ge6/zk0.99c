#include "../../../msm/src/bn254.cuh"
#include "../../../mont/src/bn254_scalar.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

using bn254::Point;
using bn254::PointAffine;
using bn254_scalar::Element;

bool cuda_msm(unsigned int len, Element* scalers, PointAffine* points, Point& res);
