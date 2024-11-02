#pragma once

#include "../../../msm/src/bn254.cuh"
#include "../../../msm/src/msm.cuh"
#include "../../../mont/src/bn254_scalar.cuh"

#include <iostream>
#include <fstream>


using bn254::Point;
using bn254::PointAffine;
using bn254_scalar::Element;

bool cuda_msm(unsigned int len, Element* scalers, PointAffine* points, Point& res);
