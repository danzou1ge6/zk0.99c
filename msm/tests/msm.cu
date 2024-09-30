#include "../src/bn254.cuh"
#include "../src/msm.cuh"
#include "../../mont/src/bn254_scalar.cuh"

#include <iostream>
#include <fstream>

using bn254::Point;
using bn254::PointAffine;
using bn254_scalar::Element;
using bn254_scalar::Number;
using mont::u32;
using mont::u64;

struct MsmProblem
{
  u32 len;
  u32 *scalers, *points;
};

std::istream &
operator>>(std::istream &is, MsmProblem &msm)
{
  is >> msm.len;
  msm.scalers = new u32[msm.len * Number::LIMBS];
  msm.points = new u32[msm.len * PointAffine::N_WORDS];
  for (u32 i = 0; i < msm.len; i++)
  {
    Number scaler;
    PointAffine point;
    char _;
    is >> scaler >> _ >> point;
    scaler.store(msm.scalers + i * Number::LIMBS);
    point.store(msm.points + i * PointAffine::N_WORDS);
  }
  return is;
}

std::ostream &
operator<<(std::ostream &os, const MsmProblem &msm)
{

  for (u32 i = 0; i < msm.len; i++)
  {
    auto scaler = Element::load(msm.scalers + i * Number::LIMBS);
    auto point = PointAffine::load(msm.points + i * PointAffine::N_WORDS);
    os << scaler << '|' << point << std::endl;
  }
  return os;
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "usage: <prog> input_file" << std::endl;
    return 2;
  }

  std::ifstream rf(argv[1]);
  if (!rf.is_open())
  {
    std::cout << "open file " << argv[1] << " failed" << std::endl;
    return 3;
  }

  MsmProblem msm;

  rf >> msm;
  // std::cout << msm;
  Point r;
  msm::run<msm::MsmConfig>(msm.scalers, msm.points, msm.len, r);
  std::cout << r;
  return 0;
}
