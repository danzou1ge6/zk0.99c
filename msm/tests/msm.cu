#include "../src/bn254.cuh"
#include "../src/msm_bay.cuh"
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
  PointAffine *points;
  Element *scalers;
};

std::istream &
operator>>(std::istream &is, MsmProblem &msm)
{
  is >> msm.len;
  msm.scalers = new Element[msm.len];
  msm.points = new PointAffine[msm.len];
  for (u32 i = 0; i < msm.len; i++)
  {
    char _;
    is >> msm.scalers[i].n >> _ >> msm.points[i];
  }
  return is;
}

std::ostream &
operator<<(std::ostream &os, const MsmProblem &msm)
{

  for (u32 i = 0; i < msm.len; i++)
  {
    os << msm.scalers[i].n << '|' << msm.points[i] << std::endl;
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
  Point r;
  msm::run<msm::MsmConfig>((u32*)msm.scalers, (u32*)msm.points, msm.len, r);
  std::cout << r.to_affine();
  return 0;
}
