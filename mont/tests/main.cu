#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <iostream>

#include "../src/bn254_scalar.cuh"
using bn254_scalar::Element;

using Number = mont::Number<8>;
using Number2 = mont::Number<16>;
using mont::u32;
using mont::u64;

__global__ void bn_add(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto nb = Number::load(b);
  auto nr = na + nb;
  nr.store(r);
}

__global__ void bn_sub(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto nb = Number::load(b);
  auto nr = na - nb;
  nr.store(r);
}

__global__ void mont_add(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Element::load(a);
  auto nb = Element::load(b);
  auto nr = na + nb;
  nr.store(r);
}

__global__ void mont_sub(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Element::load(a);
  auto nb = Element::load(b);
  auto nr = na - nb;
  nr.store(r);
}

__global__ void bn_mul(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto nb = Number::load(b);
  auto nr = na * nb;
  nr.store(r);
}

__global__ void bn_square(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto nr = na.square();
  nr.store(r);
}

__global__ void mont_mul(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Element::load(a);
  auto nb = Element::load(b);
  auto nr = na * nb;
  nr.store(r);
}

__global__ void convert_to_mont(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto elem = Element::from_number(na);
  elem.store(r);
}

__global__ void convert_from_mont(u32 *r, const u32 *a, const u32 *b)
{
  auto ea = Element::load(a);
  auto na = ea.to_number();
  na.store(r);
}

__global__ void mont_neg(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Element::load(a);
  auto elem = na.neg();
  elem.store(r);
}

__global__ void mont_square(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto ea = Element::from_number(na);
  auto elem = ea * ea;
  elem.store(r);
}

__global__ void mont_pow(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto ea = Element::from_number(na);
  auto nb = Number::load(b);
  auto ea_pow = ea.pow(nb);
  ea_pow.store(r);
}

__global__ void mont_inv(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto ea = Element::from_number(na);
  auto ea_inv = ea.invert();
  ea_inv.store(r);
}

__global__ void bn_slr(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto nr = na.slr(*b);
  nr.store(r);
}

template <u32 WORDS>
void test_mont_kernel(const u32 r[WORDS], const u32 a[WORDS],
                      const u32 b[WORDS], void kernel(u32 *, const u32 *, const u32 *))
{
  u32 *dr, *da, *db;
  size_t bytes = WORDS * sizeof(u32);
  cudaMalloc(&da, bytes);
  cudaMalloc(&db, bytes);
  cudaMalloc(&dr, bytes);
  cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  kernel(dr, da, db);

  err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  u32 got_r[WORDS];
  cudaMemcpy(got_r, dr, bytes, cudaMemcpyDeviceToHost);

  const auto n_got_r = Number::load(got_r);
  const auto nr = Number::load(r);

  if (nr != n_got_r)
    FAIL("Expected ", nr, ", but got ", n_got_r);
}

template <u32 WORDS>
void test_mont_kernel2(const u32 r[WORDS], const u32 a[WORDS],
                       const u32 b[WORDS], void kernel(u32 *, const u32 *, const u32 *))
{
  u32 *dr, *da, *db;
  size_t bytes = WORDS * sizeof(u32);
  cudaMalloc(&da, bytes);
  cudaMalloc(&db, bytes);
  cudaMalloc(&dr, bytes * 2);
  cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  kernel(dr, da, db);

  err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  u32 got_r[WORDS * 2];
  cudaMemcpy(got_r, dr, bytes * 2, cudaMemcpyDeviceToHost);

  const auto n_got_r = Number2::load(got_r);
  const auto nr = Number2::load(r);

  if (n_got_r != nr)
  {
    FAIL("Expected\n  ", nr, ", but got\n  ", n_got_r);
  }
}

template <u32 WORDS>
void test_host(const u32 r[WORDS], const u32 a[WORDS],
               const u32 b[WORDS], Number f(const Number &a, const Number &b))
{
  const auto nr = Number::load(r);
  const auto na = Number::load(a);
  const auto nb = Number::load(b);
  const auto n_got_r = f(na, nb);
  REQUIRE(nr == n_got_r);
}

template <u32 WORDS>
void test_host2(const u32 r[WORDS * 2], const u32 a[WORDS],
                const u32 b[WORDS], Number2 f(const Number &a, const Number &b))
{
  const auto nr = Number2::load(r);
  const auto na = Number::load(a);
  const auto nb = Number::load(b);
  const auto n_got_r = f(na, nb);

  if (nr != n_got_r)
  {
    FAIL("Expected\n  ", nr, ", but got\n  ", n_got_r);
  }
}

const u32 WORDS = 8;
namespace instance1
{ // 20950602143661580501922242635873223609458169555687161247076590552793214934125
  const u32 a[WORDS] = BIG_INTEGER_CHUNKS8(0x2e519edf, 0x51995825, 0xd06ddd1c, 0x66a9d40d, 0x07c15275, 0x3fc9b502, 0xb284765a, 0x0bc0dc6d);
  // 8534682880701732357289068089474441931417200410519833184118768407212130707740
  const u32 b[WORDS] = BIG_INTEGER_CHUNKS8(0x12de7596, 0x5a479a08, 0x55d31bc9, 0x619f0e62, 0x4d8d3d3b, 0x1b26ec31, 0x3acab18e, 0xadfe291c);
  const u32 sum[WORDS] = BIG_INTEGER_CHUNKS8(0x41301475, 0xabe0f22e, 0x2640f8e5, 0xc848e26f, 0x554e8fb0, 0x5af0a133, 0xed4f27e8, 0xb9bf0589);
  const u32 sum_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x10cbc602, 0xcaaf5204, 0x6df0b32f, 0x46c78a11, 0xbdcd251e, 0xf27ed6a6, 0xb12e9bd1, 0xe1420842);
  const u32 sub[WORDS] = BIG_INTEGER_CHUNKS8(0x1b732948, 0xf751be1d, 0x7a9ac153, 0x050ac5aa, 0xba34153a, 0x24a2c8d1, 0x77b9c4cb, 0x5dc2b351);
  const u32 sub_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x1b732948, 0xf751be1d, 0x7a9ac153, 0x050ac5aa, 0xba34153a, 0x24a2c8d1, 0x77b9c4cb, 0x5dc2b351);
  const u32 prod[WORDS * 2] = BIG_INTEGER_CHUNKS16(0x369fd39, 0xfb2a58ae, 0x2cdb8bde, 0xc1849b9c, 0x7a5614f8, 0xd6840711, 0x6a71ca00, 0xfd4ca2f2, 0x236aa040, 0xe35a81ee, 0xd6dd3928, 0xc3119d8c, 0x58790fd2, 0x5aaf6b59, 0x821dfd81, 0x898b90ec);
  const u32 prod_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x2af695a5, 0x37fa68a8, 0x52594465, 0x1c545319, 0x2a450da4, 0x126bb45c, 0xf6618de3, 0xe05185b2);
  const u32 a_square[WORDS * 2] = BIG_INTEGER_CHUNKS16(0x8616f1e, 0x2d3d2a3f, 0xeace05c7, 0x46a6f401, 0x130a3cac, 0xcb0e8bab, 0x1feeff4f, 0x3a33eb89, 0x56e7f892, 0xd99e0fab, 0x975f45ca, 0xfdfb647b, 0x5ef25820, 0x2bcb07a0, 0xf3c19ce0, 0xbf4b8669);
  const u32 a_square_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x2f6dfb3c, 0xa2a0b555, 0xbaf95a0f, 0x780404e0, 0xc1617b88, 0xc7ce4e91, 0xc4883fa2, 0x0e6e37be);
  const u32 pow_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x2da9299a, 0x632a0883, 0xfd72a055, 0xe82a7d5b, 0xec24fd02, 0xbbc88ce3, 0x0725256c, 0x62453be5);
  const u32 a_inv_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x24b92e1c, 0xbb93836e, 0xa0302f6c, 0x306427dc, 0x7e7593af, 0x07c2de33, 0x0c839600, 0x62ef5059);
  const u32 a_slr125[WORDS] = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x1, 0x728cf6fa, 0x8ccac12e, 0x836ee8e3, 0x354ea068);
  TEST_CASE("Big number subtraction 1")
  {
    test_mont_kernel<WORDS>(sub, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { bn_sub<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number addition 1")
  {
    test_mont_kernel<WORDS>(sum, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { bn_add<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Fp addition 1")
  {
    test_mont_kernel<WORDS>(sum_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_add<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Fp subtraction 1")
  {
    test_mont_kernel<WORDS>(sub_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_sub<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number multiplication 1")
  {
    test_mont_kernel2<WORDS>(prod, a, b, [](u32 *r, const u32 *a, const u32 *b)
                             { bn_mul<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number square 1")
  {
    test_mont_kernel2<WORDS>(a_square, a, b, [](u32 *r, const u32 *a, const u32 *b)
                             { bn_square<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number shift logical right 1")
  {
    const u32 k[WORDS] = {125};
    test_mont_kernel<WORDS>(a_slr125, a, k, [](u32 *r, const u32 *a, const u32 *b)
                            { bn_slr<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery multiplication 1")
  {
    // Here a, b are viewed as elements
    test_mont_kernel<WORDS>(prod_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_mul<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery square 1")
  {
    test_mont_kernel<WORDS>(a_square_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_square<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery power 1")
  {
    test_mont_kernel<WORDS>(pow_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_pow<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery inversion 1")
  {
    test_mont_kernel<WORDS>(a_inv_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_inv<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number addition 1 (host)")
  {
    test_host<WORDS>(sum, a, b, [](const Number &a, const Number &b)
                     { return a + b; });
  }

  TEST_CASE("Big number subtraction 1 (host)")
  {
    test_host<WORDS>(sub, a, b, [](const Number &a, const Number &b)
                     { return a - b; });
  }

  TEST_CASE("Big number multiplication 1 (host)")
  {
    test_host2<WORDS>(prod, a, b, [](const Number &a, const Number &b)
                      { return a * b; });
  }

  TEST_CASE("Big number square 1 (host)")
  {
    test_host2<WORDS>(a_square, a, b, [](const Number &a, const Number &b)
                      { return a.square(); });
  }

  TEST_CASE("Montgomery multiplication 1 (host)")
  {
    test_host<WORDS>(prod_mont, a, b, [](const Number &a, const Number &b)
                     {
        // Here a, b are viewd as elements. This is a break of abstraction.
        Element ea, eb;
        ea.n = a;
        eb.n = b;
        auto er = ea * eb;
        return er.n; });
  }

  TEST_CASE("Montgomery power 1 (host)")
  {
    test_host<WORDS>(pow_mont, a, b, [](const Number &a, const Number &b)
                     { 
        auto ea = Element::from_number(a);
        auto er = ea.pow(b);
        return er.n; });
  }
}

namespace instance2
{ // 21614655952834921457016191407777979025428180395005350954076529249519191920543
  const u32 a[WORDS] = BIG_INTEGER_CHUNKS8(0x2fc97634, 0x0d6f2ef0, 0xf6d392cb, 0x72108b39, 0x58342771, 0xda355afe, 0x49eacd22, 0xeceb0f9f);
  // 14916816494250498795815920831287468076482586307060395802685550524019546555211
  const u32 b[WORDS] = BIG_INTEGER_CHUNKS8(0x20fa9e72, 0xe43908f3, 0x00627591, 0x4c4b288f, 0x0e291038, 0x4d2b5d6a, 0xb3a36f24, 0xb438834b);
  const u32 sum[WORDS] = BIG_INTEGER_CHUNKS8(0x50c414a6, 0xf1a837e3, 0xf736085c, 0xbe5bb3c8, 0x665d37aa, 0x2760b868, 0xfd8e3c47, 0xa12392ea);
  const u32 sum_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x205fc634, 0x107697ba, 0x3ee5c2a6, 0x3cda5b6a, 0xcedbcd18, 0xbeeeeddb, 0xc16db030, 0xc8a695a3);
  const u32 sub[WORDS] = BIG_INTEGER_CHUNKS8(0xeced7c1, 0x293625fd, 0xf6711d3a, 0x25c562aa, 0x4a0b1739, 0x8d09fd93, 0x96475dfe, 0x38b28c54);
  const u32 sub_mont[WORDS] = BIG_INTEGER_CHUNKS8(0xeced7c1, 0x293625fd, 0xf6711d3a, 0x25c562aa, 0x4a0b1739, 0x8d09fd93, 0x96475dfe, 0x38b28c54);
  const u32 prod[WORDS * 2] = BIG_INTEGER_CHUNKS16(0x627f717, 0xb9c7dab7, 0xe6345511, 0x038997b6, 0x932eddeb, 0xd17673a5, 0x497a67f6, 0x18ce45dd, 0xee540c42, 0x11c70613, 0xa06b647b, 0x5c4fc659, 0x5a14d6e8, 0x1c5d79fc, 0x990d363a, 0xe8a3f095);
  const u32 prod_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x20938ba8, 0xb245dc2b, 0xe4fa53ec, 0x5951491d, 0xbb1675b4, 0x8bbb457a, 0x4bdf9617, 0x8304a559);
  const u32 a_square[WORDS * 2] = BIG_INTEGER_CHUNKS16(0x8eb97f1, 0xf13ff2cc, 0x262af8a0, 0xbf4564cc, 0xfef354e8, 0x56f52db0, 0xaac7c748, 0x774b49d3, 0x4490caaa, 0x000a242a, 0x3c5a8dd0, 0x5be57f36, 0xb6580c8a, 0x409cec59, 0x7d6308de, 0xd6de04c1);
  const u32 a_square_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x1c53376, 0x556cdbf4, 0x5bf47b95, 0x25a7bbab, 0x2d2a34dd, 0xba31603d, 0x641f7760, 0x6ad8a20f);
  const u32 pow_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x2614df2c, 0x4e6462b9, 0x7e15c29c, 0x162826fb, 0x928b9dd8, 0xdb0e6ad3, 0x239a1389, 0xa36392a9);
  const u32 a_inv_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x2332a25, 0x7c6e9948, 0x823de74d, 0x2ee06219, 0x670daf76, 0xd138b4ea, 0x412edb53, 0x3a394b82);
  const u32 a_slr125[WORDS] = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x1, 0x7e4bb1a0, 0x6b797787, 0xb69c965b, 0x908459ca);
  TEST_CASE("Big number addition 2")
  {
    test_mont_kernel<WORDS>(sum, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { bn_add<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number subtraction 2")
  {
    test_mont_kernel<WORDS>(sub, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { bn_sub<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Fp addition 2")
  {
    test_mont_kernel<WORDS>(sum_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_add<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Fp subtraction 2")
  {
    test_mont_kernel<WORDS>(sub_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_sub<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number multiplication 2")
  {
    test_mont_kernel2<WORDS>(prod, a, b, [](u32 *r, const u32 *a, const u32 *b)
                             { bn_mul<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number square 2")
  {
    test_mont_kernel2<WORDS>(a_square, a, b, [](u32 *r, const u32 *a, const u32 *b)
                             { bn_square<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number shift logical right 2")
  {
    const u32 k[WORDS] = BIG_INTEGER_CHUNKS8(0, 0, 0, 0, 0, 0, 0, 125);
    test_mont_kernel<WORDS>(a_slr125, a, k, [](u32 *r, const u32 *a, const u32 *b)
                            { bn_slr<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery multiplication 2")
  {
    test_mont_kernel<WORDS>(prod_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_mul<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery square 2")
  {
    test_mont_kernel<WORDS>(a_square_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_square<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery power 2")
  {
    test_mont_kernel<WORDS>(pow_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_pow<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery inversion 2")
  {
    test_mont_kernel<WORDS>(a_inv_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_inv<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number addition 2 (host)")
  {
    test_host<WORDS>(sum, a, b, [](const Number &a, const Number &b)
                     { return a + b; });
  }

  TEST_CASE("Big number subtraction 2 (host)")
  {
    test_host<WORDS>(sub, a, b, [](const Number &a, const Number &b)
                     { return a - b; });
  }

  TEST_CASE("Big number multiplication 2 (host)")
  {
    test_host2<WORDS>(prod, a, b, [](const Number &a, const Number &b)
                      { return a * b; });
  }

  TEST_CASE("Big number square 2 (host)")
  {
    test_host2<WORDS>(a_square, a, b, [](const Number &a, const Number &b)
                      { return a.square(); });
  }

  TEST_CASE("Montgomery multiplication 2 (host)")
  {
    test_host<WORDS>(prod_mont, a, b, [](const Number &a, const Number &b)
                     {
        // Here a, b are viewd as elements. This is a break of abstraction.
        Element ea, eb;
        ea.n = a;
        eb.n = b;
        auto er = ea * eb;
        return er.n; });
  }

  TEST_CASE("Montgomery power 2 (host)")
  {
    test_host<WORDS>(pow_mont, a, b, [](const Number &a, const Number &b)
                     { 
        auto ea = Element::from_number(a);
        auto er = ea.pow(b);
        return er.n; });
  }
}

TEST_CASE("Convert to and from Montgomery")
{
  const u32 x[WORDS] = BIG_INTEGER_CHUNKS8(0x14021876, 0x4dbe5ba4, 0xabcc4ca3, 0x4be34308, 0x508480a4, 0xcb5d23b7, 0xdd6e0720, 0xb40134fb);
  const u32 x_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x31b695, 0x97ad5cd7, 0x1ad2620f, 0xd05f08cd, 0x52bf5551, 0x785f15bd, 0xecfe190a, 0x6128e2ff
);

  test_mont_kernel<WORDS>(x_mont, x, x, [](u32 *r, const u32 *a, const u32 *b)
                          { convert_to_mont<<<1, 1>>>(r, a, b); });
  test_mont_kernel<WORDS>(x, x_mont, x_mont, [](u32 *r, const u32 *a, const u32 *b)
                          { convert_from_mont<<<1, 1>>>(r, a, b); });
}

TEST_CASE("Fp negation")
{
  const u32 x[WORDS] = {0, 0, 0, 0, 0, 0, 0, 0};
  test_mont_kernel<WORDS>(x, x, x, [](u32 *r, const u32 *a, const u32 *b)
                          { mont_neg<<<1, 1>>>(r, a, b); });
}

TEST_CASE("Convert to and from Montgomery (host)")
{
  auto e = Element::host_random();
  auto n = e.to_number();
  auto e1 = Element::from_number(n);
  REQUIRE(e1 == e);
}
