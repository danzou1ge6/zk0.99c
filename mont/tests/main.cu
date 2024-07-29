#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <iostream>

#include "../src/mont.cuh"

using namespace mont256;

__global__ void bn_add(u32 *r, const u32 *a, const u32 *b,
                       const Params *p)
{
  auto na = Number::load(a);
  auto nb = Number::load(b);
  auto nr = na + nb;
  nr.store(r);
}

__global__ void bn_sub(u32 *r, const u32 *a, const u32 *b,
                       const Params *p)
{
  auto na = Number::load(a);
  auto nb = Number::load(b);
  auto nr = na - nb;
  nr.store(r);
}

__global__ void mont_add(u32 *r, const u32 *a, const u32 *b, const Params *p)
{
  Env env(*p);
  auto na = Number::load(a);
  auto nb = Number::load(b);
  auto nr = env.add_modulo(na, nb);
  nr.store(r);
}

__global__ void mont_sub(u32 *r, const u32 *a, const u32 *b, const Params *p)
{
  Env env(*p);
  auto na = Number::load(a);
  auto nb = Number::load(b);
  auto nr = env.sub_modulo(na, nb);
  nr.store(r);
}

__global__ void bn_mul(u32 *r, const u32 *a, const u32 *b, const Params *p)
{
  auto na = Number::load(a);
  auto nb = Number::load(b);
  auto nr = na * nb;
  nr.store(r);
}

__global__ void bn_square(u32 *r, const u32 *a, const u32 *b, const Params *p)
{
  auto na = Number::load(a);
  auto nr = na.square();
  nr.store(r);
}

__global__ void mont_mul(u32 *r, const u32 *a, const u32 *b, const Params *p)
{
  Env env(*p);
  auto na = Element::load(a);
  auto nb = Element::load(b);
  auto nr = env.mul(na, nb);
  nr.store(r);
}

__global__ void convert_to_mont(u32 *r, const u32 *a, const u32 *b, const Params *p)
{
  Env env(*p);
  auto na = Number::load(a);
  auto elem = env.from_number(na);
  elem.store(r);
}

__global__ void mont_neg(u32 *r, const u32 *a, const u32 *b, const Params *p)
{
  Env env(*p);
  auto na = Element::load(a);
  auto elem = env.neg(na);
  elem.store(r);
}

__global__ void mont_square(u32 *r, const u32 *a, const u32 *b, const Params *p)
{
  Env env(*p);
  auto na = Element::load(a);
  auto elem = env.square(na);
  elem.store(r);
}

__global__ void mont_pow(u32 *r, const u32 *a, const u32 *b, const Params *p)
{
  Env env(*p);
  auto na = Number::load(a);
  auto ea = env.from_number(na);
  auto nb = Number::load(b);
  auto ea_pow = env.pow(ea, nb);
  ea_pow.store(r);
}

__global__ void mont_inv(u32 *r, const u32 *a, const u32 *b, const Params *p)
{
  Env env(*p);
  auto na = Number::load(a);
  auto ea = env.from_number(na);
  auto ea_inv = env.invert(ea);
  ea_inv.store(r);
}

__global__ void bn_slr(u32 *r, const u32 *a, const u32 *b, const Params *p)
{
  auto na = Number::load(a);
  auto nr = na.slr(*b);
  nr.store(r);
}

template <u32 WORDS>
void test_mont_kernel(const u32 r[WORDS], const u32 a[WORDS],
                      const u32 b[WORDS], const Params params, void kernel(u32 *, const u32 *, const u32 *, const Params *))
{
  u32 *dr, *da, *db;
  Params *dp;
  size_t bytes = WORDS * sizeof(u32);
  cudaMalloc(&da, bytes);
  cudaMalloc(&db, bytes);
  cudaMalloc(&dr, bytes);
  cudaMalloc(&dp, sizeof(Params));
  cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dp, &params, sizeof(Params), cudaMemcpyHostToDevice);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  kernel(dr, da, db, dp);

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
                       const u32 b[WORDS], const Params params, void kernel(u32 *, const u32 *, const u32 *, const Params *))
{
  u32 *dr, *da, *db;
  Params *dp;
  size_t bytes = WORDS * sizeof(u32);
  cudaMalloc(&da, bytes);
  cudaMalloc(&db, bytes);
  cudaMalloc(&dr, bytes * 2);
  cudaMalloc(&dp, sizeof(Params));
  cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dp, &params, sizeof(Params), cudaMemcpyHostToDevice);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  kernel(dr, da, db, dp);

  err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  u32 got_r[WORDS * 2];
  cudaMemcpy(got_r, dr, bytes * 2, cudaMemcpyDeviceToHost);

  const auto n_got_r = Number2::load(got_r);
  const auto nr = Number2::load(r);
  Number n_got_r_hi, n_got_r_lo, nr_hi, nr_lo;
  n_got_r.split(n_got_r_hi, n_got_r_lo);
  nr.split(nr_hi, nr_lo);

  if (nr_lo != n_got_r_lo || nr_hi != n_got_r_hi)
  {
    FAIL("Expected\n  ", nr_hi, ", ", nr_lo, ", but got\n  ", n_got_r_hi, ", ", n_got_r_lo);
  }
}

template <u32 WORDS>
void test_host(const u32 r[WORDS], const u32 a[WORDS],
               const u32 b[WORDS], const Params params, Number f(const Number &a, const Number &b, Env &env))
{
  const auto nr = Number::load(r);
  const auto na = Number::load(a);
  const auto nb = Number::load(b);
  auto env = Env::host_new(params);
  const auto n_got_r = f(na, nb, env);
  REQUIRE(nr == n_got_r);
}

template <u32 WORDS>
void test_host2(const u32 r[WORDS * 2], const u32 a[WORDS],
                const u32 b[WORDS], const Params params, Number2 f(const Number &a, const Number &b, Env &env))
{
  const auto nr = Number2::load(r);
  const auto na = Number::load(a);
  const auto nb = Number::load(b);
  auto env = Env::host_new(params);
  const auto n_got_r = f(na, nb, env);

  Number n_got_r_hi, n_got_r_lo, nr_hi, nr_lo;
  n_got_r.split(n_got_r_hi, n_got_r_lo);
  nr.split(nr_hi, nr_lo);
  if (nr_lo != n_got_r_lo || nr_hi != n_got_r_hi)
  {
    FAIL("Expected\n  ", nr_hi, ", ", nr_lo, ", but got\n  ", n_got_r_hi, ", ", n_got_r_lo);
  }
}

#define BIG_INTEGER_CHUNKS(c7, c6, c5, c4, c3, c2, c1, c0) {c0, c1, c2, c3, c4, c5, c6, c7}
#define BIG_INTEGER_CHUNKS2(c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0) \
  {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15}

const u32 WORDS = 8;
const auto params = Params{
    // 8749054268878081845992735117657085490803352298049958388812839746200388362933
    .m = BIG_INTEGER_CHUNKS(0x1357ca0b, 0x1175f673, 0x618376f7, 0xbe5cb471, 0x29552c58, 0xfd07e66d, 0x5c09a3fc, 0x1951e2b5),
    .r_mod = BIG_INTEGER_CHUNKS(0x48abd70, 0x1d027c24, 0x0c52f56b, 0x554ad640, 0xe6acbf7b, 0x26994c72, 0x5382ac32, 0xb6d77ccf),
    .r2_mod = BIG_INTEGER_CHUNKS(0xaf6c7c1, 0x4cbccf2f, 0x2335d990, 0x8329a189, 0xd86803f4, 0x9da2f940, 0x61ee8dd3, 0x97f473f0),
    .m_prime = 2695922787};

namespace instance1
{
  // 4019478546065350814798867468142334808935292970159923177730183147609940437278
  const u32 a[WORDS] = BIG_INTEGER_CHUNKS(0x8e2f1b9, 0x74caa8b2, 0xa201f5ce, 0xdd06a772, 0x33525f1a, 0xc8794b1e, 0x460dd0e8, 0x3abe291e);
  // 379489414366113194947093122905543766951565025151396298424585213707929480463
  const u32 b[WORDS] = BIG_INTEGER_CHUNKS(0xd6c88c, 0xb2e981fc, 0x0acbd667, 0x745cc76e, 0x828cb275, 0xb3413d58, 0x35d4ba8a, 0xd2b36d0f);
  const u32 sum[WORDS] = BIG_INTEGER_CHUNKS(0x9b9ba46, 0x27b42aae, 0xaccdcc36, 0x51636ee0, 0xb5df1190, 0x7bba8876, 0x7be28b73, 0x0d71962d);
  const u32 sum_mont[WORDS] = BIG_INTEGER_CHUNKS(0x9b9ba46, 0x27b42aae, 0xaccdcc36, 0x51636ee0, 0xb5df1190, 0x7bba8876, 0x7be28b73, 0x0d71962d);
  const u32 sub[WORDS] = BIG_INTEGER_CHUNKS(0x80c292c, 0xc1e126b6, 0x97361f67, 0x68a9e003, 0xb0c5aca5, 0x15380dc6, 0x1039165d, 0x680abc0f);
  const u32 sub_mont[WORDS] = BIG_INTEGER_CHUNKS(0x80c292c, 0xc1e126b6, 0x97361f67, 0x68a9e003, 0xb0c5aca5, 0x15380dc6, 0x1039165d, 0x680abc0f);
  const u32 prod[WORDS * 2] = BIG_INTEGER_CHUNKS2(0x774ac, 0x40329d37, 0x4887cae6, 0xabcd9018, 0x5fbb6628, 0x03cbd7db, 0xc8d6bc53, 0xfd26ce8a,
                                                  0x65af4e11, 0x927fcb4e, 0x42fef4f7, 0xa5f0dbab, 0xcd81b0d6, 0x8d373fcc, 0xa68d257b, 0xc4a02ec2);
  const u32 prod_mont[WORDS] = BIG_INTEGER_CHUNKS(0xa0b4f94, 0x130a8861, 0xaabacbd5, 0x67a95bc8, 0x940e2073, 0xad9f431c, 0x6e99ad03, 0x42f2abaa);
  const u32 a_square[WORDS * 2] = BIG_INTEGER_CHUNKS2(0x4ef84b, 0x46fd949c, 0x99569b59, 0x5297ce7c, 0x6a5d7232, 0x5b118f8f, 0x85c70946, 0xf18e468c, 0xb95b8544,
                                                      0x68c79459, 0x84d592ba, 0x629083cc, 0x4a7dddf0, 0xd1355ef4, 0x1e01fe42, 0xa7229f84);
  const u32 a_square_mont[WORDS] = BIG_INTEGER_CHUNKS(0xdf5cb5a, 0x096834f9, 0x4a6de8da, 0xc658cfdf, 0xa741c620, 0x551a3a6c, 0xbfbc374c, 0xeeb5da6e);
  // a^b R mod m
  const u32 pow_mont[WORDS] = BIG_INTEGER_CHUNKS(0xbd92e68, 0x3407ccf8, 0x5be63308, 0xb5207210, 0x907bef0a, 0xf2ac6db5, 0x37e9b8d8, 0xf87662fa);
  const u32 a_inv_mont[WORDS] = BIG_INTEGER_CHUNKS(0x1e46804, 0x440e153b, 0xcfdcb1c1, 0x4c66dfe6, 0x6a669456, 0x0e18dec0, 0x8e8bfa6e, 0x48c9128c);
  // a >> 125
  const u32 a_slr125[WORDS] = BIG_INTEGER_CHUNKS(0x0, 0x0, 0x0, 0x0, 0x47178dcb, 0xa6554595, 0x100fae76, 0xe8353b91);

  TEST_CASE("Big number subtraction 1")
  {
    test_mont_kernel<WORDS>(sub, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { bn_sub<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Big number addition 1")
  {
    test_mont_kernel<WORDS>(sum, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { bn_add<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Fp addition 1")
  {
    test_mont_kernel<WORDS>(sum_mont, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { mont_add<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Fp subtraction 1")
  {
    test_mont_kernel<WORDS>(sub_mont, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { mont_sub<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Big number multiplication 1")
  {
    test_mont_kernel2<WORDS>(prod, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                             { bn_mul<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Big number square 1")
  {
    test_mont_kernel2<WORDS>(a_square, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                             { bn_square<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Big number shift logical right 1")
  {
    const u32 k[WORDS] = BIG_INTEGER_CHUNKS(0, 0, 0, 0, 0, 0, 0, 125);
    test_mont_kernel<WORDS>(a_slr125, a, k, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { bn_slr<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Montgomery multiplication 1")
  {
    // Here a, b are viewed as elements
    test_mont_kernel<WORDS>(prod_mont, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { mont_mul<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Montgomery square 1")
  {
    test_mont_kernel<WORDS>(a_square_mont, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { mont_square<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Montgomery power 1")
  {
    test_mont_kernel<WORDS>(pow_mont, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { mont_pow<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Montgomery inversion 1")
  {
    test_mont_kernel<WORDS>(a_inv_mont, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { mont_inv<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Big number addition 1 (host)")
  {
    test_host<WORDS>(sum, a, b, params, [](const Number &a, const Number &b, Env &env)
                     { return a.host_add(b); });
  }

  TEST_CASE("Big number subtraction 1 (host)")
  {
    test_host<WORDS>(sub, a, b, params, [](const Number &a, const Number &b, Env &env)
                     { return a.host_sub(b); });
  }

  TEST_CASE("Big number multiplication 1 (host)")
  {
    test_host2<WORDS>(prod, a, b, params, [](const Number &a, const Number &b, Env &env)
                      { return a.host_mul(b); });
  }

  TEST_CASE("Big number square 1 (host)")
  {
    test_host2<WORDS>(a_square, a, b, params, [](const Number &a, const Number &b, Env &env)
                      { return a.host_square(); });
  }

  TEST_CASE("Montgomery multiplication 1 (host)")
  {
    test_host<WORDS>(prod_mont, a, b, params, [](const Number &a, const Number &b, Env &env)
                     {
        // Here a, b are viewd as elements. This is a break of abstraction.
        Element ea, eb;
        ea.n = a;
        eb.n = b;
        auto er = env.host_mul(ea, eb);
        return er.n; });
  }

  TEST_CASE("Montgomery inversion 1 (host)")
  {
    test_host<WORDS>(pow_mont, a, b, params, [](const Number &a, const Number &b, Env &env)
                     { 
        auto ea = env.host_from_number(a);
        auto er = env.host_pow(ea, b);
        return er.n; });
  }
}

namespace instance2
{
  // 7601573515458430373530446595112607744880779795719120405816581808942487411900
  const u32 a[WORDS] = BIG_INTEGER_CHUNKS(0x10ce5690, 0x6268b3fc, 0x98655707, 0xf2924657, 0xc411e4e4, 0x67a23af3, 0x1d73262b, 0x21fb9cbc);
  // 4647671637959955902843015869877048340263466261541785797712932949582077663836
  const u32 b[WORDS] = BIG_INTEGER_CHUNKS(0xa467d2a, 0x424b9042, 0x34e56518, 0x96c3bce9, 0x45cd6895, 0xdf44eaa7, 0xc97d7e3b, 0x90569a5c);
  const u32 sum[WORDS] = BIG_INTEGER_CHUNKS(0x1b14d3ba, 0xa4b4443e, 0xcd4abc20, 0x89560341, 0x09df4d7a, 0x46e7259a, 0xe6f0a466, 0xb2523718);
  const u32 sum_mont[WORDS] = BIG_INTEGER_CHUNKS(0x7bd09af, 0x933e4dcb, 0x6bc74528, 0xcaf94ecf, 0xe08a2121, 0x49df3f2d, 0x8ae7006a, 0x99005463);
  const u32 sub[WORDS] = BIG_INTEGER_CHUNKS(0x687d966, 0x201d23ba, 0x637ff1ef, 0x5bce896e, 0x7e447c4e, 0x885d504b, 0x53f5a7ef, 0x91a50260);
  const u32 sub_mont[WORDS] = BIG_INTEGER_CHUNKS(0x687d966, 0x201d23ba, 0x637ff1ef, 0x5bce896e, 0x7e447c4e, 0x885d504b, 0x53f5a7ef, 0x91a50260);
  const u32 prod[WORDS * 2] = BIG_INTEGER_CHUNKS2(0xacb004, 0xd5cbdba3, 0x591dd36a, 0x7ed69756, 0xf3480eea, 0xaa33697c, 0x149a2485, 0x707be2ca,
                                                  0x7a7cb2d0, 0x8ce8f496, 0x220d4021, 0xf2d0727f, 0x5a1ce3c9, 0xe9685c76, 0xbfe043f5, 0xf9dd6b90);
  const u32 prod_mont[WORDS] = BIG_INTEGER_CHUNKS(0xa4d8112, 0xe4160087, 0x732df738, 0x92988df5, 0x358c970e, 0x34dfabf9, 0x3c172b02, 0xba57ea39);
  const u32 a_square[WORDS * 2] = BIG_INTEGER_CHUNKS2(0x11a7121, 0x79f0ca8e, 0x721e7528, 0xb80c8b72, 0x3fb1eceb, 0x3825c3e1, 0x0b0c3d7d, 0x36bbf5e6,
                                                      0xc3c79c07, 0x3f169f49, 0xb48894fd, 0xe2fef33b, 0x858ed52b, 0x42c51bd4, 0x8a794cdd, 0x309daa10);
  const u32 a_square_mont[WORDS] = BIG_INTEGER_CHUNKS(0xcc71e43, 0x5420feed, 0xca1a0867, 0xa0807fcd, 0x1948fd76, 0xb4ad5ad7, 0x93ad57f9, 0x6b7942d5);
  // a^b R mod m
  const u32 pow_mont[WORDS] = BIG_INTEGER_CHUNKS(0x10c99ac6, 0xb563f419, 0x8775e73e, 0x5ce7be6f, 0xaa2bf8d0, 0x54a445fa, 0xaa6941d2, 0x614f6e74);
  const u32 a_inv_mont[WORDS] = BIG_INTEGER_CHUNKS(0xb6ebbd1, 0xc0fd36a5, 0xad4f0ad6, 0x4c55c6f2, 0xe6eba872, 0xaf521f6e, 0x24ef45dc, 0xc51967b2);
  const u32 a_slr125[WORDS] = BIG_INTEGER_CHUNKS(0x0, 0x0, 0x0, 0x0, 0x8672b483, 0x13459fe4, 0xc32ab83f, 0x949232be);

  TEST_CASE("Big number addition 2")
  {
    test_mont_kernel<WORDS>(sum, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { bn_add<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Big number subtraction 2")
  {
    test_mont_kernel<WORDS>(sub, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { bn_sub<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Fp addition 2")
  {
    test_mont_kernel<WORDS>(sum_mont, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { mont_add<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Fp subtraction 2")
  {
    test_mont_kernel<WORDS>(sub_mont, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { mont_sub<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Big number multiplication 2")
  {
    test_mont_kernel2<WORDS>(prod, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                             { bn_mul<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Big number square 2")
  {
    test_mont_kernel2<WORDS>(a_square, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                             { bn_square<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Big number shift logical right 2")
  {
    const u32 k[WORDS] = BIG_INTEGER_CHUNKS(0, 0, 0, 0, 0, 0, 0, 125);
    test_mont_kernel<WORDS>(a_slr125, a, k, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { bn_slr<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Montgomery multiplication 2")
  {
    test_mont_kernel<WORDS>(prod_mont, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { mont_mul<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Montgomery square 2")
  {
    test_mont_kernel<WORDS>(a_square_mont, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { mont_square<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Montgomery power 2")
  {
    test_mont_kernel<WORDS>(pow_mont, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { mont_pow<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Montgomery inversion 2")
  {
    test_mont_kernel<WORDS>(a_inv_mont, a, b, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                            { mont_inv<<<1, 1>>>(r, a, b, p); });
  }

  TEST_CASE("Big number addition 2 (host)")
  {
    test_host<WORDS>(sum, a, b, params, [](const Number &a, const Number &b, Env &env)
                     { return a.host_add(b); });
  }

  TEST_CASE("Big number subtraction 2 (host)")
  {
    test_host<WORDS>(sub, a, b, params, [](const Number &a, const Number &b, Env &env)
                     { return a.host_sub(b); });
  }

  TEST_CASE("Big number multiplication 2 (host)")
  {
    test_host2<WORDS>(prod, a, b, params, [](const Number &a, const Number &b, Env &env)
                      { return a.host_mul(b); });
  }

  TEST_CASE("Big number square 2 (host)")
  {
    test_host2<WORDS>(a_square, a, b, params, [](const Number &a, const Number &b, Env &env)
                      { return a.host_square(); });
  }

  TEST_CASE("Montgomery multiplication 2 (host)")
  {
    test_host<WORDS>(prod_mont, a, b, params, [](const Number &a, const Number &b, Env &env)
                     {
        // Here a, b are viewd as elements. This is a break of abstraction.
        Element ea, eb;
        ea.n = a;
        eb.n = b;
        auto er = env.host_mul(ea, eb);
        return er.n; });
  }

  TEST_CASE("Montgomery inversion 1 (host)")
  {
    test_host<WORDS>(pow_mont, a, b, params, [](const Number &a, const Number &b, Env &env)
                     { 
        auto ea = env.host_from_number(a);
        auto er = env.host_pow(ea, b);
        return er.n; });
  }
}

TEST_CASE("Convert to Montgomery")
{
  const u32 x[WORDS] = BIG_INTEGER_CHUNKS(0xf87ecc5, 0x4ffca37c, 0x180a10f6, 0x4e41de59, 0x69467642, 0x88d72303, 0x354d1651, 0xbdd2cbc4);
  const u32 x_mont[WORDS] = BIG_INTEGER_CHUNKS(0x4226847, 0xbc719d69, 0x4de580ff, 0x9a26805e, 0xfc3ff9c8, 0xabef6da5, 0x948971bc, 0x76bb9582);

  test_mont_kernel<WORDS>(x_mont, x, x, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                          { convert_to_mont<<<1, 1>>>(r, a, b, p); });
}

TEST_CASE("Fp negation")
{
  const u32 x[WORDS] = BIG_INTEGER_CHUNKS(0, 0, 0, 0, 0, 0, 0, 0);
  test_mont_kernel<WORDS>(x, x, x, params, [](u32 *r, const u32 *a, const u32 *b, const Params *p)
                          { mont_neg<<<1, 1>>>(r, a, b, p); });
}
