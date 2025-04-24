# Accelerating zk-SNARK with GPU

A high-performance CUDA-based library for accelerating zero-knowledge proof systems on GPUs.

## Key Features

- Efficient GPU Montgomery arithmetic implementation
- Optimized Multi-scalar Multiplication (MSM) computation
- High-performance Number Theoretic Transform (NTT)
- C/Rust language bindings

## Components

- `mont` - GPU implementation of Montgomery arithmetic
- `msm` - Efficient Multi-scalar Multiplication implementation
- `ntt` - Various optimized implementations of Number Theoretic Transform
- `poly` - Polynomial operations
- `runtime` - Runtime system(deprecated)
- `wrapper` - C/Rust language binding interfaces
- `doc` - Development documentation

## Build

Build the entire project using xmake:
```sh
xmake build
```

## Test

CUDA tests using xmake:

```sh
# Montgomery arithmetic tests
xmake run bench-mont
xmake run bench-mont0
xmake run test-mont

# MSM tests
xmake run test-msm
xmake run test-bn254

# NTT tests
xmake run bench-ntt
xmake run bench-ntt-4step
xmake run bench-ntt-end2end
xmake run test-ntt-4step
xmake run test-ntt-big
xmake run test-ntt-int
xmake run test-ntt-numpy
xmake run test-ntt-parallel
xmake run test-ntt-recompute
xmake run test-ntt-transpose
xmake run transpose

# Polynomial tests
xmake run test-poly
xmake run test-poly-eval
xmake run test-poly-kate

# Runtime tests (deprecated)
xmake run test-graph
xmake run test-json
```

For Rust binding tests:
```sh
cargo test
cargo test --package zk0d99c_msm --release --test msm -- --nocapture # test msm
```
# Cost Model

You can run `utils/cost_model.py` to get the best configuration for MSM.
use `python ./utils/cost_model.py -h` to see the parameters.

The output will looks like this:
```
alpha: 8
s: 16
c: 262144
divide: 4
h: 1
```
To apply this to the code, you need change `wrapper/msm/c_api/msm_c_api.cu`
where
```
using Config = msm::MsmConfig<field-bits, s, alpha, false>;
u32 batch_size = you batch size;
u32 batch_per_run = h;
u32 parts = divide;
u32 stage_scalers = 2;
u32 stage_points = 2;
```

## Documentation

Documentation is built with [mdBook](https://github.com/rust-lang/mdBook). To view:

```sh
cd doc
mdbook build --open
```

## Acknowledgments

The Montgomery arithmetic implementation in the `mont/field` directory incorporates code from [sppark](https://github.com/supranational/sppark) (Apache-2.0 licensed). We gratefully acknowledge their contribution.
