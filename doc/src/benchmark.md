# Benchmark

这里记录benchmark的测试方法

## Halo2版本

-  halo2版本 halo2_proofs = { git = "https://github.com/privacyscaling-explorations/halo2.git ", tag = "v0.3.0", features =  ["circuit-params", "derive_serde"] }

-  实际部署运行的上位机zkevm版本： 在chainweaver开源平台上： https://git.chainweaver.org.cn/zkp/zkevm-circuits。（默认分支） 

## MSM算子

基准程序：

```rust
//! This benchmarks Multi Scalar Multiplication (MSM).
//! It measures `G1` from the BN256 curve.
//!
//! To run this benchmark:
//!
//!     cargo bench --bench msm
//!
//! Caveat:  The multicore benchmark assumes:
//!     1. a multi-core system
//!     2. that the `multicore` feature is enabled.  It is by default.

#[macro_use]
extern crate criterion;

use criterion::{BenchmarkId, Criterion};
use ff::{Field, PrimeField};
use halo2_proofs::arithmetic::best_multiexp;
use halo2curves::bn256::{Fr as Scalar, G1Affine as Point};
use rand_core::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;
use rayon::current_thread_index;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::time::SystemTime;

const SAMPLE_SIZE: usize = 10;
const SINGLECORE_RANGE: [u8; 6] = [3, 8, 10, 12, 14, 16];
const MULTICORE_RANGE: [u8; 9] = [3, 8, 10, 12, 14, 16, 18, 20, 22];
const SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

fn generate_curvepoints(k: u8) -> Vec<Point> {
    let n: u64 = {
        assert!(k < 64);
        1 << k
    };

    println!("Generating 2^{k} = {n} curve points..",);
    let timer = SystemTime::now();
    let bases = (0..n)
        .into_par_iter()
        .map_init(
            || {
                let mut thread_seed = SEED;
                let uniq = current_thread_index().unwrap().to_ne_bytes();
                assert!(std::mem::size_of::<usize>() == 8);
                for i in 0..uniq.len() {
                    thread_seed[i] += uniq[i];
                    thread_seed[i + 8] += uniq[i];
                }
                XorShiftRng::from_seed(thread_seed)
            },
            |rng, _| Point::random(rng),
        )
        .collect();
    let end = timer.elapsed().unwrap();
    println!(
        "Generating 2^{k} = {n} curve points took: {} sec.\n\n",
        end.as_secs()
    );
    bases
}

fn generate_coefficients(k: u8, bits: usize) -> Vec<Scalar> {
    let n: u64 = {
        assert!(k < 64);
        1 << k
    };
    let max_val: Option<u128> = match bits {
        1 => Some(1),
        8 => Some(0xff),
        16 => Some(0xffff),
        32 => Some(0xffff_ffff),
        64 => Some(0xffff_ffff_ffff_ffff),
        128 => Some(0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff),
        256 => None,
        _ => panic!("unexpected bit size {}", bits),
    };

    println!("Generating 2^{k} = {n} coefficients..",);
    let timer = SystemTime::now();
    let coeffs = (0..n)
        .into_par_iter()
        .map_init(
            || {
                let mut thread_seed = SEED;
                let uniq = current_thread_index().unwrap().to_ne_bytes();
                assert!(std::mem::size_of::<usize>() == 8);
                for i in 0..uniq.len() {
                    thread_seed[i] += uniq[i];
                    thread_seed[i + 8] += uniq[i];
                }
                XorShiftRng::from_seed(thread_seed)
            },
            |rng, _| {
                if let Some(max_val) = max_val {
                    let v_lo = rng.next_u64() as u128;
                    let v_hi = rng.next_u64() as u128;
                    let mut v = v_lo + (v_hi << 64);
                    v &= max_val; // Mask the 128bit value to get a lower number of bits
                    Scalar::from_u128(v)
                } else {
                    Scalar::random(rng)
                }
            },
        )
        .collect();
    let end = timer.elapsed().unwrap();
    println!(
        "Generating 2^{k} = {n} coefficients took: {} sec.\n\n",
        end.as_secs()
    );
    coeffs
}

fn msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm");
    let max_k = *SINGLECORE_RANGE
        .iter()
        .chain(MULTICORE_RANGE.iter())
        .max()
        .unwrap_or(&16);
    let bases: Vec<Point> = generate_curvepoints(max_k);
    let bits = [256];
    let coeffs: Vec<_> = bits
        .iter()
        .map(|b| generate_coefficients(max_k, *b))
        .collect();

    for (b_index, b) in bits.iter().enumerate() {
        // for k in SINGLECORE_RANGE {
        //     let id = format!("{b}b_{k}");
        //     group
        //         .bench_function(BenchmarkId::new("singlecore", id), |b| {
        //             assert!(k < 64);
        //             let n: usize = 1 << k;
        //             let mut acc = Point::identity().into();
        //             b.iter(|| multiexp_serial(&coeffs[b_index][..n], &bases[..n], &mut acc));
        //         })
        //         .sample_size(10);
        // }
        for k in MULTICORE_RANGE {
            let id = format!("{b}b_{k}");
            group
                .bench_function(BenchmarkId::new("multicore", id), |b| {
                    assert!(k < 64);
                    let n: usize = 1 << k;
                    b.iter(|| {
                        best_multiexp(&coeffs[b_index][..n], &bases[..n]);
                    })
                })
                .sample_size(SAMPLE_SIZE);
        }
    }
    group.finish();
}

criterion_group!(benches, msm);
criterion_main!(benches);
// criterion_group!(benches, criterion_benchmark);
// criterion_main!(benches);

```

运行指令：

```
cargo bench --package halo2_proofs --bench msm
```



## NTT算子

运算FFT的基准程序：https://github.com/privacy-scalingexplorations/halo2/blob/v0.3.0/halo2_proofs/benches/fft.rs

运行指令：

```
cargo bench --package halo2_proofs --bench fft
```

## 整体基准

运行的基准程序：https://github.com/privacyscalingexplorations/halo2/blob/v0.3.0/halo2_proofs/benches/plonk.rs 

运行指令：

```
cargo bench --package halo2_proofs --bench plonk
```

## zk-evm测试

运行指令：

```
cargo test -p zkevm-circuits --profile bench --features "evm" --no-default-features --bench benchmark_list -- super_circuit --nocapture
```

