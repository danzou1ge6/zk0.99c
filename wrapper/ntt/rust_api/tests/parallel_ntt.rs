use arithmetic::parallelize;
use group::ff::Field;
use halo2_proofs::*;
use halo2curves::pasta::Fp;
use rand_core::OsRng;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use std::time::Instant;
use zk0d99c_ntt::gpu_ntt;

#[test]
fn compare_with_halo2() {
    let max_k = 21;
    let iterations = 100;
    println!("testing with pasta Fp...");
    for k in 1..=max_k {
        println!("generating data for k = {k}...");
        let data: Vec<Fp> = (0..(1 << k))
            .into_par_iter()
            .map(|_| Fp::random(XorShiftRng::from_rng(OsRng).unwrap()))
            .collect();

        let omega = Fp::random(XorShiftRng::from_rng(OsRng).unwrap()); // would be weird if this mattered

        let mut data_rust: Vec<Vec<Fp>> = (0..iterations)
            .into_par_iter()
            .map(|_| data.clone())
            .collect();
        let mut data_cuda: Vec<Vec<Fp>> = data_rust.clone();

        println!("testing for k = {k}:");
        let start1 = Instant::now();

        parallelize(&mut data_rust, |slice, _| {
            for it in slice.iter_mut() {
                arithmetic::best_fft(it, omega, k as u32);
            }
        });

        let time1 = start1.elapsed().as_micros();
        println!("cpu time: {time1}");

        let start2 = Instant::now();

        parallelize(&mut data_cuda, |slice, _| {
            for it in slice.iter_mut() {
                gpu_ntt(it, omega, k as u32).unwrap();
            }
        });

        let time2 = start2.elapsed().as_micros();
        println!("gpu time: {time2}, {}x faster", time1 as f32 / time2 as f32);

        data_rust
            .par_iter()
            .zip(data_cuda.par_iter())
            .for_each(|(a, b)| {
                a.iter().zip(b.iter()).for_each(|(x, y)| {
                    assert_eq!(x, y);
                });
            });
    }
}
