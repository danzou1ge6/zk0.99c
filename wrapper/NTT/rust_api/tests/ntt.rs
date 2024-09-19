include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::time::Instant;
use group::ff::Field;
use halo2_proofs::*;
use halo2curves::pasta::Fp;
use rand_core::OsRng;
use rayon::prelude::*;

#[test]
fn compare_with_halo2() {
    for k in 1..=20 {
        println!("generating data for k = {k}...");
        let mut data_rust: Vec<Fp> = (0..(1 << k))
            .into_par_iter()
            .map(|_| Fp::random(OsRng))
            .collect();
        let mut data_cuda: Vec<Fp> = data_rust.par_iter().map(|x| x.clone()).collect();

        let omega: Fp = Fp::random(OsRng); // would be weird if this mattered

        println!("testing for k = {k}:");
        let start1 = Instant::now();
        arithmetic::best_fft(&mut data_rust, omega, k as u32);
        let time1 = start1.elapsed().as_micros();
        println!("cpu time: {time1}");
        
        let start2 = Instant::now();
        unsafe {
            cuda_ntt(data_cuda.as_mut_ptr() as *mut u32, (&omega) as *const Fp as *const u32 , k);
        }
        let time2 = start2.elapsed().as_micros();
        println!("gpu time: {time2}, {}x faster", time1 as f32 / time2 as f32);

        data_cuda.par_iter().zip(data_rust.par_iter()).for_each(|(a, b)| {
            assert_eq!(a, b);
        });

    }
}