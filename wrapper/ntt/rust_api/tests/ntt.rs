use std::time::Instant;
use group::ff::Field;
use halo2_proofs::*;
use halo2curves::pasta::Fp;
use rand_core::OsRng;
use rayon::prelude::*;
use zk0d99c_ntt::gpu_ntt;
use halo2curves::bn256::Fr;
use rand_xorshift::XorShiftRng;
use rand_core::SeedableRng;

#[test]
fn compare_with_halo2() {
    let max_k = 20;
    println!("testing with pasta Fp...");
    for k in 1..=max_k {
        println!("generating data for k = {k}...");
        let mut data_rust: Vec<Fp> = (0..(1 << k))
            .into_par_iter()
            .map(|_| Fp::random(XorShiftRng::from_rng(OsRng).unwrap()))
            .collect();
        let mut data_cuda = data_rust.clone();

        let omega = Fp::random(XorShiftRng::from_rng(OsRng).unwrap()); // would be weird if this mattered

        println!("testing for k = {k}:");
        let start1 = Instant::now();
        arithmetic::best_fft(&mut data_rust, omega, k as u32);
        let time1 = start1.elapsed().as_micros();
        println!("cpu time: {time1}");

        gpu_ntt(&mut data_cuda.clone(), omega, k as u32).unwrap();
        
        let start2 = Instant::now();

        match gpu_ntt(&mut data_cuda, omega, k as u32) {
            Ok(_) => {},
            Err(e) => panic!("{e}"),  
        };

        let time2 = start2.elapsed().as_micros();
        println!("gpu time: {time2}, {}x faster", time1 as f32 / time2 as f32);

        data_cuda.par_iter().zip(data_rust.par_iter()).for_each(|(a, b)| {
            assert_eq!(a, b);
        });

    }
    println!("testing with pasta Fr...");
    for k in 1..=max_k {
        println!("generating data for k = {k}...");
        let mut data_rust: Vec<Fr> = (0..(1 << k))
            .into_par_iter()
            .map(|_| Fr::random(XorShiftRng::from_rng(OsRng).unwrap()))
            .collect();
        let mut data_cuda = data_rust.clone();

        let omega = Fr::random(XorShiftRng::from_rng(OsRng).unwrap()); // would be weird if this mattered

        println!("testing for k = {k}:");
        let start1 = Instant::now();
        arithmetic::best_fft(&mut data_rust, omega, k as u32);
        let time1 = start1.elapsed().as_micros();
        println!("cpu time: {time1}");
        
        let start2 = Instant::now();

        match gpu_ntt(&mut data_cuda, omega, k as u32) {
            Ok(_) => {},
            Err(e) => panic!("{e}"),  
        };

        let time2 = start2.elapsed().as_micros();
        println!("gpu time: {time2}, {}x faster", time1 as f32 / time2 as f32);

        data_cuda.par_iter().zip(data_rust.par_iter()).for_each(|(a, b)| {
            assert_eq!(a, b);
        });

    }
}