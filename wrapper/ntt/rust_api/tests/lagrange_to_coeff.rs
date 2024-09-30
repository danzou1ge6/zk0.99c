use group::ff::Field;
use halo2_proofs::poly::EvaluationDomain;
use halo2curves::bn256::Fr;
use halo2curves::pasta::Fp;
use rand_core::OsRng;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use std::time::Instant;
use zk0d99c_ntt::gpu_intt;

#[test]
fn compare_with_halo2() {
    let max_k = 20;
    println!("testing with pasta Fp...");
    for k in 1..=max_k {
        println!("generating data for k = {k}...");
        let data_rust: Vec<Fp> = (0..(1 << k))
            .into_par_iter()
            .map(|_| Fp::random(XorShiftRng::from_rng(OsRng).unwrap()))
            .collect();
        let mut data_cuda = data_rust.clone();

        let domain = EvaluationDomain::<Fp>::new(4, k);

        let poly_rust = domain.lagrange_from_vec(data_rust);

        println!("testing for k = {k}:");
        let start1 = Instant::now();
        let poly_rust = domain.lagrange_to_coeff(poly_rust);
        let time1 = start1.elapsed().as_micros();
        println!("cpu time: {time1}");

        let ifft_divisor = Fp::from(1 << k).invert().unwrap();

        gpu_intt(
            &mut data_cuda.clone(),
            domain.get_omega_inv(),
            domain.k(),
            ifft_divisor.clone(),
        )
        .unwrap();

        let start2 = Instant::now();

        match gpu_intt(
            &mut data_cuda,
            domain.get_omega_inv(),
            domain.k(),
            ifft_divisor,
        ) {
            Ok(_) => {}
            Err(e) => panic!("{e}"),
        };

        let time2 = start2.elapsed().as_micros();
        println!("gpu time: {time2}, {}x faster", time1 as f32 / time2 as f32);

        let poly_cuda = domain.lagrange_from_vec(data_cuda);

        poly_cuda
            .par_iter()
            .zip(poly_rust.par_iter())
            .for_each(|(a, b)| {
                assert_eq!(a, b);
            });
    }
    println!("testing with pasta Fr...");
    for k in 1..=max_k {
        println!("generating data for k = {k}...");
        let data_rust: Vec<Fr> = (0..(1 << k))
            .into_par_iter()
            .map(|_| Fr::random(XorShiftRng::from_rng(OsRng).unwrap()))
            .collect();
        let mut data_cuda = data_rust.clone();

        let domain = EvaluationDomain::<Fr>::new(4, k);

        let poly_rust = domain.lagrange_from_vec(data_rust);

        println!("testing for k = {k}:");
        let start1 = Instant::now();
        let poly_rust = domain.lagrange_to_coeff(poly_rust);
        let time1 = start1.elapsed().as_micros();
        println!("cpu time: {time1}");

        let ifft_divisor = Fr::from(1 << k).invert().unwrap();

        let start2 = Instant::now();

        match gpu_intt(
            &mut data_cuda,
            domain.get_omega_inv(),
            domain.k(),
            ifft_divisor,
        ) {
            Ok(_) => {}
            Err(e) => panic!("{e}"),
        };

        let time2 = start2.elapsed().as_micros();
        println!("gpu time: {time2}, {}x faster", time1 as f32 / time2 as f32);

        let poly_cuda = domain.lagrange_from_vec(data_cuda);

        poly_cuda
            .par_iter()
            .zip(poly_rust.par_iter())
            .for_each(|(a, b)| {
                assert_eq!(a, b);
            });
    }
}
