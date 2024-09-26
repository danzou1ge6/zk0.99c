use group::ff::Field;
use halo2_proofs::poly::EvaluationDomain;
use halo2curves::bn256::Fr;
use halo2curves::pasta::Fp;
use rand_core::OsRng;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use std::time::Instant;
use zk0d99c_ntt::gpu_coeff_to_extended;
use group::ff::WithSmallOrderMulGroup;

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

        let poly_rust = domain.coeff_from_vec(data_rust);
        let g_coset = Fp::ZETA;
        let g_coset_inv = g_coset.square();
        let zeta = vec![g_coset, g_coset_inv];

        println!("testing for k = {k}:");
        let start1 = Instant::now();
        let poly_rust = domain.coeff_to_extended(poly_rust);
        let time1 = start1.elapsed().as_micros();
        println!("cpu time: {time1}");

        gpu_coeff_to_extended(
            &mut data_cuda.clone(),
            domain.extended_len(),
            domain.get_extended_omega(),
            domain.extended_k(),
            &zeta,
        )
        .unwrap();

        let start2 = Instant::now();

        match gpu_coeff_to_extended(
            &mut data_cuda,
            domain.extended_len(),
            domain.get_extended_omega(),
            domain.extended_k(),
            &zeta,
        ) {
            Ok(_) => {}
            Err(e) => panic!("{e}"),
        };

        let time2 = start2.elapsed().as_micros();
        println!("gpu time: {time2}, {}x faster", time1 as f32 / time2 as f32);

        data_cuda
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

        let poly_rust = domain.coeff_from_vec(data_rust);
        let g_coset = Fr::ZETA;
        let g_coset_inv = g_coset.square();
        let zeta = vec![g_coset, g_coset_inv];

        println!("testing for k = {k}:");
        let start1 = Instant::now();
        let poly_rust = domain.coeff_to_extended(poly_rust);
        let time1 = start1.elapsed().as_micros();
        println!("cpu time: {time1}");

        let start2 = Instant::now();

        match gpu_coeff_to_extended(
            &mut data_cuda,
            domain.extended_len(),
            domain.get_extended_omega(),
            domain.extended_k(),
            &zeta,
        ) {
            Ok(_) => {}
            Err(e) => panic!("{e}"),
        };

        let time2 = start2.elapsed().as_micros();
        println!("gpu time: {time2}, {}x faster", time1 as f32 / time2 as f32);

        data_cuda
            .par_iter()
            .zip(poly_rust.par_iter())
            .for_each(|(a, b)| {
                assert_eq!(a, b);
            });
    }
}
