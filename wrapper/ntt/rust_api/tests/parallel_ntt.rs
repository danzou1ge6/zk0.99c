use halo2_proofs::arithmetic::parallelize;
use group::ff::{Field, WithSmallOrderMulGroup};
use halo2_proofs::poly::EvaluationDomain;
use halo2curves::pasta::Fp;
use rand_core::{OsRng, SeedableRng};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use std::time::Instant;
use zk0d99c_ntt::gpu_coeff_to_extended;

#[test]
fn compare_with_halo2() {
    let max_k = 20;
    let iterations = 20;
    println!("testing with pasta Fp...");
    for k in 1..=max_k {
        println!("generating data for k = {k}...");
        let data_rust: Vec<Fp> = (0..(1 << k))
            .into_par_iter()
            .map(|_| Fp::random(XorShiftRng::from_rng(OsRng).unwrap()))
            .collect();
        let data_cuda = data_rust.clone();

        let domain = EvaluationDomain::<Fp>::new(4, k);

        // let poly_rust = domain.coeff_from_vec(data_rust);
        let vec_rust_data = (0..iterations).into_par_iter().map(|_| data_rust.clone()).collect::<Vec<_>>();
            
        let mut vec_data_cuda = vec_rust_data.clone();

        let g_coset = Fp::ZETA;
        let g_coset_inv = g_coset.square();
        let zeta = vec![g_coset, g_coset_inv];

        println!("testing for k = {k}:");
        let start1 = Instant::now();
        
        let vec_poly_rust: Vec<_> = vec_rust_data.par_iter().map(|data| {
            let poly = domain.coeff_from_vec(data.clone());
            domain.coeff_to_extended(poly)
        }).collect();

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

        parallelize(&mut vec_data_cuda, |slice, _| {
            for it in slice.iter_mut() {
                gpu_coeff_to_extended(
                    it,
                    domain.extended_len(),
                    domain.get_extended_omega(),
                    domain.extended_k(),
                    &zeta,
                )
                .unwrap();
            }
        });

        let time2 = start2.elapsed().as_micros();
        println!("gpu time: {time2}, {}x faster", time1 as f32 / time2 as f32);

        vec_data_cuda
            .par_iter()
            .zip(vec_poly_rust.par_iter())
            .for_each(|(a, b)| {
                a.par_iter().zip(b.par_iter()).for_each(|(a, b)| {
                    assert_eq!(a, b);
                });
            });
    }
}