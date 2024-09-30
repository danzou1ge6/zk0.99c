use group::ff::{Field, WithSmallOrderMulGroup};
use halo2_proofs::poly::{EvaluationDomain, ExtendedLagrangeCoeff};
use halo2curves::{bn256::Fr, pasta::Fp};
use rand_core::{OsRng, SeedableRng};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use std::{marker::PhantomData, time::Instant};
use zk0d99c_ntt::gpu_extended_to_coeff;

#[test]
fn compare_with_halo2() {
    let max_k = 20;
    println!("testing with pasta Fp...");
    for k in 1..=max_k {
        println!("generating data for k = {k}...");

        let j = 4;

        let domain = EvaluationDomain::<Fp>::new(j, k);

        let data_rust: Vec<Fp> = (0..(1 << domain.extended_k()))
            .into_par_iter()
            .map(|_| Fp::random(XorShiftRng::from_rng(OsRng).unwrap()))
            .collect();
        let mut data_cuda = data_rust.clone();

        let g_coset = Fp::ZETA;
        let g_coset_inv = g_coset.square();
        let zeta = vec![g_coset_inv, g_coset];

        unsafe {
            let e_pointer = (&(data_rust, PhantomData::<ExtendedLagrangeCoeff>))
                as *const (Vec<Fp>, PhantomData<ExtendedLagrangeCoeff>)
                as *const halo2_proofs::poly::Polynomial<
                    Fp,
                    halo2_proofs::poly::ExtendedLagrangeCoeff,
                >;
            println!("testing for k = {k}:");
            let start1 = Instant::now();
            let poly_rust = (*e_pointer).clone();

            let res_rust = domain.extended_to_coeff(poly_rust);
            let time1 = start1.elapsed().as_micros();
            println!("cpu time: {time1}");

            let extended_intt_divisor = Fp::from(1 << domain.extended_k()).invert().unwrap();
            let truncate_len: usize = (1 << k) * (j - 1) as usize;

            gpu_extended_to_coeff(
                &mut data_cuda.clone(),
                domain.get_extended_omega().invert().unwrap(),
                domain.extended_k(),
                extended_intt_divisor,
                &zeta,
                truncate_len,
            )
            .unwrap();

            let start2 = Instant::now();
            match gpu_extended_to_coeff(
                &mut data_cuda,
                domain.get_extended_omega().invert().unwrap(),
                domain.extended_k(),
                extended_intt_divisor,
                &zeta,
                truncate_len,
            ) {
                Ok(_) => {}
                Err(e) => panic!("{e}"),
            };
            let time2 = start2.elapsed().as_micros();
            println!("gpu time: {time2}, {}x faster", time1 as f32 / time2 as f32);

            data_cuda
                .par_iter()
                .zip(res_rust.par_iter())
                .for_each(|(a, b)| {
                    assert_eq!(a, b);
                });
        }
    }
    println!("testing with pasta Fr...");
    for k in 1..=max_k {
        println!("generating data for k = {k}...");

        let j = 2;

        let domain = EvaluationDomain::<Fr>::new(j, k);

        let data_rust: Vec<Fr> = (0..(1 << domain.extended_k()))
            .into_par_iter()
            .map(|_| Fr::random(XorShiftRng::from_rng(OsRng).unwrap()))
            .collect();
        let mut data_cuda = data_rust.clone();

        let g_coset = Fr::ZETA;
        let g_coset_inv = g_coset.square();
        let zeta = vec![g_coset_inv, g_coset];

        unsafe {
            let e_pointer = (&(data_rust, PhantomData::<ExtendedLagrangeCoeff>))
                as *const (Vec<Fr>, PhantomData<ExtendedLagrangeCoeff>)
                as *const halo2_proofs::poly::Polynomial<
                    Fr,
                    halo2_proofs::poly::ExtendedLagrangeCoeff,
                >;
            println!("testing for k = {k}:");
            let start1 = Instant::now();
            let poly_rust = (*e_pointer).clone();

            let res_rust = domain.extended_to_coeff(poly_rust);
            let time1 = start1.elapsed().as_micros();
            println!("cpu time: {time1}");

            let extended_intt_divisor = Fr::from(1 << domain.extended_k()).invert().unwrap();
            let truncate_len: usize = (1 << k) * (j - 1) as usize;

            let start2 = Instant::now();
            match gpu_extended_to_coeff(
                &mut data_cuda,
                domain.get_extended_omega().invert().unwrap(),
                domain.extended_k(),
                extended_intt_divisor,
                &zeta,
                truncate_len,
            ) {
                Ok(_) => {}
                Err(e) => panic!("{e}"),
            };
            let time2 = start2.elapsed().as_micros();
            println!("gpu time: {time2}, {}x faster", time1 as f32 / time2 as f32);

            data_cuda
                .par_iter()
                .zip(res_rust.par_iter())
                .for_each(|(a, b)| {
                    assert_eq!(a, b);
                });
        }
    }
}
