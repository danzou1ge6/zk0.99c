use group::ff::Field;
use group::ff::PrimeField;
use halo2_proofs::*;
use halo2curves::bn256::Fr as Scalar;
use halo2curves::bn256::G1Affine as Point;
pub use halo2curves::CurveAffine;
use rand_core::RngCore;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;
use rayon::current_thread_index;
use rayon::prelude::*;
use std::time::Instant;
use std::time::SystemTime;
use zk0d99c_msm::gpu_msm;

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

#[test]
fn compare_with_halo2() {
    let max_k = 24;
    for k in 10..=max_k {
        println!("generating data for k = {k}...");
        let bases: Vec<Point> = generate_curvepoints(k);
        let bits = [256];
        let coeffs: Vec<_> = bits
            .iter()
            .map(|b| generate_coefficients(k, *b))
            .collect();

        println!("testing for k = {k}:");
        let n: usize = 1 << k;

        let start1 = Instant::now();
        let cpu_result = arithmetic::best_multiexp(&coeffs[0][..n], &bases[..n]);
        let time1 = start1.elapsed().as_micros();
        println!("cpu time: {time1}");

        let mut gpu_result = cpu_result;


        let start2 = Instant::now();

        match gpu_msm(&coeffs[0][..n], &bases[..n], &mut gpu_result) {
            Ok(_) => {}
            Err(e) => panic!("{e}"),
        };

        let time2 = start2.elapsed().as_micros();
        println!("gpu time: {time2}, {}x faster", time1 as f32 / time2 as f32);

        let x1 = cpu_result.x * gpu_result.z;
        let y1 = cpu_result.y * gpu_result.z;
        let x2 = gpu_result.x * cpu_result.z;
        let y2 = gpu_result.y * cpu_result.z;
        // let cpu_z = cpu_result.z.invert().unwrap();
        // let x1 = cpu_result.x * cpu_z;
        // let y1 = cpu_result.y * cpu_z;
        // let gpu_z = gpu_result.z.invert().unwrap();
        // let x2 = gpu_result.x * gpu_z;
        // let y2 = gpu_result.y * gpu_z;
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }
}
