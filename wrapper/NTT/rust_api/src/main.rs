
use group::ff::Field;
use halo2_proofs::*;
use halo2curves::pasta::Fp;
use rand_core::OsRng;

extern "C" {
    fn cuda_ntt(data: *mut ::std::os::raw::c_uint, omega: *const ::std::os::raw::c_uint, log_n: ::std::os::raw::c_int);
}

#[test]
fn compare_with_halo2() {
    for k in 1..18 {
        let mut data_rust = (0..(1 << k)).map(|_| Fp::random(OsRng)).collect::<Vec<_>>();
        let mut data_cuda = data_rust.clone();

        let omega: Fp = Fp::random(OsRng); // would be weird if this mattered

        arithmetic::best_fft(&mut data_rust, omega, k as u32);
        
        unsafe {
            cuda_ntt(data_cuda.as_mut_ptr() as *mut u32, (&omega) as *const Fp as *const u32 , k);
        }

        for i in 0..(1 << k) {
            assert_eq!(data_cuda[i], data_rust[i]);
        }
    }
}

fn main() {
    let log_n = 4;

    let omega : [u32; 8] = [5, 0, 0, 0, 0, 0, 0, 0];

    let mut data = Vec::new();
    data.resize((1 << log_n) * 8, 0);
    for i in 0..(1 << log_n) {
        data[i * 8] = i as u32;
        print!("{} ", data[i * 8]);
    }
    println!();

    unsafe {
        cuda_ntt(data.as_mut_ptr(), omega.as_ptr(), log_n);
    }

    for i in 0..(1 << log_n) {
        print!("{} ", data[i * 8]);
    }
    println!();
}
