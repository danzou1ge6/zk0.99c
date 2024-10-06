use group::{
    ff::{Field, WithSmallOrderMulGroup},
    GroupOpsOwned, ScalarMulOwned,
};
use std::{
    // alloc::{ alloc, Layout},
    any::type_name,
    // mem::size_of,
    // os::raw::c_void,
    // ptr::{addr_of_mut, null, null_mut},
    ptr::null,
};

// use zk0d99c_cuda_api::{cpp_free, cuda_device_to_host_sync, cuda_free, cuda_unregister};
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Copyed from halo2_proof
/// This represents an element of a group with basic operations that can be
/// performed. This allows an FFT implementation (for example) to operate
/// generically over either a field or elliptic curve group.
pub trait FftGroup<Scalar: Field>:
    Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>
{
}

impl<T, Scalar> FftGroup<Scalar> for T
where
    Scalar: Field,
    T: Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>,
{
}

pub fn gpu_ntt<Scalar: Field, G: FftGroup<Scalar>>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
) -> Result<(), String> {
    match type_name::<G>() {
        "pasta_curves::fields::fp::Fp" => {
            let res = unsafe {
                cuda_ntt(
                    a.as_mut_ptr() as *mut u32,
                    (&omega) as *const Scalar as *const u32,
                    log_n,
                    FIELD_PASTA_CURVES_FIELDS_FP,
                    false,
                    false,
                    null(),
                    null(),
                    0,
                )
            };
            if res {
                Ok(())
            } else {
                Err(String::from("cuda failed to operate"))
            }
        }
        "halo2curves::bn256::fr::Fr" => {
            let res = unsafe {
                cuda_ntt(
                    a.as_mut_ptr() as *mut u32,
                    (&omega) as *const Scalar as *const u32,
                    log_n,
                    FIELD_HALO2CURVES_BN256_FR,
                    false,
                    false,
                    null(),
                    null(),
                    0,
                )
            };
            if res {
                Ok(())
            } else {
                Err(String::from("cuda failed to operate"))
            }
        }
        _ => Err(format!("Unsupported field type: {}", type_name::<G>())),
    }
}

pub fn gpu_intt<F>(
    values: &mut Vec<F>,
    omega_inv: F,
    log_n: u32,
    intt_divisor: F,
) -> Result<(), String>
where
    F: WithSmallOrderMulGroup<3>,
{
    match type_name::<F>() {
        "pasta_curves::fields::fp::Fp" => {
            let res = unsafe {
                cuda_ntt(
                    values.as_mut_ptr() as *mut u32,
                    (&omega_inv) as *const F as *const u32,
                    log_n,
                    FIELD_PASTA_CURVES_FIELDS_FP,
                    true,
                    false,
                    (&intt_divisor) as *const F as *const u32,
                    null(),
                    0,
                )
            };
            if res {
                Ok(())
            } else {
                Err(String::from("cuda failed to operate"))
            }
        }
        "halo2curves::bn256::fr::Fr" => {
            let res = unsafe {
                cuda_ntt(
                    values.as_mut_ptr() as *mut u32,
                    (&omega_inv) as *const F as *const u32,
                    log_n,
                    FIELD_HALO2CURVES_BN256_FR,
                    true,
                    false,
                    (&intt_divisor) as *const F as *const u32,
                    null(),
                    0,
                )
            };
            if res {
                Ok(())
            } else {
                Err(String::from("cuda failed to operate"))
            }
        }
        _ => Err(format!("Unsupported field type: {}", type_name::<F>())),
    }
}

// this is the version that try to hide the host memory allocation, but it's not working

// pub fn gpu_coeff_to_extended<F>(
//     values: &mut Vec<F>,
//     extended_len: usize,
//     extended_omega: F,
//     extended_k: u32,
//     zeta: &[F],
// ) -> Result<(), String>
// where
//     F: WithSmallOrderMulGroup<3>,
// {
//     // values.resize(extended_len, F::ZERO);
//     let mut dev_ptr: *mut u32 = null_mut();
//     let mut stream: *mut c_void = null_mut();
//     match type_name::<F>() {
//         "pasta_curves::fields::fp::Fp" => {
//             let res = unsafe {
//                 cuda_coeff_to_extended(
//                     values.as_mut_ptr() as *mut u32,
//                     (&extended_omega) as *const F as *const u32,
//                     extended_k,
//                     FIELD_PASTA_CURVES_FIELDS_FP,
//                     zeta.as_ptr() as *const u32,
//                     addr_of_mut!(dev_ptr),
//                     values.len().try_into().unwrap(),
//                     addr_of_mut!(stream),
//                 )
//             };
//             if !res {
//                 return Err(String::from("cuda failed to operate"));
//             }
//         }
//         "halo2curves::bn256::fr::Fr" => {
//             let res = unsafe {
//                 cuda_coeff_to_extended(
//                     values.as_mut_ptr() as *mut u32,
//                     (&extended_omega) as *const F as *const u32,
//                     extended_k,
//                     FIELD_HALO2CURVES_BN256_FR,
//                     zeta.as_ptr() as *const u32,
//                     addr_of_mut!(dev_ptr),
//                     values.len().try_into().unwrap(),
//                     addr_of_mut!(stream),
//                 )
//             };
//             if !res {
//                 return Err(String::from("cuda failed to operate"));
//             }
//         }
//         _ => return Err(format!("Unsupported field type: {}", type_name::<F>())),
//     };
//     let buffer: *mut u8;
//     unsafe {
//         buffer = alloc(Layout::array::<F>(extended_len).unwrap());
//     }

//     let res = unsafe {
//         cuda_device_to_host_sync(
//             buffer as *mut c_void,
//             dev_ptr as *const c_void,
//             (extended_len * size_of::<F>()).try_into().unwrap(),
//             stream,
//         )
//     };

//     if !res {
//         return Err(String::from("cuda failed to operate"));
//     }
//     unsafe {
//         cuda_free(dev_ptr as *mut c_void, stream);
//         cpp_free(stream);
//         cuda_unregister(values.as_mut_ptr() as *mut c_void);
//     }
//     *values = unsafe { Vec::<F>::from_raw_parts(buffer as *mut F, extended_len, extended_len) };
//     Ok(())
// }

pub fn gpu_coeff_to_extended<F>(
    values: &mut Vec<F>,
    extended_len: usize,
    extended_omega: F,
    extended_k: u32,
    zeta: &[F],
) -> Result<(), String>
where
    F: WithSmallOrderMulGroup<3>,
{
    let start_len = values.len();
    println!("start_len: {}", start_len);
    values.resize(extended_len, F::ZERO);
    match type_name::<F>() {
        "pasta_curves::fields::fp::Fp" => {
            let res = unsafe {
                cuda_ntt(
                    values.as_mut_ptr() as *mut u32,
                    (&extended_omega) as *const F as *const u32,
                    extended_k,
                    FIELD_PASTA_CURVES_FIELDS_FP,
                    false,
                    true,
                    null(),
                    zeta.as_ptr() as *const u32,
                    start_len.try_into().unwrap(),
                )
            };
            if res {
                Ok(())
            } else {
                Err(String::from("cuda failed to operate"))
            }
        }
        "halo2curves::bn256::fr::Fr" => {
            let res = unsafe {
                cuda_ntt(
                    values.as_mut_ptr() as *mut u32,
                    (&extended_omega) as *const F as *const u32,
                    extended_k,
                    FIELD_HALO2CURVES_BN256_FR,
                    false,
                    true,
                    null(),
                    zeta.as_ptr() as *const u32,
                    start_len.try_into().unwrap(),
                )
            };
            if res {
                Ok(())
            } else {
                Err(String::from("cuda failed to operate"))
            }
        }
        _ => Err(format!("Unsupported field type: {}", type_name::<F>())),
    }
}

pub fn gpu_extended_to_coeff<F>(
    values: &mut Vec<F>,
    extended_omega_inv: F,
    extended_k: u32,
    extended_intt_divisor: F,
    zeta: &[F],
    truncate_len: usize,
) -> Result<(), String>
where
    F: WithSmallOrderMulGroup<3>,
{
    match type_name::<F>() {
        "pasta_curves::fields::fp::Fp" => {
            let res = unsafe {
                cuda_ntt(
                    values.as_mut_ptr() as *mut u32,
                    (&extended_omega_inv) as *const F as *const u32,
                    extended_k,
                    FIELD_PASTA_CURVES_FIELDS_FP,
                    true,
                    true,
                    (&extended_intt_divisor) as *const F as *const u32,
                    zeta.as_ptr() as *const u32,
                    0,
                )
            };
            if res {
                values.truncate(truncate_len);
                Ok(())
            } else {
                Err(String::from("cuda failed to operate"))
            }
        }
        "halo2curves::bn256::fr::Fr" => {
            let res = unsafe {
                cuda_ntt(
                    values.as_mut_ptr() as *mut u32,
                    (&extended_omega_inv) as *const F as *const u32,
                    extended_k,
                    FIELD_HALO2CURVES_BN256_FR,
                    true,
                    true,
                    (&extended_intt_divisor) as *const F as *const u32,
                    zeta.as_ptr() as *const u32,
                    0,
                )
            };
            if res {
                values.truncate(truncate_len);
                Ok(())
            } else {
                Err(String::from("cuda failed to operate"))
            }
        }
        _ => Err(format!("Unsupported field type: {}", type_name::<F>())),
    }
}
