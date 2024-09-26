use group::{
    ff::{Field, WithSmallOrderMulGroup},
    GroupOpsOwned, ScalarMulOwned,
};
use std::{any::type_name, ptr::null};
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
