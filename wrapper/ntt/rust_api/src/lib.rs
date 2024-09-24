use std::any::type_name;
use group::{
    ff::Field,
    GroupOpsOwned, ScalarMulOwned,
};
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

pub fn gpu_ntt<Scalar: Field, G: FftGroup<Scalar>>(a: &mut [G], omega: Scalar, log_n: u32) -> Result<(), String> {
    match type_name::<G>() {
        "pasta_curves::fields::fp::Fp" => {
            let res = unsafe {
                cuda_ntt(a.as_mut_ptr() as *mut u32, (&omega) as *const Scalar as *const u32, log_n, FIELD_PASTA_CURVES_FIELDS_FP)
            };
            if res {
                Ok(())
            }
            else {
                Err(String::from("cuda failed to operate"))
            }
        },
        "halo2curves::bn256::fr::Fr" => {
            let res = unsafe {
                cuda_ntt(a.as_mut_ptr() as *mut u32, (&omega) as *const Scalar as *const u32, log_n, FIELD_HALO2CURVES_BN256_FR)
            };
            if res {
                Ok(())
            }
            else {
                Err(String::from("cuda failed to operate"))
            }
        },
        _ => Err(format!("Unsupported field type: {}", type_name::<G>())),
    }
}