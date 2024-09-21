use std::any::type_name;
use group::ff::Field;
use halo2_proofs::arithmetic::FftGroup;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));


pub fn gpu_ntt<Scalar: Field, G: FftGroup<Scalar>>(a: &mut [G], omega: Scalar, log_n: u32) -> Result<(), String> {
    match type_name::<G>() {
        "pasta_curves::fields::fp::Fp" => {
            unsafe {
                cuda_ntt(a.as_mut_ptr() as *mut u32, (&omega) as *const Scalar as *const u32, log_n, FIELD_PASTA_CURVES_FIELDS_FP);
            }
            Ok(())
        },
        "halo2curves::bn256::fr::Fr" => {
            unsafe {
                cuda_ntt(a.as_mut_ptr() as *mut u32, (&omega) as *const Scalar as *const u32, log_n, FIELD_HALO2CURVES_BN256_FR);
            }
            Ok(())
        },
        _ => return Err(format!("Unsupported field type: {}", type_name::<G>())),
    }
}