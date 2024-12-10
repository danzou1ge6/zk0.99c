include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use group::ff::Field;
use group::{
    ff::{PrimeField},
    GroupOpsOwned, ScalarMulOwned,
};

pub use halo2curves::{CurveAffine, CurveExt};

pub fn gpu_msm<C: CurveAffine>(
    coeffs: &[C::Scalar], 
    bases: &[C],
    acc: &mut C::Curve
) -> Result<(), String> {
    let len = coeffs.len();
    // let mut res = vec![0u32; 16]; // 假设 PointAffine 有 16 个 u32 值 (8 对应 x 和 8 对应 y)

    // 将 scalers 和 points 转换为指针
    let coeffs1: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();
    let scalers_ptr = coeffs1.as_ptr() as *const u32;
    let points_ptr = bases.as_ptr() as *const u32;

    let acc_ptr = acc as *mut C::Curve as *mut u32;

    // 调用 CUDA 函数
    let success = unsafe {
        cuda_msm(len as u32, scalers_ptr, points_ptr, acc_ptr)
    };

    if !success {
        return Err("Failed to execute cuda_msm".to_string());
    }

    Ok(())
}