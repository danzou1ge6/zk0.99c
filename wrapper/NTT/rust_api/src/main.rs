
extern "C" {
    fn cuda_ntt(data: *mut ::std::os::raw::c_uint, omega: *const ::std::os::raw::c_uint, log_n: ::std::os::raw::c_int);
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
