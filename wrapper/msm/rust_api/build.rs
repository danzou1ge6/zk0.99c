use std::env;
use std::path::PathBuf;

fn main() {
    // This is the directory where the `c` library is located.
    let libdir_path = PathBuf::from("../../..")
        // Canonicalize the path as `rustc-link-search` requires an absolute
        // path.
        .canonicalize()
        .expect("cannot canonicalize path")
        .join("lib/");

    // This is the path to the `c` headers file.
    let headers_path = PathBuf::from("../c_api/msm_c_api.h");
    let headers_path_str = headers_path.to_str().expect("Path is not a valid string");

    if !std::process::Command::new("xmake")
        .arg("build")
        .arg("cuda_msm")
        .current_dir("../../..")
        .output()
        .expect("could not spawn `xmake`")
        .status
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not build the library");
    }

    println!("cargo:rerun-if-changed={}", "../c_api");
    println!("cargo:rerun-if-changed={}", "../../../msm/src");
    println!("cargo:rerun-if-changed={}", "../../../mont/src");
    println!("cargo:rerun-if-changed={}", libdir_path.to_str().unwrap());

    // Tell cargo to look for libraries in the specified directory
    println!("cargo:rustc-link-search={}", libdir_path.to_str().unwrap());
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");

    println!("cargo:rustc-link-lib=static=cuda_msm");

    println!("cargo:rustc-link-lib=dylib=cudart"); // 使用动态链接
    println!("cargo:rustc-link-lib=dylib=cudadevrt");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stub");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(headers_path_str)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .clang_arg("-xc++") // Enable C++ mode
        .clang_arg("-std=c++17") // Specify the C++ standard
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings!");
}
