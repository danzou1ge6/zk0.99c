# 编写build.rs

此处参考的是bindgen的实例代码。

实现确定路径

```rust
// This is the directory where the `c` library is located.
    let libdir_path = PathBuf::from("../../../lib")
        // Canonicalize the path as `rustc-link-search` requires an absolute
        // path.
        .canonicalize()
        .expect("cannot canonicalize path");

    // This is the path to the `c` headers file.
    let headers_path = PathBuf::from("../c_api/ntt_c_api.h");
    let headers_path_str = headers_path.to_str().expect("Path is not a valid string");
```

然后调用xmake

```rust
if !std::process::Command::new("xmake")
        .arg("build")
        .arg("cuda_ntt")
        .current_dir("../../..")
        .output()
        .expect("could not spawn `xmake`")
        .status
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not build the library");
    }
```

接着控制重新编译的条件

```rust
println!("cargo:rerun-if-changed={}", "../c_api");
println!("cargo:rerun-if-changed={}", "../../../NTT/src");
```

引入需要链接的库，需要注意被依赖的动态库需要在依赖的库后引入

```rust
    // Tell cargo to look for libraries in the specified directory
    println!("cargo:rustc-link-search={}", libdir_path.to_str().unwrap());
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
    
    println!("cargo:rustc-link-lib=static=cuda_ntt");
    
    println!("cargo:rustc-link-lib=dylib=cudart");  // 使用动态链接
    println!("cargo:rustc-link-lib=dylib=cudadevrt");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stub");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");
```

最后调用`bindgen`

```rust
    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(headers_path_str)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .clang_arg("-xc++")  // Enable C++ mode
        .clang_arg("-std=c++20")  // Specify the C++ standard
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
```

