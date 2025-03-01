fn main() {
    #[cfg(feature = "static")]
    static_link_faiss();
    #[cfg(not(feature = "static"))]
    println!("cargo:rustc-link-lib=faiss_c");
}

#[cfg(feature = "static")]
fn static_link_faiss() {
    use std::path::PathBuf;

    let mut cfg = cmake::Config::new("faiss");

    cfg.define("FAISS_ENABLE_C_API", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define(
            "FAISS_ENABLE_GPU",
            if cfg!(feature = "gpu") { "ON" } else { "OFF" },
        )
        .define("FAISS_ENABLE_PYTHON", "OFF")
        .define("BUILD_TESTING", "OFF")
        .profile("RelWithDebInfo")
        .very_verbose(true);

    let profile = cfg.get_profile().to_owned();
    let dst = cfg.build();

    let faiss_location = dst.join("lib");

    // CMake on Windows puts the C API library in a subfolder based on the build
    // profile
    #[cfg(windows)]
    let faiss_c_location = dst.join("build\\c_api").join(profile);

    #[cfg(not(windows))]
    let faiss_c_location = dst.join("build/c_api");

    println!(
        "cargo:rustc-link-search=native={}",
        faiss_location.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        faiss_c_location.display()
    );
    println!("cargo:rustc-link-lib=static=faiss_c");
    println!("cargo:rustc-link-lib=static=faiss");

    if let Ok(oneapi_root) = std::env::var("ONEAPI_ROOT") {
        println!("using ONEAPI_ROOT={oneapi_root:?}");

        // if using Intel oneAPI (such as on Windows), we need libraries from
        // the mkl folder and the compiler folder
        let oneapi_root = PathBuf::from(oneapi_root);
        let mkl_root = oneapi_root.join("mkl").join("latest");
        let compiler_root = oneapi_root.join("compiler").join("latest");

        println!(
            "cargo:rustc-link-search=native={}",
            mkl_root.join("bin").display()
        );
        println!(
            "cargo:rustc-link-search={}",
            mkl_root.join("lib").display()
        );

        println!(
            "cargo:rustc-link-search=native={}",
            compiler_root.join("bin").display()
        );
        println!(
            "cargo:rustc-link-search={}",
            compiler_root.join("lib").display()
        );

        println!("cargo:rustc-link-lib=mkl_intel_lp64_dll");
        println!("cargo:rustc-link-lib=mkl_intel_thread_dll");
        println!("cargo:rustc-link-lib=mkl_core_dll");
        println!("cargo:rustc-link-lib=libiomp5md");
    } else if let Ok(mkl_root) = std::env::var("MKLROOT") {
        println!("using MKLROOT={mkl_root:?}");
        let mkl_root = PathBuf::from(mkl_root);

        println!(
            "cargo:rustc-link-search=native={}",
            mkl_root.join("bin").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            mkl_root.join("lib").display()
        );

        println!("cargo:rustc-link-lib=dylib=mkl_core_dll");
        println!("cargo:rustc-link-lib=dylib=mkl_intel_lp64_dll");
        println!("cargo:rustc-link-lib=dylib=mkl_intel_thread_dll");
        println!("cargo:rustc-link-lib=dylib=libiomp5md");
    } else {
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=blas");
        println!("cargo:rustc-link-lib=lapack");
    }

    println!("cargo:rerun-if-env-changed=ONEAPI_ROOT");
    println!("cargo:rerun-if-env-changed=MKLROOT");

    link_cxx();

    if cfg!(feature = "gpu") {
        let cuda_path = cuda_lib_path();
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
    }
}

#[cfg(feature = "static")]
fn link_cxx() {
    let cxx = match std::env::var("CXXSTDLIB") {
        Ok(s) if s.is_empty() => None,
        Ok(s) => Some(s),
        Err(_) => {
            let target = std::env::var("TARGET").unwrap();
            if target.contains("msvc") {
                None
            } else if target.contains("apple")
                | target.contains("freebsd")
                | target.contains("openbsd")
            {
                Some("c++".to_string())
            } else {
                Some("stdc++".to_string())
            }
        }
    };
    if let Some(cxx) = cxx {
        println!("cargo:rustc-link-lib={}", cxx);
    }
}

#[cfg(feature = "static")]
fn cuda_lib_path() -> String {
    // look for CUDA_PATH in environment,
    // then CUDA_LIB_PATH,
    // then CUDA_INCLUDE_PATH
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        return cuda_path;
    }
    if let Ok(cuda_lib_path) = std::env::var("CUDA_LIB_PATH") {
        return cuda_lib_path;
    }
    if let Ok(cuda_include_path) = std::env::var("CUDA_INCLUDE_PATH") {
        return cuda_include_path;
    }

    panic!("Could not find CUDA: environment variables `CUDA_PATH`, `CUDA_LIB_PATH`, or `CUDA_INCLUDE_PATH` must be set");
}
