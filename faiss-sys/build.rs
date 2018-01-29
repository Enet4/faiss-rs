
#[cfg(feature = "gpu")]
const LIBNAME: &str = "gpufaiss_c";
#[cfg(not(feature = "gpu"))]
const LIBNAME: &str = "faiss_c";

fn main() {
    println!("cargo:rustc-link-lib={}", LIBNAME);
}
