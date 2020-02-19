#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub type faiss_usize = i64;

#[cfg(target_arch = "x86")]
pub type faiss_usize = i32;
