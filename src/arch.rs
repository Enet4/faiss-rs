#[cfg(
    any(
	target_arch = "x86_64",
	target_arch = "aarch64",
	target_arch = "powerpc64",
	target_arch = "mips64",
	target_arch = "nvptx"
    )
)]
pub type faiss_usize = i64;

#[cfg(
    any(
	target_arch = "x86",
	target_arch = "arm",
	target_arch = "powerpc",
	target_arch = "mips",
	target_arch = "wasm32"
    )
)]
pub type faiss_usize = i32;
