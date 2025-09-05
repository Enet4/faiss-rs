//! This crate provides high-level bindings to Faiss, the
//! vector similarity search engine.
//!
//! # Preparing
//!
//! This crate requires Faiss and the C API to be built beforehand by the
//! developer. Please follow the instructions
//! [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md),
//! and build the dynamic library with the C API (instructions
//! [here](https://github.com/facebookresearch/faiss/blob/master/c_api/INSTALL.md))
//!
//! This will result in the dynamic library `faiss_c` ("libfaiss_c.so" in Linux),
//! which needs to be installed in a place where your system will pick up. In
//! Linux, try somewhere in the `LD_LIBRARY_PATH` environment variable, such as
//! "/usr/lib", or try adding a new path to this variable.
//!
//! ## GPU support
//!
//! In order to have GPU support, the `gpufaiss_c` library from the main project
//! needs to be built instead. Then, enable the `gpu` Cargo feature for this crate.
//!
//! ```toml
//! [dependencies]
//! "faiss" = {version = "0.12.0", features = ["gpu"]}
//! ```
//!
//! # Examples
//!
//! The [`Index`] trait is one of the center-pieces of this library. Index
//! implementations can be requested using the [`index_factory`] function:
//!
//! [`Index`]: index/trait.Index.html
//! [`index_factory`]: index/fn.index_factory.html
//!
//! ```no_run
//! use faiss::{Index, index_factory, MetricType};
//! # use faiss::error::Result;
//! # fn run() -> Result<()> {
//! let mut index = index_factory(8, "Flat", MetricType::L2)?;
//! # let my_data = unimplemented!();
//! index.add(my_data)?;
//! # let my_query = unimplemented!();
//! let result = index.search(my_query, 5)?;
//! for (i, (l, d)) in result.labels.iter()
//!     .zip(result.distances.iter())
//!     .enumerate()
//! {
//!     println!("#{}: {} (D={})", i + 1, *l, *d);
//! }
//! # Ok(())
//! # }
//! # run().unwrap()
//! ```
//!
//! With GPU support, create a [`StandardGpuResources`] and use the
//! [`into_gpu`] and [`into_cpu`] methods to move an index to and from the GPU.
//!
//! [`StandardGpuResources`]: gpu/struct.StandardGpuResources.html
//! [`into_gpu`]: index/struct.IndexImpl.html#method.into_gpu
//! [`into_cpu`]: index/gpu/struct.GpuIndexImpl.html#method.into_cpu
//!
//! ```
//! # #[cfg(feature = "gpu")]
//! use faiss::{GpuResources, StandardGpuResources, Index, index_factory, MetricType};
//! # use faiss::error::Result;
//!
//! # #[cfg(feature = "gpu")]
//! # fn run() -> Result<()> {
//! let index = index_factory(8, "Flat", MetricType::L2)?;
//! let gpu_res = StandardGpuResources::new()?;
//! let index = index.into_gpu(&gpu_res, 0)?;
//! # Ok(())
//! # }
//! # #[cfg(feature = "gpu")]
//! # run().unwrap()
//! ```
//!
//! Unless otherwise indicated, vectors are added and retrieved from the
//! library under the form of contiguous column-first slices of `f32` elements.
//!
//! Details from the official Faiss APIs still apply. Please visit
//! the [Faiss wiki](https://github.com/facebookresearch/faiss/wiki)
//! for additional guidance.
//!
#![doc(html_root_url = "https://docs.rs/faiss/0.12.0")]

#[macro_use]
mod macros;

pub mod cluster;
pub mod error;
pub mod index;
pub mod metric;
pub mod selector;
pub mod utils;
pub mod vector_transform;
pub mod search_params;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use index::flat::FlatIndex;
pub use index::id_map::IdMap;
pub use index::io::{read_index, write_index, read_index_binary, write_index_binary};
pub use index::lsh::LshIndex;
pub use index::{index_factory, index_binary_factory, ConcurrentIndex, Idx, Index};
pub use metric::MetricType;

#[cfg(feature = "gpu")]
pub use gpu::{GpuResources, StandardGpuResources};
#[cfg(feature = "gpu")]
pub use index::gpu::GpuIndexImpl;

pub(crate) fn faiss_try(code: std::os::raw::c_int) -> Result<(), crate::error::NativeError> {
    if code != 0 {
        Err(crate::error::NativeError::from_last_error(code))
    } else {
        Ok(())
    }
}
