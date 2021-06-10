//! Contents for GPU support

use crate::error::Result;
use crate::faiss_try;
use faiss_sys::*;
use std::ptr;

/// Common interface for GPU resources used by Faiss.
pub trait GpuResources {
    /// Obtain a raw pointer to the native GPU resources object.
    fn inner_ptr(&self) -> *mut FaissGpuResourcesProvider;

    /// Disable allocation of temporary memory; all temporary memory
    /// requests will call `cudaMalloc` / `cudaFree` at the point of use
    fn no_temp_memory(&mut self) -> Result<()>;

    /// Specify that we wish to use a certain fixed size of memory on
    /// all devices as temporary memory
    fn set_temp_memory(&mut self, size: usize) -> Result<()>;

    /// Set amount of pinned memory to allocate, for async GPU <-> CPU
    /// transfers
    fn set_pinned_memory(&mut self, size: usize) -> Result<()>;
}

/// Common interface for a GPU resource provider.
pub trait GpuResourcesProvider {
    /// Obtain a raw pointer to the native GPU resource provider object.
    fn inner_ptr(&self) -> *mut FaissGpuResourcesProvider;
}

/// Standard GPU resources descriptor.
///
/// # Examples
///
/// GPU resources are meant to be passed to an index implementation's
/// [`into_gpu`] or [`to_gpu`] methods.
///
/// [`to_gpu`]: ../index/struct.IndexImpl.html#method.to_gpu
/// [`into_gpu`]: ../index/struct.IndexImpl.html#method.into_gpu
///
/// ```
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// use faiss::{StandardGpuResources, MetricType};
/// use faiss::index::flat::FlatIndex;
///
/// let gpu = StandardGpuResources::new()?;
/// let index = FlatIndex::new(64, MetricType::L2)?;
/// let gpu_index = index.into_gpu(&gpu, 0)?;
/// # Ok(())
/// # }
/// # run().unwrap();
/// ```
///
/// Since GPU implementations are not thread-safe, attempting to use the GPU
/// resources from another thread is not allowed.
///
/// ```compile_fail
/// use faiss::{GpuResources, StandardGpuResources};
/// use faiss::index::flat::FlatIndex;
/// use std::sync::Arc;
/// use std::thread;
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let gpu = Arc::new(StandardGpuResources::new()?);
/// let gpu_rc = gpu.clone();
/// thread::spawn(move || {
///     let index = FlatIndex::new_l2(64)?;
///     let gpu_index = index.into_gpu(&*gpu_rc, 0)?; // will not compile
///     Ok(())
/// });
/// # Ok(())
/// # }
/// # run().unwrap();
/// ```
///
/// Other than that, indexes can share the same GPU resources, so long as
/// neither of them cross any thread boundaries.
///
/// ```
/// use faiss::{GpuResources, StandardGpuResources, MetricType, index_factory};
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let mut gpu = StandardGpuResources::new()?;
/// let index1 = index_factory(64, "Flat", MetricType::L2)?
///     .into_gpu(&gpu, 0)?;
/// let index2 = index_factory(32, "Flat", MetricType::InnerProduct)?
///     .into_gpu(&gpu, 0)?;
/// # Ok(())
/// # }
/// # run().unwrap();
/// ```
///
#[derive(Debug)]
pub struct StandardGpuResources {
    inner: *mut FaissGpuResourcesProvider,
}

// Deliberately _not_ Sync!
unsafe impl Send for StandardGpuResources {}

impl StandardGpuResources {
    /// Create a standard GPU resources object.
    pub fn new() -> Result<Self> {
        unsafe {
            let mut ptr = ptr::null_mut();
            faiss_try(faiss_StandardGpuResources_new(&mut ptr))?;
            Ok(StandardGpuResources { inner: ptr })
        }
    }
}

impl GpuResourcesProvider for StandardGpuResources {
    fn inner_ptr(&self) -> *mut FaissGpuResourcesProvider {
        self.inner as *mut _
    }
}

impl GpuResources for StandardGpuResources {
    fn inner_ptr(&self) -> *mut FaissGpuResourcesProvider {
        self.inner
    }

    fn no_temp_memory(&mut self) -> Result<()> {
        unsafe {
            faiss_try(faiss_StandardGpuResources_noTempMemory(self.inner))?;
            Ok(())
        }
    }

    fn set_temp_memory(&mut self, size: usize) -> Result<()> {
        unsafe {
            faiss_try(faiss_StandardGpuResources_setTempMemory(self.inner, size))?;
            Ok(())
        }
    }

    fn set_pinned_memory(&mut self, size: usize) -> Result<()> {
        unsafe {
            faiss_try(faiss_StandardGpuResources_setPinnedMemory(self.inner, size))?;
            Ok(())
        }
    }
}

impl<'g> GpuResources for &'g mut StandardGpuResources {
    fn inner_ptr(&self) -> *mut FaissGpuResourcesProvider {
        self.inner
    }

    fn no_temp_memory(&mut self) -> Result<()> {
        (**self).no_temp_memory()
    }

    fn set_temp_memory(&mut self, size: usize) -> Result<()> {
        (**self).set_temp_memory(size)
    }

    fn set_pinned_memory(&mut self, size: usize) -> Result<()> {
        (**self).set_pinned_memory(size)
    }
}

#[cfg(test)]
mod tests {
    use super::StandardGpuResources;

    #[test]
    fn smoke_detector() {
        StandardGpuResources::new().unwrap();
    }
}
