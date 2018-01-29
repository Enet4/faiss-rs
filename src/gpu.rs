//! Contents for GPU support

use faiss_sys::*;
use error::Result;
use std::ptr;

/// Common interface for GPU resources used by Faiss.
pub trait GpuResources {
    /// Obtain a raw pointer to the native GPU resources object.
    fn inner_ptr_mut(&mut self) -> *mut FaissGpuResources;

    /// Disable allocation of temporary memory; all temporary memory
    /// requests will call `cudaMalloc` / `cudaFree` at the point of use
    fn no_temp_memory(&mut self) -> Result<()>;

    /// Specify that we wish to use a certain fixed size of memory on
    /// all devices as temporary memory
    fn set_temp_memory(&mut self, size: usize) -> Result<()>;

    /// Specify that we wish to use a certain fraction of memory on
    /// all devices as temporary memory
    fn set_temp_memory_fraction(&mut self, fraction: f32) -> Result<()>;

    /// Set amount of pinned memory to allocate, for async GPU <-> CPU
    /// transfers
    fn set_pinned_memory(&mut self, size: usize) -> Result<()>;
}

/// Standard GPU resources descriptor.
pub struct StandardGpuResources {
    inner: *mut FaissGpuResources,
}

impl StandardGpuResources {

    /// Create a standard GPU resources object.
    pub fn new() -> Result<Self> {
        unsafe {
            let mut ptr = ptr::null_mut();
            faiss_try!(faiss_StandardGpuResources_new(&mut ptr));
            Ok(StandardGpuResources { inner: ptr })
        }
    }
}

impl GpuResources for StandardGpuResources {
    fn inner_ptr_mut(&mut self) -> *mut FaissGpuResources {
        self.inner
    }

    fn no_temp_memory(&mut self) -> Result<()> {
        unsafe {
            faiss_try!(faiss_StandardGpuResources_noTempMemory(self.inner));
            Ok(())
        }
    }

    fn set_temp_memory(&mut self, size: usize) -> Result<()> {
        unsafe {
            faiss_try!(faiss_StandardGpuResources_setTempMemory(self.inner, size));
            Ok(())
        }
    }


    fn set_temp_memory_fraction(&mut self, fraction: f32) -> Result<()> {
        unsafe {
            faiss_try!(faiss_StandardGpuResources_setTempMemoryFraction(self.inner, fraction));
            Ok(())
        }
    }

    fn set_pinned_memory(&mut self, size: usize) -> Result<()> {
        unsafe {
            faiss_try!(faiss_StandardGpuResources_setPinnedMemory(self.inner, size));
            Ok(())
        }
    }
}