use faiss_sys::*;
use super::{CpuIndex, NativeIndex};
use error::Result;
use std::ptr;

#[derive(Debug)]
pub struct LshIndex {
    inner: *mut FaissIndexLSH,
}

unsafe impl Send for LshIndex {}
unsafe impl Sync for LshIndex {}

impl CpuIndex for LshIndex {}

impl NativeIndex for LshIndex {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl LshIndex {
    pub fn new(d: u32, nbits: u32) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try!(faiss_IndexLSH_new(
                &mut inner,
                d as idx_t,
                nbits as ::std::os::raw::c_int,
            ));
            Ok(LshIndex { inner })
        }
    }

    pub fn new_with_options(d: u32, nbits: u32, rotate_data: bool, train_thresholds: bool) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try!(faiss_IndexLSH_new_with_options(
                &mut inner,
                d as idx_t,
                nbits as ::std::os::raw::c_int,
                rotate_data as ::std::os::raw::c_int,
                train_thresholds as ::std::os::raw::c_int,
            ));
            Ok(LshIndex { inner })
        }
    }
}

impl_native_index!(LshIndex);

// TODO tests
#[cfg(test)]
mod tests {

}
