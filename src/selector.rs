//! Abstract Faiss ID selector
use crate::error::Result;
use crate::index::Idx;
use faiss_sys::*;
use std::ptr;

/// Abstraction over IDSelectorRange and IDSelectorBatch
#[derive(Debug)]
pub struct IdSelector {
    inner: *mut FaissIDSelector,
}

impl IdSelector {
    /// Create new range selector
    pub fn range(min: Idx, max: Idx) -> Result<Self> {
        let mut p_sel = ptr::null_mut();
        unsafe {
            faiss_try!(faiss_IDSelectorRange_new(
                &mut p_sel,
                min.to_native(),
                max.to_native()
            ));
        };
        Ok(IdSelector {
            inner: p_sel as *mut _,
        })
    }

    /// Create new batch selector
    pub fn batch(indices: &[Idx]) -> Result<Self> {
        let n = indices.len();
        let mut p_sel = ptr::null_mut();
        unsafe {
            faiss_try!(faiss_IDSelectorBatch_new(
                &mut p_sel,
                n,
                indices.as_ptr() as *const _
            ));
        };
        Ok(IdSelector {
            inner: p_sel as *mut _,
        })
    }

    /// Return the inner pointer
    pub fn inner_ptr(&self) -> *mut FaissIDSelector {
        self.inner
    }
}

impl Drop for IdSelector {
    fn drop(&mut self) {
        unsafe {
            faiss_IDSelector_free(self.inner);
        }
    }
}

unsafe impl Send for IdSelector {}
unsafe impl Sync for IdSelector {}
