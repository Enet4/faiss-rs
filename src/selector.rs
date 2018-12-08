//! Abstract Faiss ID selector
use faiss_sys::*;

/// Abstraction over IDSelectorRange and IDSelectorBatch
#[derive(Debug)]
pub struct IdSelector {
    inner: *mut FaissIDSelector,
}

impl IdSelector {
    /// Create new range selector
    pub fn range(min: idx_t, max: idx_t) -> IdSelector {
        let mut sel = FaissIDSelector_H {_unused: []};
        let _ = faiss_IDSelectorRange_new(&mut &mut sel, min, max);
        IdSelector { inner: &mut sel }
    }

    /// Create new batch selector
    pub fn batch(n: i64, indices: &idx_t) -> IdSelector {
        let mut sel = FaissIDSelector_H {_unused: []};
        let _ = faiss_IDSelectorBatch_new(&mut &mut sel, n, indices);
        IdSelector { inner: &mut sel }
    }

    /// Return the inner pointer
    pub fn inner_ptr(&self) -> *mut FaissIDSelector {
        self.inner
    }
    
}
