//! Interface and implementation to Locality-Sensitive Hashing (LSH) index type.

use super::{
    AssignSearchResult, CpuIndex, FromInnerPtr, Idx, Index, IndexImpl, NativeIndex,
    RangeSearchResult, SearchResult, TryFromInnerPtr,
};
use crate::error::{Error, Result};
use crate::faiss_try;
use crate::selector::IdSelector;
use faiss_sys::*;
use std::mem;
use std::ptr;

#[derive(Debug)]
pub struct LshIndex {
    inner: *mut FaissIndexLSH,
}

unsafe impl Send for LshIndex {}
unsafe impl Sync for LshIndex {}

impl CpuIndex for LshIndex {}

impl Drop for LshIndex {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexLSH_free(self.inner);
        }
    }
}

impl NativeIndex for LshIndex {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl FromInnerPtr for LshIndex {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        LshIndex { inner: inner_ptr }
    }
}

impl TryFromInnerPtr for LshIndex {
    fn try_from_inner_ptr(inner_ptr: *mut FaissIndex) -> Result<Self>
    where
        Self: Sized,
    {
        unsafe {
            let new_inner = faiss_IndexLSH_cast(inner_ptr);
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                Ok(LshIndex { inner: new_inner })
            }
        }
    }
}

impl LshIndex {
    /// Create a new LSH index.
    pub fn new(d: u32, nbits: u32) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexLSH_new(
                &mut inner,
                d as idx_t,
                nbits as ::std::os::raw::c_int,
            ))?;
            Ok(LshIndex { inner })
        }
    }

    /// Create a new LSH index.
    pub fn new_with_options(
        d: u32,
        nbits: u32,
        rotate_data: bool,
        train_thresholds: bool,
    ) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexLSH_new_with_options(
                &mut inner,
                d as idx_t,
                nbits as ::std::os::raw::c_int,
                rotate_data as ::std::os::raw::c_int,
                train_thresholds as ::std::os::raw::c_int,
            ))?;
            Ok(LshIndex { inner })
        }
    }

    pub fn nbits(&self) -> u32 {
        unsafe { faiss_IndexLSH_nbits(self.inner) as u32 }
    }

    pub fn rotate_data(&self) -> bool {
        unsafe { faiss_IndexLSH_rotate_data(self.inner) != 0 }
    }

    pub fn train_thresholds(&self) -> bool {
        unsafe { faiss_IndexLSH_rotate_data(self.inner) != 0 }
    }

    pub fn bytes_per_vec(&self) -> usize {
        unsafe { faiss_IndexLSH_bytes_per_vec(self.inner) as usize }
    }
}

impl_native_index!(LshIndex);

impl_native_index_clone!(LshIndex);

impl IndexImpl {
    /// Attempt a dynamic cast of an index to the LSH index type.
    #[deprecated(
        since = "0.8.0",
        note = "Non-idiomatic name, prefer `into_lsh` instead"
    )]
    pub fn as_lsh(self) -> Result<LshIndex> {
        self.into_lsh()
    }

    /// Attempt a dynamic cast of an index to the LSH index type.
    pub fn into_lsh(self) -> Result<LshIndex> {
        unsafe {
            let new_inner = faiss_IndexLSH_cast(self.inner_ptr());
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                mem::forget(self);
                Ok(LshIndex { inner: new_inner })
            }
        }
    }
}

impl_concurrent_index!(LshIndex);

#[cfg(test)]
mod tests {
    use super::LshIndex;
    use crate::error::Result;
    use crate::index::{index_factory, ConcurrentIndex, FromInnerPtr, Idx, Index, NativeIndex};
    use crate::metric::MetricType;

    const D: u32 = 8;

    #[test]
    fn index_from_cast() {
        let index = index_factory(8, "Flat", MetricType::L2).unwrap();
        let r: Result<LshIndex> = index.into_lsh();
        assert!(r.is_err());
    }

    #[test]
    fn index_search() {
        let mut index = LshIndex::new(D, 16).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
            -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16., 32.,
            25., 20., 20., 40., 15.,
        ];
        index.train(some_data).unwrap();
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = index.search(&my_query, 3).unwrap();
        assert_eq!(result.labels.len(), 3);
        assert!(result.labels.into_iter().all(Idx::is_some));
        assert_eq!(result.distances.len(), 3);
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; D as usize];
        // flat index can be used behind an immutable ref
        let result = (&index).search(&my_query, 3).unwrap();
        assert_eq!(result.labels.len(), 3);
        assert!(result.labels.into_iter().all(Idx::is_some));
        assert_eq!(result.distances.len(), 3);
        assert!(result.distances.iter().all(|x| *x > 0.));

        index.reset().unwrap();
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn index_assign() {
        let mut index = LshIndex::new(D, 16).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
            -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16., 32.,
            25., 20., 20., 40., 15.,
        ];
        index.train(some_data).unwrap();
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = index.assign(&my_query, 3).unwrap();
        assert_eq!(result.labels.len(), 3);
        assert!(result.labels.into_iter().all(Idx::is_some));

        let my_query = [100.; D as usize];
        // flat index can be used behind an immutable ref
        let result = (&index).assign(&my_query, 3).unwrap();
        assert_eq!(result.labels.len(), 3);
        assert!(result.labels.into_iter().all(Idx::is_some));

        index.reset().unwrap();
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn index_transition() {
        let index = {
            let mut index = LshIndex::new(D, 16).unwrap();
            assert_eq!(index.d(), D);
            assert_eq!(index.ntotal(), 0);
            let some_data = &[
                7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
                -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16.,
                32., 25., 20., 20., 40., 15.,
            ];
            index.train(some_data).unwrap();
            assert!(index.is_trained());
            index.add(some_data).unwrap();
            assert_eq!(index.ntotal(), 5);

            unsafe {
                let inner = index.inner_ptr();
                // forget index, rebuild it into another object
                ::std::mem::forget(index);
                LshIndex::from_inner_ptr(inner)
            }
        };
        assert!(index.is_trained());
        assert_eq!(index.ntotal(), 5);
    }
}
