//! Interface and implementation to IVFFlat index type.

use super::*;

use crate::error::Result;
use crate::faiss_try;
use std::mem;
use std::ptr;

/// Alias for the native implementation of a flat index.
pub type IVFFlatIndex = IVFFlatIndexImpl;

/// Native implementation of a flat index.
#[derive(Debug)]
pub struct IVFFlatIndexImpl {
    inner: *mut FaissIndexIVFFlat,
}

unsafe impl Send for IVFFlatIndexImpl {}
unsafe impl Sync for IVFFlatIndexImpl {}

impl CpuIndex for IVFFlatIndexImpl {}

impl Drop for IVFFlatIndexImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexIVFFlat_free(self.inner);
        }
    }
}

impl IVFFlatIndexImpl {
    /// Create a new IVF flat index.
    pub fn new(quantizer: flat::FlatIndex, d: u32, nlist: u32, metric: MetricType) -> Result<Self> {
        unsafe {
            let metric = metric as c_uint;
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexIVFFlat_new_with_metric(
                &mut inner,
                quantizer.inner_ptr(),
                d as usize,
                nlist as usize,
                metric,
            ))?;

            mem::forget(quantizer); // own_fields default == true
            Ok(IVFFlatIndexImpl { inner })
        }
    }

    /// Create a new IVF flat index with L2 as the metric type.
    pub fn new_l2(quantizer: flat::FlatIndex, d: u32, nlist: u32) -> Result<Self> {
        IVFFlatIndexImpl::new(quantizer, d, nlist, MetricType::L2)
    }

    /// Create a new IVF flat index with IP (inner product) as the metric type.
    pub fn new_ip(quantizer: flat::FlatIndex, d: u32, nlist: u32) -> Result<Self> {
        IVFFlatIndexImpl::new(quantizer, d, nlist, MetricType::InnerProduct)
    }
}

impl NativeIndex for IVFFlatIndexImpl {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl FromInnerPtr for IVFFlatIndexImpl {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        IVFFlatIndexImpl {
            inner: inner_ptr as *mut FaissIndexIVFFlat,
        }
    }
}

impl_native_index!(IVFFlatIndex);

impl_native_index_clone!(IVFFlatIndex);

impl ConcurrentIndex for IVFFlatIndexImpl {
    fn assign(&self, query: &[f32], k: usize) -> Result<AssignSearchResult> {
        unsafe {
            let nq = query.len() / self.d() as usize;
            let mut out_labels = vec![Idx::none(); k * nq];
            faiss_try(faiss_Index_assign(
                self.inner,
                nq as idx_t,
                query.as_ptr(),
                out_labels.as_mut_ptr() as *mut _,
                k as i64,
            ))?;
            Ok(AssignSearchResult { labels: out_labels })
        }
    }
    fn search(&self, query: &[f32], k: usize) -> Result<SearchResult> {
        unsafe {
            let nq = query.len() / self.d() as usize;
            let mut distances = vec![0_f32; k * nq];
            let mut labels = vec![Idx::none(); k * nq];
            faiss_try(faiss_Index_search(
                self.inner,
                nq as idx_t,
                query.as_ptr(),
                k as idx_t,
                distances.as_mut_ptr(),
                labels.as_mut_ptr() as *mut _,
            ))?;
            Ok(SearchResult { distances, labels })
        }
    }
    fn range_search(&self, query: &[f32], radius: f32) -> Result<RangeSearchResult> {
        unsafe {
            let nq = (query.len() / self.d() as usize) as idx_t;
            let mut p_res: *mut FaissRangeSearchResult = ptr::null_mut();
            faiss_try(faiss_RangeSearchResult_new(&mut p_res, nq))?;
            faiss_try(faiss_Index_range_search(
                self.inner,
                nq,
                query.as_ptr(),
                radius,
                p_res,
            ))?;
            Ok(RangeSearchResult { inner: p_res })
        }
    }
}

#[cfg(test)]
mod tests {

    use super::IVFFlatIndex;
    use crate::index::flat::FlatIndexImpl;
    use crate::index::{ConcurrentIndex, Idx, Index};

    const D: u32 = 8;

    #[test]
    // #[ignore]
    fn index_search() {
        let q = FlatIndexImpl::new_l2(D).unwrap();
        let mut index = IVFFlatIndex::new_l2(q, D, 1).unwrap();
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
        let q = FlatIndexImpl::new_l2(D).unwrap();
        let mut index = IVFFlatIndex::new_l2(q, D, 1).unwrap();
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
}
