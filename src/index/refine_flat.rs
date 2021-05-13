//! Interface and implementation to RefineFlat index type.

use super::*;

use crate::error::Result;
use crate::faiss_try;
use std::mem;
use std::ptr;

/// Alias for the native implementation of a index.
pub type RefineFlatIndex = RefineFlatIndexImpl;

/// Native implementation of a RefineFlat index.
#[derive(Debug)]
pub struct RefineFlatIndexImpl {
    inner: *mut FaissIndexRefineFlat,
}

unsafe impl Send for RefineFlatIndexImpl {}
unsafe impl Sync for RefineFlatIndexImpl {}

impl CpuIndex for RefineFlatIndexImpl {}

impl Drop for RefineFlatIndexImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexRefineFlat_free(self.inner);
        }
    }
}

impl RefineFlatIndexImpl {
    pub fn new<I: NativeIndex>(base_index: I) -> Result<Self> {
        let index = RefineFlatIndexImpl::new_helper(&base_index, true)?;
        mem::forget(base_index);
        Ok(index)
    }

    pub fn new_by_ref<I: NativeIndex>(base_index: &I) -> Result<Self> {
        RefineFlatIndexImpl::new_helper(base_index, false)
    }

    fn new_helper<I: NativeIndex>(base_index: &I, own_fields: bool) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexRefineFlat_new(
                &mut inner,
                base_index.inner_ptr(),
            ))?;
            faiss_IndexRefineFlat_set_own_fields(inner, own_fields as i32);
            Ok(RefineFlatIndexImpl { inner })
        }
    }

    pub fn set_k_factor(&mut self, kf: f32) {
        unsafe {
            faiss_IndexRefineFlat_set_k_factor(self.inner_ptr(), kf);
        }
    }

    pub fn k_factor(&self) -> f32 {
        unsafe { faiss_IndexRefineFlat_k_factor(self.inner_ptr()) }
    }
}

impl NativeIndex for RefineFlatIndexImpl {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl FromInnerPtr for RefineFlatIndexImpl {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        RefineFlatIndexImpl {
            inner: inner_ptr as *mut FaissIndexFlat,
        }
    }
}

impl_native_index!(RefineFlatIndexImpl);

impl_native_index_clone!(RefineFlatIndexImpl);

impl ConcurrentIndex for RefineFlatIndexImpl {
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
    use super::RefineFlatIndexImpl;
    use crate::index::{flat::FlatIndexImpl, ConcurrentIndex, Idx, Index};

    const D: u32 = 8;

    #[test]
    fn refine_flat_index_search() {
        let index = FlatIndexImpl::new_l2(D).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);

        let mut refine = RefineFlatIndexImpl::new(index).unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        refine.add(some_data).unwrap();
        assert_eq!(refine.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = refine.search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![2, 1, 0, 3, 4]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; D as usize];
        // flat index can be used behind an immutable ref
        let result = (&refine).search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![3, 4, 0, 1, 2]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        refine.reset().unwrap();
        assert_eq!(refine.ntotal(), 0);
    }

    #[test]
    fn refine_flat_index_ref_search() {
        let index = FlatIndexImpl::new_l2(D).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);

        let mut refine = RefineFlatIndexImpl::new_by_ref(&index).unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        refine.add(some_data).unwrap();
        assert_eq!(refine.ntotal(), 5);
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = refine.search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![2, 1, 0, 3, 4]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; D as usize];
        // flat index can be used behind an immutable ref
        let result = (&refine).search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![3, 4, 0, 1, 2]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        refine.reset().unwrap();
        assert_eq!(refine.ntotal(), 0);
        assert_eq!(index.ntotal(), 0);
    }
}
