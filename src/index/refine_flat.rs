//! Interface and implementation to RefineFlat index type.

use super::*;

use crate::error::Result;
use crate::faiss_try;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_int;
use std::ptr;

/// Alias for the native implementation of a index.
pub type RefineFlatIndex<BI> = RefineFlatIndexImpl<BI>;

/// Native implementation of a RefineFlat index.
#[derive(Debug)]
pub struct RefineFlatIndexImpl<BI> {
    inner: *mut FaissIndexRefineFlat,
    base_index: PhantomData<BI>,
}

unsafe impl<BI: Send> Send for RefineFlatIndexImpl<BI> {}
unsafe impl<BI: Sync> Sync for RefineFlatIndexImpl<BI> {}

impl<BI: CpuIndex> CpuIndex for RefineFlatIndexImpl<BI> {}

impl<BI> Drop for RefineFlatIndexImpl<BI> {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexRefineFlat_free(self.inner);
        }
    }
}

impl<BI: NativeIndex> RefineFlatIndexImpl<BI> {
    pub fn new(base_index: BI) -> Result<Self> {
        let index = RefineFlatIndexImpl::new_helper(&base_index, true)?;
        mem::forget(base_index);
        Ok(index)
    }

    fn new_helper<I: NativeIndex>(base_index: &I, own_fields: bool) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexRefineFlat_new(
                &mut inner,
                base_index.inner_ptr(),
            ))?;
            faiss_IndexRefineFlat_set_own_fields(inner, c_int::from(own_fields));
            Ok(RefineFlatIndexImpl {
                inner,
                base_index: PhantomData,
            })
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

impl<BI> NativeIndex for RefineFlatIndexImpl<BI> {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl<BI> FromInnerPtr for RefineFlatIndexImpl<BI> {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        RefineFlatIndexImpl {
            inner: inner_ptr as *mut FaissIndexFlat,
            base_index: PhantomData,
        }
    }
}

impl<BI> Index for RefineFlatIndexImpl<BI> {
    fn is_trained(&self) -> bool {
        unsafe { faiss_Index_is_trained(self.inner_ptr()) != 0 }
    }

    fn ntotal(&self) -> u64 {
        unsafe { faiss_Index_ntotal(self.inner_ptr()) as u64 }
    }

    fn d(&self) -> u32 {
        unsafe { faiss_Index_d(self.inner_ptr()) as u32 }
    }

    fn metric_type(&self) -> MetricType {
        unsafe { MetricType::from_code(faiss_Index_metric_type(self.inner_ptr()) as u32).unwrap() }
    }

    fn add(&mut self, x: &[f32]) -> Result<()> {
        unsafe {
            let n = x.len() / self.d() as usize;
            faiss_try(faiss_Index_add(self.inner_ptr(), n as i64, x.as_ptr()))?;
            Ok(())
        }
    }

    fn add_with_ids(&mut self, x: &[f32], xids: &[Idx]) -> Result<()> {
        unsafe {
            let n = x.len() / self.d() as usize;
            faiss_try(faiss_Index_add_with_ids(
                self.inner_ptr(),
                n as i64,
                x.as_ptr(),
                xids.as_ptr() as *const _,
            ))?;
            Ok(())
        }
    }
    fn train(&mut self, x: &[f32]) -> Result<()> {
        unsafe {
            let n = x.len() / self.d() as usize;
            faiss_try(faiss_Index_train(self.inner_ptr(), n as i64, x.as_ptr()))?;
            Ok(())
        }
    }
    fn assign(&mut self, query: &[f32], k: usize) -> Result<AssignSearchResult> {
        unsafe {
            let nq = query.len() / self.d() as usize;
            let mut out_labels = vec![Idx::none(); k * nq];
            faiss_try(faiss_Index_assign(
                self.inner_ptr(),
                nq as idx_t,
                query.as_ptr(),
                out_labels.as_mut_ptr() as *mut _,
                k as i64,
            ))?;
            Ok(AssignSearchResult { labels: out_labels })
        }
    }
    fn search(&mut self, query: &[f32], k: usize) -> Result<SearchResult> {
        unsafe {
            let nq = query.len() / self.d() as usize;
            let mut distances = vec![0_f32; k * nq];
            let mut labels = vec![Idx::none(); k * nq];
            faiss_try(faiss_Index_search(
                self.inner_ptr(),
                nq as idx_t,
                query.as_ptr(),
                k as idx_t,
                distances.as_mut_ptr(),
                labels.as_mut_ptr() as *mut _,
            ))?;
            Ok(SearchResult { distances, labels })
        }
    }
    fn range_search(&mut self, query: &[f32], radius: f32) -> Result<RangeSearchResult> {
        unsafe {
            let nq = (query.len() / self.d() as usize) as idx_t;
            let mut p_res: *mut FaissRangeSearchResult = ::std::ptr::null_mut();
            faiss_try(faiss_RangeSearchResult_new(&mut p_res, nq))?;
            faiss_try(faiss_Index_range_search(
                self.inner_ptr(),
                nq,
                query.as_ptr(),
                radius,
                p_res,
            ))?;
            Ok(RangeSearchResult { inner: p_res })
        }
    }

    fn reset(&mut self) -> Result<()> {
        unsafe {
            faiss_try(faiss_Index_reset(self.inner_ptr()))?;
            Ok(())
        }
    }

    fn remove_ids(&mut self, sel: &IdSelector) -> Result<usize> {
        unsafe {
            let mut n_removed = 0;
            faiss_try(faiss_Index_remove_ids(
                self.inner_ptr(),
                sel.inner_ptr(),
                &mut n_removed,
            ))?;
            Ok(n_removed)
        }
    }

    fn verbose(&self) -> bool {
        unsafe { faiss_Index_verbose(self.inner) != 0 }
    }

    fn set_verbose(&mut self, value: bool) {
        unsafe {
            faiss_Index_set_verbose(self.inner, std::os::raw::c_int::from(value));
        }
    }
}

impl<BI> RefineFlatIndexImpl<BI> {
    /// Create an independent clone of this index.
    ///
    /// # Errors
    ///
    /// May result in a native error if the clone operation is not
    /// supported for the internal type of index.
    pub fn try_clone(&self) -> Result<Self> {
        unsafe {
            let mut new_index_ptr = ::std::ptr::null_mut();
            faiss_try(faiss_clone_index(self.inner_ptr(), &mut new_index_ptr))?;
            Ok(crate::index::FromInnerPtr::from_inner_ptr(new_index_ptr))
        }
    }
}

impl<BI> ConcurrentIndex for RefineFlatIndexImpl<BI>
where
    BI: ConcurrentIndex,
{
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
    use crate::index::{flat::FlatIndexImpl, ConcurrentIndex, Idx, Index, UpcastIndex};

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
    fn refine_flat_index_upcast() {
        let index = FlatIndexImpl::new_l2(D).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);

        let refine = RefineFlatIndexImpl::new(index).unwrap();

        let index_impl = refine.upcast();
        assert_eq!(index_impl.d(), D);
    }
}
