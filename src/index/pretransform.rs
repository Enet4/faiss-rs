//! Implementation to PreTransform index type.

use super::*;

use crate::error::{Error, Result};
use crate::faiss_try;
use crate::vector_transform::NativeVectorTransform;
use std::mem;
use std::os::raw::c_int;
use std::ptr;

/// Alias for the native implementation of a PreTransform index.
pub type PreTransformIndex = PreTransformIndexImpl;

/// Native implementation of a flat index.
#[derive(Debug)]
pub struct PreTransformIndexImpl {
    inner: *mut FaissIndexPreTransform,
}

unsafe impl Send for PreTransformIndexImpl {}
unsafe impl Sync for PreTransformIndexImpl {}

impl CpuIndex for PreTransformIndexImpl {}

impl Drop for PreTransformIndexImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexPreTransform_free(self.inner);
        }
    }
}

impl PreTransformIndexImpl {
    pub fn new<LT: NativeVectorTransform, I: NativeIndex>(lt: LT, sub_index: I) -> Result<Self> {
        let index = PreTransformIndexImpl::new_helper(&lt, &sub_index, true)?;
        mem::forget(lt);
        mem::forget(sub_index);

        Ok(index)
    }

    pub fn new_by_ref<LT: NativeVectorTransform, I: NativeIndex>(
        lt: &LT,
        sub_index: &I,
    ) -> Result<Self> {
        PreTransformIndexImpl::new_helper(lt, sub_index, false)
    }

    pub fn new_helper<LT: NativeVectorTransform, I: NativeIndex>(
        lt: &LT,
        sub_index: &I,
        own_fields: bool,
    ) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexPreTransform_new_with_transform(
                &mut inner,
                lt.inner_ptr(),
                sub_index.inner_ptr(),
            ))?;
            faiss_IndexPreTransform_set_own_fields(inner, c_int::from(own_fields));
            Ok(PreTransformIndexImpl { inner })
        }
    }
}

impl IndexImpl {
    pub fn into_pre_transform(self) -> Result<PreTransformIndexImpl> {
        unsafe {
            let new_inner = faiss_IndexPreTransform_cast(self.inner_ptr());
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                mem::forget(self);
                Ok(PreTransformIndexImpl { inner: new_inner })
            }
        }
    }
}

impl NativeIndex for PreTransformIndexImpl {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl FromInnerPtr for PreTransformIndexImpl {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        PreTransformIndexImpl {
            inner: inner_ptr as *mut FaissIndexPreTransform,
        }
    }
}

impl_native_index!(PreTransformIndexImpl);

impl_native_index_clone!(PreTransformIndexImpl);

impl ConcurrentIndex for PreTransformIndexImpl {
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
    use super::*;
    use crate::metric::MetricType;
    use crate::{
        index::{index_factory, ConcurrentIndex, Idx, Index},
        vector_transform::PCAMatrixImpl,
    };

    const D: u32 = 8;

    #[test]
    fn pre_transform_index_from_cast() {
        let mut index = index_factory(D, "PCA4,Flat", MetricType::L2).unwrap();

        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        if !index.is_trained() {
            index.train(some_data).unwrap();
        }
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let index: PreTransformIndexImpl = index.into_pre_transform().unwrap();
        assert_eq!(index.is_trained(), true);
        assert_eq!(index.ntotal(), 5);
        assert_eq!(index.d(), 8);
    }

    #[test]
    fn pre_transform_index_search() {
        const D_OUT: u32 = D / 2;
        let index = crate::index::flat::FlatIndexImpl::new_l2(D_OUT).unwrap();
        assert_eq!(index.d(), D_OUT);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];

        let vt = PCAMatrixImpl::new(D, D_OUT, 0f32, false).unwrap();
        let mut pre_transform_index = PreTransformIndexImpl::new(vt, index).unwrap();
        assert_eq!(pre_transform_index.d(), D);

        if !pre_transform_index.is_trained() {
            pre_transform_index.train(some_data).unwrap();
        }
        pre_transform_index.add(some_data).unwrap();
        assert_eq!(pre_transform_index.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = pre_transform_index.search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![2, 1, 0, 3, 4]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; D as usize];
        let result = (&pre_transform_index).search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![3, 4, 0, 1, 2]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        pre_transform_index.reset().unwrap();
        assert_eq!(pre_transform_index.ntotal(), 0);
    }
}
