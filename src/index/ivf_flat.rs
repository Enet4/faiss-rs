//! Interface and implementation to IVFFlat index type.

use super::*;

use crate::error::Result;
use crate::faiss_try;
use std::mem;
use std::os::raw::c_int;
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
    fn new_helper(
        quantizer: &flat::FlatIndex,
        d: u32,
        nlist: u32,
        metric: MetricType,
        own_fields: bool,
    ) -> Result<Self> {
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
            faiss_IndexIVFFlat_set_own_fields(inner, c_int::from(own_fields));
            Ok(IVFFlatIndexImpl { inner })
        }
    }

    /// Create a new IVF flat index.
    // The index owns the quantizer.
    pub fn new(quantizer: flat::FlatIndex, d: u32, nlist: u32, metric: MetricType) -> Result<Self> {
        let index = IVFFlatIndexImpl::new_helper(&quantizer, d, nlist, metric, true)?;
        std::mem::forget(quantizer);

        Ok(index)
    }

    /// Create a new IVF flat index with L2 as the metric type.
    // The index owns the quantizer.
    pub fn new_l2(quantizer: flat::FlatIndex, d: u32, nlist: u32) -> Result<Self> {
        IVFFlatIndexImpl::new(quantizer, d, nlist, MetricType::L2)
    }

    /// Create a new IVF flat index with IP (inner product) as the metric type.
    // The index owns the quantizer.
    pub fn new_ip(quantizer: flat::FlatIndex, d: u32, nlist: u32) -> Result<Self> {
        IVFFlatIndexImpl::new(quantizer, d, nlist, MetricType::InnerProduct)
    }

    /// Get number of probes at query time
    pub fn nprobe(&self) -> u32 {
        unsafe { faiss_IndexIVFFlat_nprobe(self.inner_ptr()) as u32 }
    }

    /// Set number of probes at query time
    pub fn set_nprobe(&mut self, value: u32) {
        unsafe {
            faiss_IndexIVFFlat_set_nprobe(self.inner_ptr(), value as usize);
        }
    }

    /// Get number of possible key values
    pub fn nlist(&self) -> u32 {
        unsafe { faiss_IndexIVFFlat_nlist(self.inner_ptr()) as u32 }
    }

    /// Get train type
    pub fn train_type(&self) -> Option<TrainType> {
        unsafe {
            let code = faiss_IndexIVFFlat_quantizer_trains_alone(self.inner_ptr());
            TrainType::from_code(code)
        }
    }
}

/**
 * = 0: use the quantizer as index in a kmeans training
 * = 1: just pass on the training set to the train() of the quantizer
 * = 2: kmeans training on a flat index + add the centroids to the quantizer
 */
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
pub enum TrainType {
    /// use the quantizer as index in a kmeans training
    QuantizerAsIndex,
    /// just pass on the training set to the train() of the quantizer
    QuantizerTrainsAlone,
    /// kmeans training on a flat index + add the centroids to the quantizer
    FlatIndexAndQuantizer,
}

impl TrainType {
    pub(crate) fn from_code(code: i8) -> Option<Self> {
        match code {
            0 => Some(TrainType::QuantizerAsIndex),
            1 => Some(TrainType::QuantizerTrainsAlone),
            2 => Some(TrainType::FlatIndexAndQuantizer),
            _ => None,
        }
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

impl_concurrent_index!(IVFFlatIndexImpl);

impl IndexImpl {
    /// Attempt a dynamic cast of an index to the IVF flat index type.
    pub fn into_ivf_flat(self) -> Result<IVFFlatIndexImpl> {
        unsafe {
            let new_inner = faiss_IndexIVFFlat_cast(self.inner_ptr());
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                mem::forget(self);
                Ok(IVFFlatIndexImpl { inner: new_inner })
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::IVFFlatIndexImpl;
    use crate::index::flat::FlatIndexImpl;
    use crate::index::{index_factory, ConcurrentIndex, Idx, Index, UpcastIndex};
    use crate::MetricType;

    const D: u32 = 8;

    #[test]
    // #[ignore]
    fn index_search() {
        let q = FlatIndexImpl::new_l2(D).unwrap();
        let mut index = IVFFlatIndexImpl::new_l2(q, D, 1).unwrap();
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
    fn index_search_own() {
        let q = FlatIndexImpl::new_l2(D).unwrap();
        let mut index = IVFFlatIndexImpl::new_l2(q, D, 1).unwrap();
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
        let mut index = IVFFlatIndexImpl::new_l2(q, D, 1).unwrap();
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
    fn ivf_flat_index_from_cast() {
        let mut index = index_factory(8, "IVF1,Flat", MetricType::L2).unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.train(some_data).unwrap();
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let index: IVFFlatIndexImpl = index.into_ivf_flat().unwrap();
        assert_eq!(index.is_trained(), true);
        assert_eq!(index.ntotal(), 5);
    }

    #[test]
    fn index_upcast() {
        let q = FlatIndexImpl::new_l2(D).unwrap();
        let index = IVFFlatIndexImpl::new_l2(q, D, 1).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);

        let index_impl = index.upcast();
        assert_eq!(index_impl.d(), D);
    }
}
