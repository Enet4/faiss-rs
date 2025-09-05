//! Interface and implementation to Flat index type.

use super::*;

/// Alias for the native implementation of a flat index.
pub type FlatIndex = FlatIndexImpl;

/// Native implementation of a flat index.
#[derive(Debug)]
pub struct FlatIndexImpl {
    inner: *mut FaissIndexFlat,
}

unsafe impl Send for FlatIndexImpl {}
unsafe impl Sync for FlatIndexImpl {}

impl CpuIndex for FlatIndexImpl {}

impl Drop for FlatIndexImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_Index_free(self.inner);
        }
    }
}

impl FlatIndexImpl {
    /// Create a new flat index.
    pub fn new(d: u32, metric: MetricType) -> Result<Self> {
        unsafe {
            let metric = metric as c_uint;
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexFlat_new_with(
                &mut inner,
                (d & 0x7FFF_FFFF) as idx_t,
                metric,
            ))?;
            Ok(FlatIndexImpl { inner })
        }
    }

    /// Create a new flat index with L2 as the metric type.
    pub fn new_l2(d: u32) -> Result<Self> {
        FlatIndexImpl::new(d, MetricType::L2)
    }

    /// Create a new flat index with IP (inner product) as the metric type.
    pub fn new_ip(d: u32) -> Result<Self> {
        FlatIndexImpl::new(d, MetricType::InnerProduct)
    }

    /// Obtain a reference to the indexed data.
    pub fn xb(&self) -> &[f32] {
        unsafe {
            let mut xb = ptr::null_mut();
            let mut len = 0;
            faiss_IndexFlat_xb(self.inner, &mut xb, &mut len);
            ::std::slice::from_raw_parts(xb, len)
        }
    }

    /// Compute distance with a subset of vectors. `x` is a sequence of query
    /// vectors, size `n * d`, where `n` is inferred from the length of `x`.
    /// `labels` is a sequence of indexed vector ID's that should be compared
    /// for each query vector, size `n * k`, where `k` is inferred from the
    /// length of `labels`. Returns the corresponding output distances, size
    /// `n * k`.
    pub fn compute_distance_subset(&mut self, x: &[f32], labels: &[Idx]) -> Result<Vec<f32>> {
        unsafe {
            let n = x.len() / self.d() as usize;
            let k = labels.len() / n;
            let mut distances = vec![0.; n * k];
            faiss_try(faiss_IndexFlat_compute_distance_subset(
                self.inner,
                n as idx_t,
                x.as_ptr(),
                k as idx_t,
                distances.as_mut_ptr(),
                labels.as_ptr() as *const _,
            ))?;
            Ok(distances)
        }
    }
}

impl IndexImpl {
    /// Attempt a dynamic cast of an index to the flat index type.
    #[deprecated(
        since = "0.8.0",
        note = "Non-idiomatic name, prefer `into_flat` instead"
    )]
    pub fn as_flat(self) -> Result<FlatIndexImpl> {
        self.into_flat()
    }

    /// Attempt a dynamic cast of an index to the flat index type.
    pub fn into_flat(self) -> Result<FlatIndexImpl> {
        unsafe {
            let new_inner = faiss_IndexFlat_cast(self.inner_ptr());
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                mem::forget(self);
                Ok(FlatIndexImpl { inner: new_inner })
            }
        }
    }
}

impl NativeIndex for FlatIndexImpl {
    type Inner = FaissIndex;
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl FromInnerPtr for FlatIndexImpl {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        FlatIndexImpl {
            inner: inner_ptr as *mut FaissIndexFlat,
        }
    }
}

impl TryFromInnerPtr for FlatIndexImpl {
    unsafe fn try_from_inner_ptr(inner_ptr: *mut FaissIndex) -> Result<Self>
    where
        Self: Sized,
    {
        // safety: `inner_ptr` is documented to be a valid pointer to an index,
        // so the dynamic cast should be safe.
        #[allow(unused_unsafe)]
        unsafe {
            let new_inner = faiss_IndexFlat_cast(inner_ptr);
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                Ok(FlatIndexImpl { inner: new_inner })
            }
        }
    }
}

impl_native_index!(FlatIndex);

impl TryClone for FlatIndexImpl {
    fn try_clone(&self) -> Result<Self>
    where
        Self: Sized,
    {
        try_clone_from_inner_ptr(self)
    }
}

impl_concurrent_index!(FlatIndexImpl);

#[cfg(test)]
mod tests {
    use super::FlatIndexImpl;
    use crate::index::{
        index_factory, ConcurrentIndex, FromInnerPtr, Idx, Index, NativeIndex, SearchWithParams, TryClone, UpcastIndex
    };
    use crate::metric::MetricType;
    use crate::search_params::SearchParametersImpl;
    use crate::selector::IdSelector;

    const D: u32 = 8;

    #[test]
    fn flat_index_from_upcast() {
        let index = FlatIndexImpl::new_l2(D).unwrap();

        let index_impl = index.upcast();
        assert_eq!(index_impl.d(), D);
    }

    #[test]
    fn flat_index_from_cast() {
        let mut index = index_factory(8, "Flat", MetricType::L2).unwrap();
        assert_eq!(index.is_trained(), true); // Flat index does not need training
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let index: FlatIndexImpl = index.into_flat().unwrap();
        assert_eq!(index.is_trained(), true);
        assert_eq!(index.ntotal(), 5);
        let xb = index.xb();
        assert_eq!(xb.len(), 8 * 5);
    }

    #[test]
    fn flat_index_boxed() {
        let mut index = FlatIndexImpl::new_l2(8).unwrap();
        assert_eq!(index.is_trained(), true); // Flat index does not need training
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let boxed = Box::new(index);
        assert_eq!(boxed.is_trained(), true);
        assert_eq!(boxed.ntotal(), 5);
        let xb = boxed.xb();
        assert_eq!(xb.len(), 8 * 5);
    }

    #[test]
    fn index_verbose() {
        let mut index = FlatIndexImpl::new_l2(D).unwrap();
        assert_eq!(index.is_trained(), true); // Flat index does not need training
        index.set_verbose(true);
        assert_eq!(index.verbose(), true);
        index.set_verbose(false);
        assert_eq!(index.verbose(), false);
    }

    #[test]
    fn index_clone() {
        let mut index = FlatIndexImpl::new_l2(D).unwrap();
        assert_eq!(index.is_trained(), true); // Flat index does not need training
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        {
            let mut index: FlatIndexImpl = index.try_clone().unwrap();
            assert_eq!(index.is_trained(), true);
            assert_eq!(index.ntotal(), 5);
            {
                let xb = index.xb();
                assert_eq!(xb.len(), 8 * 5);
            }
            index.reset().unwrap();
            assert_eq!(index.ntotal(), 0);
        }
        assert_eq!(index.ntotal(), 5);
    }

    #[test]
    fn flat_index_search() {
        let mut index = FlatIndexImpl::new_l2(D).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = index.search(&my_query, 5).unwrap();
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
        let result = (&index).search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![3, 4, 0, 1, 2]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        index.reset().unwrap();
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn flat_index_assign() {
        let mut index = FlatIndexImpl::new(D, MetricType::L2).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = index.assign(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![2, 1, 0, 3, 4]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );

        let my_query = [100.; D as usize];
        // flat index can be used behind an immutable ref
        let result = (&index).assign(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![3, 4, 0, 1, 2]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );

        index.reset().unwrap();
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn flat_index_range_search() {
        let mut index = FlatIndexImpl::new(D, MetricType::L2).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = (&index).range_search(&my_query, 8.125).unwrap();
        let (distances, labels) = result.distance_and_labels();
        assert!(labels == &[Idx::new(1), Idx::new(2)] || labels == &[Idx::new(2), Idx::new(1)]);
        assert!(distances.iter().all(|x| *x > 0.));
    }

    #[test]
    fn index_transition() {
        let index = {
            let mut index = FlatIndexImpl::new_l2(D).unwrap();
            assert_eq!(index.d(), D);
            assert_eq!(index.ntotal(), 0);
            let some_data = &[
                7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
                -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16.,
                32., 25., 20., 20., 40., 15.,
            ];
            index.add(some_data).unwrap();
            assert_eq!(index.ntotal(), 5);

            unsafe {
                let inner = index.inner_ptr();
                // forget index, rebuild it into another object
                ::std::mem::forget(index);
                FlatIndexImpl::from_inner_ptr(inner)
            }
        };
        assert_eq!(index.ntotal(), 5);
    }

    #[test]
    fn flat_index_search_with_params() {
        let mut index = FlatIndexImpl::new(D, MetricType::L2).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let selector = IdSelector::range(Idx::new(0), Idx::new(10)).unwrap();
        let search_params = SearchParametersImpl::new(selector).unwrap();

        let my_query = [0.; D as usize];
        let result = index.search_with_params(&my_query, 5, &search_params).unwrap();
        
        assert_eq!(
            result.labels,
            vec![2, 1, 0, 3, 4]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

    }
}
