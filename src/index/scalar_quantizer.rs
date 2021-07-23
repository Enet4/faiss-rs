//! Interface and implementation to ScalarQuantizer index type.

use super::*;

use crate::error::Result;
use crate::faiss_try;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_int;
use std::ptr;

/// Alias for the native implementation of a scalar quantizer index.
pub type ScalarQuantizerIndex = ScalarQuantizerIndexImpl;

/// Enumerate type describing the type of metric assumed by an index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QuantizerType {
    /// 8 bits per component
    QT_8bit = 0,
    /// 4 bits per component
    QT_4bit = 1,
    /// same, shared range for all dimensions
    QT_8bit_uniform = 2,
    /// same, shared range for all dimensions
    QT_4bit_uniform = 3,
    QT_fp16 = 4,
    /// fast indexing of uint8s
    QT_8bit_direct = 5,
    /// 6 bits per component
    QT_6bit = 6,
}

impl QuantizerType {
    /// Obtain the native code which identifies this quantizer type.
    pub fn code(self) -> u32 {
        self as u32
    }

    /// Obtain a quantizer type value from the native code.
    pub fn from_code(v: u32) -> Option<Self> {
        match v {
            0 => Some(QuantizerType::QT_8bit),
            1 => Some(QuantizerType::QT_4bit),
            2 => Some(QuantizerType::QT_8bit_uniform),
            3 => Some(QuantizerType::QT_4bit_uniform),
            4 => Some(QuantizerType::QT_fp16),
            5 => Some(QuantizerType::QT_8bit_direct),
            6 => Some(QuantizerType::QT_6bit),
            _ => None,
        }
    }
}

/// Native implementation of a scalar quantizer index.
#[derive(Debug)]
pub struct ScalarQuantizerIndexImpl {
    inner: *mut FaissIndexScalarQuantizer,
}

unsafe impl Send for ScalarQuantizerIndexImpl {}
unsafe impl Sync for ScalarQuantizerIndexImpl {}

impl CpuIndex for ScalarQuantizerIndexImpl {}

impl Drop for ScalarQuantizerIndexImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexScalarQuantizer_free(self.inner);
        }
    }
}

impl ScalarQuantizerIndexImpl {
    /// Create a new scalar quantizer index.
    pub fn new(d: u32, qt: QuantizerType, metric: MetricType) -> Result<Self> {
        unsafe {
            let metric = metric as c_uint;
            let qt_ = qt as c_uint;
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexScalarQuantizer_new_with(
                &mut inner,
                (d & 0x7FFF_FFFF) as idx_t,
                qt_,
                metric,
            ))?;
            Ok(ScalarQuantizerIndexImpl { inner })
        }
    }
}

impl NativeIndex for ScalarQuantizerIndexImpl {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl FromInnerPtr for ScalarQuantizerIndexImpl {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        ScalarQuantizerIndexImpl {
            inner: inner_ptr as *mut FaissIndexScalarQuantizer,
        }
    }
}

impl_native_index!(ScalarQuantizerIndexImpl);

impl_native_index_clone!(ScalarQuantizerIndexImpl);

impl IndexImpl {
    /// Attempt a dynamic cast of an index to the Scalar Quantizer index type.
    pub fn into_scalar_quantizer(self) -> Result<ScalarQuantizerIndexImpl> {
        unsafe {
            let new_inner = faiss_IndexScalarQuantizer_cast(self.inner_ptr());
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                mem::forget(self);
                Ok(ScalarQuantizerIndexImpl { inner: new_inner })
            }
        }
    }
}

impl_concurrent_index!(ScalarQuantizerIndexImpl);

/// Alias for the native implementation of a IVF scalar quantizer index.
pub type IVFScalarQuantizerIndex<Q> = IVFScalarQuantizerIndexImpl<Q>;

/// Native implementation of a scalar quantizer index.
#[derive(Debug)]
pub struct IVFScalarQuantizerIndexImpl<Q> {
    inner: *mut FaissIndexIVFScalarQuantizer,
    quantizer: PhantomData<Q>,
}

unsafe impl<Q: Send> Send for IVFScalarQuantizerIndexImpl<Q> {}
unsafe impl<Q: Sync> Sync for IVFScalarQuantizerIndexImpl<Q> {}

impl<Q: CpuIndex> CpuIndex for IVFScalarQuantizerIndexImpl<Q> {}

impl<Q> Drop for IVFScalarQuantizerIndexImpl<Q> {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexIVFScalarQuantizer_free(self.inner);
        }
    }
}

impl<Q> IVFScalarQuantizerIndexImpl<Q>
where
    Q: NativeIndex,
{
    /// Create a new IVF scalar quantizer index with metric.
    /// The index owns the quantizer.
    pub fn new_with_metric(
        quantizer: Q,
        d: u32,
        qt: QuantizerType,
        nlist: u32,
        metric: MetricType,
        encode_residual: Option<bool>,
    ) -> Result<Self> {
        IVFScalarQuantizerIndexImpl::new_owned(quantizer, d, qt, nlist, metric, encode_residual)
    }

    fn new_owned(
        quantizer: Q,
        d: u32,
        qt: QuantizerType,
        nlist: u32,
        metric: MetricType,
        encode_residual: Option<bool>,
    ) -> Result<Self> {
        let index = IVFScalarQuantizerIndexImpl::new_helper(
            &quantizer,
            d,
            qt,
            nlist,
            metric,
            encode_residual,
            true,
        )?;
        std::mem::forget(quantizer);

        Ok(index)
    }

    /// Create a new IVF scalar quantizer index with L2 metric.
    /// The index owns the quantizer.
    pub fn new_l2(quantizer: Q, d: u32, qt: QuantizerType, nlist: u32) -> Result<Self> {
        IVFScalarQuantizerIndexImpl::new_owned(quantizer, d, qt, nlist, MetricType::L2, None)
    }

    /// Create a new IVF scalar quantizer index with IP metric.
    /// The index owns the quantizer.
    pub fn new_ip(quantizer: Q, d: u32, qt: QuantizerType, nlist: u32) -> Result<Self> {
        IVFScalarQuantizerIndexImpl::new_owned(
            quantizer,
            d,
            qt,
            nlist,
            MetricType::InnerProduct,
            None,
        )
    }

    fn new_helper(
        quantizer: &Q,
        d: u32,
        qt: QuantizerType,
        nlist: u32,
        metric: MetricType,
        encode_residual: Option<bool>,
        own_fields: bool,
    ) -> Result<Self> {
        unsafe {
            let metric_ = metric as c_uint;
            let qt_ = qt as c_uint;
            let mut inner = ptr::null_mut();
            let quantizer_ = quantizer.inner_ptr();
            let encode_residual_ = c_int::from(encode_residual.unwrap_or(true));
            faiss_try(faiss_IndexIVFScalarQuantizer_new_with_metric(
                &mut inner,
                quantizer_,
                d as usize,
                nlist as usize,
                qt_,
                metric_,
                encode_residual_,
            ))?;

            faiss_IndexIVFScalarQuantizer_set_own_fields(inner, c_int::from(own_fields));
            Ok(IVFScalarQuantizerIndexImpl {
                inner,
                quantizer: PhantomData,
            })
        }
    }

    /// Get number of possible key values
    pub fn nlist(&self) -> u32 {
        unsafe { faiss_IndexIVFScalarQuantizer_nlist(self.inner_ptr()) as u32 }
    }

    /// Get number of probes at query time
    pub fn nprobe(&self) -> u32 {
        unsafe { faiss_IndexIVFScalarQuantizer_nprobe(self.inner_ptr()) as u32 }
    }

    /// Set number of probes at query time
    pub fn set_nprobe(&mut self, value: u32) {
        unsafe {
            faiss_IndexIVFScalarQuantizer_set_nprobe(self.inner_ptr(), value as usize);
        }
    }

    pub fn train_residual(&mut self, x: &[f32]) -> Result<()> {
        unsafe {
            let n = x.len() / self.d() as usize;
            faiss_try(faiss_IndexIVFScalarQuantizer_train_residual(
                self.inner_ptr(),
                n as i64,
                x.as_ptr(),
            ))?;
            Ok(())
        }
    }
}

impl<Q> NativeIndex for IVFScalarQuantizerIndexImpl<Q> {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl<Q> FromInnerPtr for IVFScalarQuantizerIndexImpl<Q> {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        IVFScalarQuantizerIndexImpl {
            inner: inner_ptr as *mut FaissIndexIVFScalarQuantizer,
            quantizer: PhantomData,
        }
    }
}

impl<Q> Index for IVFScalarQuantizerIndexImpl<Q> {
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

impl<Q> IVFScalarQuantizerIndexImpl<Q> {
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

impl<Q> ConcurrentIndex for IVFScalarQuantizerIndexImpl<Q>
where
    Q: ConcurrentIndex,
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

impl IndexImpl {
    /// Attempt a dynamic cast of an index to the IVF Scalar Quantizer index type.
    pub fn into_ivf_scalar_quantizer(self) -> Result<IVFScalarQuantizerIndexImpl<IndexImpl>> {
        unsafe {
            let new_inner = faiss_IndexIVFScalarQuantizer_cast(self.inner_ptr());
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                mem::forget(self);
                Ok(IVFScalarQuantizerIndexImpl {
                    inner: new_inner,
                    quantizer: PhantomData,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{IVFScalarQuantizerIndexImpl, QuantizerType, ScalarQuantizerIndexImpl};
    use crate::index::{flat, index_factory, ConcurrentIndex, Idx, Index, UpcastIndex};
    use crate::metric::MetricType;

    const D: u32 = 8;

    #[test]
    fn sq_index_search() {
        let mut index =
            ScalarQuantizerIndexImpl::new(D, QuantizerType::QT_fp16, MetricType::L2).unwrap();
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
        // index can be used behind an immutable ref
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
    fn ivf_sq_index_nlist() {
        let quantizer = flat::FlatIndex::new_l2(D).unwrap();
        let index =
            IVFScalarQuantizerIndexImpl::new_l2(quantizer, D, QuantizerType::QT_fp16, 1).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        assert_eq!(index.nlist(), 1);
    }

    #[test]
    fn ivf_sq_index_nprobe() {
        let quantizer = flat::FlatIndex::new_l2(D).unwrap();
        let mut index =
            IVFScalarQuantizerIndexImpl::new_l2(quantizer, D, QuantizerType::QT_fp16, 1).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        assert_eq!(index.nlist(), 1);

        index.set_nprobe(10);
        assert_eq!(index.nprobe(), 10);
    }

    #[test]
    fn ivf_sq_index_search() {
        let quantizer = flat::FlatIndex::new_l2(D).unwrap();
        let mut index =
            IVFScalarQuantizerIndexImpl::new_l2(quantizer, D, QuantizerType::QT_fp16, 1).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.train(some_data).unwrap();
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
        // index can be used behind an immutable ref
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
    fn ivf_sq_index_own_search() {
        let quantizer = flat::FlatIndex::new_l2(D).unwrap();
        let mut index =
            IVFScalarQuantizerIndexImpl::new_l2(quantizer, D, QuantizerType::QT_fp16, 1).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.train(some_data).unwrap();
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
        // index can be used behind an immutable ref
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
    fn sq_index_from_cast() {
        let mut index = index_factory(8, "SQfp16", MetricType::L2).unwrap();
        assert_eq!(index.is_trained(), true); // fp16 index does not need training
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let index: ScalarQuantizerIndexImpl = index.into_scalar_quantizer().unwrap();
        assert_eq!(index.is_trained(), true);
        assert_eq!(index.ntotal(), 5);
    }

    #[test]
    fn ivf_sq_index_from_cast() {
        let mut index = index_factory(8, "IVF1,SQfp16", MetricType::L2).unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.train(some_data).unwrap();
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let index = index.into_ivf_scalar_quantizer().unwrap();
        assert_eq!(index.is_trained(), true);
        assert_eq!(index.ntotal(), 5);
    }

    #[test]
    fn ivf_sq_index_upcast() {
        let quantizer = flat::FlatIndex::new_l2(D).unwrap();
        let index =
            IVFScalarQuantizerIndexImpl::new_l2(quantizer, D, QuantizerType::QT_fp16, 1).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);

        let index_impl = index.upcast();
        assert_eq!(index_impl.d(), D);
    }
}
