//! Interface and implementation to ScalarQuantizer index type.

use super::*;

use crate::error::{Result};
use crate::faiss_try;
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

impl ConcurrentIndex for ScalarQuantizerIndexImpl {
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
    use super::{ScalarQuantizerIndexImpl, QuantizerType};
    use crate::index::{ConcurrentIndex, Idx, Index};
    use crate::metric::MetricType;

    const D: u32 = 8;

    #[test]
    fn sq_index_search() {
        let mut index = ScalarQuantizerIndexImpl::new(D, QuantizerType::QT_fp16, MetricType::L2).unwrap();
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
}