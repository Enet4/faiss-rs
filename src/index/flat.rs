//! Interface and implementation to Flat index type.

use faiss_sys::*;
use super::*;

use error::{Error, Result};
use std::mem;
use std::ptr;

/// Alias for the native implementation of a flat index.
pub type FlatIndex = FlatIndexImpl;

#[derive(Debug)]
pub struct FlatIndexImpl {
    inner: *mut FaissIndexFlat,
}

unsafe impl Send for FlatIndexImpl {}
unsafe impl Sync for FlatIndexImpl {}

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
            faiss_try!(faiss_IndexFlat_new_with(
                &mut inner,
                (d & 0x7FFFFFFF) as idx_t,
                metric
            ));
            Ok(FlatIndexImpl { inner })
        }
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
            faiss_try!(faiss_IndexFlat_compute_distance_subset(
                self.inner,
                n as idx_t,
                x.as_ptr(),
                k as idx_t,
                distances.as_mut_ptr(),
                labels.as_ptr()
            ));
            Ok(distances)
        }
    }
}

impl IndexImpl {
    /// Attempt a dynamic cast of an index to the flat index type.
    pub fn as_flat(mut self) -> Result<FlatIndexImpl> {
        unsafe {
            let new_inner = faiss_IndexFlat_cast(self.inner_ptr_mut());
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
    fn inner_ptr(&self) -> *const FaissIndex {
        self.inner
    }

    fn inner_ptr_mut(&mut self) -> *mut FaissIndex {
        self.inner
    }
}

impl Index for FlatIndexImpl {
    fn is_trained(&self) -> bool {
        unsafe { faiss_Index_is_trained(self.inner) != 0 }
    }

    fn ntotal(&self) -> u64 {
        unsafe { faiss_Index_ntotal(self.inner) as u64 }
    }

    fn d(&self) -> u32 {
        unsafe { faiss_Index_d(self.inner) as u32 }
    }

    fn metric_type(&self) -> MetricType {
        unsafe { MetricType::from_code(faiss_Index_metric_type(self.inner) as u32).unwrap() }
    }

    fn add(&mut self, x: &[f32]) -> Result<()> {
        unsafe {
            let n = x.len() / self.d() as usize;
            faiss_try!(faiss_Index_add(self.inner, n as i64, x.as_ptr()));
            Ok(())
        }
    }

    fn add_with_ids(&mut self, x: &[f32], xids: &[Idx]) -> Result<()> {
        unsafe {
            let n = x.len() / self.d() as usize;
            faiss_try!(faiss_Index_add_with_ids(
                self.inner,
                n as i64,
                x.as_ptr(),
                xids.as_ptr()
            ));
            Ok(())
        }
    }
    fn train(&mut self, x: &[f32]) -> Result<()> {
        unsafe {
            let n = x.len() / self.d() as usize;
            faiss_try!(faiss_Index_train(self.inner, n as i64, x.as_ptr()));
            Ok(())
        }
    }
    fn assign(&mut self, query: &[f32], k: usize) -> Result<AssignSearchResult> {
        unsafe {
            let nq = query.len() / self.d() as usize;
            let mut out_labels = vec![0 as Idx; k * nq];
            faiss_try!(faiss_Index_assign(
                self.inner,
                nq as idx_t,
                query.as_ptr(),
                out_labels.as_mut_ptr(),
                k as i64
            ));
            Ok(AssignSearchResult { labels: out_labels })
        }
    }
    fn search(&mut self, query: &[f32], k: usize) -> Result<SearchResult> {
        unsafe {
            let nq = query.len() / self.d() as usize;
            let mut distances = vec![0_f32; k * nq];
            let mut labels = vec![0 as Idx; k * nq];
            faiss_try!(faiss_Index_search(
                self.inner,
                nq as idx_t,
                query.as_ptr(),
                k as idx_t,
                distances.as_mut_ptr(),
                labels.as_mut_ptr()
            ));
            Ok(SearchResult { distances, labels })
        }
    }
    fn range_search(&mut self, query: &[f32], radius: f32) -> Result<RangeSearchResult> {
        unsafe {
            let nq = (query.len() / self.d() as usize) as idx_t;
            let mut p_res: *mut FaissRangeSearchResult = ptr::null_mut();
            faiss_try!(faiss_RangeSearchResult_new(&mut p_res, nq));
            faiss_try!(faiss_Index_range_search(
                self.inner,
                nq,
                query.as_ptr(),
                radius,
                p_res
            ));
            Ok(RangeSearchResult { inner: p_res })
        }
    }

    fn reset(&mut self) -> Result<()> {
        unsafe {
            faiss_try!(faiss_Index_reset(self.inner));
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Index;
    use super::FlatIndexImpl;
    use metric::MetricType;

    const D: u32 = 8;

    #[test]
    fn flat_index_search() {
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
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![2, 1, 0, 3, 4]);
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; D as usize];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![3, 4, 0, 1, 2]);
        assert!(result.distances.iter().all(|x| *x > 0.));

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
        let result = index.range_search(&my_query, 8.125).unwrap();
        let (distances, labels) = result.distance_and_labels();
        assert!(labels == &[1, 2] || labels == &[2, 1]);
        assert!(distances.iter().all(|x| *x > 0.));
    }
}
