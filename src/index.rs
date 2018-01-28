use error::Result;
use metric::MetricType;
use std::fmt;
use std::os::raw::c_uint;
use std::ptr;
use std::ffi::CString;

use faiss_sys::*;

pub type Idx = idx_t;

#[derive(Debug, Clone, PartialEq)]
pub struct AssignSearchResult {
    pub labels: Vec<Idx>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub distances: Vec<f32>,
    pub labels: Vec<Idx>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeSearchResult {
    inner: *mut FaissRangeSearchResult,
}
impl RangeSearchResult {
    pub fn distances(&self) -> &[f32] {
        unimplemented!()
    }

    pub fn labels(&self) -> &[Idx] {
        unimplemented!()
    }
}

impl Drop for RangeSearchResult {
    fn drop(&mut self) {
        unsafe {
            faiss_RangeSearchResult_free(self.inner);
        }
    }
}

/// Interface for a Faiss index
pub trait Index {
    /// Whether the index is trained
    fn is_trained(&self) -> bool;

    /// The total number of vectors indexed
    fn ntotal(&self) -> u64;

    /// The dimensionality of the indexed vectors
    fn d(&self) -> u32;

    /// The metric type assumed by the index
    fn metric_type(&self) -> MetricType;

    /// Add new data vectors to the index.
    /// This assumes a contiguous memory slice of vectors, where the total
    /// number of vectors is `x.len() / d`.
    fn add(&mut self, x: &[f32]) -> Result<()>;

    /// Add new data vectors to the index with ids.
    /// This assumes a contiguous memory slice of vectors, where the total
    /// number of vectors is `x.len() / d`.
    /// Not all index types may support this operation.
    fn add_with_ids(&mut self, x: &[f32], xids: &[Idx]) -> Result<()>;

    fn train(&mut self, x: &[f32]) -> Result<()>;
    fn assign(&mut self, q: &[f32], k: usize) -> Result<AssignSearchResult>;
    fn search(&mut self, q: &[f32], k: usize) -> Result<SearchResult>;
    fn range_search(&mut self, q: &[f32], radius: f32) -> Result<RangeSearchResult>;

    fn reset(&mut self) -> Result<()>;
}

pub struct IndexImpl {
    inner: *mut FaissIndex,
}

impl fmt::Debug for IndexImpl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("IndexImpl")
    }
}

impl Drop for IndexImpl {
    fn drop(&mut self) {
        unsafe { faiss_Index_free(self.inner); }
    }
}

impl Index for IndexImpl {
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
        unsafe {
            MetricType::from_code(
                faiss_Index_metric_type(self.inner) as u32).unwrap()
        }
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
            faiss_try!(faiss_Index_add_with_ids(self.inner, n as i64, x.as_ptr(), xids.as_ptr()));
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
                self.inner, nq as idx_t, query.as_ptr(), out_labels.as_mut_ptr(), k as i64));
            Ok(AssignSearchResult {
                labels: out_labels
            })
        }
    }
    fn search(&mut self, query: &[f32], k: usize) -> Result<SearchResult> {
        unsafe {
            let nq = query.len() / self.d() as usize;
            let mut distances = vec![0_f32; k * nq];
            let mut labels = vec![0 as Idx; k * nq];
            faiss_try!(faiss_Index_search(
                self.inner, nq as idx_t, query.as_ptr(), k as idx_t, distances.as_mut_ptr(), labels.as_mut_ptr()));
            Ok(SearchResult {
                distances,
                labels
            })
        }
    }
    fn range_search(&mut self, query: &[f32], radius: f32) -> Result<RangeSearchResult> {
        unsafe {
            let nq = (query.len() / self.d() as usize) as idx_t;
            let mut p_res: *mut FaissRangeSearchResult = ptr::null_mut();
            faiss_try!(faiss_RangeSearchResult_new(&mut p_res, nq));
            faiss_try!(faiss_Index_range_search(
                self.inner, nq, query.as_ptr(), radius, p_res));
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

pub fn index_factory<D: AsRef<str>>(d: u32, description: D, metric: MetricType) -> Result<IndexImpl> {
    unsafe {
        let metric = metric as c_uint;
        let description = CString::new(description.as_ref()).unwrap();
        let mut index_ptr = ::std::ptr::null_mut();
        faiss_try!(faiss_index_factory(&mut index_ptr, (d & 0x7FFFFFFF) as i32, description.as_ptr(), metric));
        Ok(IndexImpl {
            inner: index_ptr,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::index_factory;
    use metric::MetricType;

    #[test]
    fn index_factory_flat() {
        let r = index_factory(8, "Flat", MetricType::L2);
        assert!(r.is_ok());
    }

    #[test]
    fn bad_index_factory_description() {
        let r = index_factory(8, "fdnoyq", MetricType::L2);
        assert!(r.is_err());
    }

}