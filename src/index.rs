//! Index interface and implementations

use error::Result;
use metric::MetricType;
use std::fmt;
use std::os::raw::c_uint;
use std::ptr;
use std::ffi::CString;

use faiss_sys::*;

/// Primitive data type for identifying a vector in an index.
pub type Idx = idx_t;

/// The outcome of an index assign operation.
#[derive(Debug, Clone, PartialEq)]
pub struct AssignSearchResult {
    pub labels: Vec<Idx>,
}

/// The outcome of an index search operation.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub distances: Vec<f32>,
    pub labels: Vec<Idx>,
}

/// The outcome of an index range search operation.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeSearchResult {
    inner: *mut FaissRangeSearchResult,
}

impl RangeSearchResult {

    pub fn nq(&self) -> usize {
        unsafe {
            faiss_RangeSearchResult_nq(self.inner)
        }
    }

    pub fn lims(&self) -> &[usize] {
        unsafe {
            let mut lims_ptr = ptr::null_mut();
            faiss_RangeSearchResult_lims(self.inner, &mut lims_ptr);
            ::std::slice::from_raw_parts(lims_ptr, self.nq() + 1)
        }
    }

    /// getter for labels and respective distances (not sorted):
    /// result for query `i` is `labels[lims[i] .. lims[i+1]]`
    pub fn distance_and_labels(&self) -> (&[f32], &[Idx]) {
        let lims = self.lims();
        let full_len = lims.last().map(|x| *x).unwrap_or(0);
        unsafe {
            let mut distances_ptr = ptr::null_mut();
            let mut labels_ptr = ptr::null_mut();
            faiss_RangeSearchResult_labels(self.inner, &mut labels_ptr, &mut distances_ptr);
            let distances = ::std::slice::from_raw_parts(distances_ptr, full_len);
            let labels = ::std::slice::from_raw_parts(labels_ptr, full_len);
            (distances, labels)
        }
    }

    /// getter for labels and respective distances (not sorted):
    /// result for query `i` is `labels[lims[i] .. lims[i+1]]`
    pub fn distance_and_labels_mut(&self) -> (&mut [f32], &mut [Idx]) {
        unsafe {
            let buf_size = faiss_RangeSearchResult_buffer_size(self.inner);
            let mut distances_ptr = ptr::null_mut();
            let mut labels_ptr = ptr::null_mut();
            faiss_RangeSearchResult_labels(self.inner, &mut labels_ptr, &mut distances_ptr);
            let distances = ::std::slice::from_raw_parts_mut(distances_ptr, buf_size);
            let labels = ::std::slice::from_raw_parts_mut(labels_ptr, buf_size);
            (distances, labels)
        }
    }

    pub fn distances(&self) -> &[f32] {
        self.distance_and_labels().0
    }

    pub fn distances_mut(&mut self) -> &mut [f32] {
        self.distance_and_labels_mut().0
    }

    pub fn labels(&self) -> &[Idx] {
        self.distance_and_labels().1
    }

    pub fn labels_mut(&mut self) -> &mut [Idx] {
        self.distance_and_labels_mut().1
    }
}

impl Drop for RangeSearchResult {
    fn drop(&mut self) {
        unsafe {
            faiss_RangeSearchResult_free(self.inner);
        }
    }
}

/// Interface for a Faiss index.
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

    /// Train the underlying index with the given data.
    fn train(&mut self, x: &[f32]) -> Result<()>;

    /// Similar to `search`, but only provides the labels.
    fn assign(&mut self, q: &[f32], k: usize) -> Result<AssignSearchResult>;

    /// Perform a search for the `k` closest vectors to the given query vectors.
    fn search(&mut self, q: &[f32], k: usize) -> Result<SearchResult>;

    /// Perform a ranged search for the vectors closest to the given query vectors
    /// by the given radius.
    fn range_search(&mut self, q: &[f32], radius: f32) -> Result<RangeSearchResult>;

    /// Clear the entire index.
    fn reset(&mut self) -> Result<()>;
}

/// Native implementation of a Faiss Index.
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

impl IndexImpl {
    pub fn inner_ptr(&self) -> *const FaissIndex {
        self.inner
    }

    pub fn inner_ptr_mut(&mut self) -> *mut FaissIndex {
        self.inner
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

#[cfg(feature = "gpu")]
pub mod gpu {
    //! GPU Index implementation

    use std::marker::PhantomData;
    use std::mem;
    use std::ptr;
    use faiss_sys::*;
    use error::Result;
    use gpu::GpuResources;
    use metric::MetricType;
    use super::{Index, IndexImpl, Idx, AssignSearchResult, SearchResult, RangeSearchResult};

    pub struct GpuIndexImpl<'gpu> {
        inner: IndexImpl,
        phantom: PhantomData<&'gpu ()>,
    }

    impl IndexImpl {
        pub fn to_gpu<'gpu, G>(self, gpu_res: &'gpu mut G, device: i32) -> Result<GpuIndexImpl<'gpu>>
        where
            G: GpuResources,
        {
            unsafe {
                let mut gpuindex_ptr = ptr::null_mut(); 
                faiss_try!(faiss_index_cpu_to_gpu(gpu_res.inner_ptr_mut(), device, self.inner_ptr(), &mut gpuindex_ptr));
                mem::forget(self); // don't free the index
                Ok(GpuIndexImpl {
                    inner: IndexImpl { inner: gpuindex_ptr },
                    phantom: PhantomData,
                })
            }
        }
    }

    impl<'gpu> GpuIndexImpl<'gpu> {
        pub fn inner_ptr(&self) -> *const FaissIndex {
            self.inner.inner_ptr()
        }

        pub fn inner_ptr_mut(&mut self) -> *mut FaissIndex {
            self.inner.inner_ptr_mut()
        }

        pub fn to_cpu(self) -> Result<IndexImpl> {
            unsafe {
                let mut cpuindex_ptr = ptr::null_mut(); 
                faiss_try!(faiss_index_gpu_to_cpu(self.inner.inner_ptr(), &mut cpuindex_ptr));
                mem::forget(self); // don't free the index
                Ok(IndexImpl {
                    inner: cpuindex_ptr,
                })
            }
        }
    }

    impl<'gpu> Index for GpuIndexImpl<'gpu> {
        fn is_trained(&self) -> bool {
            self.inner.is_trained()
        }

        fn ntotal(&self) -> u64 {
            self.inner.ntotal()
        }

        fn d(&self) -> u32 {
            self.inner.d()
        }

        fn metric_type(&self) -> MetricType {
            self.inner.metric_type()
        }

        fn add(&mut self, x: &[f32]) -> Result<()> {
            self.inner.add(x)
        }

        fn add_with_ids(&mut self, x: &[f32], xids: &[Idx]) -> Result<()> {
            self.inner.add_with_ids(x, xids)
        }

        fn train(&mut self, x: &[f32]) -> Result<()> {
            self.inner.train(x)
        }

        fn assign(&mut self, query: &[f32], k: usize) -> Result<AssignSearchResult> {
            self.inner.assign(query, k)
        }

        fn search(&mut self, query: &[f32], k: usize) -> Result<SearchResult> {
            self.inner.search(query, k)
        }

        fn range_search(&mut self, query: &[f32], radius: f32) -> Result<RangeSearchResult> {
            self.inner.range_search(query, radius)
        }

        fn reset(&mut self) -> Result<()> {
            self.inner.reset()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::index_factory;
        use super::super::Index;
        use gpu::StandardGpuResources;
        use metric::MetricType;

        #[test]
        fn flat_index_search() {
            let mut res = StandardGpuResources::new().unwrap();

            let mut index = index_factory(8, "Flat", MetricType::L2).unwrap()
                .to_gpu(&mut res, 0).unwrap();
            let some_data = &[
                7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5,
                -1., 1., 1., 1., 1., 1., 1., -1.,
                0., 0., 0., 1., 1., 0., 0., -1.,
                100., 100., 100., 100., -100., 100., 100., 100.,
                120., 100., 100., 105., -100., 100., 100., 105.,
            ];
            index.add(some_data).unwrap();
            assert_eq!(index.ntotal(), 5);

            let my_query = [0. ; 8];
            let result = index.search(&my_query, 5).unwrap();
            assert_eq!(result.labels, vec![2, 1, 0, 3, 4]);
            assert!(result.distances.iter().all(|x| *x > 0.));

            let my_query = [100. ; 8];
            let result = index.search(&my_query, 5).unwrap();
            assert_eq!(result.labels, vec![3, 4, 0, 1, 2]);
            assert!(result.distances.iter().all(|x| *x > 0.));
        }

    }
}

#[cfg(test)]
mod tests {
    use super::{Index, index_factory};
    use metric::MetricType;

    #[test]
    fn index_factory_flat() {
        let r = index_factory(64, "Flat", MetricType::L2);
        assert!(r.is_ok());
        let index = r.unwrap();
        assert_eq!(index.is_trained(), true); // Flat index does not need training
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn bad_index_factory_description() {
        let r = index_factory(64, "fdnoyq", MetricType::L2);
        assert!(r.is_err());
    }

    #[test]
    fn flat_index_search() {
        let mut index = index_factory(8, "Flat", MetricType::L2).unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5,
            -1., 1., 1., 1., 1., 1., 1., -1.,
            0., 0., 0., 1., 1., 0., 0., -1.,
            100., 100., 100., 100., -100., 100., 100., 100.,
            120., 100., 100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0. ; 8];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![2, 1, 0, 3, 4]);
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100. ; 8];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![3, 4, 0, 1, 2]);
        assert!(result.distances.iter().all(|x| *x > 0.));
    }

    #[test]
    fn flat_index_range_search() {
        let mut index = index_factory(8, "Flat", MetricType::L2).unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5,
            -1., 1., 1., 1., 1., 1., 1., -1.,
            0., 0., 0., 1., 1., 0., 0., -1.,
            100., 100., 100., 100., -100., 100., 100., 100.,
            120., 100., 100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0. ; 8];
        let result = index.range_search(&my_query, 8.125).unwrap();
        let (distances, labels) = result.distance_and_labels();
        assert!(labels == &[1, 2] || labels == &[2, 1]);
        assert!(distances.iter().all(|x| *x > 0.));
    }

}