//! GPU Index implementation

use std::marker::PhantomData;
use std::mem;
use std::ptr;
use faiss_sys::*;
use error::Result;
use gpu::GpuResources;
use metric::MetricType;
use super::{AssignSearchResult, Idx, Index, IndexImpl, NativeIndex, RangeSearchResult,
            SearchResult};

#[derive(Debug)]
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
            faiss_try!(faiss_index_cpu_to_gpu(
                gpu_res.inner_ptr_mut(),
                device,
                self.inner_ptr(),
                &mut gpuindex_ptr
            ));
            mem::forget(self); // don't free the index
            Ok(GpuIndexImpl {
                inner: IndexImpl {
                    inner: gpuindex_ptr,
                },
                phantom: PhantomData,
            })
        }
    }
}

impl<'gpu> GpuIndexImpl<'gpu> {
    pub fn to_cpu(self) -> Result<IndexImpl> {
        unsafe {
            let mut cpuindex_ptr = ptr::null_mut();
            faiss_try!(faiss_index_gpu_to_cpu(
                self.inner.inner_ptr(),
                &mut cpuindex_ptr
            ));
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

impl<'g> NativeIndex for GpuIndexImpl<'g> {
    fn inner_ptr(&self) -> *const FaissIndex {
        self.inner.inner_ptr()
    }

    fn inner_ptr_mut(&mut self) -> *mut FaissIndex {
        self.inner.inner_ptr_mut()
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

        let mut index = index_factory(8, "Flat", MetricType::L2)
            .unwrap()
            .to_gpu(&mut res, 0)
            .unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; 8];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![2, 1, 0, 3, 4]);
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; 8];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![3, 4, 0, 1, 2]);
        assert!(result.distances.iter().all(|x| *x > 0.));
    }
}
