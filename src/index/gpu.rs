//! GPU Index implementation

use std::marker::PhantomData;
use std::mem;
use std::ptr;
use faiss_sys::*;
use error::Result;
use gpu::GpuResources;
use metric::MetricType;
use super::{AssignSearchResult, CpuIndex, FromInnerPtr, Idx, Index, IndexImpl, NativeIndex,
            RangeSearchResult, SearchResult};
use super::flat::FlatIndexImpl;

/// Trait for Faiss index types known to be running on the GPU.
pub trait GpuIndex: Index {}

/// Native GPU implementation of a Faiss index. GPU indexes in Faiss are first
/// built on the CPU, and subsequently transferred to one or more GPU's via the
/// [`into_gpu`] method. Calling [`to_cpu`] enables the user to bring the index
/// back to CPU memory.
/// 
/// The `'gpu` lifetime ensures that the [GPU resources] are in scope for as
/// long as the index lives.
///
/// [`into_gpu`]: ../struct.IndexImpl.html#method.into_gpu
/// [`into_cpu`]: struct.GpuIndexImpl.html#method.into_cpu
#[derive(Debug)]
pub struct GpuIndexImpl<'gpu, I> {
    /// We keep the descriptor of the original (CPU) index, so it's
    /// cleaned up alongside the GPU index. It also allows us to
    /// recover the original CPU index type.
    index: I,
    inner: *mut FaissGpuIndex,
    phantom: PhantomData<&'gpu ()>,
}

impl<'g, I> GpuIndex for GpuIndexImpl<'g, I>
where
    I: NativeIndex,
{
}

// `GpuIndexImpl` is deliberately not `Sync`!
unsafe impl<'g, I> Send for GpuIndexImpl<'g, I>
where
    I: Send,
{
}

impl<'g, I> Drop for GpuIndexImpl<'g, I> {
    fn drop(&mut self) {
        unsafe {
            faiss_Index_free(self.inner);
        }
    }
}

impl<'g, I> GpuIndexImpl<'g, I>
where
    I: CpuIndex,
{
    /// Build a GPU in from the given CPU native index. The operation fails if the
    /// index does not provide GPU support.
    /// Users should prefer calling [`into_gpu`] instead.
    /// 
    /// [`into_gpu`]: ../struct.IndexImpl.html#method.into_gpu
    pub fn from_cpu<G>(index: I, gpu_res: &G, device: i32) -> Result<Self>
    where
        I: NativeIndex,
        I: CpuIndex,
        G: GpuResources,
    {
        unsafe {
            let mut gpuindex_ptr = ptr::null_mut();
            faiss_try!(faiss_index_cpu_to_gpu(
                gpu_res.inner_ptr(),
                device,
                index.inner_ptr(),
                &mut gpuindex_ptr
            ));
            Ok(GpuIndexImpl {
                index: index,
                inner: gpuindex_ptr,
                phantom: PhantomData,
            })
        }
    }
}

impl IndexImpl {
    /// Build a GPU in from the given CPU native index. The operation fails if the
    /// index does not provide GPU support.
    pub fn into_gpu<'gpu, G: 'gpu>(
        self,
        gpu_res: &'gpu G,
        device: i32,
    ) -> Result<GpuIndexImpl<'gpu, IndexImpl>>
    where
        G: GpuResources,
    {
        GpuIndexImpl::from_cpu(self, gpu_res, device)
    }
}

impl<'gpu, I> GpuIndexImpl<'gpu, I>
where
    I: NativeIndex,
    I: FromInnerPtr,
{
    /// Transfer the GPU index back to its original CPU implementation.
    pub fn into_cpu(self) -> Result<I> {
        unsafe {
            let mut cpuindex_ptr = ptr::null_mut();
            faiss_try!(faiss_index_gpu_to_cpu(self.inner, &mut cpuindex_ptr));
            mem::forget(self); // don't free the index
            Ok(I::from_inner_ptr(cpuindex_ptr))
        }
    }
}

impl<'gpu, I> Index for GpuIndexImpl<'gpu, I>
where
    I: Index,
{
    fn is_trained(&self) -> bool {
        self.index.is_trained()
    }

    fn ntotal(&self) -> u64 {
        self.index.ntotal()
    }

    fn d(&self) -> u32 {
        self.index.d()
    }

    fn metric_type(&self) -> MetricType {
        self.index.metric_type()
    }

    fn add(&mut self, x: &[f32]) -> Result<()> {
        self.index.add(x)
    }

    fn add_with_ids(&mut self, x: &[f32], xids: &[Idx]) -> Result<()> {
        self.index.add_with_ids(x, xids)
    }

    fn train(&mut self, x: &[f32]) -> Result<()> {
        self.index.train(x)
    }

    fn assign(&mut self, query: &[f32], k: usize) -> Result<AssignSearchResult> {
        self.index.assign(query, k)
    }

    fn search(&mut self, query: &[f32], k: usize) -> Result<SearchResult> {
        self.index.search(query, k)
    }

    fn range_search(&mut self, query: &[f32], radius: f32) -> Result<RangeSearchResult> {
        self.index.range_search(query, radius)
    }

    fn reset(&mut self) -> Result<()> {
        self.index.reset()
    }
}

impl<'g, I> NativeIndex for GpuIndexImpl<'g, I>
where
    I: NativeIndex,
{
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl FlatIndexImpl {
    /// Build a GPU in from the given CPU native index. The operation fails if the
    /// index does not provide GPU support.
    pub fn into_gpu<'gpu, G>(
        self,
        gpu_res: &'gpu G,
        device: i32,
    ) -> Result<GpuIndexImpl<'gpu, FlatIndexImpl>>
    where
        G: GpuResources,
    {
        unsafe {
            let mut gpuindex_ptr = ptr::null_mut();
            faiss_try!(faiss_index_cpu_to_gpu(
                gpu_res.inner_ptr(),
                device,
                self.inner_ptr(),
                &mut gpuindex_ptr
            ));
            Ok(GpuIndexImpl {
                index: self,
                inner: gpuindex_ptr,
                phantom: PhantomData,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::index_factory;
    use super::super::{CpuIndex, Index};
    use super::super::flat::FlatIndex;
    use super::GpuIndex;
    use gpu::{GpuResources, StandardGpuResources};
    use metric::MetricType;

    fn is_in_gpu<I: GpuIndex>(_: &I) {}
    fn is_in_cpu<I: CpuIndex>(_: &I) {}

    #[test]
    fn flat_in_and_out() {
        let mut res = StandardGpuResources::new().unwrap();
        res.set_temp_memory(10).unwrap();

        let mut index: FlatIndex = FlatIndex::new(8, MetricType::L2).unwrap();
        assert_eq!(index.d(), 8);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();

        for _ in 0..10 {
            let gpu_index = index.into_gpu(&res, 0).unwrap();
            is_in_gpu(&gpu_index);
            index = gpu_index.into_cpu().unwrap();
            is_in_cpu(&index);
        }
    }

    #[test]
    fn flat_index_search() {
        let res = StandardGpuResources::new().unwrap();

        let mut index = index_factory(8, "Flat", MetricType::L2)
            .unwrap()
            .into_gpu(&res, 0)
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
