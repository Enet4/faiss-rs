//! GPU Index implementation

use std::marker::PhantomData;
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
/// [`into_gpu`] or [`to_gpu`] methods. Calling [`into_cpu`] (or [`to_cpu`])
/// enables the user to bring the index back to CPU memory.
/// 
/// When using [`to_gpu`]() or [`to_cpu`](), the indexes will contain the same
/// indexed vectors, but are independent at the point of creation. The use of
/// [`into_gpu`] or [`into_cpu`] isn't necessarily faster, but will automatically
/// free the originating index.
/// 
/// The `'gpu` lifetime ensures that the [GPU resources] are in scope for as
/// long as the index lives.
///
/// [`into_gpu`]: ../struct.IndexImpl.html#method.into_gpu
/// [`to_gpu`]: ../struct.IndexImpl.html#method.to_gpu
/// [`into_cpu`]: struct.GpuIndexImpl.html#method.into_cpu
/// [`to_cpu`]: struct.GpuIndexImpl.html#method.to_cpu
/// [GPU resources]: ../gpu/index.html
#[derive(Debug)]
pub struct GpuIndexImpl<'gpu, I> {
    inner: *mut FaissGpuIndex,
    /// retaining the GPU resources' lifetime,
    /// plus the original index type `I`
    phantom: PhantomData<(&'gpu (), I)>,
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
    /// Users will indirectly use this through [`to_gpu`] or [`into_gpu`].
    /// 
    /// [`to_gpu`]: ../struct.IndexImpl.html#method.to_gpu
    /// [`into_gpu`]: ../struct.IndexImpl.html#method.into_gpu
    pub(crate) fn from_cpu<G>(index: &I, gpu_res: &G, device: i32) -> Result<Self>
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
                inner: gpuindex_ptr,
                phantom: PhantomData,
            })
        }
    }
}

impl IndexImpl {
    /// Build a GPU index from the given CPU native index. The operation fails if the
    /// index type does not provide GPU support.
    pub fn to_gpu<'gpu, G: 'gpu>(
        &self,
        gpu_res: &'gpu G,
        device: i32,
    ) -> Result<GpuIndexImpl<'gpu, IndexImpl>>
    where
        G: GpuResources,
    {
        GpuIndexImpl::from_cpu(&self, gpu_res, device)
    }

    /// Build a GPU index from the given CPU native index. The operation fails if the
    /// index does not provide GPU support.
    pub fn into_gpu<'gpu, G: 'gpu>(
        self,
        gpu_res: &'gpu G,
        device: i32,
    ) -> Result<GpuIndexImpl<'gpu, IndexImpl>>
    where
        G: GpuResources,
    {
        self.to_gpu(gpu_res, device)
        // let the CPU index drop naturally
    }
}

impl<'gpu, I> GpuIndexImpl<'gpu, I>
where
    I: NativeIndex,
    I: FromInnerPtr,
{
    /// Transfer the GPU index back to its original CPU implementation.
    pub fn to_cpu(&self) -> Result<I> {
        unsafe {
            let mut cpuindex_ptr = ptr::null_mut();
            faiss_try!(faiss_index_gpu_to_cpu(self.inner, &mut cpuindex_ptr));
            Ok(I::from_inner_ptr(cpuindex_ptr))
        }
    }

    /// Transfer the GPU index back to its original CPU implementation,
    /// freeing the GPU-backed index in the process.
    pub fn into_cpu(self) -> Result<I> {
        self.to_cpu()
        // let the GPU index drop naturally
    }
}

impl<'gpu, I> Index for GpuIndexImpl<'gpu, I>
where
    I: Index,
{
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

impl<'g, I> NativeIndex for GpuIndexImpl<'g, I>
where
    I: NativeIndex,
{
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl FlatIndexImpl {
    /// Build a GPU in from the given CPU native index, yielding two
    /// independent indices. The operation fails if the index does
    /// not provide GPU support.
    pub fn to_gpu<'gpu, G>(
        &self,
        gpu_res: &'gpu G,
        device: i32,
    ) -> Result<GpuIndexImpl<'gpu, FlatIndexImpl>>
    where
        G: GpuResources,
    {
        GpuIndexImpl::from_cpu(self, gpu_res, device)
    }
    
    /// Build a GPU in from the given CPU native index, discarding the
    /// CPU-backed index. The operation fails if the index does not
    /// provide GPU support.
    pub fn into_gpu<'gpu, G>(
        self,
        gpu_res: &'gpu G,
        device: i32,
    ) -> Result<GpuIndexImpl<'gpu, FlatIndexImpl>>
    where
        G: GpuResources,
    {
        self.to_gpu(gpu_res, device)
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
        assert_eq!(index.ntotal(), 5);

        let mut gpu_index = index.into_gpu(&res, 0).unwrap();
        is_in_gpu(&gpu_index);
        for _ in 0..3 {
            let index = gpu_index.into_cpu().unwrap();
            is_in_cpu(&index);
            gpu_index = index.into_gpu(&res, 0).unwrap();
            is_in_gpu(&gpu_index);
        }
        assert_eq!(gpu_index.ntotal(), 5); // indexed vectors should be retained
    }

    #[test]
    fn flat_index_search_into_gpu() {
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

        // now back to the CPU
        let mut index = index.into_cpu().unwrap();

        let my_query = [0.; 8];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![2, 1, 0, 3, 4]);
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; 8];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![3, 4, 0, 1, 2]);
        assert!(result.distances.iter().all(|x| *x > 0.));
    }

    #[test]
    fn flat_index_search_to_gpu() {
        let res = StandardGpuResources::new().unwrap();

        let mut index_cpu = index_factory(8, "Flat", MetricType::L2)
            .unwrap();
        let mut index_gpu = index_cpu.to_gpu(&res, 0)
            .unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index_cpu.add(some_data).unwrap();
        assert_eq!(index_cpu.ntotal(), 5);

        let my_query = [0.; 8];
        let result = index_cpu.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![2, 1, 0, 3, 4]);
        assert!(result.distances.iter().all(|x| *x > 0.));

        index_gpu.add(some_data).unwrap();
        assert_eq!(index_gpu.ntotal(), 5);

        let my_query = [0.; 8];
        let result = index_gpu.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![2, 1, 0, 3, 4]);
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; 8];
        let result = index_cpu.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![3, 4, 0, 1, 2]);
        assert!(result.distances.iter().all(|x| *x > 0.));

        // add more data to CPU index, see it in effect
        // (but it won't be visible on the GPU)
        let more_data = &[32.; 8];
        index_cpu.add(more_data).unwrap();
        assert_eq!(index_cpu.ntotal(), 6);
        assert_eq!(index_gpu.ntotal(), 5);

        let my_query = [0.; 8];
        let result = index_cpu.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![2, 1, 0, 5, 3]);
        assert!(result.distances.iter().all(|x| *x > 0.));

        drop(index_cpu);
        // won't use CPU index anymore, but GPU index should still work

        let my_query = [100.; 8];
        let result = index_gpu.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![3, 4, 0, 1, 2]);
        assert!(result.distances.iter().all(|x| *x > 0.));
    }
}
