//! Module for the ID map wrapper.
//!
//! Most index implementations will bind a sequential ID to each vector by default.
//! However, some specific implementations support binding each vector an arbitrary
//! ID. When supported, this can be done with the [`Index#add_with_ids`] method.
//! Please see the [Faiss wiki] for more information.
//!
//! For implementations which do not support arbitrary IDs, this module provides
//! the [`IdMap`] wrapper type. An `IdMap<I>` retains the algorithm and compile
//! time properties of the index type `I`, while ensuring the extra ID mapping
//! functionality.
//!
//! [`Index#add_with_ids`]: ../trait.Index.html#tymethod.add_with_ids
//! [Faiss wiki]: https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing#faiss-id-mapping
//! [`IdMap`]: struct.IdMap.html
//!
//! # Examples
//!
//! A flat index does not support arbitrary ID mapping, but `IdMap` solves this:
//!
//! ```
//! use faiss::{IdMap, Idx, Index, FlatIndex};
//! # fn run() -> Result<(), Box<dyn std::error::Error>>  {
//! let mut index = FlatIndex::new_l2(4)?;
//! assert!(index.add_with_ids(&[0., 1., 0., 1.], &[Idx::new(5)]).is_err());
//!
//! let mut index = IdMap::new(index)?;
//! index.add_with_ids(&[0., 1., 0., 1.], &[Idx::new(5)])?;
//! assert_eq!(index.ntotal(), 1);
//! # Ok(())
//! # }
//! # run().unwrap();
//! ```
//!
//! `IdMap` also works for GPU backed indexes, but the index map will reside
//! in CPU memory. Once an index map is made, moving an index to/from the GPU
//! is not possible.
//!
//! ```
//! # #[cfg(feature = "gpu")]
//! # use faiss::{GpuResources, StandardGpuResources, Index, FlatIndex, IdMap};
//! # #[cfg(feature = "gpu")]
//! # use faiss::error::Result;
//! # #[cfg(feature = "gpu")]
//! # fn run() -> Result<()> {
//! let index = FlatIndex::new_l2(8)?;
//! let gpu_res = StandardGpuResources::new()?;
//! let index: IdMap<_> = IdMap::new(index.into_gpu(&gpu_res, 0)?)?;
//! # Ok(())
//! # }
//! # #[cfg(feature = "gpu")]
//! # run().unwrap()
//! ```
//!

use crate::error::{Error, Result};
use crate::index::{
    self, AssignSearchResult, ConcurrentIndex, CpuIndex, FromInnerPtr, Idx, Index, NativeIndex,
    RangeSearchResult, SearchResult,
};
use crate::selector::IdSelector;
use crate::{faiss_try, MetricType};
use faiss_sys::*;

use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_int;
use std::ptr;

use super::IndexImpl;

/// Wrapper for implementing arbitrary ID mapping to an index.
///
/// See the [module level documentation] for more information.
///
/// [module level documentation]: ./index.html
#[derive(Debug)]
pub struct IdMap<I> {
    inner: *mut FaissIndexIDMap,
    index_inner: *mut FaissIndex,
    phantom: PhantomData<I>,
}

unsafe impl<I: Send> Send for IdMap<I> {}
unsafe impl<I: Sync> Sync for IdMap<I> {}
impl<I: CpuIndex> CpuIndex for IdMap<I> {}

impl<I> NativeIndex for IdMap<I> {
    type Inner = FaissIndex;
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl<I> Drop for IdMap<I> {
    fn drop(&mut self) {
        unsafe {
            faiss_Index_free(self.inner);
        }
    }
}

impl<I> IdMap<I>
where
    I: NativeIndex<Inner = FaissIndex>,
{
    /// Augment an index with arbitrary ID mapping.
    pub fn new(index: I) -> Result<Self> {
        unsafe {
            let index_inner = index.inner_ptr();
            let mut inner_ptr = ptr::null_mut();
            faiss_try(faiss_IndexIDMap_new(&mut inner_ptr, index_inner))?;
            // let IDMap take ownership of the index
            faiss_IndexIDMap_set_own_fields(inner_ptr, 1);
            mem::forget(index);

            Ok(IdMap {
                inner: inner_ptr,
                index_inner,
                phantom: PhantomData,
            })
        }
    }

    /// Retrieve a slice of the internal ID map.
    pub fn id_map(&self) -> &[Idx] {
        unsafe {
            let mut id_ptr = ptr::null_mut();
            let mut psize = 0;
            faiss_IndexIDMap_id_map(self.inner, &mut id_ptr, &mut psize);
            ::std::slice::from_raw_parts(id_ptr as *const _, psize)
        }
    }

    /// Obtain the raw pointer to the internal index.
    ///
    /// # Safety
    ///
    /// While this method is safe, note that the returned index pointer is
    /// already owned by this ID map. Therefore, it is undefined behavior to
    /// create a high-level index value from this pointer without first
    /// decoupling this ownership. See [`into_inner`](Self::into_inner) for a safe alternative.
    pub fn index_inner_ptr(&self) -> *mut FaissIndex {
        self.index_inner
    }

    /// Discard the ID map, recovering the index originally created without it.
    pub fn into_inner(self) -> I
    where
        I: FromInnerPtr,
    {
        unsafe {
            // make id map disown the index
            faiss_IndexIDMap_set_own_fields(self.inner, 0);
            // now it's safe to build a managed index
            // (`index_inner` is expected to always point to a valid index)
            I::from_inner_ptr(self.index_inner)
        }
    }

    /// Discard the ID map, recovering the index originally created without it.
    /// Safety build managed index from pointer.
    pub fn try_into_inner(self) -> Result<I>
    where
        I: index::TryFromInnerPtr,
    {
        unsafe {
            // make id map disown the index
            faiss_IndexIDMap_set_own_fields(self.inner, 0);
            // now it's safe to build a managed index
            // (`index_inner` is expected to always point to a valid index)
            I::try_from_inner_ptr(self.index_inner)
        }
    }

    /// Specialization of the index type inside `IdMap`.
    pub fn try_cast_inner_index<B>(self) -> Result<IdMap<B>>
    where
        B: index::TryFromInnerPtr<Inner = FaissIndex>,
    {
        // safety: index_inner is expected to always point to a valid index
        let r = unsafe { B::try_from_inner_ptr(self.index_inner) };
        if let Ok(index) = r {
            let res = IdMap {
                inner: self.inner,
                index_inner: index.inner_ptr(),
                phantom: PhantomData,
            };
            mem::forget(index);
            mem::forget(self);

            Ok(res)
        } else {
            Err(Error::BadCast)
        }
    }
}

impl<I> Index for IdMap<I> {
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
        unsafe { faiss_Index_verbose(self.inner_ptr()) != 0 }
    }

    fn set_verbose(&mut self, value: bool) {
        unsafe {
            faiss_Index_set_verbose(self.inner_ptr(), c_int::from(value));
        }
    }

    
            
    fn reconstruct(
        &self,
        idx: Idx,
        output: &mut [f32]
    ) -> Result<()> {
        unsafe {
            let d = self.d() as usize;
            if d != output.len() {
                return Err(crate::error::Error::BadDimension);
            }
            
            faiss_try(faiss_Index_reconstruct(
                self.inner_ptr(),
                idx.0,
                output.as_mut_ptr()
            ))?;

            Ok(())
        }
    }

    fn reconstruct_n(
        &self, 
        first_key: Idx, 
        count: usize, 
        output: &mut [f32]
    ) -> Result<()> {
        unsafe {
            let d = self.d() as usize;
            if count * d != output.len() {
                return Err(crate::error::Error::BadDimension);
            }
            
            faiss_try(faiss_Index_reconstruct_n(
                self.inner_ptr(),
                first_key.0,
                count as i64,
                output.as_mut_ptr()
            ))?;

            Ok(())
        }
    }
}

impl<I> ConcurrentIndex for IdMap<I>
where
    I: ConcurrentIndex,
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
    /// Attempt a dynamic cast of the index to one that is [ID-mapped][1].
    ///
    /// [1]: crate::IdMap
    pub fn into_id_map(self) -> Result<IdMap<IndexImpl>> {
        unsafe {
            let new_inner = faiss_IndexIDMap_cast(self.inner_ptr());
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                mem::forget(self);
                let index_inner = faiss_IndexIDMap_sub_index(new_inner);
                Ok(IdMap {
                    inner: new_inner,
                    index_inner,
                    phantom: PhantomData,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::IdMap;
    use crate::index::{flat::FlatIndexImpl, index_factory, Idx, Index, IndexImpl};
    use crate::selector::IdSelector;
    use crate::MetricType;

    #[test]
    fn flat_index_search_ids() {
        let index = index_factory(8, "Flat", MetricType::L2).unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        let some_ids = &[
            Idx::new(3),
            Idx::new(6),
            Idx::new(9),
            Idx::new(12),
            Idx::new(15),
        ];
        let mut index = IdMap::new(index).unwrap();
        index.add_with_ids(some_data, some_ids).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; 8];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![9, 6, 3, 12, 15]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; 8];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![12, 15, 3, 6, 9]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 100., 100., 100., 100., 100., 100., 100., 100.,
        ];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![9, 6, 3, 12, 15, 12, 15, 3, 6, 9]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));
    }

    #[test]
    fn index_remove_ids() {
        let index = index_factory(4, "Flat", MetricType::L2).unwrap();
        let mut id_index = IdMap::new(index).unwrap();
        let some_data = &[2.3_f32, 0.0, -1., 1., 1., 1., 1., 4.5, 2.3, 7.6, 1., 2.2];

        id_index
            .add_with_ids(some_data, &[Idx::new(4), Idx::new(8), Idx::new(12)])
            .unwrap();
        assert_eq!(id_index.ntotal(), 3);

        let id_sel = IdSelector::batch(&[Idx::new(4), Idx::new(12)])
            .ok()
            .unwrap();

        id_index.remove_ids(&id_sel).unwrap();
        assert_eq!(id_index.ntotal(), 1);
    }

    #[test]
    fn try_from_inner_ptr() {
        let index = index_factory(4, "Flat", MetricType::L2).unwrap();
        let id_index = IdMap::new(index).unwrap();

        let index: IndexImpl = id_index.try_into_inner().unwrap();
        assert_eq!(index.d(), 4);
    }

    #[test]
    fn try_cast_inner_index() {
        let index = index_factory(4, "Flat", MetricType::L2).unwrap();
        let id_index = IdMap::new(index).unwrap();

        let index: IdMap<FlatIndexImpl> = id_index.try_cast_inner_index::<FlatIndexImpl>().unwrap();
        assert_eq!(index.d(), 4);
    }

    #[test]
    fn flat_try_from_inner_ptr() {
        let index = FlatIndexImpl::new_l2(4).unwrap();
        let id_index = IdMap::new(index).unwrap();

        let flat_index: FlatIndexImpl = id_index.try_into_inner().unwrap();
        assert_eq!(flat_index.d(), 4);
    }

    #[test]
    fn index_impl_to_id_map() {
        let index = index_factory(4, "IDMap,Flat", MetricType::L2).unwrap();
        let id_map = index.into_id_map().unwrap();

        assert_eq!(id_map.d(), 4);
    }
}
