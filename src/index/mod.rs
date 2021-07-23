//! Index interface and native implementations.
//!
//! This module hosts the vital [`Index`] trait, through which index building
//! and searching is made. It also contains [`index_factory`], a generic
//! function through which the user can retrieve most of the available index
//! implementations. A very typical usage scenario of this crate is to create
//! the index through this function, but some statically verified index types
//! are available as well.
//!
//! [`Index`]: trait.Index.html
//! [`index_factory`]: fn.index_factory.html

use crate::error::{Error, Result};
use crate::faiss_try;
use crate::metric::MetricType;
use crate::selector::IdSelector;
use std::ffi::CString;
use std::fmt::{self, Display, Formatter, Write};
use std::os::raw::c_uint;
use std::{mem, ptr};

use faiss_sys::*;

pub mod autotune;
pub mod flat;
pub mod id_map;
pub mod io;
pub mod ivf_flat;
pub mod lsh;
pub mod pretransform;
pub mod refine_flat;
pub mod scalar_quantizer;

#[cfg(feature = "gpu")]
pub mod gpu;

/// Primitive data type for identifying a vector in an index (or lack thereof).
///
/// Depending on the kind of index, it may be possible for vectors to share the
/// same ID.
#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
pub struct Idx(idx_t);

impl Display for Idx {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self.get() {
            None => f.write_char('x'),
            Some(i) => i.fmt(f),
        }
    }
}

impl From<idx_t> for Idx {
    fn from(x: idx_t) -> Self {
        Idx(x)
    }
}

impl Idx {
    /// Create a vector identifier.
    ///
    /// # Panic
    ///
    /// Panics if the ID is too large (`>= 2^63`)
    #[inline]
    pub fn new(idx: u64) -> Self {
        assert!(
            idx < 0x8000_0000_0000_0000,
            "too large index value provided to Idx::new"
        );
        let idx = idx as idx_t;
        Idx(idx)
    }

    /// Create an identifier referring to no vector.
    #[inline]
    pub fn none() -> Self {
        Idx(-1)
    }

    /// Check whether the vector identifier does not point to anything.
    #[inline]
    pub fn is_none(self) -> bool {
        self.0 == -1
    }

    /// Check whether the vector identifier is not "none".
    #[inline]
    pub fn is_some(self) -> bool {
        self.0 != -1
    }

    /// Retrieve the vector identifier as a primitive unsigned integer.
    pub fn get(self) -> Option<u64> {
        match self.0 {
            -1 => None,
            x => Some(x as u64),
        }
    }

    /// Convert the vector identifier into a native `idx_t` value.
    pub fn to_native(self) -> idx_t {
        self.0
    }
}

/// This comparison is not entirely reflexive: it returns always `false` if at
/// least one of the values is a `none`.
impl PartialEq<Idx> for Idx {
    fn eq(&self, idx: &Idx) -> bool {
        self.0 != -1 && idx.0 != -1 && self.0 == idx.0
    }
}

/// This comparison is not entirely reflexive: it returns always `None` if at
/// least one of the values is a `none`.
impl PartialOrd<Idx> for Idx {
    fn partial_cmp(&self, idx: &Idx) -> Option<std::cmp::Ordering> {
        match (self.get(), idx.get()) {
            (None, _) => None,
            (_, None) => None,
            (Some(a), Some(b)) => Some(a.cmp(&b)),
        }
    }
}

/// Interface for a Faiss index. Most methods in this trait match the ones in
/// the native library, whereas some others serve as getters to the index'
/// parameters.
///
/// Although all methods appear to be available for all index implementations,
/// some methods may not be supported. For instance, a [`FlatIndex`] stores
/// vectors sequentially, and so does not support `add_with_ids` nor
/// `remove_ids`. Users are advised to read the Faiss wiki pages in order
/// to understand which index algorithms support which operations.
///
/// [`FlatIndex`]: flat/struct.FlatIndex.html
pub trait Index {
    /// Whether the Index does not require training, or if training is done already
    fn is_trained(&self) -> bool;

    /// The total number of vectors indexed
    fn ntotal(&self) -> u64;

    /// The dimensionality of the indexed vectors
    fn d(&self) -> u32;

    /// The metric type assumed by the index
    fn metric_type(&self) -> MetricType;

    /// Add new data vectors to the index.
    /// This assumes a C-contiguous memory slice of vectors, where the total
    /// number of vectors is `x.len() / d`.
    fn add(&mut self, x: &[f32]) -> Result<()>;

    /// Add new data vectors to the index with IDs.
    /// This assumes a C-contiguous memory slice of vectors, where the total
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

    /// Remove data vectors represented by IDs.
    fn remove_ids(&mut self, sel: &IdSelector) -> Result<usize>;

    /// Index verbosity level
    fn verbose(&self) -> bool;

    /// Set Index verbosity level
    fn set_verbose(&mut self, value: bool);
}

impl<I> Index for Box<I>
where
    I: Index,
{
    fn is_trained(&self) -> bool {
        (**self).is_trained()
    }

    fn ntotal(&self) -> u64 {
        (**self).ntotal()
    }

    fn d(&self) -> u32 {
        (**self).d()
    }

    fn metric_type(&self) -> MetricType {
        (**self).metric_type()
    }

    fn add(&mut self, x: &[f32]) -> Result<()> {
        (**self).add(x)
    }

    fn add_with_ids(&mut self, x: &[f32], xids: &[Idx]) -> Result<()> {
        (**self).add_with_ids(x, xids)
    }

    fn train(&mut self, x: &[f32]) -> Result<()> {
        (**self).train(x)
    }

    fn assign(&mut self, q: &[f32], k: usize) -> Result<AssignSearchResult> {
        (**self).assign(q, k)
    }

    fn search(&mut self, q: &[f32], k: usize) -> Result<SearchResult> {
        (**self).search(q, k)
    }

    fn range_search(&mut self, q: &[f32], radius: f32) -> Result<RangeSearchResult> {
        (**self).range_search(q, radius)
    }

    fn reset(&mut self) -> Result<()> {
        (**self).reset()
    }

    fn remove_ids(&mut self, sel: &IdSelector) -> Result<usize> {
        (**self).remove_ids(sel)
    }

    fn verbose(&self) -> bool {
        (**self).verbose()
    }

    fn set_verbose(&mut self, value: bool) {
        (**self).set_verbose(value)
    }
}

/// Sub-trait for native implementations of a Faiss index.
pub trait NativeIndex: Index {
    /// Retrieve a pointer to the native index object.
    fn inner_ptr(&self) -> *mut FaissIndex;
}

impl<NI: NativeIndex> NativeIndex for Box<NI> {
    fn inner_ptr(&self) -> *mut FaissIndex {
        (**self).inner_ptr()
    }
}

/// Trait for a Faiss index that can be safely searched over multiple threads.
/// Operations which do not modify the index are given a method taking an
/// immutable reference. This is not the default for every index type because
/// some implementations (such as the ones running on the GPU) do not allow
/// concurrent searches.
///
/// Users of these methods should still note that batched querying is
/// considerably faster than running queries one by one, even in parallel.
pub trait ConcurrentIndex: Index {
    /// Similar to `search`, but only provides the labels.
    fn assign(&self, q: &[f32], k: usize) -> Result<AssignSearchResult>;

    /// Perform a search for the `k` closest vectors to the given query vectors.
    fn search(&self, q: &[f32], k: usize) -> Result<SearchResult>;

    /// Perform a ranged search for the vectors closest to the given query vectors
    /// by the given radius.
    fn range_search(&self, q: &[f32], radius: f32) -> Result<RangeSearchResult>;
}

impl<CI: ConcurrentIndex> ConcurrentIndex for Box<CI> {
    fn assign(&self, q: &[f32], k: usize) -> Result<AssignSearchResult> {
        (**self).assign(q, k)
    }

    fn search(&self, q: &[f32], k: usize) -> Result<SearchResult> {
        (**self).search(q, k)
    }

    fn range_search(&self, q: &[f32], radius: f32) -> Result<RangeSearchResult> {
        (**self).range_search(q, radius)
    }
}

/// Trait for Faiss index types known to be running on the CPU.
pub trait CpuIndex: Index {}

impl<CI: CpuIndex> CpuIndex for Box<CI> {}

/// Trait for Faiss index types which can be built from a pointer
/// to a native implementation.
pub trait FromInnerPtr: NativeIndex {
    /// Create an index using the given pointer to a native object.
    ///
    /// # Safety
    ///
    /// `inner_ptr` must point to a valid, non-freed CPU index, and cannot be
    /// shared across multiple instances. The inner index must also be
    /// compatible with the target `NativeIndex` type according to the native
    /// class hierarchy. For example, creating an `IndexImpl` out of a pointer
    /// to `FaissIndexFlatL2` is valid, but creating a `FlatIndex` out of a
    /// plain `FaissIndex` can cause undefined behavior.
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self;
}

/// Trait for Faiss index types which can be built from a pointer
/// to a native implementation.
pub trait TryFromInnerPtr: NativeIndex {
    fn try_from_inner_ptr(inner_ptr: *mut FaissIndex) -> Result<Self>
    where
        Self: Sized;
}

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
        unsafe { faiss_RangeSearchResult_nq(self.inner) }
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
        let full_len = lims.last().cloned().unwrap_or(0);
        unsafe {
            let mut distances_ptr = ptr::null_mut();
            let mut labels_ptr = ptr::null_mut();
            faiss_RangeSearchResult_labels(self.inner, &mut labels_ptr, &mut distances_ptr);
            let distances = ::std::slice::from_raw_parts(distances_ptr, full_len);
            let labels = ::std::slice::from_raw_parts(labels_ptr as *const Idx, full_len);
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
            let labels = ::std::slice::from_raw_parts_mut(labels_ptr as *mut Idx, buf_size);
            (distances, labels)
        }
    }

    /// getter for distances (not sorted):
    /// result for query `i` is `distances[lims[i] .. lims[i+1]]`
    pub fn distances(&self) -> &[f32] {
        self.distance_and_labels().0
    }

    /// getter for distances (not sorted):
    /// result for query `i` is `distances[lims[i] .. lims[i+1]]`
    pub fn distances_mut(&mut self) -> &mut [f32] {
        self.distance_and_labels_mut().0
    }

    /// getter for labels (not sorted):
    /// result for query `i` is `labels[lims[i] .. lims[i+1]]`
    pub fn labels(&self) -> &[Idx] {
        self.distance_and_labels().1
    }

    /// getter for labels (not sorted):
    /// result for query `i` is `labels[lims[i] .. lims[i+1]]`
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

/// Native implementation of a Faiss Index
/// running on the CPU.
#[derive(Debug)]
pub struct IndexImpl {
    inner: *mut FaissIndex,
}

unsafe impl Send for IndexImpl {}
unsafe impl Sync for IndexImpl {}

impl CpuIndex for IndexImpl {}

impl Drop for IndexImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_Index_free(self.inner);
        }
    }
}

impl IndexImpl {
    pub fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl NativeIndex for IndexImpl {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl FromInnerPtr for IndexImpl {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        IndexImpl { inner: inner_ptr }
    }
}

impl TryFromInnerPtr for IndexImpl {
    fn try_from_inner_ptr(inner_ptr: *mut FaissIndex) -> Result<Self>
    where
        Self: Sized,
    {
        if inner_ptr.is_null() {
            Err(Error::BadCast)
        } else {
            Ok(IndexImpl { inner: inner_ptr })
        }
    }
}

/// Index upcast trait.
/// 
/// If you need to store several different types of indexes in one collection,
/// you can cast all indexes to the common type `IndexImpl`.
/// # Examples
///
/// ```
/// # use faiss::{index::{IndexImpl, UpcastIndex}, FlatIndex, index_factory, MetricType};
/// let f1 = FlatIndex::new_l2(128).unwrap();
/// let f2 = index_factory(128, "Flat", MetricType::L2).unwrap();
/// let v: Vec<IndexImpl> = vec![
///     f1.upcast(),
///     f2,
/// ];
/// ```
///
pub trait UpcastIndex: NativeIndex {
    /// Convert an index to the base `IndexImpl` type
    fn upcast(self) -> IndexImpl;
}

impl<NI: NativeIndex> UpcastIndex for NI {
    fn upcast(self) -> IndexImpl {
        let inner_ptr = self.inner_ptr();
        mem::forget(self);

        unsafe { IndexImpl::from_inner_ptr(inner_ptr) }
    }
}

impl_native_index!(IndexImpl);

impl_native_index_clone!(IndexImpl);

/// Use the index factory to create a native instance of a Faiss index, for `d`-dimensional
/// vectors. `description` should follow the exact guidelines as the native Faiss interface
/// (see the [Faiss wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) for examples).
///
/// # Error
///
/// This function returns an error if the description contains any byte with the value `\0` (since
/// it cannot be converted to a C string), or if the internal index factory operation fails.
pub fn index_factory<D>(d: u32, description: D, metric: MetricType) -> Result<IndexImpl>
where
    D: AsRef<str>,
{
    unsafe {
        let metric = metric as c_uint;
        let description =
            CString::new(description.as_ref()).map_err(|_| Error::IndexDescription)?;
        let mut index_ptr = ::std::ptr::null_mut();
        faiss_try(faiss_index_factory(
            &mut index_ptr,
            (d & 0x7FFF_FFFF) as i32,
            description.as_ptr(),
            metric,
        ))?;
        Ok(IndexImpl { inner: index_ptr })
    }
}

#[cfg(test)]
mod tests {
    use super::{index_factory, Idx, Index};
    use crate::metric::MetricType;

    #[test]
    fn index_factory_flat() {
        let index = index_factory(64, "Flat", MetricType::L2).unwrap();
        assert_eq!(index.is_trained(), true); // Flat index does not need training
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn index_factory_flat_boxed() {
        let index = index_factory(64, "Flat", MetricType::L2).unwrap();
        let boxed = Box::new(index);
        assert_eq!(boxed.is_trained(), true); // Flat index does not need training
        assert_eq!(boxed.ntotal(), 0);
    }

    #[test]
    fn index_factory_ivf_flat() {
        let index = index_factory(64, "IVF8,Flat", MetricType::L2).unwrap();
        assert_eq!(index.is_trained(), false);
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn index_factory_sq() {
        let index = index_factory(64, "SQ8", MetricType::L2).unwrap();
        assert_eq!(index.is_trained(), false);
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn index_factory_pq() {
        let index = index_factory(64, "PQ8", MetricType::L2).unwrap();
        assert_eq!(index.is_trained(), false);
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn index_factory_ivf_sq() {
        let index = index_factory(64, "IVF8,SQ4", MetricType::L2).unwrap();
        assert_eq!(index.is_trained(), false);
        assert_eq!(index.ntotal(), 0);

        let index = index_factory(64, "IVF8,SQ8", MetricType::L2).unwrap();
        assert_eq!(index.is_trained(), false);
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn index_factory_hnsw() {
        let index = index_factory(64, "HNSW8", MetricType::L2).unwrap();
        assert_eq!(index.is_trained(), true); // training is not required
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn bad_index_factory_description() {
        let r = index_factory(64, "fdnoyq", MetricType::L2);
        assert!(r.is_err());
        let r = index_factory(64, "Flat\0Flat", MetricType::L2);
        assert!(r.is_err());
    }

    #[test]
    fn index_clone() {
        let mut index = index_factory(4, "Flat", MetricType::L2).unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1.,
        ];

        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 6);

        let mut index2 = index.try_clone().unwrap();
        assert_eq!(index2.ntotal(), 6);

        let some_more_data = &[
            100., 100., 100., 100., -100., 100., 100., 100., 120., 100., 100., 105., -100., 100.,
            100., 105.,
        ];

        index2.add(some_more_data).unwrap();
        assert_eq!(index.ntotal(), 6);
        assert_eq!(index2.ntotal(), 10);
    }

    #[test]
    fn flat_index_search() {
        let mut index = index_factory(8, "Flat", MetricType::L2).unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; 8];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![Idx(2), Idx(1), Idx(0), Idx(3), Idx(4)]);
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; 8];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![Idx(3), Idx(4), Idx(0), Idx(1), Idx(2)]);
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 100., 100., 100., 100., 100., 100., 100., 100.,
        ];
        let result = index.search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![
                Idx(2),
                Idx(1),
                Idx(0),
                Idx(3),
                Idx(4),
                Idx(3),
                Idx(4),
                Idx(0),
                Idx(1),
                Idx(2)
            ]
        );
        assert!(result.distances.iter().all(|x| *x > 0.));
    }

    #[test]
    fn flat_index_assign() {
        let mut index = index_factory(8, "Flat", MetricType::L2).unwrap();
        assert_eq!(index.d(), 8);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; 8];
        let result = index.assign(&my_query, 5).unwrap();
        assert_eq!(result.labels, vec![Idx(2), Idx(1), Idx(0), Idx(3), Idx(4)]);

        let my_query = [0.; 32];
        let result = index.assign(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![2, 1, 0, 3, 4, 2, 1, 0, 3, 4, 2, 1, 0, 3, 4, 2, 1, 0, 3, 4]
                .into_iter()
                .map(Idx)
                .collect::<Vec<_>>()
        );

        let my_query = [100.; 8];
        let result = index.assign(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![3, 4, 0, 1, 2].into_iter().map(Idx).collect::<Vec<_>>()
        );

        let my_query = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 100., 100., 100., 100., 100., 100., 100., 100.,
        ];
        let result = index.assign(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![2, 1, 0, 3, 4, 3, 4, 0, 1, 2]
                .into_iter()
                .map(Idx)
                .collect::<Vec<_>>()
        );

        index.reset().unwrap();
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn flat_index_range_search() {
        let mut index = index_factory(8, "Flat", MetricType::L2).unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; 8];
        let result = index.range_search(&my_query, 8.125).unwrap();
        let (distances, labels) = result.distance_and_labels();
        assert!(labels == &[Idx(1), Idx(2)] || labels == &[Idx(2), Idx(1)]);
        assert!(distances.iter().all(|x| *x > 0.));
    }
}
