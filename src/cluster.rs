//! Vector clustering interface and implementation.

use crate::error::Result;
use crate::index::NativeIndex;
use faiss_sys::*;
use std::os::raw::c_int;
use std::{mem, ptr};

/// Parameters for the clustering algorithm.
pub struct ClusteringParameters {
    inner: FaissClusteringParameters,
}

impl Default for ClusteringParameters {
    fn default() -> Self {
        ClusteringParameters::new()
    }
}

impl ClusteringParameters {
    /// Create a new clustering parameters object.
    pub fn new() -> Self {
        unsafe {
            let mut inner: FaissClusteringParameters = mem::zeroed();
            faiss_ClusteringParameters_init(&mut inner);
            ClusteringParameters { inner }
        }
    }

    pub fn niter(&self) -> i32 {
        self.inner.niter as i32
    }

    pub fn nredo(&self) -> i32 {
        self.inner.nredo as i32
    }

    pub fn min_points_per_centroid(&self) -> i32 {
        self.inner.min_points_per_centroid as i32
    }

    pub fn max_points_per_centroid(&self) -> i32 {
        self.inner.max_points_per_centroid as i32
    }

    pub fn frozen_centroids(&self) -> bool {
        self.inner.frozen_centroids != 0
    }

    pub fn spherical(&self) -> bool {
        self.inner.spherical != 0
    }

    pub fn update_index(&self) -> bool {
        self.inner.update_index != 0
    }

    pub fn verbose(&self) -> bool {
        self.inner.verbose != 0
    }

    pub fn seed(&self) -> u32 {
        self.inner.seed as u32
    }

    pub fn set_niter(&mut self, niter: u32) {
        self.inner.niter = (niter & 0x7FFF_FFFF) as i32;
    }

    pub fn set_nredo(&mut self, nredo: u32) {
        self.inner.nredo = (nredo & 0x7FFF_FFFF) as i32;
    }

    pub fn set_min_points_per_centroid(&mut self, min_points_per_centroid: u32) {
        self.inner.min_points_per_centroid = (min_points_per_centroid & 0x7FFF_FFFF) as i32;
    }

    pub fn set_max_points_per_centroid(&mut self, max_points_per_centroid: u32) {
        self.inner.max_points_per_centroid = (max_points_per_centroid & 0x7FFF_FFFF) as i32;
    }

    pub fn set_frozen_centroids(&mut self, frozen_centroids: bool) {
        self.inner.frozen_centroids = if frozen_centroids { 1 } else { 0 };
    }

    pub fn set_update_index(&mut self, update_index: bool) {
        self.inner.update_index = if update_index { 1 } else { 0 };
    }

    pub fn set_spherical(&mut self, spherical: bool) {
        self.inner.spherical = if spherical { 1 } else { 0 };
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        self.inner.verbose = if verbose { 1 } else { 0 };
    }

    pub fn set_seed(&mut self, seed: u32) {
        self.inner.seed = seed as i32;
    }
}

/// The clustering algorithm.
pub struct Clustering {
    inner: *mut FaissClustering,
}

unsafe impl Send for Clustering {}
unsafe impl Sync for Clustering {}

impl Drop for Clustering {
    fn drop(&mut self) {
        unsafe {
            faiss_Clustering_free(self.inner);
        }
    }
}

impl Clustering {
    /**
     * Obtain a new clustering object with the given dimensionality
     * `d` and number of centroids `k`.
     */
    pub fn new(d: u32, k: u32) -> Result<Self> {
        unsafe {
            let d = d as c_int;
            let k = k as c_int;
            let mut inner: *mut FaissClustering = ptr::null_mut();
            faiss_try!(faiss_Clustering_new(&mut inner, d, k));
            Ok(Clustering { inner })
        }
    }

    /**
     * Obtain a new clustering object, with the given clustering parameters.
     */
    pub fn new_with_params(d: u32, k: u32, params: &ClusteringParameters) -> Result<Self> {
        unsafe {
            let d = d as c_int;
            let k = k as c_int;
            let mut inner: *mut FaissClustering = ptr::null_mut();
            faiss_try!(faiss_Clustering_new_with_params(
                &mut inner,
                d,
                k,
                &params.inner
            ));
            Ok(Clustering { inner })
        }
    }

    /**
     * Perform the clustering algorithm with the given data and index.
     * The index is used during the assignment stage.
     */
    pub fn train<I: ?Sized>(&mut self, x: &[f32], index: &mut I) -> Result<()>
    where
        I: NativeIndex,
    {
        unsafe {
            let n = x.len() / self.d() as usize;
            faiss_try!(faiss_Clustering_train(
                self.inner,
                n as idx_t,
                x.as_ptr(),
                index.inner_ptr()
            ));
            Ok(())
        }
    }

    /**
     * Retrieve the centroids from the clustering process. Returns
     * a vector of `k` slices of size `d`.
     */
    pub fn centroids(&self) -> Result<Vec<&[f32]>> {
        unsafe {
            let mut data = ptr::null_mut();
            let mut size = 0;
            faiss_Clustering_centroids(self.inner, &mut data, &mut size);
            Ok(::std::slice::from_raw_parts(data, size)
                .chunks(self.d() as usize)
                .collect())
        }
    }

    /**
     * Retrieve the centroids from the clustering process. Returns
     * a vector of `k` slices of size `d`.
     */
    pub fn centroids_mut(&mut self) -> Result<Vec<&mut [f32]>> {
        unsafe {
            let mut data = ptr::null_mut();
            let mut size = 0;
            faiss_Clustering_centroids(self.inner, &mut data, &mut size);
            Ok(::std::slice::from_raw_parts_mut(data, size)
                .chunks_mut(self.d() as usize)
                .collect())
        }
    }

    /**
     * Retrieve the stats achieved from the clustering process.
     * Returns as many values as the number of iterations made.
     */
    pub fn iteration_stats(&self) -> Result<&[FaissClusteringIterationStats]> {
        unsafe {
            let mut data = ptr::null_mut();
            let mut size = 0;
            faiss_Clustering_iteration_stats(self.inner, &mut data, &mut size);
            Ok(::std::slice::from_raw_parts(data, size))
        }
    }

    /**
     * Retrieve the stats.
     * Returns as many values as the number of iterations made.
     */
    pub fn iteration_stats_mut(&mut self) -> Result<&mut [FaissClusteringIterationStats]> {
        unsafe {
            let mut data = ptr::null_mut();
            let mut size = 0;
            faiss_Clustering_iteration_stats(self.inner, &mut data, &mut size);
            Ok(::std::slice::from_raw_parts_mut(data, size))
        }
    }

    /** Getter for the clustering object's vector dimensionality. */
    pub fn d(&self) -> u32 {
        unsafe { faiss_Clustering_d(self.inner) as u32 }
    }

    /** Getter for the number of centroids. */
    pub fn k(&self) -> u32 {
        unsafe { faiss_Clustering_k(self.inner) as u32 }
    }

    /** Getter for the number of k-means iterations. */
    pub fn niter(&self) -> u32 {
        unsafe { faiss_Clustering_niter(self.inner) as u32 }
    }

    /** Getter for the `nredo` property of `Clustering`. */
    pub fn nredo(&self) -> u32 {
        unsafe { faiss_Clustering_nredo(self.inner) as u32 }
    }

    /** Getter for the `verbose` property of `Clustering`. */
    pub fn verbose(&self) -> bool {
        unsafe { faiss_Clustering_niter(self.inner) != 0 }
    }

    /** Getter for whether spherical clustering is intended. */
    pub fn spherical(&self) -> bool {
        unsafe { faiss_Clustering_spherical(self.inner) != 0 }
    }

    /** Getter for the `update_index` property of `Clustering`. */
    pub fn update_index(&self) -> bool {
        unsafe { faiss_Clustering_update_index(self.inner) != 0 }
    }

    /** Getter for the `frozen_centroids` property of `Clustering`. */
    pub fn frozen_centroids(&self) -> bool {
        unsafe { faiss_Clustering_frozen_centroids(self.inner) != 0 }
    }

    /** Getter for the `seed` property of `Clustering`. */
    pub fn seed(&self) -> u32 {
        unsafe { faiss_Clustering_seed(self.inner) as u32 }
    }

    /** Getter for the minimum number of points per centroid. */
    pub fn min_points_per_centroid(&self) -> u32 {
        unsafe { faiss_Clustering_min_points_per_centroid(self.inner) as u32 }
    }

    /** Getter for the maximum number of points per centroid. */
    pub fn max_points_per_centroid(&self) -> u32 {
        unsafe { faiss_Clustering_max_points_per_centroid(self.inner) as u32 }
    }
}

/// Plain data structure for the outcome of the simple k-means clustering
/// function (see [`kmeans_clustering`]).
///
/// [`kmeans_clustering`]: fn.kmeans_clustering.html
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct KMeansResult {
    /// The centroids of each cluster as a single contiguous vector (size `k * d`)
    pub centroids: Vec<f32>,
    /// The quantization error
    pub q_error: f32,
}

/// Simplified interface for k-means clustering.
///
/// - `d`: dimension of the data
/// - `k`: nb of output centroids
/// - `x`: training set (size `n * d`)
///
/// The number of points is inferred from `x` and `k`.
///
/// Returns the final quantization error and centroids (size `k * d`).
///
pub fn kmeans_clustering(d: u32, k: u32, x: &[f32]) -> Result<KMeansResult> {
    unsafe {
        let n = x.len() / d as usize;
        let mut centroids = vec![0_f32; (d * k) as usize];
        let mut q_error: f32 = 0.;
        faiss_try!(faiss_kmeans_clustering(
            d as usize,
            n,
            k as usize,
            x.as_ptr(),
            centroids.as_mut_ptr(),
            &mut q_error
        ));
        Ok(KMeansResult { centroids, q_error })
    }
}

#[cfg(test)]
mod tests {
    use super::{kmeans_clustering, Clustering, ClusteringParameters};
    use crate::index::index_factory;
    use crate::MetricType;

    #[test]
    fn test_clustering() {
        const D: u32 = 8;
        const K: u32 = 3;
        const NITER: u32 = 12;
        let mut params = ClusteringParameters::default();
        params.set_niter(NITER);
        params.set_min_points_per_centroid(1);
        params.set_max_points_per_centroid(10);

        let some_data = [
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., -7., 1., 4.,
            1., 2., 1., 3., -1., 120., 100., 100., 120., -100., 100., 100., 120., 0., 0., -12., 1.,
            1., 0., 6., -1., 0., 0., -0.25, 1., 16., 24., 0., -1., 100., 10., 100., 100., 10.,
            100., 50., 10., 20., 22., 4.5, -2., -100., 0., 0., 100.,
        ];

        let mut clustering = Clustering::new_with_params(D, K, &params).unwrap();
        let mut index = index_factory(D, "Flat", MetricType::L2).unwrap();
        clustering.train(&some_data, &mut index).unwrap();

        let centroids: Vec<_> = clustering.centroids().unwrap();
        assert_eq!(centroids.len(), K as usize);

        for c in centroids {
            assert_eq!(c.len(), D as usize);
        }

        let stats = clustering.iteration_stats().unwrap();
        assert_eq!(stats.len(), NITER as usize);
    }

    #[test]
    fn test_simple_clustering() {
        const D: u32 = 8;
        const K: u32 = 2;
        let some_data = [
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., -7., 1., 4.,
            1., 2., 1., 3., -1., 120., 100., 100., 120., -100., 100., 100., 120., 0., 0., -12., 1.,
            1., 0., 6., -1., 0., 0., -0.25, 1., 16., 24., 0., -1., 100., 10., 100., 100., 10.,
            100., 50., 10., 20., 22., 4.5, -2., -100., 0., 0., 100.,
        ];

        let out = kmeans_clustering(D, K, &some_data).unwrap();
        assert!(out.q_error > 0.);
        assert_eq!(out.centroids.len(), (D * K) as usize);
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    #[cfg(test)]
    mod tests {
        use super::super::{Clustering, ClusteringParameters};
        use crate::gpu::StandardGpuResources;
        use crate::index::index_factory;
        use crate::MetricType;

        #[test]
        fn test_clustering() {
            const D: u32 = 8;
            const K: u32 = 3;
            const NITER: u32 = 12;
            let mut params = ClusteringParameters::default();
            params.set_niter(NITER);
            params.set_min_points_per_centroid(1);
            params.set_max_points_per_centroid(10);

            let some_data = [
                7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0.,
                0., 0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., -7.,
                1., 4., 1., 2., 1., 3., -1., 120., 100., 100., 120., -100., 100., 100., 120., 0.,
                0., -12., 1., 1., 0., 6., -1., 0., 0., -0.25, 1., 16., 24., 0., -1., 100., 10.,
                100., 100., 10., 100., 50., 10., 20., 22., 4.5, -2., -100., 0., 0., 100.,
            ];

            let mut clustering = Clustering::new_with_params(D, K, &params).unwrap();
            let res = StandardGpuResources::new().unwrap();
            let mut index = index_factory(D, "Flat", MetricType::L2)
                .unwrap()
                .into_gpu(&res, 0)
                .unwrap();
            clustering.train(&some_data, &mut index).unwrap();

            let centroids: Vec<_> = clustering.centroids().unwrap();
            assert_eq!(centroids.len(), K as usize);

            for c in centroids {
                assert_eq!(c.len(), D as usize);
            }
            let objectives = clustering.objectives().unwrap();
            assert_eq!(objectives.len(), NITER as usize);
        }
    }
}
