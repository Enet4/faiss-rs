use error::Result;
use std::{mem, ptr};
use std::os::raw::{c_int};
use faiss_sys::*;

pub struct ClusteringParameters {
    inner: FaissClusteringParameters,
}

impl ClusteringParameters {

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
        self.inner.niter = (niter & 0x7FFFFFFF) as i32;
    }

    pub fn set_nredo(&mut self, nredo: u32) {
        self.inner.nredo = (nredo & 0x7FFFFFFF) as i32;
    }

    pub fn set_min_points_per_centroid(&mut self, min_points_per_centroid: u32) {
        self.inner.min_points_per_centroid = (min_points_per_centroid & 0x7FFFFFFF) as i32;
    }

    pub fn set_max_points_per_centroid(&mut self, max_points_per_centroid: u32) {
        self.inner.max_points_per_centroid = (max_points_per_centroid & 0x7FFFFFFF) as i32;
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
     * Obtain a new clustering object.
     * 
     * # Panic
     * 
     * On invalid parameters of `d` or `k`, the function will panic.
     */
    pub fn new(d: u32, k: u32) -> Result<Self> {
        unsafe {
            let d = d as c_int;
            let k = k as c_int;
            assert!(d > 0);
            assert!(k > 0);
            let mut inner: *mut FaissClustering = ptr::null_mut();
            faiss_try!(faiss_Clustering_new(&mut inner, d, k));
            Ok(Clustering { inner })
        }
    }

    /** 
     * Obtain a new clustering object with the given parameters.
     * 
     * # Panic
     * 
     * On invalid parameters of `d` or `k`, the function will panic.
     */
    pub fn new_with_params(d: u32, k: u32, params: &ClusteringParameters) -> Result<Self> {
        unsafe {
            let d = d as c_int;
            let k = k as c_int;
            assert!(d > 0);
            assert!(k > 0);
            let mut inner: *mut FaissClustering = ptr::null_mut();
            let params_inner = params.inner;
            faiss_try!(faiss_Clustering_new_with_params(&mut inner, d, k, &params_inner));
            Ok(Clustering { inner })
        }
    }

    pub fn seed(&self) -> u32 {
        unsafe {
            faiss_Clustering_seed(self.inner) as u32
        }
    }
}
