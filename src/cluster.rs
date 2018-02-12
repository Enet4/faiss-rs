use error::Result;
use std::{mem, ptr};
use std::os::raw::{c_int};
use faiss_sys::*;
use index::IndexImpl;

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
     * Obtain a new clustering object.
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
     * Obtain a new clustering object with the given parameters.
     */
    pub fn new_with_params(d: u32, k: u32, params: &ClusteringParameters) -> Result<Self> {
        unsafe {
            let d = d as c_int;
            let k = k as c_int;
            let mut inner: *mut FaissClustering = ptr::null_mut();
            faiss_try!(faiss_Clustering_new_with_params(&mut inner, d, k, &params.inner));
            Ok(Clustering { inner })
        }
    }

    pub fn train(&mut self, x: &[f32], index: &mut IndexImpl) -> Result<()>
    {
        unsafe {
            let n = x.len() / self.d() as usize;
            faiss_try!(faiss_Clustering_train(self.inner, n as idx_t, x.as_ptr(), index.inner_ptr_mut()));
            Ok(())
        }
    }

    pub fn centroids(&self) -> Result<&[f32]> {
        unsafe {
            let mut data = ptr::null_mut();
            let mut size = 0;
            faiss_Clustering_centroids(self.inner, &mut data, &mut size);
            Ok(::std::slice::from_raw_parts(data, size))
        }
    }

    pub fn centroids_mut(&mut self) -> Result<&mut [f32]> {
        unsafe {
            let mut data = ptr::null_mut();
            let mut size = 0;
            faiss_Clustering_centroids(self.inner, &mut data, &mut size);
            Ok(::std::slice::from_raw_parts_mut(data, size))
        }
    }

    pub fn obj(&self) -> Result<&[f32]> {
        unsafe {
            let mut data = ptr::null_mut();
            let mut size = 0;
            faiss_Clustering_obj(self.inner, &mut data, &mut size);
            Ok(::std::slice::from_raw_parts(data, size))
        }
    }

    pub fn obj_mut(&mut self) -> Result<&mut [f32]> {
        unsafe {
            let mut data = ptr::null_mut();
            let mut size = 0;
            faiss_Clustering_obj(self.inner, &mut data, &mut size);
            Ok(::std::slice::from_raw_parts_mut(data, size))
        }
    }

    pub fn d(&self) -> u32 {
        unsafe {
            faiss_Clustering_d(self.inner) as u32
        }
    }

    pub fn k(&self) -> u32 {
        unsafe {
            faiss_Clustering_k(self.inner) as u32
        }
    }

    pub fn seed(&self) -> u32 {
        unsafe {
            faiss_Clustering_seed(self.inner) as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Clustering, ClusteringParameters};
    use index::index_factory;
    use MetricType;

    #[test]
    fn test_clustering() {
        const D: u32 = 8;
        const K: u32 = 3;
        let mut params = ClusteringParameters::default();
        params.set_niter(5);
        params.set_min_points_per_centroid(1);
        params.set_max_points_per_centroid(10);

        let some_data = [
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5,
            -1., 1., 1., 1., 1., 1., 1., -1.,
            0., 0., 0., 1., 1., 0., 0., -1.,
            100., 100., 100., 100., -100., 100., 100., 100.,
            -7., 1., 4., 1., 2., 1., 3., -1.,
            120., 100., 100., 120., -100., 100., 100., 120.,
            0., 0., -12., 1., 1., 0., 6., -1.,
            0., 0., -0.25, 1., 16., 24., 0., -1.,
            100., 10., 100., 100., 10., 100., 50., 10.,
            20., 22., 4.5, -2., -100., 0., 0., 100.,
        ];

        let mut clustering = Clustering::new_with_params(D, K, &params).unwrap();
        let mut index = index_factory(D, "Flat", MetricType::L2).unwrap();
        clustering.train(&some_data, &mut index).unwrap();

        let centroids: Vec<_> = clustering.centroids().unwrap().chunks(D as usize).collect();
        assert_eq!(centroids.len(), K as usize);
    }
}
