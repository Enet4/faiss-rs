#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[cfg(feature = "gpu")]
mod bindings_gpu;
#[cfg(feature = "gpu")]
pub use bindings_gpu::*;

#[cfg(not(feature = "gpu"))]
mod bindings;
#[cfg(not(feature = "gpu"))]
pub use bindings::*;

mod iobridge;
pub use iobridge::{
    ffi::{faiss_read_index_br, faiss_write_index_bs},
    new_bufreceiver, new_bufsender, BufReceiver, BufSender,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::mem;
    use std::os::raw::c_char;
    use std::ptr;

    #[test]
    fn getting_last_error() {
        unsafe {
            let mut index_ptr: *mut FaissIndexFlatL2 = ptr::null_mut();
            let desc = CString::new("noooo").unwrap();
            let c = faiss_index_factory(&mut index_ptr, 4, desc.as_ptr(), 0);
            assert_ne!(c, 0);
            let last_error: *const c_char = faiss_get_last_error();
            assert!(!last_error.is_null());
        }
    }

    #[test]
    fn flat_index() {
        const D: usize = 8;
        unsafe {
            let mut index_ptr: *mut FaissIndexFlatL2 = ptr::null_mut();
            let c = faiss_IndexFlatL2_new_with(&mut index_ptr as *mut _, D as idx_t);
            assert_eq!(c, 0);
            assert!(!index_ptr.is_null());
            let some_data = [
                7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0.,
                0., 0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 200.,
                100., 100., 500., -100., 100., 100., 500.,
            ];
            let some_data_ptr = some_data.as_ptr();
            assert_eq!(faiss_Index_is_trained(index_ptr) != 0, true);
            let c = faiss_Index_add(index_ptr, (some_data.len() / D) as idx_t, some_data_ptr);
            assert_eq!(c, 0);
            assert_eq!(faiss_Index_ntotal(index_ptr), 5);

            let some_query = [0.0_f32; D];
            // output vectors (with canary values at the end)
            let mut distances = [0_f32, 0., 0., 0., -1.];
            let mut labels = [0 as idx_t, 0, 0, 0, -1];
            // search for vectors closest to the origin
            let c = faiss_Index_search(
                index_ptr,
                1,
                some_query.as_ptr(),
                4,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            );
            assert_eq!(c, 0);
            assert_eq!(labels, [2, 1, 0, 3, -1]);
            assert!(distances[0] > 0.);
            assert!(distances[1] > 0.);
            assert!(distances[2] > 0.);
            assert!(distances[3] > 0.);
            assert_eq!(distances[4], -1.);
            faiss_Index_free(index_ptr);
        }
    }

    #[test]
    fn clustering() {
        const D: usize = 8;
        unsafe {
            let mut params = mem::MaybeUninit::<FaissClusteringParameters>::uninit();
            faiss_ClusteringParameters_init(params.as_mut_ptr());
            let mut params = params.assume_init();
            assert_eq!(params.verbose, 0);
            assert_eq!(params.spherical, 0);
            assert_eq!(params.frozen_centroids, 0);
            assert_eq!(params.update_index, 0);
            assert!(params.niter > 0);
            params.niter = 5;
            assert!(params.min_points_per_centroid > 0);
            assert!(params.max_points_per_centroid > 0);
            params.min_points_per_centroid = 1;
            params.max_points_per_centroid = 10;

            let some_data = [
                7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0.,
                0., 0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., -7.,
                1., 4., 1., 2., 1., 3., -1., 120., 100., 100., 120., -100., 100., 100., 120., 0.,
                0., -12., 1., 1., 0., 6., -1., 0., 0., -0.25, 1., 16., 24., 0., -1., 100., 10.,
                100., 100., 10., 100., 50., 10., 20., 22., 4.5, -2., -100., 0., 0., 100.,
            ];

            let mut clustering_ptr: *mut FaissClustering = ptr::null_mut();
            let c = faiss_Clustering_new_with_params(&mut clustering_ptr, D as i32, 2, &params);
            assert_eq!(c, 0);
            assert_ne!(clustering_ptr, ptr::null_mut());
            let mut index_ptr: *mut FaissIndexFlatL2 = ptr::null_mut();
            let desc = CString::new("Flat").unwrap();
            let c = faiss_index_factory(
                &mut index_ptr,
                D as i32,
                desc.as_ptr(),
                FaissMetricType_METRIC_L2,
            );
            assert_eq!(c, 0);
            assert_ne!(index_ptr, ptr::null_mut());

            let c = faiss_Clustering_train(clustering_ptr, 10, some_data.as_ptr(), index_ptr);
            assert_eq!(c, 0);

            faiss_Clustering_free(clustering_ptr);
        }
    }
}
