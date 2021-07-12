use faiss_sys::*;

#[doc = " Setter of threshold value on nx above which we switch to BLAS to compute"]
#[doc = " distances"]
pub fn set_distance_compute_blas_threshold(value: u32) {
    unsafe {
        let v = (value & 0x7FFF_FFFF) as i32;
        faiss_set_distance_compute_blas_threshold(v);
    }
}

#[doc = " Getter of threshold value on nx above which we switch to BLAS to compute"]
#[doc = " distances"]
pub fn get_distance_compute_blas_threshold() -> u32 {
    unsafe { faiss_get_distance_compute_blas_threshold() as u32 }
}

#[doc = " Setter of block sizes value for BLAS distance computations"]
pub fn set_distance_compute_blas_query_bs(value: u32) {
    unsafe {
        let v = (value & 0x7FFF_FFFF) as i32;
        faiss_set_distance_compute_blas_query_bs(v);
    }
}

#[doc = " Getter of block sizes value for BLAS distance computations"]
pub fn get_distance_compute_blas_query_bs() -> u32 {
    unsafe { faiss_get_distance_compute_blas_query_bs() as u32 }
}

#[doc = " Setter of block sizes value for BLAS distance computations"]
pub fn set_distance_compute_blas_database_bs(value: u32) {
    unsafe {
        let v = (value & 0x7FFF_FFFF) as i32;
        faiss_set_distance_compute_blas_database_bs(v);
    }
}

#[doc = " Getter of block sizes value for BLAS distance computations"]
pub fn get_distance_compute_blas_database_bs() -> u32 {
    unsafe { faiss_get_distance_compute_blas_database_bs() as u32 }
}

#[doc = " Setter of number of results we switch to a reservoir to collect results"]
#[doc = " rather than a heap"]
pub fn set_distance_compute_min_k_reservoir(value: u32) {
    unsafe {
        let v = (value & 0x7FFF_FFFF) as i32;
        faiss_set_distance_compute_min_k_reservoir(v);
    }
}

#[doc = " Getter of number of results we switch to a reservoir to collect results"]
#[doc = " rather than a heap"]
pub fn get_distance_compute_min_k_reservoir() -> u32 {
    unsafe { faiss_get_distance_compute_min_k_reservoir() as u32 }
}
