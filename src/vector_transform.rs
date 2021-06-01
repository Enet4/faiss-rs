//! Vector transformation implementation

use crate::error::Result;
use crate::faiss_try;
use faiss_sys::*;
use std::os::raw::c_int;
use std::ptr;

/// Trait for native implementations of a Faiss VectorTransform.
pub trait NativeVectorTransform {
    /// Retrieve a pointer to the native object.
    fn inner_ptr(&self) -> *mut FaissVectorTransform;
}

pub trait VectorTransform {
    /// Getter for is_trained
    fn is_trained(&self) -> bool;

    /// Getter for input dimension
    fn d_in(&self) -> u32;

    /// Getter for output dimension
    fn d_out(&self) -> u32;

    /// Perform training on a representative set of vectors
    fn train(&mut self, n: usize, x: &[f32]) -> Result<()>;

    /// apply transformation and result is pre-allocated
    fn apply_noalloc(&self, x: &[f32]) -> Vec<f32>;

    /// reverse transformation. May not be implemented or may return
    /// approximate result
    fn reverse_transform(&self, xt: &[f32]) -> Vec<f32>;
}

impl<T> VectorTransform for T
where
    T: NativeVectorTransform,
{
    fn is_trained(&self) -> bool {
        unsafe { faiss_VectorTransform_is_trained(self.inner_ptr()) != 0 }
    }

    fn d_in(&self) -> u32 {
        unsafe { faiss_VectorTransform_d_in(self.inner_ptr()) as u32 }
    }

    fn d_out(&self) -> u32 {
        unsafe { faiss_VectorTransform_d_out(self.inner_ptr()) as u32 }
    }

    fn train(&mut self, n: usize, x: &[f32]) -> Result<()> {
        unsafe {
            faiss_try(faiss_VectorTransform_train(
                self.inner_ptr(),
                n as i64,
                x.as_ptr(),
            ))?;
            Ok(())
        }
    }

    fn apply_noalloc(&self, x: &[f32]) -> Vec<f32> {
        unsafe {
            let n = x.len() / self.d_in() as usize;
            let mut xt = Vec::with_capacity(n * self.d_out() as usize);
            faiss_VectorTransform_apply_noalloc(
                self.inner_ptr(),
                n as i64,
                x.as_ptr(),
                xt.as_mut_ptr(),
            );

            xt
        }
    }

    fn reverse_transform(&self, xt: &[f32]) -> Vec<f32> {
        unsafe {
            let n = xt.len() / self.d_out() as usize;
            let mut x = Vec::with_capacity(n * self.d_in() as usize);
            faiss_VectorTransform_reverse_transform(
                self.inner_ptr(),
                n as i64,
                xt.as_ptr(),
                x.as_mut_ptr(),
            );

            x
        }
    }
}

pub trait LinearTransform: VectorTransform {
    /// compute x = A^T * (x - b)
    /// is reverse transform if A has orthonormal lines
    fn transform_transpose(&self, y: &[f32]) -> Vec<f32>;

    /// compute A^T * A to set the is_orthonormal flag
    fn set_is_orthonormal(&mut self);

    /// Getter for have_bias
    fn have_bias(&self) -> bool;

    /// Getter for is_orthonormal
    fn is_orthonormal(&self) -> bool;
}

pub type RandomRotationMatrix = RandomRotationMatrixImpl;

pub struct RandomRotationMatrixImpl {
    inner: *mut FaissRandomRotationMatrix,
}

unsafe impl Send for RandomRotationMatrixImpl {}
unsafe impl Sync for RandomRotationMatrixImpl {}

impl Drop for RandomRotationMatrixImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_RandomRotationMatrix_free(self.inner);
        }
    }
}

impl RandomRotationMatrixImpl {
    pub fn new(d_in: u32, d_out: u32) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_RandomRotationMatrix_new_with(
                &mut inner,
                d_in as i32,
                d_out as i32,
            ))?;

            Ok(RandomRotationMatrixImpl { inner })
        }
    }
}

impl NativeVectorTransform for RandomRotationMatrixImpl {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.inner
    }
}

impl_native_linear_transform!(RandomRotationMatrixImpl);

pub type PCAMatrix = PCAMatrixImpl;

pub struct PCAMatrixImpl {
    inner: *mut FaissPCAMatrix,
}

unsafe impl Send for PCAMatrixImpl {}
unsafe impl Sync for PCAMatrixImpl {}

impl Drop for PCAMatrixImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_PCAMatrix_free(self.inner);
        }
    }
}

impl PCAMatrixImpl {
    pub fn new(d_in: u32, d_out: u32, eigen_power: f32, random_rotation: bool) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_PCAMatrix_new_with(
                &mut inner,
                d_in as i32,
                d_out as i32,
                eigen_power,
                c_int::from(random_rotation),
            ))?;

            Ok(PCAMatrixImpl { inner })
        }
    }

    pub fn eigen_power(&self) -> f32 {
        unsafe { faiss_PCAMatrix_eigen_power(self.inner_ptr()) }
    }

    pub fn random_rotation(&self) -> bool {
        unsafe { faiss_PCAMatrix_random_rotation(self.inner_ptr()) != 0 }
    }
}

impl NativeVectorTransform for PCAMatrixImpl {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.inner
    }
}

impl_native_linear_transform!(PCAMatrixImpl);

pub type ITQMatrix = ITQMatrixImpl;

pub struct ITQMatrixImpl {
    inner: *mut FaissITQMatrix,
}

unsafe impl Send for ITQMatrixImpl {}
unsafe impl Sync for ITQMatrixImpl {}

impl Drop for ITQMatrixImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_ITQMatrix_free(self.inner);
        }
    }
}

impl ITQMatrixImpl {
    pub fn new(d: u32) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_ITQMatrix_new_with(&mut inner, d as i32))?;

            Ok(ITQMatrixImpl { inner })
        }
    }
}

impl NativeVectorTransform for ITQMatrixImpl {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.inner
    }
}

impl_native_linear_transform!(ITQMatrixImpl);

pub type ITQTransform = ITQTransformImpl;

pub struct ITQTransformImpl {
    inner: *mut FaissITQTransform,
}

unsafe impl Send for ITQTransformImpl {}
unsafe impl Sync for ITQTransformImpl {}

impl Drop for ITQTransformImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_ITQTransform_free(self.inner);
        }
    }
}

impl ITQTransformImpl {
    pub fn new(d_in: u32, d_out: u32, do_pca: bool) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_ITQTransform_new_with(
                &mut inner,
                d_in as i32,
                d_out as i32,
                c_int::from(do_pca),
            ))?;

            Ok(ITQTransformImpl { inner })
        }
    }

    pub fn get_do_pca(&self) -> bool {
        unsafe { faiss_ITQTransform_do_pca(self.inner_ptr()) != 0 }
    }
}

impl NativeVectorTransform for ITQTransformImpl {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.inner
    }
}

pub type OPQMatrix = OPQMatrixImpl;

pub struct OPQMatrixImpl {
    inner: *mut FaissOPQMatrix,
}

unsafe impl Send for OPQMatrixImpl {}
unsafe impl Sync for OPQMatrixImpl {}

impl Drop for OPQMatrixImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_OPQMatrix_free(self.inner);
        }
    }
}

impl OPQMatrixImpl {
    pub fn new(d: u32, m: u32, d2: u32) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_OPQMatrix_new_with(
                &mut inner, d as i32, m as i32, d2 as i32,
            ))?;

            Ok(OPQMatrixImpl { inner })
        }
    }

    pub fn set_verbose(&mut self, value: bool) {
        unsafe { faiss_OPQMatrix_set_verbose(self.inner_ptr(), c_int::from(value)) }
    }

    pub fn verbose(&self) -> bool {
        unsafe { faiss_OPQMatrix_verbose(self.inner_ptr()) != 0 }
    }

    pub fn set_niter(&mut self, value: u32) {
        unsafe { faiss_OPQMatrix_set_niter(self.inner_ptr(), value as i32) }
    }

    pub fn niter(&self) -> u32 {
        unsafe { faiss_OPQMatrix_niter(self.inner_ptr()) as u32 }
    }

    pub fn set_niter_pq(&mut self, value: u32) {
        unsafe { faiss_OPQMatrix_set_niter_pq(self.inner_ptr(), value as i32) }
    }

    pub fn niter_pq(&self) -> u32 {
        unsafe { faiss_OPQMatrix_niter_pq(self.inner_ptr()) as u32 }
    }
}

impl NativeVectorTransform for OPQMatrixImpl {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.inner
    }
}

impl_native_linear_transform!(OPQMatrixImpl);

pub type RemapDimensionsTransform = RemapDimensionsTransformImpl;

pub struct RemapDimensionsTransformImpl {
    inner: *mut FaissRemapDimensionsTransform,
}

unsafe impl Send for RemapDimensionsTransformImpl {}
unsafe impl Sync for RemapDimensionsTransformImpl {}

impl Drop for RemapDimensionsTransformImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_RemapDimensionsTransform_free(self.inner);
        }
    }
}

impl RemapDimensionsTransformImpl {
    pub fn new(d_in: u32, d_out: u32, uniform: bool) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_RemapDimensionsTransform_new_with(
                &mut inner,
                d_in as i32,
                d_out as i32,
                c_int::from(uniform),
            ))?;

            Ok(RemapDimensionsTransformImpl { inner })
        }
    }
}

impl NativeVectorTransform for RemapDimensionsTransformImpl {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.inner
    }
}

pub type NormalizationTransform = NormalizationTransformImpl;

pub struct NormalizationTransformImpl {
    inner: *mut FaissNormalizationTransform,
}

unsafe impl Send for NormalizationTransformImpl {}
unsafe impl Sync for NormalizationTransformImpl {}

impl Drop for NormalizationTransformImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_NormalizationTransform_free(self.inner);
        }
    }
}

impl NormalizationTransformImpl {
    pub fn new(d: u32, norm: f32) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_NormalizationTransform_new_with(
                &mut inner, d as i32, norm,
            ))?;

            Ok(NormalizationTransformImpl { inner })
        }
    }

    pub fn norm(&self) -> f32 {
        unsafe { faiss_NormalizationTransform_norm(self.inner_ptr()) }
    }
}

impl NativeVectorTransform for NormalizationTransformImpl {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.inner
    }
}

pub type CenteringTransform = CenteringTransformImpl;

pub struct CenteringTransformImpl {
    inner: *mut FaissCenteringTransform,
}

unsafe impl Send for CenteringTransformImpl {}
unsafe impl Sync for CenteringTransformImpl {}

impl Drop for CenteringTransformImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_CenteringTransform_free(self.inner);
        }
    }
}

impl CenteringTransformImpl {
    pub fn new(d: u32) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_CenteringTransform_new_with(&mut inner, d as i32))?;

            Ok(CenteringTransformImpl { inner })
        }
    }
}

impl NativeVectorTransform for CenteringTransformImpl {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.inner
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn random_rotation_matrix_base_checks() {
        let rrt = RandomRotationMatrix::new(512, 256).unwrap();
        // vector transform
        assert_eq!(rrt.d_in(), 512);
        assert_eq!(rrt.d_out(), 256);
        assert_eq!(rrt.is_trained(), false);
    }
}
