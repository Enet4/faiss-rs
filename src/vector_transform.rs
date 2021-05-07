//! Vector transformation implementation

use crate::error::Result;
use crate::faiss_try;
use faiss_sys::*;
use std::ptr;

/// Sub-trait for native implementations of a Faiss VectorTransform.
pub trait NativeVectorTransform {
    /// Retrieve a pointer to the native object.
    fn vector_transform_ptr(&self) -> *mut FaissVectorTransform;
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
        unsafe { faiss_VectorTransform_is_trained(self.vector_transform_ptr()) != 0 }
    }

    fn d_in(&self) -> u32 {
        unsafe { faiss_VectorTransform_d_in(self.vector_transform_ptr()) as u32 }
    }

    fn d_out(&self) -> u32 {
        unsafe { faiss_VectorTransform_d_out(self.vector_transform_ptr()) as u32 }
    }

    fn train(&mut self, n: usize, x: &[f32]) -> Result<()> {
        unsafe {
            faiss_try(faiss_VectorTransform_train(
                self.vector_transform_ptr(),
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
                self.vector_transform_ptr(),
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
                self.vector_transform_ptr(),
                n as i64,
                xt.as_ptr(),
                x.as_mut_ptr(),
            );

            x
        }
    }
}

/// Sub-trait for native implementations of a Faiss LinearTransform.
pub trait NativeLinearTransform {
    /// Retrieve a pointer to the native object.
    fn linear_transform_ptr(&self) -> *mut FaissLinearTransform;
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

impl<T> LinearTransform for T
where
    T: NativeLinearTransform + VectorTransform,
{
    fn transform_transpose(&self, y: &[f32]) -> Vec<f32> {
        unsafe {
            let n = y.len() / self.d_in() as usize;
            let mut x = Vec::with_capacity(n * self.d_out() as usize);
            faiss_LinearTransform_transform_transpose(
                self.linear_transform_ptr(),
                n as i64,
                y.as_ptr(),
                x.as_mut_ptr(),
            );

            x
        }
    }

    fn set_is_orthonormal(&mut self) {
        unsafe {
            faiss_LinearTransform_set_is_orthonormal(self.linear_transform_ptr());
        }
    }

    fn have_bias(&self) -> bool {
        unsafe { faiss_LinearTransform_have_bias(self.linear_transform_ptr()) != 0 }
    }

    fn is_orthonormal(&self) -> bool {
        unsafe { faiss_LinearTransform_is_orthonormal(self.linear_transform_ptr()) != 0 }
    }
}
pub struct RandomRotationMatrixImpl {
    inner: *mut FaissRandomRotationMatrix,
}

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
            let d_in_ = d_in as i32;
            let d_out_ = d_out as i32;
            faiss_try(faiss_RandomRotationMatrix_new_with(
                &mut inner, d_in_, d_out_,
            ))?;

            Ok(RandomRotationMatrixImpl { inner })
        }
    }
}

// impl NativeLinearTransform for RandomRotationMatrixImpl {
//     fn linear_transform_ptr(&self) -> *mut FaissLinearTransform {
//         self.inner
//     }
// }

// impl NativeVectorTransform for RandomRotationMatrixImpl {
//     fn vector_transform_ptr(&self) -> *mut FaissVectorTransform {
//         self.inner
//     }
// }
