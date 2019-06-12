//! Index I/O functions

use crate::error::{Error, Result};
use crate::index::{CpuIndex, FromInnerPtr, IndexImpl, NativeIndex};
use faiss_sys::*;
use std::ffi::CString;
use std::ptr;

/// Write an index to a file.
///
/// # Error
///
/// This function returns an error if the description contains any byte with the value `\0` (since
/// it cannot be converted to a C string), or if the internal index writing operation fails.
pub fn write_index<I, P>(index: &I, file_name: P) -> Result<()>
where
    I: NativeIndex,
    I: CpuIndex,
    P: AsRef<str>,
{
    unsafe {
        let f = file_name.as_ref();
        let f = CString::new(f).map_err(|_| Error::BadFilePath)?;

        faiss_try!(faiss_write_index_fname(index.inner_ptr(), f.as_ptr()));
        Ok(())
    }
}

/// Read an index from a file.
///
/// # Error
///
/// This function returns an error if the description contains any byte with the value `\0` (since
/// it cannot be converted to a C string), or if the internal index reading operation fails.
pub fn read_index<P>(file_name: P) -> Result<IndexImpl>
where
    P: AsRef<str>,
{
    unsafe {
        let f = file_name.as_ref();
        let f = CString::new(f).map_err(|_| Error::BadFilePath)?;
        let mut inner = ptr::null_mut();
        faiss_try!(faiss_read_index_fname(f.as_ptr(), 0, &mut inner));
        Ok(IndexImpl::from_inner_ptr(inner))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::flat::FlatIndex;
    use crate::index::Index;
    const D: u32 = 8;

    #[test]
    fn write_read() {
        let mut index = FlatIndex::new_l2(D).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
            -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16., 32.,
            25., 20., 20., 40., 15.,
        ];
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let filepath = ::std::env::temp_dir().join("test_write_read.index");
        let filename = filepath.to_str().unwrap();
        write_index(&index, filename).unwrap();
        let index = read_index(&filename).unwrap();
        assert_eq!(index.ntotal(), 5);
        ::std::fs::remove_file(&filepath).unwrap();
    }
}
