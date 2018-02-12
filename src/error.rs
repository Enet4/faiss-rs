//! Error handling module

use std::fmt;
use std::error::Error as StdError;
use std::ffi::CStr;
use faiss_sys::*;

/// Type alias for results of functions in this crate.
pub type Result<T> = ::std::result::Result<T, Error>;

/// The main error type.
pub type Error = NativeError;

/// An error derived from a native Faiss exception.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeError {
    /// The error code retrieved from the C API
    code: i32,
    /// The exception's message
    msg: String,
}

impl NativeError {
    /// Getter for the internal error code.
    pub fn code(&self) -> i32 {
        self.code
    }

    /// Getter for the exception's message. Same as `description()`.
    pub fn msg(&self) -> &str {
        &self.msg
    }
}

impl NativeError {
    /// Create a native error value by taking the error from
    /// the last failed operation.
    ///
    /// # Panics
    ///
    /// The operation is meant to be used immediately after
    /// a operation which returned a non-zero error code.
    /// This function might panic if no operation was made
    /// or the last operation was successful.
    pub fn from_last_error(code: i32) -> Self {
        unsafe {
            let e: *const _ = faiss_get_last_error();
            assert!(!e.is_null());
            let cstr = CStr::from_ptr(e);
            let msg: String = cstr.to_string_lossy().into_owned();
            NativeError { code, msg }
        }
    }
}

impl fmt::Display for NativeError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(&self.msg)
    }
}

impl StdError for NativeError {
    fn description(&self) -> &str {
        &self.msg
    }
}
