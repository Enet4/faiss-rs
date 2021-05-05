//! Error handling module

use faiss_sys::*;
use std::error::Error as StdError;
use std::ffi::CStr;
use std::fmt;
use std::os::raw::c_int;

/// Type alias for results of functions in this crate.
pub type Result<T> = ::std::result::Result<T, Error>;

/// The main error type.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The error came from a native Faiss exception.
    Native(NativeError),
    /// Invalid index type cast.
    BadCast,
    /// Invalid index description.
    IndexDescription,
    /// Invalid file path.
    BadFilePath,
    /// Invalid parameter name of index.
    ParameterName,
    /// The number of GPU resources and devices do not match.
    GpuResourcesMatch,
}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Native(e) => write!(fmt, "Native faiss error: {}", e.msg),
            Error::BadCast => fmt.write_str("Invalid index type cast"),
            Error::IndexDescription => fmt.write_str("Invalid index description"),
            Error::BadFilePath => fmt.write_str("Invalid file path"),
            Error::ParameterName => fmt.write_str("Invalid parameter name of index"),
            Error::GpuResourcesMatch => {
                fmt.write_str("Number of GPU resources and devices do not match")
            }
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        if let Error::Native(err) = self {
            Some(err)
        } else {
            None
        }
    }
}

/// An error derived from a native Faiss exception.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeError {
    /// The error code retrieved from the C API
    code: c_int,
    /// The exception's message
    msg: String,
}

impl NativeError {
    /// Getter for the internal error code.
    pub fn code(&self) -> c_int {
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
    pub(crate) fn from_last_error(code: c_int) -> Self {
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

impl From<NativeError> for Error {
    fn from(e: NativeError) -> Self {
        Error::Native(e)
    }
}
