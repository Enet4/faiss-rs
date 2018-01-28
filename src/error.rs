use std::fmt;
use std::error::Error as StdError;
use std::ffi::CStr;
use faiss_sys::*;

pub type Result<T> = ::std::result::Result<T, Error>;

pub type Error = NativeError;

#[derive(Debug, Clone, PartialEq)]
pub struct NativeError {
    code: i32,
    msg: String,
}

impl NativeError {
    pub fn from_last_error(code: i32) -> Self {
        unsafe {
            let cstr = CStr::from_ptr(faiss_get_last_error());
            let msg: String = cstr.to_string_lossy().into_owned();
            NativeError {
                code,
                msg,
            }
        }
    }
}

impl fmt::Display for NativeError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(&self.msg)
    }
}

impl StdError for NativeError {
    fn description(&self) -> &str { &self.msg }
}
